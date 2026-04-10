"""
src/data/splitter.py

Chronological (no-shuffle) train / validation / test split for time-series data.
Does NOT scale data. Use PipelineScaler separately to avoid leakage.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from prettytable import PrettyTable

logger = logging.getLogger(__name__)


class DataSplitter:
    """Chronological split by percentage.
    
    Does NOT scale data. Use PipelineScaler separately to avoid leakage.

    Args:
        train_pct: Fraction of data for training (default: 0.6).
        val_pct: Fraction of data for validation (default: 0.2).
        test_pct: Fraction of data for testing (default: 0.2).
    """

    def __init__(
        self,
        train_pct: float = 0.6,
        val_pct: float = 0.2,
        test_pct: float = 0.2,
    ) -> None:
        assert (
            abs(train_pct + val_pct + test_pct - 1.0) < 1e-6
        ), "Percentages must sum to 1"
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame chronologically into (train, val, test) by percentage."""
        N = len(df)
        train_end_idx = int(N * self.train_pct)
        val_end_idx = train_end_idx + int(N * self.val_pct)

        df_train = df.iloc[:train_end_idx]
        df_val = df.iloc[train_end_idx:val_end_idx]
        df_test = df.iloc[val_end_idx:]

        logger.info(
            "Split sizes — Train: %d  Val: %d  Test: %d",
            len(df_train),
            len(df_val),
            len(df_test),
        )
        return df_train, df_val, df_test

    def inspect_split(self, name: str, dfs: dict, n_head: int = 3, n_tail: int = 3):
        """
        Prints:
        - number of tickers
        - per-ticker shape
        - columns
        - head (first 3 rows)
        - tail (last 3 rows)
        """
        header = f"\n{'=' * 80}\n{name} | tickers={len(dfs)}\n{'=' * 80}"
        (logger.info if logger else print)(header)

        for ticker, df in dfs.items():
            msg = f"\n[{ticker}] shape={df.shape}"
            (logger.info if logger else print)(msg)

            # columns
            cols = list(df.columns)
            (logger.info if logger else print)(f"Columns ({len(cols)}): {cols}")

            # head
            head = df.head(n_head)
            tail = df.tail(n_tail)

            head_table = PrettyTable()
            head_table.title = f"{ticker} | FIRST {n_head} ROWS"
            head_table.field_names = ["index"] + cols

            for idx, row in head.iterrows():
                head_table.add_row([idx] + list(row.values))

            tail_table = PrettyTable()
            tail_table.title = f"{ticker} | LAST {n_tail} ROWS"
            tail_table.field_names = ["index"] + cols

            for idx, row in tail.iterrows():
                tail_table.add_row([idx] + list(row.values))

            output = f"{head_table}\n\n{tail_table}\n"
            (logger.info if logger else print)(output)


# ------------------------------------------------------------------
# Sliding window construction
# ------------------------------------------------------------------


def build_windows(
    features: np.ndarray,
    returns: np.ndarray,
    session_starts: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window (lookback, F) --> next-bar-return pairs, fully vectorized.
    Excludes windows that cross a session boundary.

    Args:
        features: [N, F] scaled feature matrix
        returns:  [N] target log returns
        session_starts: [N] bool array, True at first bar of session
        lookback: sequence length

    Returns:
        X: [M, lookback, F] input tensor
        y: [M] target log returns
    """
    N, F = features.shape

    # --- Compute session IDs ---
    session_ids = np.cumsum(session_starts)

    # --- Build valid mask ---
    valid_mask = np.ones(N, dtype=bool)
    valid_mask[:lookback] = False  # first windows too short

    # Vectorized check: no session start inside window
    # Create sliding indices
    idx_matrix = np.arange(N - lookback)[:, None] + np.arange(lookback)
    window_session_ids = session_ids[idx_matrix]
    # valid if all session IDs equal the first one
    window_valid = (window_session_ids == window_session_ids[:, 0][:, None]).all(axis=1)

    # Assign to valid_mask offset by lookback
    valid_mask[lookback:] = window_valid

    # --- Count valid windows ---
    M = valid_mask.sum()
    X = np.empty((M, lookback, F), dtype=np.float32)
    y = np.empty(M, dtype=np.float32)

    # --- Extract windows efficiently ---
    valid_indices = np.flatnonzero(valid_mask)
    window_starts = valid_indices - lookback

    for i, start in enumerate(window_starts):
        X[i] = features[start : start + lookback]
        y[i] = returns[valid_indices[i]]

    return X, y
