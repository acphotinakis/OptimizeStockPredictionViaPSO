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
    """Chronological split by fixed date boundaries (preferred) or percentages.

    Fixed date splits are strongly preferred — percentage splits shift when new
    data is appended, causing silent train/test contamination.  Pass ``train_end``
    and ``val_end`` to use date-based splitting; fall back to percentages only
    when no dates are provided.

    Args:
        train_end: ISO date string for the end of the training period (inclusive).
                   E.g. ``"2024-04-05"``.  If supplied, ``val_end`` is also required.
        val_end:   ISO date string for the end of the validation period (inclusive).
        train_pct: Fraction of data for training (default: 0.6).  Used only when
                   ``train_end`` / ``val_end`` are not provided.
        val_pct:   Fraction of data for validation (default: 0.2).
        test_pct:  Fraction of data for testing (default: 0.2).
    """

    def __init__(
        self,
        train_end: str | None = None,
        val_end: str | None = None,
        train_pct: float = 0.6,
        val_pct: float = 0.2,
        test_pct: float = 0.2,
    ) -> None:
        self._use_dates = train_end is not None and val_end is not None
        if self._use_dates:
            self.train_end = pd.Timestamp(train_end)
            self.val_end = pd.Timestamp(val_end)
        else:
            assert (
                abs(train_pct + val_pct + test_pct - 1.0) < 1e-6
            ), "Percentages must sum to 1"
            self.train_pct = train_pct
            self.val_pct = val_pct
            self.test_pct = test_pct

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame chronologically into (train, val, test)."""
        if self._use_dates:
            return self._split_by_dates(df)
        return self._split_by_pct(df)

    def _split_by_dates(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fixed date-boundary split — reproducible even when data is updated."""
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            train_end = (
                self.train_end.tz_localize(idx.tz)
                if self.train_end.tz is None
                else self.train_end.tz_convert(idx.tz)
            )
            val_end = (
                self.val_end.tz_localize(idx.tz)
                if self.val_end.tz is None
                else self.val_end.tz_convert(idx.tz)
            )
        else:
            train_end = self.train_end.tz_localize(None)
            val_end = self.val_end.tz_localize(None)

        df_train = df.loc[idx <= train_end]
        df_val = df.loc[(idx > train_end) & (idx <= val_end)]
        df_test = df.loc[idx > val_end]

        logger.info(
            "Date split — Train: %d (%s–%s)  Val: %d (%s–%s)  Test: %d (%s–%s)",
            len(df_train),
            df_train.index.min() if len(df_train) else "N/A",
            df_train.index.max() if len(df_train) else "N/A",
            len(df_val),
            df_val.index.min() if len(df_val) else "N/A",
            df_val.index.max() if len(df_val) else "N/A",
            len(df_test),
            df_test.index.min() if len(df_test) else "N/A",
            df_test.index.max() if len(df_test) else "N/A",
        )
        return df_train, df_val, df_test

    def _split_by_pct(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Percentage-based split — only use when fixed dates are unavailable."""
        N = len(df)
        train_end_idx = int(N * self.train_pct)
        val_end_idx = train_end_idx + int(N * self.val_pct)

        df_train = df.iloc[:train_end_idx]
        df_val = df.iloc[train_end_idx:val_end_idx]
        df_test = df.iloc[val_end_idx:]

        logger.info(
            "Pct split — Train: %d  Val: %d  Test: %d",
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
    # A window ending at index i predicts the NEXT bar at index i+1.
    # So we need i+1 < N (i.e. i < N-1) and the window [i-lookback+1 .. i]
    # must not cross a session boundary.
    valid_mask = np.zeros(N, dtype=bool)

    # Windows require at least `lookback` bars before them AND a next bar after.
    # We mark index i as valid if:
    #   - i >= lookback (enough history)
    #   - i + 1 < N    (next bar exists for the target)
    if N > lookback:
        valid_mask[lookback : N - 1] = True

    # Vectorized check: no session start inside window [i-lookback .. i-1]
    # idx_matrix rows correspond to positions lookback..N-2
    n_candidates = N - 1 - lookback  # number of candidate end-positions
    if n_candidates > 0:
        end_positions = np.arange(lookback, N - 1)  # shape (n_candidates,)
        idx_matrix = (
            end_positions[:, None] - lookback + np.arange(lookback)
        )  # (n_candidates, lookback)
        window_session_ids = session_ids[idx_matrix]
        window_valid = (window_session_ids == window_session_ids[:, 0:1]).all(axis=1)
        valid_mask[lookback : N - 1] = window_valid

    # --- Count valid windows ---
    M = valid_mask.sum()
    X = np.empty((M, lookback, F), dtype=np.float32)
    y = np.empty(M, dtype=np.float32)

    # --- Extract windows efficiently ---
    # valid_indices[i] = end of window; target = returns[valid_indices[i] + 1]
    valid_indices = np.flatnonzero(valid_mask)
    window_starts = valid_indices - lookback

    for i, start in enumerate(window_starts):
        X[i] = features[start : start + lookback]
        y[i] = returns[valid_indices[i] + 1]  # predict NEXT bar, not current

    return X, y
