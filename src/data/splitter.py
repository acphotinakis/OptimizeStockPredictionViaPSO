"""
src/data/splitter.py

Chronological (no-shuffle) train / validation / test split for time-series data.
Guarantees zero lookahead: scaler parameters are fitted only on the training set.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler

logger = logging.getLogger(__name__)

# Features whose values are bounded by construction — use MinMaxScaler
BOUNDED_FEATURES = {
    "rsi_14",
    "rsi_30",
    "stoch_k",
    "stoch_d",
    "williams_r",
    "bb_pct_b",
    "mfi_14",
    "mfi_30",
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
}


class DataSplitter:
    """Chronological split + per-feature normalization.

    Args:
        train_end: Last date of the training period (inclusive), ISO string.
        val_end: Last date of the validation period (inclusive), ISO string.
        bounded_features: Feature names that should use MinMaxScaler [0, 1].
    """

    def __init__(
        self,
        train_end: str = "2022-01-03",
        val_end: str = "2023-01-03",
        bounded_features: set | None = None,
    ) -> None:
        self.train_end = train_end
        self.val_end = val_end
        self.bounded_features = bounded_features or BOUNDED_FEATURES
        self._scalers: dict = {}
        self._feature_names: list = []

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame chronologically into (train, val, test).

        Args:
            df: Aligned feature DataFrame with DatetimeIndex.

        Returns:
            Tuple of (df_train, df_val, df_test).
        """
        df_train = df.loc[: self.train_end]
        df_val = df.loc[self.train_end : self.val_end].iloc[1:]
        df_test = df.loc[self.val_end :].iloc[1:]

        logger.info(
            "Split sizes — Train: %d  Val: %d  Test: %d",
            len(df_train),
            len(df_val),
            len(df_test),
        )
        return df_train, df_val, df_test

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def fit_transform(self, X_train: np.ndarray, feature_names: list) -> np.ndarray:
        self._feature_names = feature_names
        X_out = np.empty_like(X_train, dtype=np.float32)
        for j, name in enumerate(feature_names):
            col = X_train[:, j].reshape(-1, 1)
            scaler = (
                MinMaxScaler(feature_range=(0, 1))
                if name in self.bounded_features
                else RobustScaler()
            )
            X_out[:, j] = scaler.fit_transform(col).ravel()
            self._scalers[name] = scaler
        return X_out

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_out = np.empty_like(X, dtype=np.float32)
        for j, name in enumerate(self._feature_names):
            X_out[:, j] = self._scalers[name].transform(X[:, j].reshape(-1, 1)).ravel()
        return X_out


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
