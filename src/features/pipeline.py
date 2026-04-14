# src/features/pipeline.py
from __future__ import annotations

"""features/pipeline.py — Refactored feature engineering pipeline."""

import logging
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

from .technical import compute_technical_features
from .statistical import compute_statistical_features
from .volume import compute_volume_features
from .cross_ticker import compute_cross_ticker_features
from .selector import FeatureSelector

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume", "log_return"}

# Features to lag and their depths
LAG_SPEC: Dict[str, List[int]] = {
    "log_return": [1, 2, 3],
    "rsi_14": [1, 2],
    "vwap_dev": [1],
    "rvol_20": [1],
    "corr_spy_20": [1],
    "macd": [1],
    "atr_14": [1],
    "bb_pct_b": [1],
    "stoch_k": [1],
    "zscore_20": [1],
}


class FeaturePipeline:
    """Build and select features for a single target ticker."""

    def __init__(
        self,
        target_ticker: str,
        peer_tickers: List[str],
        selector_kwargs: Optional[Dict] = None,
    ) -> None:
        self.target_ticker = target_ticker
        self.peer_tickers = peer_tickers
        self.selector = FeatureSelector(**(selector_kwargs or {}))
        self._feature_names: List[str] = []
        self._fitted = False

    def fit_transform(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Fit on training data; return (X, y, feature_names)."""
        self._validate(dfs)
        X, y, names = self._build_features(dfs)
        X_sel, sel_names = self.selector.fit_transform(X, y, names)
        self._feature_names = names
        self._fitted = True
        logger.info(
            "[%s] fit_transform: %d --> %d features, %d samples",
            self.target_ticker,
            len(names),
            len(sel_names),
            len(X_sel),
        )
        return X_sel, y, sel_names

    def transform(self, dfs: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply fitted pipeline to val/test data."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        self._validate(dfs)
        X, y, _ = self._build_features(dfs)
        X_sel, _ = self.selector.transform(X, self._feature_names)
        logger.info(
            "[%s] transform: %d samples, %d features",
            self.target_ticker,
            len(X_sel),
            X_sel.shape[1],
        )
        return X_sel, y

    @property
    def feature_names(self) -> List[str]:
        return list(self.selector.selected_features_)

    @property
    def n_features(self) -> int:
        return len(self.selector.selected_features_)

    def _validate(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """Validate input data structure."""
        if self.target_ticker not in dfs:
            raise KeyError(f"'{self.target_ticker}' not in dfs. Available: {list(dfs)}")
        df = dfs[self.target_ticker]
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"[{self.target_ticker}] Missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"[{self.target_ticker}] Index must be DatetimeIndex.")
        if not df.index.is_monotonic_increasing:
            raise ValueError(f"[{self.target_ticker}] Index not sorted ascending.")

    def _build_features(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Main feature building orchestrator.

        Steps:
        1. Extract base price features
        2. Compute specialized feature blocks
        3. Concatenate and deduplicate
        4. Add lag features
        5. Clean and align with target
        """
        df = dfs[self.target_ticker]

        # Step 1: Build feature block registry
        blocks = self._build_feature_blocks(df, dfs)

        # Step 2: Concatenate blocks with deduplication
        X, names = self._concat_blocks(blocks)

        # Step 3: Add lag features (vectorized)
        X, names = self._add_lags(X, names)

        # Step 4: Clean NaNs and align with target
        X, y = self._clean_and_align(X, df)

        return X, y, names

    def _build_feature_blocks(
        self, df: pd.DataFrame, dfs: Dict[str, pd.DataFrame]
    ) -> List[pd.DataFrame]:
        """
        Build registry of feature blocks from different sources.

        Each block is a DataFrame with consistent index and named columns.
        """
        blocks = [
            self._base_features(df),
            compute_technical_features(df),
            compute_statistical_features(df),
            compute_volume_features(df),
            compute_cross_ticker_features(self.target_ticker, dfs, self.peer_tickers),
        ]
        return blocks

    def _base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract base price-based features."""
        C, H, L, O = df["close"], df["high"], df["low"], df["open"]
        r = df["log_return"]
        prev_C = C.shift(1)

        # True range calculation
        tr = pd.concat([(H - L), (H - prev_C).abs(), (L - prev_C).abs()], axis=1).max(
            axis=1
        )

        base = pd.DataFrame(
            {
                "log_return": r,
                "mid_price": (H + L) / 2.0,
                "hl_ratio": (H - L) / (prev_C + 1e-10),
                "co_ratio": (C - O) / (prev_C + 1e-10),
                "true_range": tr / (prev_C + 1e-10),
                "intrabar_vol": (H - L) / (O + 1e-10),
            },
            index=df.index,
        ).astype(np.float32)

        return base

    def _concat_blocks(
        self, blocks: List[pd.DataFrame]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Concatenate feature blocks with deduplication.

        Handles:
        - Type conversion to float32
        - Within-block deduplication
        - Across-block deduplication (keeps first occurrence)
        """
        arrays = []
        names = []
        seen_names = set()

        for blk in blocks:
            # Remove duplicates within block (keep first)
            blk = blk.loc[:, ~blk.columns.duplicated()]

            # Convert to array
            arr = blk.values.astype(np.float32, copy=False)

            # Track names, skipping duplicates across blocks
            block_names = []
            for col in blk.columns:
                if col not in seen_names:
                    seen_names.add(col)
                    block_names.append(col)
                else:
                    # Skip this column in array too
                    continue

            if len(block_names) < len(blk.columns):
                # Some columns were duplicates - slice array
                keep_idx = [i for i, c in enumerate(blk.columns) if c in block_names]
                arr = arr[:, keep_idx]

            arrays.append(arr)
            names.extend(block_names)

        X = np.concatenate(arrays, axis=1)
        return X, names

    def _add_lags(
        self, X: np.ndarray, names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Add lagged features (vectorized implementation).

        Returns:
            Tuple of (augmented feature matrix, updated names)
        """
        name_idx = {n: i for i, n in enumerate(names)}
        arrays = []
        lag_names = []

        for col, lags in LAG_SPEC.items():
            if col not in name_idx:
                continue

            col_data = X[:, name_idx[col]]
            n_lags = len(lags)
            max_lag = max(lags)

            # Vectorized: create all lags at once
            lagged_matrix = np.zeros((len(col_data), n_lags), dtype=np.float32)

            for i, lag in enumerate(lags):
                if lag < len(col_data):
                    lagged_matrix[lag:, i] = col_data[:-lag]

            arrays.append(lagged_matrix)
            lag_names.extend([f"{col}_lag{lag}" for lag in lags])

        if not arrays:
            return X, names

        lag_X = np.concatenate(arrays, axis=1)
        X = np.concatenate([X, lag_X], axis=1)
        names = names + lag_names

        return X, names

    def _clean_and_align(
        self, X: np.ndarray, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clean non-finite values and align with target variable.

        Target is next-bar return (shifted by -1).
        Drops rows where target or any feature is invalid.
        """
        r = df["log_return"]

        # Replace NaN/Inf in features
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Target: next bar return
        y = r.shift(-1).values.astype(np.float32)

        # Valid rows: finite target and finite features
        valid = np.isfinite(y) & np.isfinite(X).all(axis=1)

        if (~valid).any():
            n_dropped = (~valid).sum()
            logger.debug(
                "[%s] Dropping %d non-finite rows (%.2f%%)",
                self.target_ticker,
                n_dropped,
                100 * n_dropped / len(y),
            )
            X = X[valid]
            y = y[valid]

        return X, y


# from __future__ import annotations

# """features/pipeline.py — Feature engineering pipeline for LSTM stock prediction."""


# import logging
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd

# from .technical import compute_technical_features
# from .statistical import compute_statistical_features
# from .volume import compute_volume_features
# from .cross_ticker import compute_cross_ticker_features
# from .selector import FeatureSelector

# logger = logging.getLogger(__name__)

# REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume", "log_return"}

# # Features to lag and their depths
# LAG_SPEC: Dict[str, List[int]] = {
#     "log_return": [1, 2, 3],
#     "rsi_14": [1, 2],
#     "vwap_dev": [1],
#     "rvol_20": [1],
#     "corr_spy_20": [1],
#     "macd": [1],
#     "atr_14": [1],
#     "bb_pct_b": [1],
#     "stoch_k": [1],
#     "zscore_20": [1],
# }


# class FeaturePipeline:
#     """Build and select features for a single target ticker.

#     Lifecycle:
#         X_train, y_train, names = pipeline.fit_transform(dfs_train)
#         X_val,   y_val          = pipeline.transform(dfs_val)
#         X_test,  y_test         = pipeline.transform(dfs_test)

#     Args:
#         target_ticker:  Stock to predict.
#         peer_tickers:   Pre-selected correlated peers (from SymbolUniverseBuilder).
#         selector_kwargs: Forwarded to FeatureSelector.
#     """

#     def __init__(
#         self,
#         target_ticker: str,
#         peer_tickers: List[str],
#         selector_kwargs: Optional[Dict] = None,
#     ) -> None:
#         self.target_ticker = target_ticker
#         self.peer_tickers = peer_tickers
#         self.selector = FeatureSelector(**(selector_kwargs or {}))
#         self._feature_names: List[str] = []
#         self._fitted = False

#     # ── Public API ────────────────────────────────────────────────────────────

#     def fit_transform(
#         self, dfs: Dict[str, pd.DataFrame]
#     ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
#         """Fit on training data; return (X, y, feature_names)."""
#         self._validate(dfs)
#         X, y, names = self._build_features(dfs)
#         X_sel, sel_names = self.selector.fit_transform(X, y, names)
#         self._feature_names = names  # full names — needed by transform()
#         self._fitted = True
#         logger.info(
#             "[%s] fit_transform: %d --> %d features, %d samples",
#             self.target_ticker,
#             len(names),
#             len(sel_names),
#             len(X_sel),
#         )
#         return X_sel, y, sel_names

#     def transform(self, dfs: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
#         """Apply fitted pipeline to val/test data; return (X, y)."""
#         if not self._fitted:
#             raise RuntimeError("Call fit_transform() before transform().")
#         self._validate(dfs)
#         X, y, _ = self._build_features(dfs)
#         X_sel, _ = self.selector.transform(X, self._feature_names)
#         logger.info(
#             "[%s] transform: %d samples, %d features",
#             self.target_ticker,
#             len(X_sel),
#             X_sel.shape[1],
#         )
#         return X_sel, y

#     @property
#     def feature_names(self) -> List[str]:
#         return list(self.selector.selected_features_)

#     @property
#     def n_features(self) -> int:
#         return len(self.selector.selected_features_)

#     # ── Internals ─────────────────────────────────────────────────────────────

#     def _validate(self, dfs: Dict[str, pd.DataFrame]) -> None:
#         if self.target_ticker not in dfs:
#             raise KeyError(f"'{self.target_ticker}' not in dfs. Available: {list(dfs)}")
#         df = dfs[self.target_ticker]
#         missing = REQUIRED_COLUMNS - set(df.columns)
#         if missing:
#             raise ValueError(f"[{self.target_ticker}] Missing columns: {missing}")
#         if not isinstance(df.index, pd.DatetimeIndex):
#             raise TypeError(f"[{self.target_ticker}] Index must be DatetimeIndex.")
#         if not df.index.is_monotonic_increasing:
#             raise ValueError(f"[{self.target_ticker}] Index not sorted ascending.")

#     def _build_features(
#         self, dfs: Dict[str, pd.DataFrame]
#     ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
#         """Build complete feature matrix."""
#         df = dfs[self.target_ticker]
#         C, H, L, O = df["close"], df["high"], df["low"], df["open"]
#         r = df["log_return"]
#         prev_C = C.shift(1)

#         # Base price features
#         tr = pd.concat([(H - L), (H - prev_C).abs(), (L - prev_C).abs()], axis=1).max(
#             axis=1
#         )
#         base = (
#             pd.DataFrame(
#                 {
#                     "log_return": r,
#                     "mid_price": (H + L) / 2.0,
#                     "hl_ratio": (H - L) / (prev_C + 1e-10),
#                     "co_ratio": (C - O) / (prev_C + 1e-10),
#                     "true_range": tr / (prev_C + 1e-10),
#                     "intrabar_vol": (H - L) / (O + 1e-10),
#                 },
#                 index=df.index,
#             )
#             .astype(np.float32)
#             .fillna(0.0)
#         )

#         # Feature blocks
#         blocks = [
#             base,
#             compute_technical_features(df),
#             compute_statistical_features(df),
#             compute_volume_features(df),
#             compute_cross_ticker_features(self.target_ticker, dfs, self.peer_tickers),
#         ]

#         # Concatenate
#         arrays = []
#         names = []
#         for blk in blocks:
#             blk = blk.loc[:, ~blk.columns.duplicated()]
#             arrays.append(blk.values.astype(np.float32, copy=False))
#             names.extend(blk.columns.tolist())

#         X = np.concatenate(arrays, axis=1)

#         # Add lag features (vectorized)
#         lag_X, lag_names = _build_lags(X, names)
#         if lag_X.shape[1] > 0:
#             X = np.concatenate([X, lag_X], axis=1)
#             names = names + lag_names

#         # Deduplicate
#         seen = set()
#         keep_idx = []
#         keep_names = []
#         for i, n in enumerate(names):
#             if n not in seen:
#                 seen.add(n)
#                 keep_idx.append(i)
#                 keep_names.append(n)
#         X = X[:, keep_idx]

#         # Clean non-finite values
#         np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

#         # Target alignment (next bar return)
#         y = r.shift(-1).values.astype(np.float32)  # Predict next bar
#         valid = np.isfinite(y) & np.isfinite(X).all(axis=1)

#         if (~valid).any():
#             X = X[valid]
#             y = y[valid]

#         return X, y, keep_names

#     # def _build_features(
#     #     self, dfs: Dict[str, pd.DataFrame]
#     # ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
#     #     df = dfs[self.target_ticker]
#     #     C, H, L, O = df["close"], df["high"], df["low"], df["open"]
#     #     r = df["log_return"]
#     #     prev_C = C.shift(1)

#     #     # ── Block 1: Base price features ─────────────────────────────────────
#     #     tr = pd.concat([(H - L), (H - prev_C).abs(), (L - prev_C).abs()], axis=1).max(
#     #         axis=1
#     #     )
#     #     base = pd.DataFrame(
#     #         {
#     #             "log_return": r,
#     #             "mid_price": (H + L) / 2.0,
#     #             "hl_ratio": (H - L) / (prev_C + 1e-10),
#     #             "co_ratio": (C - O) / (prev_C + 1e-10),
#     #             "ho_ratio": (H - O) / (prev_C + 1e-10),
#     #             "lc_ratio": (C - L) / (prev_C + 1e-10),
#     #             "gap": (O - prev_C) / (prev_C + 1e-10),
#     #             "true_range": tr / (prev_C + 1e-10),
#     #             "intrabar_vol": (H - L) / (O + 1e-10),
#     #             "cum_return_session": _session_cum_return(df, C),
#     #         },
#     #         index=df.index,
#     #     ).astype(np.float32)

#     #     # ── Blocks 2-5: computed feature modules ─────────────────────────────
#     #     blocks = [
#     #         base,
#     #         compute_technical_features(df),
#     #         compute_statistical_features(df),
#     #         compute_volume_features(df),
#     #         compute_cross_ticker_features(self.target_ticker, dfs, self.peer_tickers),
#     #     ]

#     #     # ── Concatenate into one matrix ───────────────────────────────────────
#     #     arrays, names = [], []
#     #     for blk in blocks:
#     #         blk = blk.loc[:, ~blk.columns.duplicated()]  # drop within-block dupes
#     #         arrays.append(blk.values.astype(np.float32, copy=False))
#     #         names.extend(blk.columns.tolist())

#     #     X = np.concatenate(arrays, axis=1)

#     #     # ── Block 6: Lag features ─────────────────────────────────────────────
#     #     lag_X, lag_names = _build_lags(X, names)
#     #     if lag_X.shape[1]:
#     #         X = np.concatenate([X, lag_X], axis=1)
#     #         names = names + lag_names

#     #     # ── Deduplicate across blocks ─────────────────────────────────────────
#     #     seen, keep_idx, keep_names = set(), [], []
#     #     for i, n in enumerate(names):
#     #         if n not in seen:
#     #             seen.add(n)
#     #             keep_idx.append(i)
#     #             keep_names.append(n)
#     #     X = X[:, keep_idx]

#     #     # ── Clean non-finite values ───────────────────────────────────────────
#     #     np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

#     #     # ── Drop rows where target is non-finite ──────────────────────────────
#     #     y = r.values.astype(np.float32)
#     #     valid = np.isfinite(y)
#     #     if (~valid).any():
#     #         logger.info(
#     #             "[%s] Dropping %d non-finite target rows.",
#     #             self.target_ticker,
#     #             (~valid).sum(),
#     #         )
#     #         X, y = X[valid], y[valid]

#     #     logger.info(
#     #         "[%s] Feature matrix: %d × %d", self.target_ticker, X.shape[0], X.shape[1]
#     #     )
#     #     return X, y, keep_names


# # ── Module-level helpers ──────────────────────────────────────────────────────


# def _session_cum_return(df: pd.DataFrame, close: pd.Series) -> pd.Series:
#     """Within-session cumulative return (resets each day)."""
#     if "session_start" in df.columns:
#         session_ids = df["session_start"].astype(bool).cumsum()
#     else:
#         session_ids = pd.Series(
#             pd.factorize(pd.to_datetime(close.index).date)[0],
#             index=close.index,
#         )
#     first_close = close.groupby(session_ids).transform("first")
#     return (close / first_close - 1.0).fillna(0.0).astype(np.float32)


# def _build_lags(X: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str]]:
#     """
#     Build lag features via vectorized slicing.

#     Args:
#         X: Feature matrix [N, F]
#         names: List of feature names corresponding to columns of X

#     Returns:
#         Tuple of (lagged_features, lag_feature_names)
#     """
#     name_idx = {n: i for i, n in enumerate(names)}
#     arrays = []
#     lag_names = []

#     for col, lags in LAG_SPEC.items():
#         if col not in name_idx:
#             continue

#         col_data = X[:, name_idx[col]]
#         max_lag = max(lags)

#         # PERFORMANCE FIX: Vectorized lag construction
#         # Create matrix [N, n_lags] instead of individual arrays
#         n_lags = len(lags)
#         lagged_matrix = np.zeros((len(col_data), n_lags), dtype=np.float32)

#         # Fill all lags at once using slicing
#         for i, lag in enumerate(lags):
#             if lag < len(col_data):
#                 lagged_matrix[lag:, i] = col_data[:-lag]
#             # Else: remains zeros (acceptable for edge cases)

#         arrays.append(lagged_matrix)
#         lag_names.extend([f"{col}_lag{lag}" for lag in lags])

#     if not arrays:
#         return np.empty((X.shape[0], 0), dtype=np.float32), []

#     # Single concatenation instead of multiple appends
#     return np.concatenate(arrays, axis=1).astype(np.float32), lag_names


# # def _build_lags(X: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str]]:
# #     """
# #     Build lag features via vectorized slicing.

# #     Args:
# #         X: Feature matrix [N, F]
# #         names: List of feature names corresponding to columns of X

# #     Returns:
# #         Tuple of (lagged_features, lag_feature_names)
# #     """
# #     name_idx = {n: i for i, n in enumerate(names)}
# #     arrays = []
# #     lag_names = []

# #     for col, lags in LAG_SPEC.items():
# #         if col not in name_idx:
# #             continue

# #         col_data = X[:, name_idx[col]]
# #         max_lag = max(lags)

# #         # Create matrix [N, n_lags] instead of individual arrays
# #         n_lags = len(lags)
# #         lagged_matrix = np.zeros((len(col_data), n_lags), dtype=np.float32)

# #         # Fill all lags at once using slicing
# #         for i, lag in enumerate(lags):
# #             if lag < len(col_data):
# #                 lagged_matrix[lag:, i] = col_data[:-lag]
# #             # Else: remains zeros (acceptable for edge cases)

# #         arrays.append(lagged_matrix)
# #         lag_names.extend([f"{col}_lag{lag}" for lag in lags])

# #     if not arrays:
# #         return np.empty((X.shape[0], 0), dtype=np.float32), []

# #     # Single concatenation instead of multiple appends
# #     return np.concatenate(arrays, axis=1).astype(np.float32), lag_names
