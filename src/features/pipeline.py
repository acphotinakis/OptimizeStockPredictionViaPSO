"""
src/features/pipeline.py

Optimized feature engineering pipeline (NumPy-first, low-copy, vectorized).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .technical import compute_technical_features
from .statistical import compute_statistical_features
from .volume import compute_volume_features
from .cross_ticker import compute_cross_ticker_features
from .selector import FeatureSelector

logger = logging.getLogger(__name__)

# Lag configuration
LAG_SOURCES = [
    "log_return",
    "rsi_14",
    "vwap_dev",
    "rvol_20",
    "corr_spy_20",
    "macd",
    "atr_14",
    "bb_pct_b",
    "stoch_k",
    "zscore_20",
]

LAG_DEPTHS = {
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
    def __init__(
        self,
        target_ticker: str,
        universe_tickers: List[str],
        selector_kwargs: Optional[dict] = None,
    ) -> None:
        self.target_ticker = target_ticker
        self.universe_tickers = universe_tickers
        self.selector = FeatureSelector(**(selector_kwargs or {}))
        self._feature_names_full: List[str] = []
        self._feature_names_selected: List[str] = []
        self._peer_tickers: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        dfs_train: Dict[str, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X_full, y, names = self._compute_features(dfs_train, fit=True)
        self._feature_names_full = names

        X_sel, sel_names = self.selector.fit_transform(X_full, y, names)
        self._feature_names_selected = sel_names
        self._fitted = True

        logger.info(
            "Pipeline fit: %d raw → %d selected features (target=%s)",
            len(names),
            len(sel_names),
            self.target_ticker,
        )
        return X_sel, y, sel_names

    def transform(
        self,
        dfs: Dict[str, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")

        X_full, y, _ = self._compute_features(dfs, fit=False)
        X_sel, _ = self.selector.transform(X_full, self._feature_names_full)
        return X_sel, y

    @property
    def n_features(self) -> int:
        return len(self._feature_names_selected)

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names_selected)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        dfs: Dict[str, pd.DataFrame],
        fit: bool,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:

        df_target = dfs[self.target_ticker]

        # --- Feature blocks ---
        tech = compute_technical_features(df_target)
        stat = compute_statistical_features(df_target)
        vol = compute_volume_features(df_target)

        if fit:
            cross, peer_tickers = compute_cross_ticker_features(
                self.target_ticker,
                dfs,
                peer_tickers=None,
                return_peer_tickers=True,
            )
            self._peer_tickers = peer_tickers
        else:
            cross = compute_cross_ticker_features(
                self.target_ticker,
                dfs,
                peer_tickers=self._peer_tickers,
                return_peer_tickers=False,
            )

        # --- Base features ---
        base = pd.DataFrame(index=df_target.index, dtype=np.float32)

        C = df_target["close"]
        H = df_target["high"]
        L = df_target["low"]
        O = df_target["open"]
        r = df_target["log_return"]

        prev_close = C.shift(1)

        base["log_return"] = r
        base["mid_price"] = (H + L) / 2
        base["hl_ratio"] = (H - L) / (prev_close + 1e-10)
        base["co_ratio"] = (C - O) / (prev_close + 1e-10)
        base["ho_ratio"] = (H - O) / (prev_close + 1e-10)
        base["lc_ratio"] = (C - L) / (prev_close + 1e-10)
        base["gap"] = (O - prev_close) / (prev_close + 1e-10)

        tr = pd.concat(
            [(H - L), (H - prev_close).abs(), (L - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        base["true_range"] = tr / (prev_close + 1e-10)

        # Session cumulative return (optimized)
        # session_flag = df_target.get(
        #     "session_start", pd.Series(False, index=df_target.index)
        # ).cumsum()

        # first_close = C.groupby(session_flag).transform("first")
        # base["cum_return_session"] = (C / first_close - 1).fillna(0.0)
        # session_start = df_target.get("session_start", pd.Series(False, index=df_target.index)).values
        # session_ids = np.cumsum(session_start)
        # first_close = np.zeros_like(C.values, dtype=np.float32)
        # for sid in np.unique(session_ids):
        #     mask = session_ids == sid
        #     first_close[mask] = C.values[mask][0]
        # base["cum_return_session"] = (C.values / first_close - 1).astype(np.float32)
        session_flag = df_target.get(
            "session_start", pd.Series(False, index=df_target.index)
        ).cumsum()

        # Only compute first_close where session has at least one bar
        first_close = pd.Series(np.nan, index=df_target.index, dtype=np.float32)
        for session_id in session_flag.unique():
            mask = session_flag == session_id
            if mask.any():
                first_close[mask] = C.loc[mask].iloc[0]
            else:
                # Empty session — assign NaN or 0
                first_close[mask] = np.nan

        base["cum_return_session"] = (C / first_close - 1).fillna(0.0)

        base["intrabar_vol"] = (H - L) / (O + 1e-10)

        # --- NumPy feature assembly ---
        feature_blocks = []
        feature_names = []

        def _append_block(df_block: pd.DataFrame):
            if df_block is None or df_block.empty:
                return
            vals = df_block.values.astype(np.float32, copy=False)
            feature_blocks.append(vals)
            feature_names.extend(df_block.columns.tolist())

        _append_block(base)
        _append_block(tech)
        _append_block(stat)
        _append_block(vol)
        _append_block(cross)

        X = np.concatenate(feature_blocks, axis=1)

        # --- Lag features (NumPy) ---
        lag_X, lag_names = self._build_lag_features_np(X, feature_names)

        if lag_X.shape[1] > 0:
            X = np.concatenate([X, lag_X], axis=1)
            feature_names.extend(lag_names)

        # --- Clean invalid values ---
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Target ---
        y = r.values.astype(np.float32)
        valid_mask = ~np.isnan(y) & ~np.isinf(y)

        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            logger.warning(
                "Removing %d rows with NaN/inf in target variable (%.2f%%)",
                n_invalid,
                100 * n_invalid / len(y),
            )
            X = X[valid_mask]
            y = y[valid_mask]

        return X.astype(np.float32, copy=False), y, feature_names

    # ------------------------------------------------------------------
    # Lag features (NumPy)
    # ------------------------------------------------------------------

    # @staticmethod
    # def _build_lag_features_np(
    #     X: np.ndarray,
    #     feature_names: List[str],
    # ) -> Tuple[np.ndarray, List[str]]:
    #     name_to_idx = {n: i for i, n in enumerate(feature_names)}

    #     lag_arrays = []
    #     lag_names = []

    #     for col, lags in LAG_DEPTHS.items():
    #         if col not in name_to_idx:
    #             continue

    #         col_idx = name_to_idx[col]
    #         base_col = X[:, col_idx]

    #         for lag in lags:
    #             lagged = np.roll(base_col, lag)
    #             lagged[:lag] = 0.0
    #             lag_arrays.append(lagged)
    #             lag_names.append(f"{col}_lag{lag}")

    #     if not lag_arrays:
    #         return np.empty((X.shape[0], 0), dtype=np.float32), []

    #     return np.stack(lag_arrays, axis=1).astype(np.float32), lag_names
    @staticmethod
    def _build_lag_features_np(X, feature_names):
        lag_arrays = []
        lag_names = []
        name_to_idx = {n: i for i, n in enumerate(feature_names)}

        for col, lags in LAG_DEPTHS.items():
            if col not in name_to_idx:
                continue
            col_data = X[:, name_to_idx[col]]
            for lag in lags:
                lagged = np.zeros_like(col_data)
                lagged[lag:] = col_data[:-lag]
                lag_arrays.append(lagged)
                lag_names.append(f"{col}_lag{lag}")

        if lag_arrays:
            return np.stack(lag_arrays, axis=1).astype(np.float32), lag_names
        return np.empty((X.shape[0], 0), dtype=np.float32), []


# """
# src/features/pipeline.py

# Unified feature engineering pipeline.
# Combines all feature categories, adds lag features, and exposes
# fit_transform / transform for train/val/test consistency.
# """

# from __future__ import annotations

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

# # Top features for lag generation (F106–F117)
# LAG_SOURCES = [
#     "log_return",
#     "rsi_14",
#     "vwap_dev",
#     "rvol_20",
#     "corr_spy_20",
#     "macd",
#     "atr_14",
#     "bb_pct_b",
#     "stoch_k",
#     "zscore_20",
# ]
# LAG_DEPTHS = {
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
#     """End-to-end feature engineering and selection pipeline.

#     Args:
#         target_ticker: Ticker for which to build the feature matrix.
#         universe_tickers: All tickers in the universe (for cross-ticker features).
#         selector_kwargs: Keyword arguments forwarded to FeatureSelector.
#     """

#     def __init__(
#         self,
#         target_ticker: str,
#         universe_tickers: List[str],
#         selector_kwargs: Optional[dict] = None,
#     ) -> None:
#         self.target_ticker = target_ticker
#         self.universe_tickers = universe_tickers
#         self.selector = FeatureSelector(**(selector_kwargs or {}))
#         self._feature_names_full: List[str] = []
#         self._feature_names_selected: List[str] = []
#         self._peer_tickers: List[str] = []
#         self._fitted = False

#     # ------------------------------------------------------------------
#     # Public interface
#     # ------------------------------------------------------------------

#     def fit_transform(
#         self,
#         dfs_train: Dict[str, pd.DataFrame],
#     ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
#         """Build features for the training set and fit the selector.

#         Args:
#             dfs_train: Dict ticker → cleaned train DataFrame.

#         Returns:
#             (X_selected, y, selected_feature_names)
#             where X has shape [N_train_windows, F_selected]
#             and y has shape [N_train_windows].
#         """
#         X_full, y, names = self._compute_features(dfs_train, fit=True)
#         self._feature_names_full = names
#         X_sel, sel_names = self.selector.fit_transform(X_full, y, names)
#         self._feature_names_selected = sel_names
#         self._fitted = True
#         logger.info(
#             "Pipeline fit: %d raw → %d selected features (target=%s)",
#             len(names),
#             len(sel_names),
#             self.target_ticker,
#         )
#         return X_sel, y, sel_names

#     def transform(
#         self,
#         dfs: Dict[str, pd.DataFrame],
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """Apply fitted pipeline to val/test data.

#         Args:
#             dfs: Dict ticker → cleaned DataFrame (val or test split).

#         Returns:
#             (X_selected, y)
#         """
#         if not self._fitted:
#             raise RuntimeError("Call fit_transform before transform.")
#         X_full, y, _ = self._compute_features(dfs, fit=False)
#         X_sel, _ = self.selector.transform(X_full, self._feature_names_full)
#         return X_sel, y

#     @property
#     def n_features(self) -> int:
#         return len(self._feature_names_selected)

#     @property
#     def feature_names(self) -> List[str]:
#         return list(self._feature_names_selected)

#     # ------------------------------------------------------------------
#     # Private helpers
#     # ------------------------------------------------------------------

#     def _compute_features(
#         self,
#         dfs: Dict[str, pd.DataFrame],
#         fit: bool,
#     ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
#         """Compute the full (unselected) flat feature matrix."""
#         df_target = dfs[self.target_ticker]

#         # --- Individual feature blocks ---
#         tech = compute_technical_features(df_target)
#         stat = compute_statistical_features(df_target)
#         vol = compute_volume_features(df_target)

#         if fit:
#             # Compute cross-ticker features and store peer tickers
#             cross, peer_tickers = compute_cross_ticker_features(
#                 self.target_ticker,
#                 dfs,
#                 peer_tickers=None,
#                 return_peer_tickers=True,
#             )
#             self._peer_tickers = peer_tickers
#         else:
#             # Use stored peer tickers
#             cross = compute_cross_ticker_features(
#                 self.target_ticker,
#                 dfs,
#                 peer_tickers=self._peer_tickers,
#                 return_peer_tickers=False,
#             )

#         # --- Price/return base features ---
#         base = pd.DataFrame(index=df_target.index)
#         C = df_target["close"]
#         H, L, O = df_target["high"], df_target["low"], df_target["open"]
#         r = df_target["log_return"]
#         base["log_return"] = r
#         base["mid_price"] = (H + L) / 2
#         base["hl_ratio"] = (H - L) / (C.shift(1) + 1e-10)
#         base["co_ratio"] = (C - O) / (C.shift(1) + 1e-10)
#         base["ho_ratio"] = (H - O) / (C.shift(1) + 1e-10)
#         base["lc_ratio"] = (C - L) / (C.shift(1) + 1e-10)
#         base["gap"] = (O - C.shift(1)) / (C.shift(1) + 1e-10)
#         tr = pd.concat(
#             [(H - L), (H - C.shift(1)).abs(), (L - C.shift(1)).abs()], axis=1
#         ).max(axis=1)
#         base["true_range"] = tr / (C.shift(1) + 1e-10)

#         session_cum_return = (
#             C
#             / df_target.groupby(
#                 (
#                     df_target.get(
#                         "session_start", pd.Series(False, index=df_target.index)
#                     ).cumsum()
#                 )
#             )["close"].transform("first")
#             - 1
#         )
#         base["cum_return_session"] = session_cum_return.fillna(0.0)
#         base["intrabar_vol"] = (H - L) / (O + 1e-10)

#         # --- Concatenate all blocks ---
#         # all_frames = [base, tech, stat, vol, cross]
#         # combined = pd.concat(all_frames, axis=1)
#         # combined = combined.loc[:, ~combined.columns.duplicated()]
#         feature_blocks = []
#         feature_names = []

#         def _append_block(df_block: pd.DataFrame):
#             if df_block is None or df_block.empty:
#                 return
#             vals = df_block.values.astype(np.float32, copy=False)
#             feature_blocks.append(vals)
#             feature_names.extend(df_block.columns.tolist())

#         _append_block(base)
#         _append_block(tech)
#         _append_block(stat)
#         _append_block(vol)
#         _append_block(cross)

#         X = np.concatenate(feature_blocks, axis=1)

#         # --- Lag features ---
#         lag_df = self._build_lag_features(X)
#         combined = pd.concat([combined, lag_df], axis=1)
#         combined = combined.fillna(0.0).replace([np.inf, -np.inf], 0.0)

#         # Clean target variable: remove NaN and inf
#         valid_mask = ~r.isna() & ~np.isinf(r)
#         if not valid_mask.all():
#             n_invalid = (~valid_mask).sum()
#             logger.warning(
#                 "Removing %d rows with NaN/inf in target variable (%.2f%%)",
#                 n_invalid,
#                 100 * n_invalid / len(r),
#             )
#             combined = combined[valid_mask]
#             r = r[valid_mask]

#         feature_names = list(combined.columns)
#         # Phase 1: Data Quantization - use float32 throughout
#         X = combined.values.astype(np.float32)
#         y = r.values.astype(np.float32)

#         return X, y, feature_names

#     # @staticmethod
#     # def _build_lag_features(df: pd.DataFrame, cols, depths) -> np.ndarray:
#     #     """Create lag features for selected columns."""
#     #     # lag_frames = {}
#     #     # for col, lags in LAG_DEPTHS.items():
#     #     #     if col not in df.columns:
#     #     #         continue
#     #     #     for lag in lags:
#     #     #         lag_frames[f"{col}_lag{lag}"] = df[col].shift(lag)
#     #     # return pd.DataFrame(lag_frames, index=df.index).fillna(0.0)
#     #     arr = df[cols].values
#     #     lagged = []

#     #     for i, col in enumerate(cols):
#     #         for lag in depths[col]:
#     #             lagged_col = np.roll(arr[:, i], lag)
#     #             lagged_col[:lag] = 0.0
#     #             lagged.append(lagged_col)

#     #     return np.stack(lagged, axis=1)

#     @staticmethod
#     def _build_lag_features(
#         X: np.ndarray,
#         feature_names: List[str],
#     ) -> Tuple[np.ndarray, List[str]]:
#         """Vectorized lag feature construction (NumPy-based)."""
#         name_to_idx = {n: i for i, n in enumerate(feature_names)}
#         lag_arrays = []
#         lag_names = []

#         for col, lags in LAG_DEPTHS.items():
#             if col not in name_to_idx:
#                 continue

#             col_idx = name_to_idx[col]
#             base_col = X[:, col_idx]

#             for lag in lags:
#                 lagged = np.roll(base_col, lag)
#                 lagged[:lag] = 0.0  # zero-fill instead of NaN
#                 lag_arrays.append(lagged)
#                 lag_names.append(f"{col}_lag{lag}")

#         if not lag_arrays:
#             return np.empty((X.shape[0], 0), dtype=np.float32), []

#         return np.stack(lag_arrays, axis=1).astype(np.float32), lag_names
