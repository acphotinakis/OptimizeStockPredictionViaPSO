"""
src/features/pipeline.py

Unified feature engineering pipeline.
Combines all feature categories, adds lag features, and exposes
fit_transform / transform for train/val/test consistency.
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

# Top features for lag generation (F106–F117)
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
    """End-to-end feature engineering and selection pipeline.

    Args:
        target_ticker: Ticker for which to build the feature matrix.
        universe_tickers: All tickers in the universe (for cross-ticker features).
        selector_kwargs: Keyword arguments forwarded to FeatureSelector.
    """

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
        """Build features for the training set and fit the selector.

        Args:
            dfs_train: Dict ticker → cleaned train DataFrame.

        Returns:
            (X_selected, y, selected_feature_names)
            where X has shape [N_train_windows, F_selected]
            and y has shape [N_train_windows].
        """
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
        """Apply fitted pipeline to val/test data.

        Args:
            dfs: Dict ticker → cleaned DataFrame (val or test split).

        Returns:
            (X_selected, y)
        """
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
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        dfs: Dict[str, pd.DataFrame],
        fit: bool,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Compute the full (unselected) flat feature matrix."""
        df_target = dfs[self.target_ticker]

        # --- Individual feature blocks ---
        tech = compute_technical_features(df_target)
        stat = compute_statistical_features(df_target)
        vol = compute_volume_features(df_target)
        cross = compute_cross_ticker_features(
            self.target_ticker,
            dfs,
            peer_tickers=None if fit else self._peer_tickers,
        )

        if fit:
            # Store peer tickers for val/test consistency
            from .cross_ticker import compute_cross_ticker_features as _ct
            import re

            self._peer_tickers = [
                c.replace("peer_corr_", "")
                for c in cross.columns
                if c.startswith("peer_corr_")
            ]

        # --- Price/return base features ---
        base = pd.DataFrame(index=df_target.index)
        C = df_target["close"]
        H, L, O = df_target["high"], df_target["low"], df_target["open"]
        r = df_target["log_return"]
        base["log_return"] = r
        base["mid_price"] = (H + L) / 2
        base["hl_ratio"] = (H - L) / (C.shift(1) + 1e-10)
        base["co_ratio"] = (C - O) / (C.shift(1) + 1e-10)
        base["ho_ratio"] = (H - O) / (C.shift(1) + 1e-10)
        base["lc_ratio"] = (C - L) / (C.shift(1) + 1e-10)
        base["gap"] = (O - C.shift(1)) / (C.shift(1) + 1e-10)
        tr = pd.concat(
            [(H - L), (H - C.shift(1)).abs(), (L - C.shift(1)).abs()], axis=1
        ).max(axis=1)
        base["true_range"] = tr / (C.shift(1) + 1e-10)

        session_cum_return = (
            C
            / df_target.groupby(
                (
                    df_target.get(
                        "session_start", pd.Series(False, index=df_target.index)
                    ).cumsum()
                )
            )["close"].transform("first")
            - 1
        )
        base["cum_return_session"] = session_cum_return.fillna(0.0)
        base["intrabar_vol"] = (H - L) / (O + 1e-10)

        # --- Concatenate all blocks ---
        all_frames = [base, tech, stat, vol, cross]
        combined = pd.concat(all_frames, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]

        # --- Lag features ---
        lag_df = self._build_lag_features(combined)
        combined = pd.concat([combined, lag_df], axis=1)
        combined = combined.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        feature_names = list(combined.columns)
        X = combined.values.astype(np.float32)
        y = r.values.astype(np.float32)

        return X, y, feature_names

    @staticmethod
    def _build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for selected columns."""
        lag_frames = {}
        for col, lags in LAG_DEPTHS.items():
            if col not in df.columns:
                continue
            for lag in lags:
                lag_frames[f"{col}_lag{lag}"] = df[col].shift(lag)
        return pd.DataFrame(lag_frames, index=df.index).fillna(0.0)
