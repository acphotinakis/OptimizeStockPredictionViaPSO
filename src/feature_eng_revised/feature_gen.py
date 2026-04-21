# src/features/pipeline.py
from __future__ import annotations

"""features/pipeline.py — Refactored feature engineering pipeline."""

import logging
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

from ..features.technical import compute_technical_features
from ..features.statistical import compute_statistical_features
from ..features.volume import compute_volume_features
from ..features.cross_ticker import compute_cross_ticker_features
from ..features.selector import FeatureSelector

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
    ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
        """Fit on training data; return (X, y, feature_names, datetime_index)."""
        self._validate(dfs)
        X, y, names, idx = self._build_features(dfs)
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
        return X_sel, y, sel_names, idx

    def transform(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Apply fitted pipeline to val/test data; return (X, y, datetime_index)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        self._validate(dfs)
        X, y, names, idx = self._build_features(dfs)
        X_sel, _ = self.selector.transform(X, self._feature_names)
        logger.info(
            "[%s] transform: %d samples, %d features",
            self.target_ticker,
            len(X_sel),
            X_sel.shape[1],
        )
        return X_sel, y, idx

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
    ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
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
        X, y, idx = self._clean_and_align(X, df)

        return X, y, names, idx

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
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
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
            idx = df.index[valid]
        else:
            idx = df.index

        return X, y, idx
