"""
Feature Engineering Pipeline

Production-grade feature pipeline that integrates:
1. TRD-aligned technical indicators
2. Cross-ticker features (SPY, peers, market breadth)
3. Wavelet denoising (NEW - TRD requirement)
4. 4-stage feature selection (Variance → Pearson → VIF → MI)
5. MinMax normalization to [-1, 1]
6. Temporal windowing for LSTM sequences

TRD Compliance: TRD1, TRD2, TRD3
Paper Attribution: Zeng et al. 2025, Ji et al. 2021, Lanbouri & Achchab 2020

Author: System Architect
Version: 1.0.0 UNIFIED
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .technical import compute_trd_technical_features, compute_price_features
from .statistical import compute_statistical_features
from .volume import compute_volume_features
from .cross_ticker import compute_cross_ticker_features
from .selector import FeatureSelector
from .wavelet import apply_wavelet_denoising
from .normalization import transform_features

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume", "log_return"}


class FeaturePipeline:
    """
    Unified feature pipeline for single-ticker prediction with cross-ticker context.

    Pipeline Stages:
    1. Raw Feature Generation (Technical, Statistical, Volume, Cross-ticker)
    2. Wavelet Denoising (close → close_denoised)
    3. Feature Selection (4-stage: Variance → Pearson → VIF → MI)
    4. MinMax Normalization ([-1, 1], fit on training only)
    5. Temporal Windowing (20-timestep sequences for LSTM)

    Guarantees:
    - Strict temporal causality (no future data leakage)
    - Training-only parameter fitting (scaler, selector, wavelet threshold)
    - Deterministic feature ordering
    - Cross-ticker feature consistency
    """

    def __init__(
        self,
        target_ticker: str,
        peer_tickers: List[str],
        selector_kwargs: Optional[Dict] = None,
        enable_wavelet: bool = True,
    ) -> None:
        """
        Initialize unified feature pipeline.

        Args:
            target_ticker: Ticker symbol being predicted
            peer_tickers: List of peer ticker symbols (pre-selected on training data)
            selector_kwargs: Optional configuration for FeatureSelector
            enable_wavelet: Whether to apply wavelet denoising (default: True per TRD)
        """
        self.target_ticker = target_ticker
        self.peer_tickers = sorted(peer_tickers)
        self.enable_wavelet = enable_wavelet

        # Initialize selector
        self.selector = FeatureSelector(**(selector_kwargs or {}))

        # Pipeline state (fitted on training data)
        self._feature_names: List[str] = []
        self._selected_features: List[str] = []
        self._wavelet_threshold: Optional[float] = None

        self._fitted = False

        logger.info(
            f"Initialized FeaturePipeline for target '{self.target_ticker}' with "
            f"{self.peer_tickers} peers, wavelet enabled: {self.enable_wavelet}"
        )

    def fit_transform(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
        """
        Fit pipeline on training data and transform.

        Args:
            dfs: Dictionary mapping ticker symbols to DataFrames with OHLCV + log_return

        Returns:
            Tuple of (X_selected, y, selected_feature_names, datetime_index)
            where X_selected is (N, F) after selection, y is (N,) target returns

        Raises:
            KeyError: If target_ticker not in dfs or required columns missing
            ValueError: If data validation fails
        """
        self._validate(dfs)

        # Stage 1: Raw feature generation
        logger.info(f"[{self.target_ticker}] Stage 1: Generating raw features")
        X, y, names, idx = self._build_features(dfs)
        logger.info(f"  Generated {len(names)} raw features, {len(X)} samples")

        # Stage 2: Wavelet denoising (fit threshold on training data)
        if self.enable_wavelet:
            logger.info(f"[{self.target_ticker}] Stage 2: Wavelet denoising")
            X, names = self._apply_wavelet(X, names, fit_mode=True)
            logger.info(f"  Threshold fitted: {self._wavelet_threshold:.6f}")

        # Stage 3: Feature selection (fit on training data)
        logger.info(f"[{self.target_ticker}] Stage 3: Feature selection")
        X_sel, sel_names = self.selector.fit_transform(X, y, names)
        self._feature_names = names  # Store full feature names for transform
        self._selected_features = sel_names
        self._fitted = True
        logger.info(f"  Selected {len(sel_names)} / {len(names)} features")

        return X_sel, y, sel_names, idx

    def transform(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Transform validation/test data using fitted pipeline.

        Args:
            dfs: Dictionary mapping ticker symbols to DataFrames

        Returns:
            Tuple of (X_selected, y, datetime_index)

        Raises:
            RuntimeError: If pipeline not fitted (call fit_transform first)
        """
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() first.")

        self._validate(dfs)

        # Stage 1: Raw feature generation (same as training)
        X, y, names, idx = self._build_features(dfs)

        # Stage 2: Wavelet denoising (use fitted threshold)
        if self.enable_wavelet:
            X, names = self._apply_wavelet(X, names, fit_mode=False)

        # Stage 3: Feature selection (use fitted selector)
        X_sel, _ = self.selector.transform(X, self._feature_names)

        logger.info(
            f"[{self.target_ticker}] transform: {len(X_sel)} samples, "
            f"{X_sel.shape[1]} features"
        )

        return X_sel, y, idx

    @property
    def feature_names(self) -> List[str]:
        """Return selected feature names after fit_transform."""
        return list(self._selected_features)

    @property
    def n_features(self) -> int:
        """Return number of selected features."""
        return len(self._selected_features)

    @property
    def wavelet_threshold(self) -> float:
        if self._wavelet_threshold is None:
            raise RuntimeError("Wavelet threshold not fitted yet")

        return self._wavelet_threshold

    def _validate(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """Validate input data structure."""
        if self.target_ticker not in dfs:
            raise KeyError(
                f"Target ticker '{self.target_ticker}' not in dfs. "
                f"Available: {list(dfs.keys())}"
            )

        df = dfs[self.target_ticker]
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"[{self.target_ticker}] Missing required columns: {missing}"
            )

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                f"[{self.target_ticker}] Index must be DatetimeIndex, "
                f"got {type(df.index)}"
            )

        if not df.index.is_monotonic_increasing:
            raise ValueError(
                f"[{self.target_ticker}] Index must be sorted in ascending order"
            )

    def _build_features(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
        """
        Main feature building orchestrator.

        Steps:
        1. Compute feature blocks (technical, statistical, volume, cross-ticker)
        2. Concatenate and deduplicate features
        3. Clean NaN/Inf and align with target variable

        Returns:
            Tuple of (X_matrix, y_target, feature_names, datetime_index)
        """
        df = dfs[self.target_ticker]

        # Build feature blocks
        blocks = self._build_feature_blocks(df, dfs)

        # Concatenate with deduplication
        X, names = self._concat_blocks(blocks)

        # Clean and align with target
        X, y, idx = self._clean_and_align(X, df)

        return X, y, names, idx

    def _build_feature_blocks(
        self, df: pd.DataFrame, dfs: Dict[str, pd.DataFrame]
    ) -> List[pd.DataFrame]:
        """
        Build feature blocks from different sources.

        Returns:
            List of DataFrames, each representing a feature category
        """
        # blocks = [
        #     compute_price_features(df),               # Base price features
        #     compute_trd_technical_features(df),       # TRD technical indicators
        #     compute_statistical_features(df),         # Statistical features
        #     compute_volume_features(df),              # Volume features
        #     compute_cross_ticker_features(            # Cross-ticker features
        #         self.target_ticker, dfs, self.peer_tickers
        #     ),
        # ]

        blocks = [
            compute_price_features(df),  # Base price features
            compute_trd_technical_features(df),  # TRD technical indicators
            compute_statistical_features(df),  # Statistical features
            compute_volume_features(df),  # Volume features
            compute_cross_ticker_features(  # Cross-ticker features
                self.target_ticker, dfs, self.peer_tickers
            ),
        ]

        logger.info(
            f"[{self.target_ticker}] Built feature blocks: "
            f"{[blk.shape[1] for blk in blocks]} features in each block"
        )
        # log out other statistics about blocks (e.g. feature names, missing values)
        for i, blk in enumerate(blocks):
            n_features = blk.shape[1]
            n_missing = blk.isna().sum().sum()
            logger.debug(
                f"  Block {i}: {n_features} features, {n_missing} missing values"
            )
        return blocks

    def _concat_blocks(
        self, blocks: List[pd.DataFrame]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Concatenate feature blocks with deduplication.

        Ensures:
        - Type conversion to float32
        - Within-block deduplication (keeps first occurrence)
        - Across-block deduplication (keeps first occurrence)

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        arrays = []
        names = []
        seen_names = set()

        for blk in blocks:
            # Remove duplicates within block
            blk = blk.loc[:, ~blk.columns.duplicated()]

            # Convert to array
            arr = blk.values.astype(np.float32, copy=False)

            # Track names, skipping duplicates across blocks
            block_names = []
            keep_indices = []
            for i, col in enumerate(blk.columns):
                if col not in seen_names:
                    seen_names.add(col)
                    block_names.append(col)
                    keep_indices.append(i)

            # Slice array to keep only unique columns
            if len(keep_indices) < len(blk.columns):
                arr = arr[:, keep_indices]

            arrays.append(arr)
            names.extend(block_names)

        X = np.concatenate(arrays, axis=1)
        return X, names

    def _apply_wavelet(
        self, X: np.ndarray, names: List[str], fit_mode: bool
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply wavelet denoising to 'close' feature.

        Args:
            X: Feature matrix (N, F)
            names: List of feature names
            fit_mode: If True, compute threshold; if False, use fitted threshold

        Returns:
            Tuple of (X_updated, names_updated) with close_denoised replacing close
        """
        if "close" not in names:
            logger.warning("'close' feature not found, skipping wavelet denoising")
            return X, names

        close_idx = names.index("close")
        close_series = pd.Series(X[:, close_idx])

        if fit_mode:
            # FIT MODE: Compute threshold on training data
            close_denoised, threshold = apply_wavelet_denoising(
                close_series, threshold_train=None
            )
            self._wavelet_threshold = threshold
        else:
            # TRANSFORM MODE: Use fitted threshold
            if self._wavelet_threshold is None:
                raise RuntimeError("Wavelet threshold not fitted")
            close_denoised = apply_wavelet_denoising(
                close_series, threshold_train=self._wavelet_threshold
            )

        # Replace close with close_denoised
        X[:, close_idx] = close_denoised.values
        names[close_idx] = "close_denoised"

        return X, names

    def _clean_and_align(
        self, X: np.ndarray, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Clean non-finite values and align with target variable.

        Target Definition:
            y[t] = (close_{t+1} - close_t) / close_t  (next-period return)

        Constraints:
            - Drops rows where target or any feature is non-finite
            - Ensures strict temporal separation (target is future)

        Returns:
            Tuple of (X_clean, y_target, datetime_index)
        """
        if "log_return" not in df.columns:
            if "close" not in df.columns:
                raise ValueError("Missing 'close' column for log_return computation")

            df = df.copy()
            df["log_return"] = np.log(df["close"]).diff()

        y = df["log_return"].values.astype(np.float32)

        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        valid = np.isfinite(y) & np.isfinite(X).all(axis=1)

        if (~valid).sum() > 0:
            n_dropped = int((~valid).sum())
            logger.info(
                f"[{self.target_ticker}] Dropping {n_dropped} rows "
                f"({100 * n_dropped / len(y):.2f}%)"
            )

        X = X[valid]
        y = y[valid]
        idx = df.index[valid]

        return X, y, pd.DatetimeIndex(idx)
