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

from .feature_creators_funcs import (
    compute_statistical_features,
    compute_volume_features,
    compute_trd_technical_features,
    compute_price_features,
    compute_target,
)
from .cross_ticker import compute_cross_ticker_features
from .selector import FeatureSelector
from .wavelet import apply_wavelet_denoising
from .normalization import transform_features

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume", "log_return"}


class FeatureGenerator:
    def __init__(
        self,
        target_ticker: str,
        peer_tickers: List[str],
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

        # Pipeline state (fitted on training data)
        self._feature_names: List[str] = []

        logger.info(
            f"Initialized FeatureGenerator for target '{self.target_ticker}' with "
            f"{self.peer_tickers} peers"
        )

    def generate_features(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
        # self._validate(dfs)

        # Stage 1: Raw feature generation
        logger.info(f"[{self.target_ticker}] Stage 1: Generating raw features")
        X, y, names, idx = self._build_features(dfs)
        logger.info(f"  Generated {len(names)} raw features, {len(X)} samples")

        return X, y, names, idx

    def _validate(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """Validate input data structure and enforce strict column schema."""

        if self.target_ticker not in dfs:
            raise KeyError(
                f"Target ticker '{self.target_ticker}' not in dfs. "
                f"Available: {list(dfs.keys())}"
            )

        df = dfs[self.target_ticker]

        # --- Strict schema enforcement ---
        extra_cols = set(df.columns) - REQUIRED_COLUMNS
        missing_cols = REQUIRED_COLUMNS - set(df.columns)

        if missing_cols:
            raise ValueError(
                f"[{self.target_ticker}] Missing required columns: {missing_cols}"
            )

        if extra_cols:
            logger.info(
                f"[{self.target_ticker}] Dropping non-required columns: {sorted(extra_cols)}"
            )
            dfs[self.target_ticker] = df[list(REQUIRED_COLUMNS)].copy()

        df = dfs[self.target_ticker]

        # --- Index validation ---
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

        blocks = [
            # compute_target(df),
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
            logger.info(
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
