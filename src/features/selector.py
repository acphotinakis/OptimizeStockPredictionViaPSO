"""
src/features/selector.py

Two-stage feature selection:
  1. Variance threshold — remove near-constant features.
  2. Pearson correlation deduplication — remove redundant correlated features.
  3. XGBoost gain importance — retain top features by cumulative importance.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Select the most informative features for a target ticker.

    Args:
        variance_threshold: Drop features with variance below this value.
        correlation_threshold: Drop one of any pair with |Pearson r| > this.
        importance_cumulative: Retain features until this fraction of total
            XGBoost gain importance is covered (0 < value ≤ 1).
    """

    def __init__(
        self,
        variance_threshold: float = 1e-6,
        correlation_threshold: float = 0.98,
        importance_cumulative: float = 0.70,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.importance_cumulative = importance_cumulative
        self.selected_features_: List[str] = []
        self.feature_importances_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> "FeatureSelector":
        """Fit the selector on training data.

        Args:
            X: [N, F] feature matrix (training set, already scaled).
            y: [N] target array (next-bar log return).
            feature_names: List of feature names (length F).

        Returns:
            self
        """
        names = list(feature_names)

        # Stage 1: Variance threshold
        variances = np.var(X, axis=0)
        mask_var = variances > self.variance_threshold
        X = X[:, mask_var]
        names = [n for n, m in zip(names, mask_var) if m]
        logger.info(
            "Variance threshold: %d → %d features", len(feature_names), len(names)
        )

        # Stage 2: Pearson correlation deduplication
        names, keep_mask = self._deduplicate_correlated(X, names)
        X = X[:, keep_mask]
        logger.info("Correlation dedup: → %d features", len(names))

        # Stage 3: XGBoost importance
        names = self._xgb_importance_filter(X, y, names)
        logger.info("XGBoost importance filter: → %d features", len(names))

        self.selected_features_ = names
        return self

    def transform(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply fitted selection mask to a feature matrix.

        Args:
            X: [N, F] feature matrix.
            feature_names: Full list of feature names before selection.

        Returns:
            (X_selected, selected_names)
        """
        if not self.selected_features_:
            raise RuntimeError("Call fit() before transform().")
        idx = [i for i, n in enumerate(feature_names) if n in self.selected_features_]
        ordered = [feature_names[i] for i in idx]
        return X[:, idx], ordered

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Convenience: fit then transform."""
        self.fit(X, y, feature_names)
        return self.transform(X, feature_names)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _deduplicate_correlated(
        self, X: np.ndarray, names: List[str]
    ) -> Tuple[List[str], np.ndarray]:
        corr = np.corrcoef(X.T)
        F = len(names)
        to_drop = set()
        for i in range(F):
            if i in to_drop:
                continue
            for j in range(i + 1, F):
                if j in to_drop:
                    continue
                if abs(corr[i, j]) > self.correlation_threshold:
                    to_drop.add(j)  # Drop j; keep i (assumed higher importance)

        keep_mask = np.array([i not in to_drop for i in range(F)])
        kept_names = [n for n, k in zip(names, keep_mask) if k]
        return kept_names, keep_mask

    def _xgb_importance_filter(
        self, X: np.ndarray, y: np.ndarray, names: List[str]
    ) -> List[str]:
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not installed; skipping importance filter.")
            return names

        # Detect GPU availability
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            tree_method = "gpu_hist" if gpu_available else "hist"
            if gpu_available:
                logger.info("GPU detected - using gpu_hist for XGBoost feature selection")
        except ImportError:
            tree_method = "hist"
            gpu_available = False

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method=tree_method,
            random_state=42,
            verbosity=0,
            n_jobs=-1 if not gpu_available else 1,  # Use all CPUs if no GPU
        )
        
        logger.info("Fitting XGBoost for feature importance (method=%s)...", tree_method)
        model.fit(X, y)
        importances = model.feature_importances_  # gain-based
        self.feature_importances_ = importances

        # Sort descending; compute cumulative importance
        order = np.argsort(importances)[::-1]
        cumulative = np.cumsum(importances[order]) / (importances.sum() + 1e-10)
        cutoff = int(np.searchsorted(cumulative, self.importance_cumulative)) + 1

        selected_idx = set(order[:cutoff])
        return [n for i, n in enumerate(names) if i in selected_idx]
