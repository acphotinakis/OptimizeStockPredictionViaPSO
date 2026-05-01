from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Production feature selector for LSTM + XGBoost.

    Pipeline:
        1. Variance filter (safe pruning)
        2. Correlation clustering (anti-redundancy, NOT pairwise deletion)
        3. Mutual Information ranking (single-pass, stable)
    """

    def __init__(
        self,
        variance_threshold: float = 1e-6,
        correlation_threshold: float = 0.95,
        mi_quantile_threshold: float = 0.25,
        max_features: Optional[int] = None,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.mi_quantile_threshold = mi_quantile_threshold
        self.max_features = max_features

        self.selected_features_: List[str] = []
        self.mi_scores_: Optional[Dict[str, float]] = None
        self._is_fitted = False

    # ────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ────────────────────────────────────────────────────────────────

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> "FeatureSelector":

        if len(feature_names) != X.shape[1]:
            raise ValueError("feature_names must match X columns")

        names = list(feature_names)

        logger.info("FeatureSelector.fit(): %d features", X.shape[1])

        # Stage 1: Variance filter
        X, names = self._stage_variance(X, names)
        self.selected_features_ = list(names)

        # Stage 2: Correlation clustering
        X, names = self._stage_correlation_cluster(X, names)
        self.selected_features_ = list(names)

        # Stage 3: Mutual Information ranking
        mi_scores = _compute_mi(X, y)
        names, mi_scores = self._stage_mi(names, mi_scores)

        self.selected_features_ = list(names)
        self.mi_scores_ = dict(zip(names, mi_scores.tolist()))
        self._is_fitted = True

        logger.info(
            "FeatureSelector: %d -> %d features",
            len(feature_names),
            len(names),
        )

        return self

    def transform(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:

        if not self._is_fitted:
            raise RuntimeError("Call fit() first")

        name_idx = {n: i for i, n in enumerate(feature_names)}
        cols = [name_idx[n] for n in self.selected_features_]

        return X[:, cols], list(self.selected_features_)

    # ────────────────────────────────────────────────────────────────
    # STAGES
    # ────────────────────────────────────────────────────────────────

    def _stage_variance(self, X, names):
        var = np.var(X.astype(np.float64), axis=0)
        keep = var > self.variance_threshold
        return X[:, keep], _mask(names, keep)

    def _stage_correlation_cluster(self, X, names):
        """
        Replace Pearson + VIF with stable clustering behavior:
        - remove near-duplicate features
        - preserve representative per correlated group
        """

        Xf = X.astype(np.float64)
        corr = np.corrcoef(Xf, rowvar=False)

        n = corr.shape[0]
        keep = np.ones(n, dtype=bool)

        for i in range(n):
            if not keep[i]:
                continue

            for j in range(i + 1, n):
                if not keep[j]:
                    continue

                if abs(corr[i, j]) > self.correlation_threshold:
                    # keep both for LSTM robustness? NO → keep both only if MI later separates
                    # deterministic rule: keep first, drop second
                    keep[j] = False

        return X[:, keep], _mask(names, keep)

    def _stage_mi(self, names, mi_scores):
        cutoff = np.quantile(mi_scores, self.mi_quantile_threshold)

        keep = mi_scores >= cutoff

        if self.max_features is not None and np.sum(keep) > self.max_features:
            topk = np.argsort(-mi_scores)[: self.max_features]
            keep = np.zeros_like(mi_scores, dtype=bool)
            keep[topk] = True

        return _mask(names, keep), mi_scores[keep]


# ────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────


def _mask(lst: list, mask: np.ndarray) -> list:
    return [x for x, k in zip(lst, mask) if k]


def _compute_mi(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    return mutual_info_regression(
        X,
        y,
        random_state=42,
        n_neighbors=5,
    ).astype(np.float64)
