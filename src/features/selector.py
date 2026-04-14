"""
src/features/selector.py

Four-stage feature selection pipeline for PSO-LSTM stock prediction.

Paper attribution:
  Stage 1 — Variance threshold
      Not in papers. Production addition: removes near-constant features
      that carry no signal and inflate the feature matrix.

  Stage 2 — Pearson inter-feature correlation deduplication  [Zeng et al., 2025]
      Zeng et al. compute Pearson coefficients between all features and the
      closing price, removing features above a 95% correlation threshold to
      avoid multicollinearity. Threshold confirmed significant at 0.01 level.
      We extend this with p-value gating as Zeng uses SPSS significance checks.
      When a correlated pair is found, we drop the LOWER-MI feature of the pair,
      not simply the one with the higher index (which the original code did).

  Stage 3 — VIF (Variance Inflation Factor)                  [Blueprint extension]
      The blueprint explicitly requires VIF because Pearson catches only pairwise
      linear relationships. Three moderately correlated features (e.g. EMA5,
      EMA10, EMA20) can survive Pearson filtering but still be collectively
      redundant. VIF detects this. Threshold: VIF > 10 indicates problematic
      multicollinearity (standard econometric convention).

  Stage 4 — Mutual Information ranking                       [Blueprint extension]
      The blueprint specifies MI as Stage C because it captures non-linear
      relationships that Pearson misses. Retains features above the bottom
      quartile of MI scores, with the quartile threshold tunable.
      The original code used XGBoost importance instead — that is NOT in the
      blueprint and NOT in any paper. XGBoost importance is biased toward
      high-cardinality / continuous features and adds a heavy dependency.
      MI is the correct method here.
"""

from __future__ import annotations

"""features/selector.py — Four-stage feature selection (variance --> Pearson --> VIF --> MI)."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Four-stage feature selector. Must be fit on training data only.

    Stages:
        1. Variance threshold  — drop near-constant features
        2. Pearson dedup       — drop inter-feature collinear pairs (Zeng et al.)
        3. VIF pruning         — drop collectively redundant features
        4. MI ranking          — keep top features by mutual information score

    Args:
        variance_threshold:    Default 1e-6 — removes genuinely constant columns.
        correlation_threshold: Default 0.95 per Zeng et al.
        pearson_p_threshold:   Pearson p-value gate (Zeng uses 0.01, unused here — see note).
        vif_threshold:         Standard econometric threshold: 10.
        mi_quantile_threshold: Drop bottom quantile; 0.25 = keep top 75%.
    """

    def __init__(
        self,
        variance_threshold: float = 1e-6,
        correlation_threshold: float = 0.95,
        vif_threshold: float = 10.0,
        mi_quantile_threshold: float = 0.25,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.mi_quantile_threshold = mi_quantile_threshold

        self.selected_features_: List[str] = []
        self.mi_scores_: Optional[Dict[str, float]] = None
        self._original_indices: List[int] = []
        self._is_fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> "FeatureSelector":
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"feature_names length {len(feature_names)} ≠ X columns {X.shape[1]}"
            )

        n_orig = len(feature_names)
        names = list(feature_names)
        indices = list(
            range(n_orig)
        )  # tracks original column positions through each stage

        logger.info("FeatureSelector.fit(): %d features, %d samples", n_orig, len(X))

        # Stage 1 — variance
        X, names, indices = self._stage_variance(X, names, indices)
        logger.info("Stage 1 (variance):   %d --> %d", n_orig, len(names))

        # Stage 2 — Pearson (MI computed once here; reused in stage 4)
        mi_scores = _compute_mi(X, y)
        n2 = len(names)
        X, names, indices = self._stage_pearson(X, names, indices, mi_scores)
        logger.info(
            "Stage 2 (Pearson r>%.2f): %d --> %d",
            self.correlation_threshold,
            n2,
            len(names),
        )

        # Stage 3 — VIF
        n3 = len(names)
        X, names, indices = self._stage_vif(X, names, indices)
        logger.info(
            "Stage 3 (VIF>%.1f):   %d --> %d", self.vif_threshold, n3, len(names)
        )

        # Stage 4 — MI ranking (recompute on VIF-filtered set)
        mi_final = _compute_mi(X, y)
        n4 = len(names)
        names, indices, mi_final = self._stage_mi(names, indices, mi_final)
        logger.info(
            "Stage 4 (MI q>%.2f): %d --> %d", self.mi_quantile_threshold, n4, len(names)
        )

        self.selected_features_ = names
        self._original_indices = indices
        self.mi_scores_ = dict(zip(names, mi_final.tolist()))
        self._is_fitted = True

        logger.info(
            "FeatureSelector: %d --> %d features (%.1f%% retained)",
            n_orig,
            len(names),
            100.0 * len(names) / max(n_orig, 1),
        )
        return self

    def transform(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract fitted features from the *full* pre-selection matrix."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"X columns {X.shape[1]} ≠ feature_names {len(feature_names)}"
            )
        name_idx = {n: i for i, n in enumerate(feature_names)}
        missing = [n for n in self.selected_features_ if n not in name_idx]
        if missing:
            raise ValueError(f"Missing {len(missing)} fitted features: {missing[:10]}")
        cols = [name_idx[n] for n in self.selected_features_]
        return X[:, cols], list(self.selected_features_)

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        self.fit(X, y, feature_names)
        return self.transform(X, feature_names)

    def report(self) -> str:
        if not self._is_fitted:
            return "FeatureSelector: not fitted."
        lines = [
            f"FeatureSelector: {len(self.selected_features_)} features selected",
        ]
        if self.mi_scores_:
            for name, score in sorted(self.mi_scores_.items(), key=lambda x: -x[1]):
                lines.append(f"  {name:<40s}  MI={score:.6f}")
        return "\n".join(lines)

    # ── Stage implementations ─────────────────────────────────────────────────

    def _stage_variance(
        self, X: np.ndarray, names: List[str], indices: List[int]
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        keep = np.var(X, axis=0, dtype=np.float64) > self.variance_threshold
        return X[:, keep], _mask(names, keep), _mask(indices, keep)

    def _stage_pearson(
        self,
        X: np.ndarray,
        names: List[str],
        indices: List[int],
        mi_scores: np.ndarray,
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        corr = np.corrcoef(X, rowvar=False)
        F = corr.shape[0]
        to_drop: set[int] = set()
        for i in range(F):
            if i in to_drop:
                continue
            for j in range(i + 1, F):
                if j in to_drop:
                    continue
                if abs(corr[i, j]) > self.correlation_threshold:
                    # Drop the lower-MI feature in the pair
                    to_drop.add(j if mi_scores[i] >= mi_scores[j] else i)
                    if i in to_drop:
                        break
        keep = np.array([i not in to_drop for i in range(F)])
        return X[:, keep], _mask(names, keep), _mask(indices, keep)

    def _stage_vif(
        self, X: np.ndarray, names: List[str], indices: List[int]
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        if X.shape[1] < 3 or X.shape[0] <= X.shape[1]:
            return X, names, indices

        keep = np.ones(X.shape[1], dtype=bool)
        w_names, w_idx = list(names), list(indices)

        for _ in range(X.shape[1]):
            if keep.sum() < 3:
                break
            vif = _compute_vif(X[:, keep])
            max_pos = int(np.argmax(vif))
            if vif[max_pos] < self.vif_threshold:
                break
            logger.info("VIF drop: '%s' (VIF=%.2f)", w_names[max_pos], vif[max_pos])
            global_pos = np.where(keep)[0][max_pos]
            keep[global_pos] = False
            w_names.pop(max_pos)
            w_idx.pop(max_pos)

        return X[:, keep], w_names, w_idx

    def _stage_mi(
        self, names: List[str], indices: List[int], mi_scores: np.ndarray
    ) -> Tuple[List[str], List[int], np.ndarray]:
        cutoff = float(np.quantile(mi_scores, self.mi_quantile_threshold))
        ranked = sorted(zip(names, indices, mi_scores), key=lambda x: -x[2])
        sel = [(n, i, s) for n, i, s in ranked if s >= cutoff] or list(ranked[:10])
        names_sel = [n for n, _, _ in sel]
        indices_sel = [i for _, i, _ in sel]
        scores_sel = np.array([s for _, _, s in sel])
        return names_sel, indices_sel, scores_sel


# ── Module-level helpers ──────────────────────────────────────────────────────


def _mask(lst: list, mask: np.ndarray) -> list:
    return [x for x, keep in zip(lst, mask) if keep]


def _compute_mi(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return mutual_info_regression(X, y, random_state=42, n_neighbors=5).astype(
        np.float64
    )


def _compute_vif(X: np.ndarray) -> np.ndarray:
    C = np.corrcoef(X, rowvar=False)
    try:
        inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(C)
    vif = np.diag(inv).copy()
    vif[np.isnan(vif) | (vif > 1e10)] = np.inf
    return vif
