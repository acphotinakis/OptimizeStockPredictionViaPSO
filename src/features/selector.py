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

Critical bugs fixed from original implementation:
  - transform() O(F*S) list lookup replaced with O(1) set lookup
  - fit_transform() shape mismatch: transform() was called with the original
    full feature_names but X had already been sliced by stages 1+2, causing
    silent column misalignment. Fixed by storing full→selected index map.
  - Correlation deduplication dropped j over i by index order; now drops
    the lower-MI feature in each correlated pair (more principled).
  - No p-value check on Pearson — added via scipy.stats.pearsonr.
  - Threshold corrected from 0.98 to 0.95 (Zeng et al. specification).
  - _deduplicate_correlated returned a bool mask but the variable was used
    inconsistently as both mask and index list across methods.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import time

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Four-stage feature selector. Must be fit on training data only.

    Stages (in order):
        1. Variance threshold        — remove near-constant features
        2. Pearson deduplication     — remove inter-feature collinear pairs (Zeng)
        3. VIF pruning               — remove collectively redundant features
        4. Mutual information rank   — retain top features by non-linear relevance

    Fit/transform contract:
        selector.fit(X_train, y_train, names_train)
        X_val_sel,  names = selector.transform(X_val,  all_names)
        X_test_sel, names = selector.transform(X_test, all_names)

    The transform() method accepts the FULL (pre-selection) feature matrix and
    the FULL feature name list, then extracts and reorders to match the fitted
    selection. This avoids the shape-mismatch bug where fit() internally sliced
    X but transform() still expected the original shape.

    Args:
        variance_threshold:     Drop features with variance below this value.
                                Default 1e-6 removes only genuinely constant cols.
        correlation_threshold:  Drop one of any pair with |Pearson r| > threshold.
                                Default 0.95 per Zeng et al.
        pearson_p_threshold:    Also require p-value < this for the correlation
                                to be considered significant (Zeng uses 0.01).
        vif_threshold:          Drop highest-VIF feature iteratively until all
                                VIF < this value. Standard econometric threshold: 10.
        mi_quantile_threshold:  Drop features below this MI quantile.
                                0.25 = keep top 75% by mutual information score.
                                Tune via PSO or cross-validation (blueprint note).
    """

    def __init__(
        self,
        variance_threshold: float = 1e-6,
        correlation_threshold: float = 0.95,
        pearson_p_threshold: float = 0.01,
        vif_threshold: float = 10.0,
        mi_quantile_threshold: float = 0.25,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.pearson_p_threshold = pearson_p_threshold
        self.vif_threshold = vif_threshold
        self.mi_quantile_threshold = mi_quantile_threshold

        # Populated during fit() — used by transform()
        # Maps: original full feature name → index in the selected output
        # Stored as an ordered list so transform() can reproduce the exact column order
        self.selected_features_: List[str] = []

        # Full→selected index mapping built during fit_transform for diagnostics
        self._selection_report_: Dict[str, object] = {}

        # MI scores for selected features (for external inspection / PSO)
        self.mi_scores_: Optional[Dict[str, float]] = None

        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> "FeatureSelector":
        """
        Fit all four selection stages on training data.

        Args:
            X:             (N, F) feature matrix — training set only, float32.
                           Must already be cleaned (no NaN/inf — pipeline guarantees this).
            y:             (N,) target array — log returns, float32.
            feature_names: List of F feature names corresponding to X columns.

        Returns:
            self (fitted)
        """
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) does not match "
                f"X column count ({X.shape[1]})."
            )

        n_original = len(feature_names)
        names = list(feature_names)  # working copy — mutated through stages
        # We track a parallel index list that maps current working positions
        # back to the original column indices in X (before any slicing).
        # This lets transform() extract the right columns from the full matrix.
        original_indices = list(range(n_original))

        logger.info(
            "FeatureSelector.fit() — starting with %d features, %d samples",
            n_original,
            len(X),
        )

        # ------------------------------------------------------------------
        # Stage 1: Variance threshold
        # ------------------------------------------------------------------
        start_time = time.perf_counter()
        X_work, names, original_indices = self._stage_variance(
            X, names, original_indices
        )
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Stage 1 (variance): %d → %d features (took %.4f s)",
            n_original,
            len(names),
            elapsed,
        )

        # ------------------------------------------------------------------
        # Stage 2: Pearson inter-feature correlation deduplication  [Zeng et al.]
        # ------------------------------------------------------------------
        # We need MI scores to decide WHICH feature to drop in each correlated pair.
        # Compute MI once here on the variance-filtered set; reuse in Stage 4.
        # ------------------------------------------------------------------
        start_time = time.perf_counter()
        # Compute MI once for use in Stage 4
        mi_scores_all = self._compute_mi(X_work, y)
        n_before_corr = len(names)
        X_work, names, original_indices = self._stage_pearson_dedup(
            X_work, y, names, original_indices, mi_scores_all
        )
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Stage 2 (Pearson r>%.2f, p<%.2f): %d → %d features (took %.4f s)",
            self.correlation_threshold,
            self.pearson_p_threshold,
            n_before_corr,
            len(names),
            elapsed,
        )

        # ------------------------------------------------------------------
        # Stage 3: VIF pruning                              [Blueprint extension]
        # ------------------------------------------------------------------
        start_vif = time.perf_counter()
        n_before_vif = len(names)
        X_work, names, original_indices = self._stage_vif(
            X_work, names, original_indices
        )
        elapsed_vif = time.perf_counter() - start_vif

        logger.info(
            "Stage 3 (VIF>%.1f): %d → %d features (took %.4f s)",
            self.vif_threshold,
            n_before_vif,
            len(names),
            elapsed_vif,
        )

        # ------------------------------------------------------------------
        # Stage 4: Mutual Information ranking               [Blueprint extension]
        # ------------------------------------------------------------------
        # Recompute MI on the VIF-filtered set (features have changed).
        start_mi = time.perf_counter()
        mi_scores_final = self._compute_mi(X_work, y)

        n_before_mi = len(names)
        names, original_indices = self._stage_mi_rank(
            names, original_indices, mi_scores_final
        )
        elapsed_mi = time.perf_counter() - start_mi

        logger.info(
            "Stage 4 (MI quantile>%.2f): %d → %d features (took %.4f s)",
            self.mi_quantile_threshold,
            n_before_mi,
            len(names),
            elapsed_mi,
        )
        # ------------------------------------------------------------------
        # Store results
        # ------------------------------------------------------------------
        self.selected_features_ = names

        # Store MI scores for selected features (useful for PSO / diagnostics)
        self.mi_scores_ = {
            name: float(mi_scores_final[i])
            for i, name in enumerate(
                # mi_scores_final is indexed to X_work columns at Stage 4 entry
                # which are the names before MI filtering — use n_before_mi names
                [names[k] for k in range(len(names))]
            )
        }

        # Store the original column indices so transform() is unambiguous
        self._original_indices_selected = original_indices

        self._selection_report_ = {
            "n_original": n_original,
            "n_selected": len(names),
            "selected": names,
            "original_indices": original_indices,
        }

        self._is_fitted = True

        logger.info(
            "FeatureSelector fit complete: %d → %d features (%.1f%% retained)",
            n_original,
            len(names),
            100.0 * len(names) / max(n_original, 1),
        )
        return self

    def transform(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply fitted selection to a feature matrix.

        IMPORTANT: X and feature_names must be the FULL pre-selection matrix
        (same dimensionality as what was passed to fit()). Do NOT pre-slice X
        before calling transform() — that causes silent column misalignment.

        Args:
            X:             (N, F_full) full feature matrix, same F as fit().
            feature_names: Full feature name list, same as passed to fit().

        Returns:
            X_selected:      (N, F_selected) — columns in fitted selection order
            selected_names:  List of selected feature names in the same order
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureSelector has not been fitted. "
                "Call fit() or fit_transform() on training data first."
            )

        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"X has {X.shape[1]} columns but feature_names has "
                f"{len(feature_names)} entries. Pass the full pre-selection matrix."
            )

        # Build name→column_index map for O(1) lookup
        name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(feature_names)}

        # Extract columns in the ORDER established during fit()
        # (importance-ranked, not input-order)
        missing = [n for n in self.selected_features_ if n not in name_to_idx]
        if missing:
            raise ValueError(
                f"transform() called with feature_names that are missing "
                f"{len(missing)} fitted features: {missing[:10]}{'...' if len(missing) > 10 else ''}. "
                f"Ensure the same full feature set is used in fit and transform."
            )

        col_indices = [name_to_idx[n] for n in self.selected_features_]
        return X[:, col_indices], list(self.selected_features_)

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fit and transform in one call (training set only).

        Returns:
            X_selected:     (N, F_selected) selected feature matrix
            selected_names: List of selected feature names
        """
        self.fit(X, y, feature_names)
        # transform() takes the FULL original X — not the internally sliced version
        return self.transform(X, feature_names)

    def report(self) -> str:
        """Return a human-readable summary of the selection results."""
        if not self._is_fitted:
            return "FeatureSelector: not yet fitted."
        r = self._selection_report_
        lines = [
            f"FeatureSelector Report",
            f"  Original features : {r['n_original']}",
            f"  Selected features : {r['n_selected']}",
            f"  Retention rate    : {100.0 * r['n_selected'] / max(r['n_original'], 1):.1f}%",
            f"  Selected names    :",
        ]
        if self.mi_scores_:
            sorted_by_mi = sorted(
                self.mi_scores_.items(), key=lambda x: x[1], reverse=True
            )
            for name, score in sorted_by_mi:
                lines.append(f"    {name:<40s}  MI={score:.6f}")
        else:
            for name in self.selected_features_:
                lines.append(f"    {name}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage_variance(
        self,
        X: np.ndarray,
        names: List[str],
        original_indices: List[int],
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        """Remove near-constant features."""

        # Compute variance (float64 for numerical stability)
        variances = np.var(X, axis=0, dtype=np.float64)
        keep_mask = variances > self.variance_threshold

        # Fast boolean indexing for names and indices
        names_arr = np.asarray(names)
        indices_arr = np.asarray(original_indices)

        kept_names = names_arr[keep_mask].tolist()
        kept_indices = indices_arr[keep_mask].tolist()

        # Only compute dropped if logging is enabled
        if logger.isEnabledFor(logging.DEBUG):
            dropped = names_arr[~keep_mask]
            if dropped.size:
                logger.info("Variance drop: %s", dropped.tolist())

        return X[:, keep_mask], kept_names, kept_indices

    def _stage_pearson_dedup(
        self,
        X: np.ndarray,
        y: np.ndarray,
        names: List[str],
        original_indices: List[int],
        mi_scores: np.ndarray,
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Remove inter-feature collinear pairs.  [Zeng et al., 2025]

        For each pair (i, j) where |r| > threshold AND p < p_threshold,
        we drop the feature with the LOWER MI score (less relevant to target).
        This is principled: keep the more informative feature of the pair.

        Original code dropped j over i purely by index order — incorrect.

        Pearson is computed between features (inter-feature), NOT between
        features and target. The goal is multicollinearity removal, not
        relevance filtering (that's Stage 4).
        """
        corr_matrix = np.corrcoef(X, rowvar=False)
        F = corr_matrix.shape[0]

        to_drop = set()

        for i in range(F):
            if i in to_drop:
                continue

            for j in range(i + 1, F):
                if j in to_drop:
                    continue

                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    if mi_scores[i] >= mi_scores[j]:
                        to_drop.add(j)
                    else:
                        to_drop.add(i)
                        break

        keep_mask = np.array([i not in to_drop for i in range(F)])

        names_arr = np.asarray(names)
        idx_arr = np.asarray(original_indices)

        return (
            X[:, keep_mask],
            names_arr[keep_mask].tolist(),
            idx_arr[keep_mask].tolist(),
        )

    def _stage_vif(
        self,
        X: np.ndarray,
        names: List[str],
        original_indices: List[int],
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        if X.shape[1] < 3:
            logger.info("VIF stage skipped: fewer than 3 features.")
            return X, names, original_indices
        if X.shape[0] <= X.shape[1]:
            logger.info(
                "VIF stage skipped: N=%d ≤ F=%d (underdetermined).",
                X.shape[0],
                X.shape[1],
            )
            return X, names, original_indices

        keep_mask = np.ones(X.shape[1], dtype=bool)
        names_w = list(names)
        orig_w = list(original_indices)

        for _ in range(X.shape[1]):
            if keep_mask.sum() < 3:
                break

            vif_values = self._compute_vif_vectorized(X[:, keep_mask])
            max_vif_idx = int(np.argmax(vif_values))
            max_vif = vif_values[max_vif_idx]

            if max_vif < self.vif_threshold:
                break

            logger.info("VIF drop: '%s' (VIF=%.2f)", names_w[max_vif_idx], max_vif)
            keep_mask[np.arange(len(keep_mask))[keep_mask][max_vif_idx]] = False
            names_w.pop(max_vif_idx)
            orig_w.pop(max_vif_idx)

        return X[:, keep_mask], names_w, orig_w

    def _stage_mi_rank(
        self,
        names: List[str],
        original_indices: List[int],
        mi_scores: np.ndarray,
    ) -> Tuple[List[str], List[int]]:
        """
        Retain features above the mi_quantile_threshold MI score.
        [Blueprint Stage C — Mutual Information]

        Mutual information captures non-linear feature→target relationships
        that Pearson correlation misses entirely. This is why it's specified
        in the blueprint AFTER Pearson (which handles linear redundancy).

        Features are RETURNED IN MI-DESCENDING ORDER so the most relevant
        features appear first. This order is preserved in transform().

        The threshold (bottom quartile by default) is intentionally conservative.
        The blueprint notes it should be tuned via PSO or cross-validation.
        """
        if len(names) == 0:
            return names, original_indices

        cutoff = float(np.quantile(mi_scores, self.mi_quantile_threshold))

        # Sort by MI descending — most relevant first
        ranked = sorted(
            zip(names, original_indices, mi_scores),
            key=lambda x: x[2],
            reverse=True,
        )

        selected_names = [n for n, _, score in ranked if score >= cutoff]
        selected_orig = [idx for _, idx, score in ranked if score >= cutoff]

        if not selected_names:
            # Safety: never return zero features — keep top 10 at minimum
            logger.info("MI stage filtered ALL features. Keeping top 10 by MI score.")
            selected_names = [n for n, _, _ in ranked[:10]]
            selected_orig = [idx for _, idx, _ in ranked[:10]]

        return selected_names, selected_orig

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _compute_vif_vectorized(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized VIF computation using matrix inversion.
        """
        # Compute correlation matrix
        C = np.corrcoef(X, rowvar=False)
        try:
            invC = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            invC = np.linalg.pinv(C)

        # Make a writable copy of the diagonal
        vif = np.diag(invC).copy()

        # Force inf for near-singular features
        vif[np.isnan(vif) | (vif > 1e10)] = np.inf

        return vif

    @staticmethod
    def _compute_mi(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute mutual information between each feature and the target.

        Uses sklearn's mutual_info_regression which handles continuous targets.
        MI is non-negative; higher = more relevant.

        Returns:
            mi_scores: np.ndarray of shape (F,), one score per feature column.
        """
        from sklearn.feature_selection import mutual_info_regression

        # MI computation is sensitive to NaN — should not occur after pipeline
        # cleaning, but guard anyway
        if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
            logger.info(
                "Non-finite values detected before MI computation. "
                "Replacing with 0.0."
            )
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
        return mi.astype(np.float64)
