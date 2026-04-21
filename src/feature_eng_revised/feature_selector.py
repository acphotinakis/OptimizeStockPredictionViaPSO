"""
Stage 5: Feature Selection Pipeline

Production-grade implementation of four-stage feature selection for PSO-LSTM
stock prediction system. Guarantees leakage-free, deterministic selection with
full state persistence for inference reuse.

Stages:
    1. Variance Threshold Filter (near-constant removal)
    2. Pearson Correlation De-duplication (Zeng et al. 2025)
    3. VIF Multicollinearity Filter (econometric standard)
    4. Mutual Information Ranking (nonlinear dependency)

Author: System Architect
Version: 1.0.0
"""

import logging
from typing import Dict, List, Tuple, Any, Set
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Four-stage feature selection pipeline for financial time-series.

    Guarantees:
    - Strict train-only fitting for all statistics
    - No data leakage from validation/test
    - Deterministic output via fixed random seeds
    - Full state serialization for inference

    Paper Attribution:
    - Stage 2 (Pearson filtering): Zeng et al. (2025) correlation-based selection
    - Stage 3 (VIF): Standard econometric multicollinearity measure (Gujarati)
    - Stage 4 (Mutual Information): Nonlinear dependency ranking
    - Explicit rejection: XGBoost feature importance NOT used (per spec)
    """

    # Stage thresholds (fixed per specification)
    VARIANCE_EPSILON = 1e-5
    PEARSON_THRESHOLD = 0.95
    VIF_THRESHOLD = 10.0
    MI_QUANTILE = 0.25

    def __init__(self):
        self._fitted = False
        self._selector_state: Dict[str, Any] = {}

    def _validate_inputs(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Validate input schema and extract feature/target separation.

        Args:
            train_df: Training data (features + target)
            val_df: Validation data (features + target)
            test_df: Test data (features + target)
            target_col: Name of target column

        Returns:
            Tuple of (X_train, X_val, X_test, y_train)

        Raises:
            KeyError: If target column missing
            ValueError: If insufficient features
        """
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if target_col not in df.columns:
                raise KeyError(
                    f"Target column '{target_col}' not found in {split_name}_df"
                )

        # Separate features and target (only y_train is used for MI)
        feature_cols = [c for c in train_df.columns if c != target_col]

        if len(feature_cols) <= 5:
            raise ValueError(f"Insufficient features: {len(feature_cols)} (need > 5)")

        X_train = train_df[feature_cols].copy()
        X_val = val_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()
        y_train = train_df[target_col].copy()

        # Validate no NaNs
        for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
            if X.isna().any().any():
                raise ValueError(f"NaN values detected in {split_name} features")

        return X_train, X_val, X_test, y_train

    def _stage1_variance_threshold(
        self, X_train: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Stage 1: Remove near-constant features (variance < epsilon).

        Fit: Training data only
        Method: Population variance

        Args:
            X_train: Training features

        Returns:
            Tuple of (filtered_train, kept_features, dropped_features)
        """
        logger.info("Stage 1: Variance Threshold Filter")

        variances = X_train.var()
        mask = variances >= self.VARIANCE_EPSILON

        kept_features = variances[mask].index.tolist()
        dropped_features = variances[~mask].index.tolist()

        logger.info(f"  Variance threshold (ε={self.VARIANCE_EPSILON}):")
        logger.info(f"    Kept: {len(kept_features)} features")
        logger.info(f"    Dropped: {len(dropped_features)} features")
        if dropped_features:
            logger.info(f"    Dropped list: {dropped_features}")

        return X_train[kept_features], kept_features, dropped_features

    def _stage2_pearson_deduplication(
        self, X_train: pd.DataFrame, y_train: pd.Series, current_features: List[str]
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Stage 2: Pearson correlation de-duplication (Zeng et al. 2025).

        Remove highly collinear features (|r| > 0.95), keeping the one with
        higher absolute correlation to target.

        Fit: Training data only
        Method: Pearson correlation matrix

        Paper Attribution:
        - Zeng et al. (2025): Pearson correlation > 0.95 threshold for redundancy

        Args:
            X_train: Training features (post-Stage 1)
            y_train: Training target
            current_features: List of features currently active

        Returns:
            Tuple of (filtered_train, kept_features, dropped_features)
        """
        logger.info("Stage 2: Pearson Correlation De-duplication (Zeng et al. 2025)")

        # Compute correlation matrix on TRAIN ONLY
        corr_matrix = X_train.corr(method="pearson")

        # Compute target correlations for tie-breaking
        target_corr = {}
        for feat in current_features:
            target_corr[feat] = abs(np.corrcoef(X_train[feat], y_train)[0, 1])

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                feat_i = corr_matrix.columns[i]
                feat_j = corr_matrix.columns[j]
                corr_val = abs(corr_matrix.iloc[i, j])

                if corr_val > self.PEARSON_THRESHOLD:
                    high_corr_pairs.append((feat_i, feat_j, corr_val))

        # Deterministic removal: lower target correlation loses
        features_to_drop = set()

        for feat_i, feat_j, corr_val in high_corr_pairs:
            if feat_i in features_to_drop or feat_j in features_to_drop:
                continue  # Already marked for removal

            # Compare absolute correlation with target
            if target_corr[feat_i] >= target_corr[feat_j]:
                loser = feat_j
            else:
                loser = feat_i

            features_to_drop.add(loser)
            logger.info(
                f"  High correlation pair: {feat_i} <-> {feat_j} (r={corr_val:.4f})"
            )
            logger.info(f"    -> Drop {loser} (lower target corr)")

        kept_features = [f for f in current_features if f not in features_to_drop]
        dropped_features = list(features_to_drop)

        logger.info(f"  Pearson threshold (|r|>{self.PEARSON_THRESHOLD}):")
        logger.info(f"    Kept: {len(kept_features)} features")
        logger.info(f"    Dropped: {len(dropped_features)} features")

        return X_train[kept_features], kept_features, dropped_features

    def _compute_vif(self, X: pd.DataFrame) -> pd.Series:
        """
        Compute Variance Inflation Factor for all features.

        VIF_i = 1 / (1 - R_i^2) where R_i^2 is from regression of feature i
        on all other features.

        Args:
            X: Feature DataFrame (with constant features already removed)

        Returns:
            Series of VIF values indexed by column
        """
        # Add constant for statsmodels OLS underlying VIF computation
        X_const = X.copy()

        vif_values = {}
        for i, col in enumerate(X_const.columns):
            try:
                vif = variance_inflation_factor(X_const.values, i)
                vif_values[col] = vif
            except Exception as e:
                logger.warning(f"VIF computation failed for {col}: {e}")
                vif_values[col] = np.inf

        return pd.Series(vif_values)

    def _stage3_vif_filter(
        self, X_train: pd.DataFrame, current_features: List[str]
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Stage 3: VIF multicollinearity filter (econometric standard).

        Iteratively remove features with VIF > 10, recomputing VIF after
        each removal.

        Fit: Training data only
        Method: statsmodels variance_inflation_factor
        Threshold: VIF > 10

        Paper Attribution:
        - Standard econometric practice (Gujarati): VIF > 10 indicates
          severe multicollinearity requiring remediation

        Args:
            X_train: Training features (post-Stage 2)
            current_features: List of features currently active

        Returns:
            Tuple of (filtered_train, kept_features, dropped_features)
        """
        logger.info("Stage 3: VIF Multicollinearity Filter (econometric standard)")

        X_work = X_train[current_features].copy()
        dropped_features = []

        iteration = 0
        while True:
            iteration += 1
            vif_series = self._compute_vif(X_work)

            # Check if all VIF <= threshold
            if (vif_series <= self.VIF_THRESHOLD).all():
                logger.info(f"  Iteration {iteration}: All VIF <= {self.VIF_THRESHOLD}")
                break

            # Find feature with highest VIF
            max_vif_feat = vif_series.idxmax()
            max_vif_val = vif_series.max()

            if max_vif_val <= self.VIF_THRESHOLD:
                break

            # Drop and recompute
            logger.info(
                f"  Iteration {iteration}: Drop {max_vif_feat} (VIF={max_vif_val:.2f})"
            )
            dropped_features.append(max_vif_feat)
            X_work = X_work.drop(columns=[max_vif_feat])

            # Safety check: don't remove all features
            if X_work.shape[1] <= 5:
                logger.warning("VIF filter stopped to preserve minimum 5 features")
                break

        kept_features = X_work.columns.tolist()

        logger.info(f"  VIF threshold (VIF<={self.VIF_THRESHOLD}):")
        logger.info(f"    Kept: {len(kept_features)} features")
        logger.info(f"    Dropped: {len(dropped_features)} features")
        if dropped_features:
            final_vif = self._compute_vif(X_work)
            logger.info(
                f"    Final VIF range: [{final_vif.min():.2f}, {final_vif.max():.2f}]"
            )

        return X_work, kept_features, dropped_features

    def _stage4_mutual_information(
        self, X_train: pd.DataFrame, y_train: pd.Series, current_features: List[str]
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Stage 4: Mutual Information ranking (nonlinear dependency).

        Keep features with MI score >= 25th percentile threshold.

        Fit: Training data only
        Method: sklearn mutual_info_regression
        Threshold: 25th percentile of MI scores

        Note: XGBoost feature importance is NOT used (explicitly forbidden)

        Args:
            X_train: Training features (post-Stage 3)
            y_train: Training target
            current_features: List of features currently active

        Returns:
            Tuple of (filtered_train, kept_features, dropped_features)
        """
        logger.info("Stage 4: Mutual Information Ranking (nonlinear dependency)")
        logger.info(
            "  Note: XGBoost feature importance NOT used (explicitly forbidden)"
        )

        X_work = X_train[current_features].copy()

        # Compute mutual information scores on TRAIN ONLY
        # random_state=0 ensures deterministic output
        mi_scores = mutual_info_regression(
            X_work.values,
            y_train.values,
            random_state=0,
            n_neighbors=3,  # Default, suitable for continuous data
        )

        mi_series = pd.Series(mi_scores, index=X_work.columns)

        # Threshold: 25th percentile
        threshold = np.quantile(mi_scores, self.MI_QUANTILE)

        mask = mi_series >= threshold
        kept_features = mi_series[mask].index.tolist()
        dropped_features = mi_series[~mask].index.tolist()

        logger.info(f"  MI threshold (quantile={self.MI_QUANTILE}): {threshold:.6f}")
        logger.info(f"    Kept: {len(kept_features)} features")
        logger.info(f"    Dropped: {len(dropped_features)} features")
        if dropped_features:
            logger.info(f"    Dropped list: {dropped_features}")

        return X_work[kept_features], kept_features, dropped_features

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = "log_return",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Execute four-stage feature selection pipeline.

        Pipeline:
        1. Variance threshold (remove near-constant)
        2. Pearson de-duplication (Zeng et al. 2025)
        3. VIF filter (econometric standard)
        4. Mutual Information ranking (nonlinear dependency)

        Final selection = intersection of all stages.

        Args:
            train_df: Training data with features and target
            val_df: Validation data with features and target
            test_df: Test data with features and target
            target_col: Name of target column

        Returns:
            Tuple of:
            - train_selected: Training features (selected only)
            - val_selected: Validation features (selected only)
            - test_selected: Test features (selected only)
            - selector_state: Serializable state for inference

        Raises:
            AssertionError: If validation checks fail
        """
        logger.info("=" * 60)
        logger.info("Feature Selection Pipeline: Starting")
        logger.info("=" * 60)

        # Validate and separate features/target
        X_train, X_val, X_test, y_train = self._validate_inputs(
            train_df, val_df, test_df, target_col
        )

        initial_features = X_train.columns.tolist()
        logger.info(f"Initial features: {len(initial_features)}")

        # Track features through each stage
        current_features = initial_features.copy()
        dropped_log = {"variance": [], "correlation": [], "vif": [], "mi": []}

        # =====================================================================
        # STAGE 1: Variance Threshold
        # =====================================================================
        X_train_s1, current_features, dropped = self._stage1_variance_threshold(
            X_train[current_features]
        )
        dropped_log["variance"] = dropped

        # =====================================================================
        # STAGE 2: Pearson Correlation De-duplication (Zeng et al. 2025)
        # =====================================================================
        X_train_s2, current_features, dropped = self._stage2_pearson_deduplication(
            X_train[current_features], y_train, current_features
        )
        dropped_log["correlation"] = dropped

        # =====================================================================
        # STAGE 3: VIF Filter (econometric standard)
        # =====================================================================
        X_train_s3, current_features, dropped = self._stage3_vif_filter(
            X_train[current_features], current_features
        )
        dropped_log["vif"] = dropped

        # =====================================================================
        # STAGE 4: Mutual Information Ranking
        # =====================================================================
        X_train_s4, current_features, dropped = self._stage4_mutual_information(
            X_train[current_features], y_train, current_features
        )
        dropped_log["mi"] = dropped

        # =====================================================================
        # FINAL SELECTION: Apply to all splits
        # =====================================================================
        selected_features = current_features

        logger.info("=" * 60)
        logger.info("Feature Selection Pipeline: Complete")
        logger.info(f"Final selected features: {len(selected_features)}")
        logger.info(f"Selected: {selected_features}")
        logger.info("=" * 60)

        # Apply selection to all splits (using same feature list)
        train_selected = X_train[selected_features].copy()
        val_selected = X_val[selected_features].copy()
        test_selected = X_test[selected_features].copy()

        # =====================================================================
        # VALIDATION CHECKS
        # =====================================================================

        # Check 1: No target leakage (target not in features)
        assert target_col not in train_selected.columns
        assert target_col not in val_selected.columns
        assert target_col not in test_selected.columns

        # Check 2: Consistency across splits
        assert list(train_selected.columns) == list(
            val_selected.columns
        ), "Feature mismatch: train vs val"
        assert list(train_selected.columns) == list(
            test_selected.columns
        ), "Feature mismatch: train vs test"

        # Check 3: No empty feature set
        assert (
            len(selected_features) > 5
        ), f"Too few features selected: {len(selected_features)} (min: 5)"

        # Check 4: No NaN values
        assert not train_selected.isna().any().any(), "NaN in train after selection"
        assert not val_selected.isna().any().any(), "NaN in val after selection"
        assert not test_selected.isna().any().any(), "NaN in test after selection"

        logger.info("Validation: All checks passed")

        # =====================================================================
        # STATE SERIALIZATION
        # =====================================================================
        self._selector_state = {
            "variance_threshold": self.VARIANCE_EPSILON,
            "pearson_threshold": self.PEARSON_THRESHOLD,
            "vif_threshold": self.VIF_THRESHOLD,
            "mi_quantile": self.MI_QUANTILE,
            "initial_features": initial_features,
            "selected_features": selected_features,
            "dropped_features": dropped_log,
            "n_features_initial": len(initial_features),
            "n_features_final": len(selected_features),
        }
        self._fitted = True

        return train_selected, val_selected, test_selected, self._selector_state

    def get_state(self) -> Dict[str, Any]:
        """Return selector state (must call fit_transform first)."""
        if not self._fitted:
            raise RuntimeError("Selector not fitted. Call fit_transform first.")
        return self._selector_state.copy()
