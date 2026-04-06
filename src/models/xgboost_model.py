"""
src/models/xgboost_model.py

Production-grade XGBoost wrapper for tabular / flattened time-series data.
Designed to mirror LSTMModel + LSTMTrainer structure for consistency
with PSO / Optuna optimization pipelines.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# Model
# ======================================================================


class XGBoostModel:
    """Configurable XGBoost model for classification/regression.

    Args:
        params: Dictionary of XGBoost hyperparameters.
    """

    def __init__(self, params: Dict) -> None:
        self.params = params.copy()

        self.model = xgb.XGBClassifier(
            objective=params.get("objective", "multi:softprob"),
            num_class=params.get("num_class", 3),
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.7),
            colsample_bytree=params.get("colsample_bytree", 0.6),
            min_child_weight=params.get("min_child_weight", 5),
            gamma=params.get("gamma", 0.1),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            tree_method=params.get("tree_method", "hist"),
            max_bin=params.get("max_bin", 128),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        self._is_fitted = False

    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Fit model with optional early stopping."""

        eval_set = None
        early_stopping_rounds = self.params.get("early_stopping_rounds", None)

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
        )

        self._is_fitted = True

    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict_proba(X)


# ======================================================================
# Trainer
# ======================================================================


class XGBoostTrainer:
    """Handles training, early stopping, and tracking.

    Args:
        model: XGBoostModel instance
    """

    def __init__(self, model: XGBoostModel) -> None:
        self.model = model
        self.history: Dict[str, list] = {"train_metric": [], "val_metric": []}

    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, list]:
        """Train with validation tracking."""

        self.model.fit(X_train, y_train, X_val, y_val)

        # Extract evaluation results if available
        evals_result = self.model.model.evals_result()

        if evals_result:
            # XGBoost stores metrics as dict[dataset][metric]
            for dataset, metrics in evals_result.items():
                for metric_name, values in metrics.items():
                    if dataset == "validation_0":
                        self.history["val_metric"] = values
                    elif dataset == "validation_1":
                        self.history["train_metric"] = values

        return self.history

    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
