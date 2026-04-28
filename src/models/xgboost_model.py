import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb

from src.evaluation.metrics import compute_and_log_all_statistical_metrics

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    Production XGBoost regressor inference wrapper.

    This class does NOT own training logic. Use XGBoostTrainer to train,
    then this wrapper for prediction, evaluation, and I/O.

    Example:
        >>> model = XGBoostModel(trained_regressor=xgb_regressor)
        >>> predictions = model.predict(X_test)
        >>> metrics = model.evaluate(X_test, y_test)
        >>> model.save("model.json")
    """

    def __init__(
        self,
        booster: xgb.Booster,
        feature_names: List[str],
        feature_importance: Dict[str, float],
        name: str = "XGBoost-v1.0",
    ):
        self.booster = booster
        self.feature_names = feature_names
        self.feature_importance = feature_importance
        self.name = name

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Feature matrix (N, F)

        Returns:
            Predictions (N,)

        Raises:
            RuntimeError: If model not trained
        """
        if self.booster is None:
            raise RuntimeError("Model not trained. Use XGBoostTrainer to train.")

        self._validate_inputs(X, None, "predict")
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        preds = self.booster.predict(dmatrix)
        return preds.astype(np.float32)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features (N_test, F)
            y_test: Test targets (N_test,)

        Returns:
            Dictionary of metrics (mse, mae, rmse, directional_accuracy)
        """
        if self.booster is None:
            raise RuntimeError("Model not trained. Use XGBoostTrainer to train.")

        self._validate_inputs(X_test, y_test, "test")

        # y_pred = self.booster.predict(X_test)
        y_pred = self.predict(X_test)

        metrics = compute_and_log_all_statistical_metrics(
            y_test, y_pred, "XGBoost Evaluate Metrics"
        )

        metrics["n_samples"] = len(y_test)

        logger.info("=" * 80)
        logger.info("TEST EVALUATION")
        logger.info("=" * 80)

        logger.info(f"RMSE: {metrics['rmse']:.6f}")
        logger.info(f"MAE:  {metrics['mae']:.6f}")
        logger.info(f"MAPE: {metrics['mape']:.6f}")
        logger.info(f"R²:   {metrics['r2']:.6f}")
        logger.info(f"DA:   {metrics['directional_accuracy']:.4f}")
        logger.info(f"F1:   {metrics['f1_ternary']:.4f}")
        logger.info(f"AUC:  {metrics['auc_ternary']:.4f}")
        logger.info(f"N:    {metrics['n_samples']}")

        logger.info("=" * 80)
        return metrics, y_pred

    def save(self, filepath: str) -> None:
        """Save model to disk (XGBoost JSON format)."""
        if self.booster is None:
            raise RuntimeError("Model not trained. Use XGBoostTrainer to train.")
        self.booster.save_model(filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, path: Path, feature_names: List[str], feature_importance: Dict):
        booster = xgb.Booster()
        booster.load_model(path)
        logger.info(f"Model loaded from {path}")
        return cls(booster, feature_names, feature_importance)

    def _validate_inputs(
        self, X: np.ndarray, y: Optional[np.ndarray], split_name: str
    ) -> None:
        """
        Validate input array shapes and dtypes.

        Args:
            X: Feature matrix (N, F)
            y: Target vector (N,) or None
            split_name: Name of split for logging

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(X, np.ndarray):
            raise ValueError(f"{split_name} X must be numpy array")

        if X.dtype not in [np.float32, np.float64]:
            raise ValueError(
                f"{split_name} X dtype must be float32 or float64, got {X.dtype}"
            )

        if X.ndim != 2:
            raise ValueError(f"{split_name} X must be 2D (N, F), got shape {X.shape}")

        if X.shape[1] < 1:
            raise ValueError(
                f"{split_name} X must have at least 1 feature, got {X.shape[1]}"
            )

        if np.isnan(X).any():
            raise ValueError(f"{split_name} X contains NaN values")

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise ValueError(f"{split_name} y must be numpy array")

            if y.dtype not in [np.float32, np.float64]:
                raise ValueError(f"{split_name} y dtype must be float32 or float64")

            if y.ndim != 1:
                raise ValueError(f"{split_name} y must be 1D (N,), got shape {y.shape}")

            if len(y) != len(X):
                raise ValueError(
                    f"{split_name} X and y length mismatch: {len(X)} vs {len(y)}"
                )

            if np.isnan(y).any():
                raise ValueError(f"{split_name} y contains NaN values")
