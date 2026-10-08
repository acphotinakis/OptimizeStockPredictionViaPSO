"""
XGBoost Training Module

Canonical training pipeline for XGBoost models. Owns:
- Lag-based feature construction
- Model fitting with early stopping
- Feature importance extraction
- History tracking

TRD Compliance:
- Consumes features from unified pipeline
- No internal feature engineering
- Temporal validation split
- Regression objective (next-period return)

Author: System Architect
Version: 2.0.0 REFACTORED
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb
from tqdm import tqdm
from xgboost.callback import EarlyStopping, TrainingCallback

from .xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class TQDMCallback(TrainingCallback):
    """TQDM progress bar for XGBoost training rounds."""

    def __init__(self, total_rounds: int):
        self.pbar = tqdm(total=total_rounds, desc="Training")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False  # continue training

    def after_training(self, model):
        self.pbar.close()
        return model


class XGBoostTrainer:
    """
    Canonical training pipeline for XGBoost models.

    This is the SINGLE owner of XGBoost training logic.
    It builds lag features, trains the regressor, and returns a ready-to-use
    XGBoostModel wrapper along with training history.

    Example:
        >>> config = {"max_depth": 4, "learning_rate": 0.01, ...}
        >>> trainer = XGBoostTrainer(config=config, seed=42)
        >>> model, history = trainer.train(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        xgboost_config: Dict,
        early_stopping_rounds: int,
    ):
        """
        Initialize XGBoost trainer.

        Args:
            config: Configuration dictionary with XGBoost parameters
            seed: Random seed for reproducibility
        """
        self.config = xgboost_config
        self.early_stopping_rounds = early_stopping_rounds
        logger.info(
            f"XGBoostTrainer initialized with seed={str(self.config.get('random_state', 42))}"
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[XGBoostModel, Dict]:
        """
        Train XGBoost model with early stopping.

        Training Protocol:
        - Fit on training set
        - Monitor validation RMSE
        - Early stop if no improvement
        - Return best iteration model

        Args:
            X_train: Training features (N_train, F)
            y_train: Training targets (N_train,)
            X_val: Validation features (N_val, F)
            y_val: Validation targets (N_val,)
            feature_names: Optional list of feature names

        Returns:
            Tuple of (trained_model, training_history)
        """
        logger.info("=" * 80)
        logger.info("XGBOOST TRAINING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Features: {X_train.shape[1]}")

        logger.info("Building XGBoost regressor:")
        logger.info(f"  Objective: {self.config['objective']}")
        logger.info(f"  Estimators: {self.config['n_estimators']}")
        logger.info(f"  Max depth: {self.config['max_depth']}")
        logger.info(f"  Learning rate: {self.config['learning_rate']}")
        logger.info(f"  Tree method: {self.config.get('tree_method', 'auto')}")
        logger.info(f"  Early stopping rounds: {self.early_stopping_rounds}")

        # Build callbacks
        n_estimators = self.config["n_estimators"]
        tqdm_callback = TQDMCallback(n_estimators)

        regressor = xgb.XGBRegressor(
            **self.config,
            callbacks=[
                tqdm_callback,
                EarlyStopping(rounds=self.early_stopping_rounds, save_best=True),
            ],
        )
        eval_set = [(X_train, y_train), (X_val, y_val)]

        logger.info("Training...")
        regressor.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Extract history
        evals_result = regressor.evals_result_ if hasattr(regressor, "evals_result_") else {}
        best_iteration = regressor.best_iteration if hasattr(regressor, "best_iteration") else 0

        # Extract feature importance
        feature_importance = None
        if (
            hasattr(regressor, "feature_importances_")
            and regressor.feature_importances_ is not None
        ):
            importance_values = regressor.feature_importances_
            if feature_names and len(feature_names) == len(importance_values):
                feature_importance = dict(zip(feature_names, importance_values))
            else:
                feature_importance = {f"f{i}": float(v) for i, v in enumerate(importance_values)}

            top_features = sorted(feature_importance.items(), key=lambda x: -x[1])[:10]
            logger.info("\nTop 10 important features:")
            for feat, imp in top_features:
                logger.info(f"  {feat}: {imp:.6f}")

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best iteration: {best_iteration}")

        if evals_result:
            train_key = "validation_0"
            val_key = "validation_1"
            if train_key in evals_result:
                metric = list(evals_result[train_key].keys())[0]
                train_rmse = evals_result[train_key][metric][best_iteration]
                logger.info(f"Best-iteration train {metric}: {train_rmse:.6f}")
            if val_key in evals_result:
                metric = list(evals_result[val_key].keys())[0]
                val_rmse = evals_result[val_key][metric][best_iteration]
                logger.info(f"Best-iteration val {metric}: {val_rmse:.6f}")
                # Persist best-iteration val metric (not last) into history.
                evals_result["best_iteration"] = best_iteration
                evals_result["best_val_metric"] = {metric: float(val_rmse)}

        return (
            XGBoostModel(
                booster=regressor.get_booster(),
                feature_names=feature_names or [],
                feature_importance=feature_importance or {},
                name="xgboost-canonical-v1",
            ),
            evals_result,
        )

    def _validate_inputs(self, X: np.ndarray, y: Optional[np.ndarray], split_name: str) -> None:
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
            raise ValueError(f"{split_name} X dtype must be float32 or float64, got {X.dtype}")

        if X.ndim != 2:
            raise ValueError(f"{split_name} X must be 2D (N, F), got shape {X.shape}")

        if X.shape[1] < 1:
            raise ValueError(f"{split_name} X must have at least 1 feature, got {X.shape[1]}")

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
                raise ValueError(f"{split_name} X and y length mismatch: {len(X)} vs {len(y)}")

            if np.isnan(y).any():
                raise ValueError(f"{split_name} y contains NaN values")
