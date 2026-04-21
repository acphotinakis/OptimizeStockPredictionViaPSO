"""
XGBoost Regression Model for Financial Time-Series

Production-grade implementation of XGBoost regressor for stock return prediction.
Strictly consumes features from the unified feature pipeline with NO internal
feature engineering or transformation.

TRD Compliance:
- Tabular input format (N, F_selected)
- No feature engineering in model layer
- No scaler fitting in model layer
- No feature selection in model layer
- Regression objective (next-period return)

Paper Attribution:
- Zeng et al. 2025: XGBoost baseline comparison
- TRD1-3: Feature pipeline compatibility

Author: System Architect
Version: 1.0.0 UNIFIED CANONICAL
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    Production XGBoost regressor for financial time-series.
    
    Architecture:
        - Gradient boosted decision trees
        - Regression objective (next-period return)
        - Early stopping on validation set
        - Feature importance tracking
    
    Constraints:
        - NO feature engineering (consumes pipeline output)
        - NO normalization (expects pre-normalized features)
        - NO feature selection (expects selected features)
        - Temporal validation split only
    
    Example:
        >>> model = XGBoostModel(config, seed=42)
        >>> model.train(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
        >>> metrics = model.evaluate(X_test, y_test)
    """
    
    def __init__(
        self,
        config: Dict,
        seed: int = 42,
    ):
        """
        Initialize XGBoost model.
        
        Args:
            config: Configuration dictionary with:
                - objective: 'reg:squarederror'
                - n_estimators: int (e.g., 200)
                - max_depth: int (e.g., 4)
                - learning_rate: float (e.g., 0.01)
                - subsample: float (e.g., 0.7)
                - colsample_bytree: float (e.g., 0.6)
                - min_child_weight: float (e.g., 5)
                - gamma: float (e.g., 0.1)
                - reg_alpha: float (L1 regularization)
                - reg_lambda: float (L2 regularization)
                - early_stopping_rounds: int (e.g., 50)
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: Optional[List[str]] = None
        self.feature_importance: Optional[Dict[str, float]] = None
        self.evals_result: Optional[Dict] = None
        self.best_iteration: int = 0
        
        logger.info("XGBoostModel initialized")
        logger.info(f"Config: {config}")
    
    def _build_model(self) -> xgb.XGBRegressor:
        """
        Build XGBoost regressor from configuration.
        
        Returns:
            Initialized XGBRegressor
        """
        # Extract parameters with defaults
        params = {
            "objective": self.config.get("objective", "reg:squarederror"),
            "n_estimators": int(self.config.get("n_estimators", 200)),
            "max_depth": int(self.config.get("max_depth", 4)),
            "learning_rate": float(self.config.get("learning_rate", 0.01)),
            "subsample": float(self.config.get("subsample", 0.7)),
            "colsample_bytree": float(self.config.get("colsample_bytree", 0.6)),
            "min_child_weight": float(self.config.get("min_child_weight", 5)),
            "gamma": float(self.config.get("gamma", 0.1)),
            "reg_alpha": float(self.config.get("reg_alpha", 0.0)),
            "reg_lambda": float(self.config.get("reg_lambda", 1.0)),
            "random_state": self.seed,
            "n_jobs": -1,  # Use all cores
            "verbosity": 0,  # Suppress XGBoost internal logging
        }
        
        # Tree method (GPU if available)
        tree_method = self.config.get("tree_method", "hist")
        if tree_method in ["gpu_hist", "hist"]:
            params["tree_method"] = tree_method
        
        # Max bin for memory control
        if "max_bin" in self.config:
            params["max_bin"] = int(self.config["max_bin"])
        
        logger.info("Building XGBoost regressor:")
        logger.info(f"  Objective: {params['objective']}")
        logger.info(f"  Estimators: {params['n_estimators']}")
        logger.info(f"  Max depth: {params['max_depth']}")
        logger.info(f"  Learning rate: {params['learning_rate']}")
        logger.info(f"  Tree method: {params.get('tree_method', 'auto')}")
        
        return xgb.XGBRegressor(**params)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "XGBoostModel":
        """
        Train XGBoost model with early stopping.
        
        Args:
            X_train: Training features (N_train, F) - TABULAR
            y_train: Training targets (N_train,)
            X_val: Validation features (N_val, F)
            y_val: Validation targets (N_val,)
            feature_names: Optional feature names for interpretability
        
        Returns:
            Self (for method chaining)
        
        Raises:
            ValueError: If input validation fails
        """
        # Validate inputs
        self._validate_inputs(X_train, y_train, "train")
        self._validate_inputs(X_val, y_val, "val")
        
        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: train={X_train.shape[1]}, "
                f"val={X_val.shape[1]}"
            )
        
        logger.info("=" * 80)
        logger.info("STARTING XGBOOST TRAINING")
        logger.info("=" * 80)
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Features: {X_train.shape[1]}")
        
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
            if len(feature_names) != X_train.shape[1]:
                logger.warning(
                    f"Feature names count ({len(feature_names)}) != "
                    f"feature dimension ({X_train.shape[1]}). Ignoring names."
                )
                self.feature_names = None
        
        # Build model
        self.model = self._build_model()
        
        # Early stopping setup
        early_stopping_rounds = int(self.config.get("early_stopping_rounds", 50))
        
        # Train with early stopping
        eval_set = [(X_train, y_train), (X_val, y_val)]
        eval_names = ["train", "validation"]
        
        logger.info(f"Training with early stopping (rounds={early_stopping_rounds})")
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,  # Suppress iteration logs
        )
        
        # Store results
        self.evals_result = self.model.evals_result()
        self.best_iteration = self.model.best_iteration
        
        # Extract feature importance
        if self.model.feature_importances_ is not None:
            importance_values = self.model.feature_importances_
            
            if self.feature_names and len(self.feature_names) == len(importance_values):
                self.feature_importance = dict(zip(self.feature_names, importance_values))
            else:
                self.feature_importance = {
                    f"f{i}": v for i, v in enumerate(importance_values)
                }
            
            # Log top features
            top_features = sorted(
                self.feature_importance.items(),
                key=lambda x: -x[1]
            )[:10]
            
            logger.info("\nTop 10 important features:")
            for feat, imp in top_features:
                logger.info(f"  {feat}: {imp:.6f}")
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best iteration: {self.best_iteration}")
        
        if self.evals_result:
            train_rmse = self.evals_result["train"]["rmse"][-1]
            val_rmse = self.evals_result["validation"]["rmse"][-1]
            logger.info(f"Final train RMSE: {train_rmse:.6f}")
            logger.info(f"Final val RMSE: {val_rmse:.6f}")
        
        return self
    
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
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self._validate_inputs(X, None, "predict")
        
        predictions = self.model.predict(X)
        
        return predictions.astype(np.float32)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features (N_test, F)
            y_test: Test targets (N_test,)
        
        Returns:
            Dictionary of metrics (mse, mae, rmse)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self._validate_inputs(X_test, y_test, "test")
        
        # Predict
        y_pred = self.predict(X_test)
        
        # Compute metrics
        mse = float(np.mean((y_test - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_test - y_pred)))
        rmse = float(np.sqrt(mse))
        
        # Directional accuracy (optional)
        correct_direction = np.sum(np.sign(y_test) == np.sign(y_pred))
        directional_accuracy = float(correct_direction / len(y_test))
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "directional_accuracy": directional_accuracy,
        }
        
        logger.info("=" * 80)
        logger.info("TEST EVALUATION")
        logger.info("=" * 80)
        logger.info(f"MSE:  {mse:.6f}")
        logger.info(f"MAE:  {mae:.6f}")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"Directional Accuracy: {directional_accuracy:.2%}")
        logger.info("=" * 80)
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk (XGBoost JSON format)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from disk."""
        if self.model is None:
            self.model = self._build_model()
        
        self.model.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def _validate_inputs(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        split_name: str
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
            raise ValueError(
                f"{split_name} X must be 2D (N, F), got shape {X.shape}"
            )
        
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
                raise ValueError(
                    f"{split_name} y dtype must be float32 or float64"
                )
            
            if y.ndim != 1:
                raise ValueError(
                    f"{split_name} y must be 1D (N,), got shape {y.shape}"
                )
            
            if len(y) != len(X):
                raise ValueError(
                    f"{split_name} X and y length mismatch: {len(X)} vs {len(y)}"
                )
            
            if np.isnan(y).any():
                raise ValueError(f"{split_name} y contains NaN values")


def create_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict,
    seed: int = 42,
) -> Tuple[XGBoostModel, Dict]:
    """
    Factory function to create and train XGBoost model in one call.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        config: Model configuration dict
        seed: Random seed
    
    Returns:
        Tuple of (XGBoostModel instance, evaluation results)
    """
    model = XGBoostModel(config, seed=seed)
    model.train(X_train, y_train, X_val, y_val)
    
    return model, model.evals_result
