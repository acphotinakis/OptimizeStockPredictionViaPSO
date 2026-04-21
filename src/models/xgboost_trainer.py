"""
XGBoost Training Module

High-level training pipeline for XGBoost models with:
- Feature representation strategy (tabular lag-based)
- Early stopping
- Feature importance analysis
- Deterministic training

TRD Compliance:
- Consumes features from unified pipeline
- No internal feature engineering
- Temporal validation split
- Regression objective (next-period return)

Author: System Architect
Version: 1.0.0 UNIFIED
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """
    Training pipeline for XGBoost models with TRD-compliant constraints.
    
    Features:
    - Early stopping on validation RMSE
    - Feature importance tracking
    - No temporal shuffling
    - Deterministic training
    
    Example:
        >>> trainer = XGBoostTrainer(config, seed=42)
        >>> model, history = trainer.train(X_train, y_train, X_val, y_val)
        >>> metrics = trainer.evaluate(model, X_test, y_test)
    """
    
    def __init__(
        self,
        config: Dict,
        seed: int = 42,
    ):
        """
        Initialize XGBoost trainer.
        
        Args:
            config: Configuration dictionary with XGBoost parameters
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        
        logger.info(f"XGBoostTrainer initialized with seed={seed}")
    
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
        
        Raises:
            ValueError: If input validation fails
        """
        logger.info("=" * 80)
        logger.info("XGBOOST TRAINING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Features: {X_train.shape[1]}")
        
        # Initialize and train model
        model = XGBoostModel(self.config, seed=self.seed)
        model.train(X_train, y_train, X_val, y_val, feature_names)
        
        # Extract history
        history = model.evals_result if model.evals_result else {}
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        
        return model, history
    
    def evaluate(
        self,
        model: XGBoostModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate trained model on test set.
        
        Args:
            model: Trained XGBoost model
            X_test: Test features (N_test, F)
            y_test: Test targets (N_test,)
        
        Returns:
            Dictionary of metrics
        """
        return model.evaluate(X_test, y_test)


def build_xgboost_lag_features(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build lag-based features for XGBoost from tabular data.
    
    XGBoost Feature Representation Strategy (TRD-Aligned):
    - Uses LAG-BASED representation (NOT flattened sequences)
    - Each sample contains current values + L previous lags
    - More interpretable for tree-based models
    - Lower dimensionality than flattened sequences
    
    Converts:
        X: (N, F) tabular features
        y: (N,) target vector
    Into:
        X_lagged: (N-lookback, F * (lookback+1)) with lag features
        y_aligned: (N-lookback,) aligned targets
    
    Args:
        X: Feature matrix (N, F)
        y: Target vector (N,)
        lookback: Number of lags to create (default: 20)
    
    Returns:
        Tuple of (X_lagged, y_aligned)
    
    Example:
        >>> X = np.random.randn(1000, 15)  # 1000 samples, 15 features
        >>> y = np.random.randn(1000)
        >>> X_lagged, y_lagged = build_xgboost_lag_features(X, y, lookback=20)
        >>> print(X_lagged.shape)  # (980, 315) = 15 features * 21 time points
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, F), got shape {X.shape}")
    
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (N,), got shape {y.shape}")
    
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    
    N, F = X.shape
    
    if N <= lookback:
        raise ValueError(
            f"Not enough samples ({N}) for lookback ({lookback}). "
            f"Need at least {lookback + 1} samples."
        )
    
    n_samples = N - lookback
    n_features_lagged = F * (lookback + 1)  # Current + lookback lags
    
    X_lagged = np.zeros((n_samples, n_features_lagged), dtype=np.float32)
    
    for i in range(n_samples):
        # For sample at position i+lookback (current time)
        # Include features from i+lookback (current) back to i (oldest)
        window = X[i : i + lookback + 1]  # Shape: (lookback+1, F)
        
        # Flatten: [f0_t, f1_t, ..., fF_t, f0_t-1, f1_t-1, ..., f0_t-lookback, ...]
        X_lagged[i] = window.flatten()
    
    # Align targets
    y_aligned = y[lookback:].astype(np.float32)
    
    logger.info(
        f"Built lag features: {X.shape} → {X_lagged.shape}, "
        f"lookback={lookback}, features={F}→{n_features_lagged}"
    )
    
    return X_lagged, y_aligned
