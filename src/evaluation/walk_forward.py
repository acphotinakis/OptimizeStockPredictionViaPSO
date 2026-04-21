"""
Canonical Walk-Forward Evaluation

Implements frozen-model, rolling-window walk-forward as defined in FINAL_PLAN.md Section 5.

CRITICAL RULES (FINAL_PLAN.md Section 5.2):
- Models are FROZEN (never retrained)
- Window size: 20 days (fixed)
- Step size: 1 day
- Window type: ROLLING (not expanding)
- NO model weight updates
- NO pipeline refitting

Author: System Architect
Version: CANONICAL 1.0
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CanonicalWalkForward:
    """
    Walk-forward evaluator with frozen models and rolling windows.
    
    FINAL_PLAN.md Section 5.1: Canonical Walk-Forward Algorithm
    
    This class enforces:
    - Static models (no retraining)
    - Rolling 20-day windows
    - 1-day ahead prediction
    - No pipeline refitting
    """
    
    def __init__(
        self,
        lookback: int = 20,
        step_size: int = 1,
    ):
        """
        Initialize walk-forward evaluator.
        
        Args:
            lookback: Window size in days (default: 20)
            step_size: Step size in days (default: 1)
        """
        self.lookback = lookback
        self.step_size = step_size
        
        logger.info("=" * 80)
        logger.info("CANONICAL WALK-FORWARD EVALUATOR (FINAL_PLAN.md)")
        logger.info("=" * 80)
        logger.info(f"Lookback: {lookback} days (rolling window)")
        logger.info(f"Step size: {step_size} day(s)")
        logger.info(f"Model state: FROZEN (never retrained)")
        logger.info(f"Pipeline state: FROZEN (never refitted)")
        logger.info("=" * 80)
    
    def evaluate_lstm(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate frozen LSTM model on test set.
        
        FINAL_PLAN.md Section 5.1: LSTM walk-forward
        
        Args:
            model: Trained LSTM model (FROZEN)
            X_test: Test features (N, F) tabular format
            y_test: Test targets (N,)
        
        Returns:
            Tuple of (predictions, actuals)
        """
        logger.info("Starting LSTM walk-forward evaluation...")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Features: {X_test.shape[1] if X_test.ndim > 1 else 'N/A'}")
        
        predictions = []
        
        # Walk through test set with rolling 20-day windows
        for t in range(self.lookback, len(X_test), self.step_size):
            # Rolling window: [t-20, t)
            X_window = X_test[t - self.lookback : t, :]  # Shape: (20, F)
            
            # Reshape for LSTM: (1, 20, F)
            X_input = X_window.reshape(1, self.lookback, -1)
            
            # Predict with FROZEN model (no weight updates)
            pred = model.predict(X_input, verbose=0)
            predictions.append(pred[0, 0] if pred.ndim > 1 else pred[0])
        
        predictions = np.array(predictions, dtype=np.float32)
        actuals = y_test[self.lookback :: self.step_size]
        
        # Align lengths (in case of truncation)
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]
        
        logger.info(f"Walk-forward complete: {len(predictions)} predictions")
        logger.info(f"  ⚠️  Model was NOT retrained (frozen)")
        
        return predictions, actuals
    
    def evaluate_xgboost(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate frozen XGBoost model on test set.
        
        FINAL_PLAN.md Section 5.1: XGBoost uses lag-based features
        
        Args:
            model: Trained XGBoost model (FROZEN)
            X_test: Test features (N, F) tabular format
            y_test: Test targets (N,)
        
        Returns:
            Tuple of (predictions, actuals)
        """
        logger.info("Starting XGBoost walk-forward evaluation...")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Features: {X_test.shape[1]}")
        
        # Build lag-based features for entire test set
        X_test_lag = self._build_lag_features(X_test)
        y_test_lag = y_test[self.lookback :]
        
        # Predict with FROZEN model
        predictions = model.predict(X_test_lag)
        
        logger.info(f"Walk-forward complete: {len(predictions)} predictions")
        logger.info(f"  ⚠️  Model was NOT retrained (frozen)")
        
        return predictions, y_test_lag
    
    def _build_lag_features(self, X: np.ndarray) -> np.ndarray:
        """
        Build lag-based features for XGBoost.
        
        FINAL_PLAN.md Section 4.3: XGBoost Feature Representation (Lag-Based)
        
        Converts tabular features to lag representation:
            Input: (N, F)
            Output: (N-lookback, F * (lookback+1))
        
        Each sample contains: [X[t], X[t-1], ..., X[t-20]]
        
        Args:
            X: Tabular features (N, F)
        
        Returns:
            Lag-based features (N-lookback, F*(lookback+1))
        """
        N, F = X.shape
        n_samples = N - self.lookback
        n_features = F * (self.lookback + 1)
        
        X_lag = np.zeros((n_samples, n_features), dtype=np.float32)
        
        for i in range(n_samples):
            # Current time: i + lookback
            # Window: [i : i+lookback+1] (includes current)
            window = X[i : i + self.lookback + 1, :]  # Shape: (21, F)
            X_lag[i, :] = window.flatten()  # Shape: (F*21,)
        
        return X_lag
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute standard evaluation metrics.
        
        Args:
            predictions: Model predictions
            actuals: Actual returns
        
        Returns:
            Dictionary of metrics
        """
        # Regression metrics
        mse = float(np.mean((actuals - predictions) ** 2))
        mae = float(np.mean(np.abs(actuals - predictions)))
        rmse = float(np.sqrt(mse))
        
        # R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
        
        # Directional accuracy
        correct_direction = np.sum(np.sign(actuals) == np.sign(predictions))
        directional_accuracy = float(correct_direction / len(actuals))
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "directional_accuracy": directional_accuracy,
            "n_predictions": len(predictions),
        }
        
        logger.info("Evaluation Metrics:")
        logger.info(f"  MSE:  {mse:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE:  {mae:.6f}")
        logger.info(f"  R²:   {r2:.4f}")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.2%}")
        
        return metrics
