"""
Base Model and Trainer Abstractions

Defines minimal protocols to enforce consistent interfaces across
LSTM and XGBoost implementations without adding unnecessary complexity.

Author: System Architect
Version: 2.0.0 - MODEL SIMPLIFICATION
Source: MODEL_SIMPLIFICATION_AUDIT.md Section 4.4
"""

from typing import Protocol, Dict, Tuple, Any, Optional
import numpy as np


class BaseModel(Protocol):
    """
    Base protocol for all prediction models.
    
    Enforces consistent API for:
    - Prediction (inference)
    - Evaluation (metric computation)
    - Persistence (save/load)
    
    This is a Protocol (structural subtyping), not a base class,
    so implementations don't need to explicitly inherit.
    """
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions from features.
        
        Args:
            X: Input features (shape depends on model type)
        
        Returns:
            Predictions as (N,) array
        """
        ...
    
    def evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        target_scaler: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X: Test features
            y: Test targets
            target_scaler: Optional scaler for inverse transform
        
        Returns:
            Dictionary of metric name -> value
        """
        ...
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: File path to save to
        """
        ...
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: File path to load from
        """
        ...


class BaseTrainer(Protocol):
    """
    Base protocol for all model trainers.
    
    Enforces consistent training API:
    - All trainers accept train + validation data
    - All trainers accept config dict (not scattered kwargs)
    - All trainers return (model, history) tuple
    - All trainers support reproducible training via seed
    
    Trainers OWN training logic. Models handle inference only.
    """
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
        seed: int = 42,
        **kwargs: Any
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Train model with validation-based early stopping.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            config: Model hyperparameters (dict for flexibility)
            seed: Random seed for reproducibility
            **kwargs: Additional trainer-specific parameters
        
        Returns:
            Tuple of (trained_model, training_history)
            
        History should include at minimum:
            - "train_loss": List[float] - Training losses per epoch
            - "val_loss": List[float] - Validation losses per epoch
        """
        ...
