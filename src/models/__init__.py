"""
Unified Model Layer - Production-Grade LSTM & XGBoost System

This module provides canonical, TRD-compliant modeling systems for
financial time-series prediction. All duplicate and experimental implementations
have been eliminated.

Public API:

LSTM Models:
- LSTMModel: Main LSTM model class (PyTorch-based)
- LSTMTrainer: Training pipeline with early stopping
- create_lstm_model: Factory function for model creation
- build_lstm_windows: Temporal windowing for LSTM sequences

XGBoost Models:
- XGBoostModel: Main XGBoost regressor class
- XGBoostTrainer: Training pipeline for XGBoost
- create_xgboost_model: Factory function for XGBoost
- build_xgboost_lag_features: Lag-based feature construction

Architecture:
- LSTM: 2-layer LSTM (configurable via PSO)
  - ReLU activation, Dropout regularization
  - Input: (batch, 20, F)
  - Output: (batch, 1) - next-period return prediction

- XGBoost: Gradient boosted trees
  - Regression objective (next-period return)
  - Input: (N, F) tabular features
  - Output: (N,) predictions

TRD Compliance:
- No temporal shuffling
- No feature engineering in model layer
- No normalization in model layer
- Early stopping (validation-based)
- Deterministic training with seed control

Version: 1.0.0 UNIFIED CANONICAL
"""

from .ipso_optimizer import IPSOOptimizer
from .baseline_lstm_model import LSTMModel, LSTMNetwork, create_lstm_model, set_seeds
from .baseline_lstm_trainer import LSTMTrainer
from .utils import build_lstm_windows, load_model_weights, save_model_weights
from .xgboost_model import XGBoostModel, create_xgboost_model
from .xgboost_trainer import XGBoostTrainer, build_xgboost_lag_features

__all__ = [
    # LSTM
    "LSTMModel",
    "LSTMNetwork",
    "LSTMTrainer",
    "create_lstm_model",
    
    # XGBoost
    "XGBoostModel",
    "XGBoostTrainer",
    "create_xgboost_model",
    
    # PSO
    "IPSOOptimizer",
    
    # Utilities
    "build_lstm_windows",
    "build_xgboost_lag_features",
    "save_model_weights",
    "load_model_weights",
    "set_seeds",
]

__version__ = "1.0.0"
