"""
Unified Model Layer - Production-Grade LSTM & XGBoost System

This module provides canonical, TRD-compliant modeling systems for
financial time-series prediction.

Public API:

LSTM Models:
- LSTMModel: Main LSTM model class (PyTorch-based)
- LSTMTrainer: Training pipeline with early stopping
- create_lstm_model: Factory function for model creation

XGBoost Models:
- XGBoostModel: Main XGBoost regressor class
- XGBoostTrainer: Training pipeline for XGBoost
- build_xgboost_lag_features: Lag-based feature construction

Data Utilities (moved to src.data):
- build_lstm_windows: Temporal windowing for LSTM sequences
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

Version: 2.0.0 REFACTORED
"""

from .lstm_model import LSTMModel, LSTMNetwork
from .lstm_trainer import LSTMTrainer
from .utils import set_seeds
from .xgboost_model import XGBoostModel
from .xgboost_trainer import XGBoostTrainer

__all__ = [
    # LSTM
    "LSTMModel",
    "LSTMNetwork",
    "LSTMTrainer",
    # XGBoost
    "XGBoostModel",
    "XGBoostTrainer",
    # Utilities
    "set_seeds",
]

__version__ = "2.0.0"
