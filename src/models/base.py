"""
src/models/base.py

Abstract base class for all models.  Every model (LSTM, XGBoost, etc.)
must implement this interface so pipelines can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseModel(ABC):
    """Minimal interface shared by all prediction models."""

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            X_train: [N_train, T, F] windowed feature tensor.
            y_train: [N_train] target log returns.
            X_val:   [N_val, T, F] windowed feature tensor.
            y_val:   [N_val] target log returns.

        Returns:
            Training history dict (keys vary by model, e.g. train_loss, val_rmse).
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            X: [N, T, F] windowed feature tensor.

        Returns:
            [N] float32 predictions.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model weights / booster to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs) -> "BaseModel":
        """Restore a saved model from disk."""

    # ------------------------------------------------------------------
    # Optional helpers (override as needed)
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameters for serialisation."""
        return {}
