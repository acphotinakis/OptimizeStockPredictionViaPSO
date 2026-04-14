"""
src/models/lstm/inference.py

LSTMWrapper — implements BaseModel.
Composes LSTMModel + LSTMTrainer into the unified interface used by pipelines.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.models.base import BaseModel
from src.models.lstm.model import LSTMModel
from src.models.lstm.trainer import LSTMTrainer

logger = logging.getLogger(__name__)

# Sensible defaults (overridden by config or CLI)
DEFAULT_PARAMS: Dict[str, Any] = {
    "num_layers": 2,
    "hidden_units": 128,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "lookback": 30,
    "max_epochs": 100,
    "patience": 10,
    "batch_size": 256,
    "grad_clip": 1.0,
    "accumulation_steps": 1,
    "use_checkpointing": False,
}


class LSTMWrapper(BaseModel):
    """Full LSTM model: architecture + training + inference.

    This is what pipelines instantiate — not LSTMModel or LSTMTrainer directly.

    Args:
        input_size: Feature dimension F (required).
        device:     Torch device override ('cuda' / 'cpu').
        **params:   Hyperparameter overrides (merged with DEFAULT_PARAMS).
    """

    def __init__(
        self,
        input_size: int,
        device: Optional[str] = None,
        **params,
    ) -> None:
        self.input_size = input_size
        self.params = {**DEFAULT_PARAMS, **params}
        self.device = device

        arch = LSTMModel(
            input_size=input_size,
            num_layers=self.params["num_layers"],
            hidden_units=self.params["hidden_units"],
            dropout=self.params["dropout"],
            use_checkpointing=self.params["use_checkpointing"],
        )
        self._trainer = LSTMTrainer(
            model=arch,
            lr=self.params["learning_rate"],
            max_epochs=self.params["max_epochs"],
            patience=self.params["patience"],
            batch_size=self.params["batch_size"],
            device=device,
            grad_clip=self.params["grad_clip"],
            accumulation_steps=self.params["accumulation_steps"],
        )

        logger.info(
            "[LSTMWrapper] input_size=%d  layers=%d  hidden=%d  lr=%.5f  lookback=%d",
            input_size,
            self.params["num_layers"],
            self.params["hidden_units"],
            self.params["learning_rate"],
            self.params["lookback"],
        )

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        return self._trainer.fit(X_train, y_train, X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._trainer._predict_batches(X)

    def save(self, path: str) -> None:
        torch.save(self._trainer.model.state_dict(), path)
        logger.info("LSTM weights saved → %s", path)

    @classmethod
    def load(
        cls, path: str, input_size: int, device: Optional[str] = None, **params
    ) -> "LSTMWrapper":
        """Restore a saved LSTMWrapper.

        The caller must supply ``input_size`` and the same hyperparams
        used during training (so the architecture matches the checkpoint).
        """
        wrapper = cls(input_size=input_size, device=device, **params)
        state = torch.load(
            path, map_location=wrapper._trainer.device, weights_only=True
        )
        wrapper._trainer.model.load_state_dict(state)
        logger.info("LSTM weights loaded ← %s", path)
        return wrapper

    def get_params(self) -> Dict[str, Any]:
        return {**self.params, "input_size": self.input_size}
