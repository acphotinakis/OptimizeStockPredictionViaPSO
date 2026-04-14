"""
src/models/lstm/trainer.py

LSTMTrainer — training loop, early stopping, gradient clipping.
Does NOT contain model architecture or inference logic.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .model import LSTMModel

logger = logging.getLogger(__name__)


class LSTMTrainer:
    """Trains an LSTMModel with early stopping.

    Args:
        model:               LSTMModel instance to train.
        lr:                  Adam learning rate.
        max_epochs:          Maximum training epochs.
        patience:            Early stopping patience (epochs without improvement).
        batch_size:          Mini-batch size.
        device:              Torch device string ('cuda' or 'cpu').
        grad_clip:           Gradient norm clipping threshold.
        accumulation_steps:  Gradient accumulation steps.
        seed:                DataLoader shuffle seed.
    """

    def __init__(
        self,
        model: LSTMModel,
        lr: float,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 256,
        device: Optional[str] = None,
        grad_clip: float = 1.0,
        accumulation_steps: int = 1,
        seed: int = 42,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.accumulation_steps = accumulation_steps
        self.seed = seed

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.criterion = nn.MSELoss()
        self._best_state: Optional[dict] = None
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, list]:
        """Train the model, return history.

        Returns:
            Dict with 'train_loss' and 'val_loss' lists (one entry per epoch).
        """
        train_dl, val_dl = self._make_loaders(X_train, y_train, X_val, y_val)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in tqdm(range(self.max_epochs), desc="Epochs", leave=False):
            train_loss = self._train_epoch(train_dl)
            val_loss = self._validate(X_val, y_val)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            logger.info(
                "[Epoch %d/%d] train_loss=%.6f  val_rmse=%.6f",
                epoch + 1,
                self.max_epochs,
                train_loss,
                val_loss,
            )

            if val_loss < best_val_loss - 1e-7:
                best_val_loss = val_loss
                patience_counter = 0
                self._best_state = {
                    k: v.clone().cpu() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if self._best_state:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self._best_state.items()}
            )

        return self.history

    # ------------------------------------------------------------------

    def _train_epoch(self, train_dl: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        self.optimizer.zero_grad()

        for batch_idx, (x_b, y_b) in enumerate(train_dl):
            pred = self.model(x_b)
            loss = self.criterion(pred, y_b) / self.accumulation_steps

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss: {loss.item()}")

            loss.backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps * len(x_b)
            n_samples += len(x_b)

        # Flush remaining gradients
        if (batch_idx + 1) % self.accumulation_steps != 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / max(n_samples, 1)

    def _validate(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Return val RMSE without building a DataLoader (fast path)."""
        preds = self._predict_batches(X_val)
        return float(np.sqrt(np.mean((y_val - preds) ** 2)))

    def _predict_batches(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                x_b = torch.as_tensor(
                    X[i : i + self.batch_size], dtype=torch.float32, device=self.device
                )
                preds.append(self.model(x_b).cpu().numpy())
        return np.concatenate(preds).reshape(-1)

    def _make_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        def _ds(X, y):
            return TensorDataset(
                torch.as_tensor(X, dtype=torch.float32),
                torch.as_tensor(y, dtype=torch.float32).unsqueeze(-1),
            )

        pin = torch.cuda.is_available()
        workers = 2 if pin else 0
        train_dl = DataLoader(
            _ds(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            pin_memory=pin,
            num_workers=workers,
        )
        val_dl = DataLoader(
            _ds(X_val, y_val),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=pin,
            num_workers=workers,
        )
        return train_dl, val_dl
