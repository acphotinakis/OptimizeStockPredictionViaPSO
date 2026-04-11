"""
src/models/lstm_model.py

Configurable stacked LSTM for many-to-one time-series regression.
All architecture parameters are passed at construction time, allowing
the PSO optimizer to vary them freely.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

from src.evaluation.metrics import all_statistical_metrics


class LSTMModel(nn.Module):
    """Stacked LSTM with many-to-one output.

    Args:
        input_size: Number of features (F) per timestep.
        num_layers: Number of stacked LSTM layers (1–4).
        hidden_units: Hidden state dimension (H).
        dropout: Dropout probability applied inter-layer and pre-output.
        output_size: Prediction dimension (1 for regression).
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        hidden_units: int,
        dropout: float,
        output_size: int = 1,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.dropout_rate = dropout
        self.use_checkpointing = use_checkpointing

        # Stacked LSTM — dropout applied between layers (not after last)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,  # input shape: [batch, seq, features]
        )

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_units, output_size)
        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier / orthogonal init; forget-gate bias = 1."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.zero_()
                # Set forget-gate bias to 1 (Jozefowicz et al., 2015)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, seq_len, input_size]

        Returns:
            out: [batch, 1]
        """
        # Phase 3: Gradient checkpointing to save memory during training
        if self.use_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            lstm_out, _ = checkpoint(self.lstm, x, use_reentrant=False)
        else:
            lstm_out, _ = self.lstm(x)  # [B, T, H]

        last_hidden = lstm_out[:, -1, :]  # many-to-one: last timestep
        out = self.dropout(last_hidden)
        out = self.fc(out)  # [B, 1]
        return out

    def predict(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Numpy convenience wrapper for inference.

        Args:
            X: [N, T, F] input array.
            device: Torch device string.

        Returns:
            [N] prediction array.
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(device)
            preds = self.forward(x_tensor).squeeze(-1)
        return preds.cpu().numpy()


# ======================================================================
# Trainer
# ======================================================================


class LSTMTrainer:
    """Manages training, early stopping, gradient accumulation, and checkpointing for LSTMModel."""

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 256,
        device: Optional[str] = None,
        grad_clip: float = 1.0,
        use_amp: bool = True,
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
        # LINE BELOW CAUSED MODEL NOT TO LEARN (IDK WHY JUST IGNORE IT)
        # self.use_amp = use_amp and torch.cuda.is_available()
        self.use_amp = False
        self.accumulation_steps = accumulation_steps
        self.seed = seed

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.criterion = nn.MSELoss()
        self.scaler = torch.amp.grad_scaler.GradScaler() if self.use_amp else None
        self._best_state: Optional[dict] = None
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}

        if self.use_amp:
            logger.info("Mixed precision training enabled (FP16)")
        if self.accumulation_steps > 1:
            logger.info(
                "Gradient accumulation: %d steps (effective batch=%d)",
                self.accumulation_steps,
                self.batch_size * self.accumulation_steps,
            )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, list]:
        """Train the model with early stopping."""

        # TensorDatasets and DataLoaders
        train_ds = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(-1)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(-1)
        )

        effective_batch = self.batch_size // self.accumulation_steps
        train_dl = DataLoader(
            train_ds,
            batch_size=effective_batch,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_dl = DataLoader(val_ds, batch_size=effective_batch, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        # self.history = {"train_loss": [], "val_loss": []}
        self.history = {
            k: []
            for k in [
                "train_loss",
                "val_loss",
                "rmse",
                "mae",
                "mape",
                "r2",
                "directional_accuracy",
                "f1_ternary",
            ]
        }

        for epoch in tqdm(range(self.max_epochs), desc="Epochs", unit="epoch"):
            self.model.train()
            epoch_loss = 0.0

            # Batch-level progress bar
            for batch_idx, (x_b, y_b) in enumerate(
                tqdm(train_dl, desc=f"Epoch {epoch+1}", leave=False, unit="batch")
            ):
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)

                # Forward + backward
                if self.use_amp:
                    with torch.amp.autocast_mode.autocast(device_type="cuda"):
                        pred = self.model(x_b)
                        loss = self.criterion(pred, y_b)
                    loss = loss / self.accumulation_steps
                    self.scaler.scale(loss).backward()
                else:
                    pred = self.model(x_b)
                    loss = self.criterion(pred, y_b)
                    loss = loss / self.accumulation_steps

                    # Check for NaN/Inf in loss
                    if not torch.isfinite(loss):
                        logger.error(f"Non-finite loss detected: {loss.item()}")
                        raise ValueError(
                            f"Training failed: non-finite loss {loss.item()}"
                        )

                    loss.backward()

                # Update weights every accumulation_steps
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += (
                    loss.item() * len(x_b) * self.accumulation_steps
                )  # restore true batch contribution

            # Apply final partial accumulation step if needed
            if (batch_idx + 1) % self.accumulation_steps != 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            avg_train_loss = epoch_loss / len(train_ds)

            # ---- Validation ----
            # ---- Validation ----
            self.model.eval()
            val_loss = 0.0

            y_val_true = []
            y_val_pred = []

            with torch.no_grad():
                for x_val_b, y_val_b in val_dl:
                    x_val_b, y_val_b = x_val_b.to(self.device), y_val_b.to(self.device)

                    preds = self.model(x_val_b)

                    # Loss accumulation
                    val_loss += self.criterion(preds, y_val_b).item() * len(x_val_b)

                    # Collect predictions for metrics
                    y_val_pred.append(preds.detach().cpu().numpy().flatten())
                    y_val_true.append(y_val_b.detach().cpu().numpy().flatten())

            val_loss /= len(val_ds)

            # Concatenate full validation predictions
            y_val_true = np.concatenate(y_val_true)
            y_val_pred = np.concatenate(y_val_pred)

            # Compute metrics
            metrics = all_statistical_metrics(
                y_val_true,
                y_val_pred,
                label=f"LSTM Epoch {epoch+1}",
            )

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["rmse"].append(metrics["rmse"])
            self.history["mae"].append(metrics["mae"])
            self.history["mape"].append(metrics["mape"])
            self.history["r2"].append(metrics["r2"])
            self.history["directional_accuracy"].append(metrics["directional_accuracy"])
            self.history["f1_ternary"].append(metrics["f1_ternary"])

            # Log everything in one line
            logger.info(
                "[EPOCH %d] train_loss=%.6f | val_loss=%.6f | RMSE=%.6f | MAE=%.6f | DA=%.4f | F1=%.4f | R2=%.4f",
                epoch + 1,
                avg_train_loss,
                val_loss,
                metrics["rmse"],
                metrics["mae"],
                metrics["directional_accuracy"],
                metrics["f1_ternary"],
                metrics["r2"],
            )

            # ---- Early stopping ----
            if val_loss < best_val_loss - 1e-7:
                best_val_loss = val_loss
                patience_counter = 0
                self._best_state = {
                    k: v.clone().cpu() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best weights
        if self._best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self._best_state.items()}
            )

        return self.history

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = torch.FloatTensor(X[i : i + batch_size]).to(self.device)
                pred_batch = self.model(X_batch).cpu().numpy()
                preds.append(pred_batch)
        return np.vstack(preds)
