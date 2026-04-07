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

logger = logging.getLogger(__name__)


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
    """Manages training, early stopping, and checkpointing for LSTMModel.

    Args:
        model: LSTMModel instance.
        lr: Adam learning rate.
        max_epochs: Hard epoch cap.
        patience: Early stopping patience (validation loss).
        batch_size: Mini-batch size.
        device: Torch device ("cuda" or "cpu").
        grad_clip: Max gradient L2 norm.
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
        use_amp: bool = True,
        accumulation_steps: int = 1,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        # Phase 3: Mixed precision training
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.grad_scaler.GradScaler() if self.use_amp else None

        # Phase 3: Gradient accumulation
        self.accumulation_steps = accumulation_steps

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self._best_state: Optional[dict] = None
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}

        if self.use_amp:
            logger.debug("Mixed precision training enabled (FP16)")
        if self.accumulation_steps > 1:
            logger.debug(
                "Gradient accumulation: %d steps (effective batch=%d)",
                self.accumulation_steps,
                self.batch_size * self.accumulation_steps,
            )

    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, list]:
        """Train the model with early stopping.

        Args:
            X_train: [N_train, T, F]
            y_train: [N_train]
            X_val:   [N_val, T, F]
            y_val:   [N_val]

        Returns:
            Training history dict with 'train_loss' and 'val_loss' lists.
        """
        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(-1),
        )

        # Adjust batch size for gradient accumulation
        physical_batch_size = self.batch_size // self.accumulation_steps

        train_dl = DataLoader(
            train_ds,
            batch_size=physical_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(42),
        )

        # Create validation DataLoader to avoid OOM
        val_ds = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val).unsqueeze(-1),
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=physical_batch_size,
            shuffle=False,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        self.history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.max_epochs):
            # ---- Training ----
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, (x_b, y_b) in enumerate(train_dl):
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)

                # Phase 3: Mixed precision training
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        pred = self.model(x_b)
                        loss = self.criterion(pred, y_b)

                    # Normalize loss for gradient accumulation
                    loss = loss / self.accumulation_steps

                    # Scaled backward pass
                    self.scaler.scale(loss).backward()

                    # Update weights every N steps
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    pred = self.model(x_b)
                    loss = self.criterion(pred, y_b)

                    # Normalize loss for gradient accumulation
                    loss = loss / self.accumulation_steps
                    loss.backward()

                    # Update weights every N steps
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                epoch_loss += loss.item() * len(x_b) * self.accumulation_steps

            avg_train = epoch_loss / len(train_ds)

            # ---- Validation ----
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_val_b, y_val_b in val_dl:
                    x_val_b = x_val_b.to(self.device)
                    y_val_b = y_val_b.to(self.device)
                    val_pred_b = self.model(x_val_b)
                    val_loss += self.criterion(val_pred_b, y_val_b).item() * len(x_val_b)
            val_loss = val_loss / len(val_ds)

            self.history["train_loss"].append(avg_train)
            self.history["val_loss"].append(val_loss)

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
                    break

        # Restore best weights
        if self._best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self._best_state.items()}
            )
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on a numpy array."""
        return self.model.predict(X, device=self.device)
