# src/models/lstm/lstm_model.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """Stacked LSTM with many-to-one output."""

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

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_units, output_size)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier / orthogonal init; forget-gate bias = 1."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.zero_()
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            lstm_out, _ = checkpoint(self.lstm, x, use_reentrant=False)
        else:
            lstm_out, _ = self.lstm(x)

        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


class LSTMTrainer:
    """Fixed trainer with proper gradient accumulation."""

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
        self.use_amp = False  # Disabled per original comment
        self.accumulation_steps = accumulation_steps
        self.seed = seed

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.criterion = nn.MSELoss()
        self._best_state: Optional[dict] = None
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "rmse": [],
        }

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def _predict_batches(self, X: np.ndarray, batch_size: int) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = self._to_tensor(X[i : i + batch_size])
                pred = self.model(X_batch).detach().cpu().numpy()
                preds.append(pred)
        return np.concatenate(preds).reshape(-1)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, list]:

        train_ds = TensorDataset(
            self._to_tensor(X_train), self._to_tensor(y_train).unsqueeze(-1)
        )
        val_ds = TensorDataset(
            self._to_tensor(X_val), self._to_tensor(y_val).unsqueeze(-1)
        )

        train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            pin_memory=torch.cuda.is_available(),
            num_workers=2 if torch.cuda.is_available() else 0,
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=2 if torch.cuda.is_available() else 0,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in tqdm(range(self.max_epochs), desc="Epochs"):
            self.model.train()
            epoch_loss = 0.0
            self.optimizer.zero_grad()  # Zero once at start of epoch

            for batch_idx, (x_b, y_b) in enumerate(train_dl):
                pred = self.model(x_b)
                # CRITICAL FIX: Scale loss by accumulation steps
                loss = self.criterion(pred, y_b) / self.accumulation_steps

                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite loss: {loss.item()}")

                loss.backward()

                # CRITICAL FIX: Only step every accumulation_steps
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Restore true loss value for logging
                epoch_loss += loss.item() * self.accumulation_steps * len(x_b)

            # Handle remaining gradients if dataset size not divisible by accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps != 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            avg_train_loss = epoch_loss / len(train_ds)

            # Validation
            y_pred = self._predict_batches(X_val, self.batch_size)
            val_loss = np.sqrt(np.mean((y_val - y_pred) ** 2))  # RMSE

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["rmse"].append(val_loss)

            logger.info(
                "[EPOCH %d] train_loss=%.6f | val_rmse=%.6f",
                epoch + 1,
                avg_train_loss,
                val_loss,
            )

            # Early stopping
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

        if self._best_state:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self._best_state.items()}
            )

        return self.history

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        return self._predict_batches(X, batch_size)
