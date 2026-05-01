import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys
import time
from scipy.stats import skew, kurtosis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .lstm_model import LSTMModel
from .utils import set_seeds
from src.evaluation.metrics import compute_and_log_all_statistical_metrics

logger = logging.getLogger(__name__)


class LSTMTrainer:
    """
    Unified training pipeline for LSTM models.

    This is the SINGLE canonical owner of LSTM training logic.
    It is used by both standalone training scripts and PSO hyperparameter loops.

    All hyperparameters are injected via the `config` dict, which must contain
    at minimum the keys required by the TRD specification.

    Example:
        >>> config = {
        ...     "optimizer": "adam",
        ...     "learning_rate": 0.001,
        ...     "loss": "mse",
        ...     "epochs": 100,
        ...     "batch_size": 32,
        ...     "shuffle": False,
        ...     "grad_clip": 1.0,
        ...     "use_amp": True,
        ...     "accumulation_steps": 1,
        ...     "early_stopping": {
        ...         "enabled": True,
        ...         "monitor": "val_loss",
        ...         "patience": 10,
        ...         "restore_best_weights": True,
        ...     },
        ... }
        >>> trainer = LSTMTrainer(config=config, seed=42)
        >>> model, history = trainer.train(X_train, y_train, X_val, y_val)
        >>> metrics = model.evaluate(X_test, y_test)
    """

    def __init__(
        self,
        lstm_model: LSTMModel,
        config: Dict,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        """
        Initialize LSTM trainer.

        Args:
            config: Configuration dictionary. See class docstring for full schema.
            seed: Random seed for reproducibility
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.config = config
        self.seed = seed
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.lstm_model = lstm_model

        # AMP scaler (lazy init)
        self._scaler: Optional[torch.cuda.amp.GradScaler] = None

        set_seeds(seed)
        logger.info(f"LSTMTrainer initialized on device: {self.device}")
        logger.info(f"Config: {config}")

    def _resolve_optimizer(self, model: nn.Module, lr: float) -> torch.optim.Optimizer:
        """Build optimizer from config string."""
        optimizer_name = str(self.config.get("optimizer", "adam")).lower()
        if optimizer_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _resolve_loss(self) -> nn.Module:
        """Build loss function from config string."""
        loss_name = str(self.config.get("loss", "mse")).lower()
        if loss_name in ("mse", "l2", "mean_squared_error"):
            return nn.MSELoss()
        elif loss_name in ("mae", "l1", "mean_absolute_error"):
            return nn.L1Loss()
        elif loss_name == "huber":
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

    def _resolve_early_stopping(self) -> Dict[str, Any]:
        """Normalize early_stopping config (supports nested dict or flat keys)."""
        es = self.config.get("early_stopping", {})
        if isinstance(es, dict):
            return {
                "enabled": bool(es.get("enabled", True)),
                "monitor": str(es.get("monitor", "val_loss")),
                "patience": int(es.get("patience", 10)),
                "restore_best_weights": bool(es.get("restore_best_weights", True)),
            }
        # Fallback to flat keys for PSO compatibility
        return {
            "enabled": bool(self.config.get("early_stopping_enabled", True)),
            "monitor": str(self.config.get("early_stopping_monitor", "val_loss")),
            "patience": int(self.config.get("patience", 10)),
            "restore_best_weights": bool(self.config.get("restore_best_weights", True)),
        }

    def _compute_grad_norm(self, model: nn.Module) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[LSTMModel, Dict[str, Any]]:

        import time

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------
        self._validate_inputs(X_train, y_train, "train")
        self._validate_inputs(X_val, y_val, "val")

        learning_rate = float(self.config["learning_rate"])
        epochs = int(self.config["epochs"])
        batch_size = int(self.config["batch_size"])
        shuffle = bool(self.config.get("shuffle", False))
        grad_clip = float(self.config.get("grad_clip", 0.0))
        use_amp = bool(self.config.get("use_amp", False)) and torch.cuda.is_available()
        accumulation_steps = max(1, int(self.config.get("accumulation_steps", 1)))

        es_cfg = self._resolve_early_stopping()
        patience = es_cfg["patience"]
        restore_best = es_cfg["restore_best_weights"]

        model = self.lstm_model.model
        if model is None:
            raise RuntimeError("Model not built")

        # -------------------------------------------------
        # DATA
        # -------------------------------------------------
        X_train_t = torch.from_numpy(X_train.astype(np.float32))
        y_train_t = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
        X_val_t = torch.from_numpy(X_val.astype(np.float32))
        y_val_t = torch.from_numpy(y_val.astype(np.float32)).view(-1, 1)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size,
            shuffle=shuffle,
        )

        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=batch_size,
            shuffle=False,
        )

        optimizer = self._resolve_optimizer(model, learning_rate)
        criterion = self._resolve_loss()

        scaler = (
            torch.amp.grad_scaler.GradScaler(device=self.device) if use_amp else None
        )

        # -------------------------------------------------
        # HISTORY (CLEAN STRUCTURE)
        # -------------------------------------------------
        history = {
            "train_loss": [],
            "val_loss": [],
            "grad_norm": [],
            "lr": [],
            "epoch_time": [],
            "train_metrics": [],
            "val_metrics": [],
            "variance_ratio": [],
            "val_loss_slope": [],
        }

        best_val_loss = float("inf")
        best_weights = None
        best_epoch = 0
        epochs_no_improve = 0

        # =================================================
        # TRAINING LOOP
        # =================================================
        for epoch in range(epochs):

            epoch_start = time.time()
            epoch_grad_norms = []

            train_losses = []
            train_preds_all = []
            train_true_all = []

            # -------------------------------------------------
            # TRAIN
            # -------------------------------------------------
            model.train()
            train_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{epochs} [Train]",
                leave=False,
            )

            for batch_idx, (batch_X, batch_y) in enumerate(train_bar):

                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                if use_amp:
                    with autocast(self.device):
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y) / accumulation_steps
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y) / accumulation_steps

                # backward
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # predictions
                train_preds_all.append(outputs.detach().cpu().numpy())
                train_true_all.append(batch_y.detach().cpu().numpy())

                # optimizer step (with accumulation). Grad norm is measured
                # AFTER AMP unscale + clipping so it reflects the gradient the
                # optimizer actually sees, not the loss-scale-inflated version.
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(
                    train_loader
                ):
                    if use_amp:
                        scaler.unscale_(optimizer)

                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    epoch_grad_norms.append(total_norm**0.5)

                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()

                train_losses.append(loss.item() * accumulation_steps)

            avg_train_loss = float(np.mean(train_losses))

            # flatten train
            train_preds_all = np.concatenate(train_preds_all).flatten()
            train_true_all = np.concatenate(train_true_all).flatten()

            # -------------------------------------------------
            # VALIDATION
            # -------------------------------------------------
            model.eval()
            val_bar = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{epochs} [Val]",
                leave=False,
            )

            val_losses = []
            val_preds_all = []
            val_true_all = []

            with torch.no_grad():
                for batch_X, batch_y in val_bar:

                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    if use_amp:
                        with autocast(self.device):
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                    else:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)

                    val_losses.append(loss.item())
                    val_preds_all.append(outputs.cpu().numpy())
                    val_true_all.append(batch_y.cpu().numpy())

            avg_val_loss = float(np.mean(val_losses))

            val_preds_all = np.concatenate(val_preds_all).flatten()
            val_true_all = np.concatenate(val_true_all).flatten()

            # -------------------------------------------------
            # METRICS (SINGLE SOURCE OF TRUTH)
            # -------------------------------------------------
            train_metrics = compute_and_log_all_statistical_metrics(
                y_true=train_true_all,
                y_pred=train_preds_all,
                label=f"Train Epoch {epoch+1}",
            )

            val_metrics = compute_and_log_all_statistical_metrics(
                y_true=val_true_all,
                y_pred=val_preds_all,
                label=f"Val Epoch {epoch+1}",
            )

            # -------------------------------------------------
            # ADDITIONAL FINANCIAL SIGNAL
            # -------------------------------------------------
            variance_ratio = float(
                np.var(val_preds_all) / (np.var(val_true_all) + 1e-8)
            )

            # -------------------------------------------------
            # LOSS SLOPE
            # -------------------------------------------------
            if len(history["val_loss"]) > 0:
                val_loss_slope = avg_val_loss - history["val_loss"][-1]
            else:
                val_loss_slope = 0.0

            # -------------------------------------------------
            # STORE HISTORY
            # -------------------------------------------------
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["grad_norm"].append(float(np.mean(epoch_grad_norms)))
            history["lr"].append(optimizer.param_groups[0]["lr"])
            history["epoch_time"].append(time.time() - epoch_start)

            history["train_metrics"].append(train_metrics)
            history["val_metrics"].append(val_metrics)

            history["variance_ratio"].append(variance_ratio)
            history["val_loss_slope"].append(val_loss_slope)

            # -------------------------------------------------
            # BEST-WEIGHTS BOOKKEEPING + EARLY STOPPING
            # -------------------------------------------------
            # Always track the best checkpoint so ``restore_best_weights`` works
            # regardless of whether early stopping is enabled. Only the
            # patience-based ``break`` is guarded by ``es_cfg["enabled"]``.
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if es_cfg["enabled"] and epochs_no_improve >= patience:
                break

        # -------------------------------------------------
        # RESTORE BEST MODEL
        # -------------------------------------------------
        if restore_best and best_weights is not None:
            model.load_state_dict(best_weights)

        model.eval()

        # -------------------------------------------------
        # FINAL RETURN STRUCTURE
        # -------------------------------------------------
        final_metrics = {
            "best_epoch": best_epoch + 1,
            "best_val_loss": float(best_val_loss),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "train_metrics_last": train_metrics,
            "val_metrics_last": val_metrics,
        }

        return self.lstm_model, {
            "history": history,
            "final_metrics": final_metrics,
        }

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray, split_name: str) -> None:
        """
        Validate input tensor shapes and dtypes.

        Args:
            X: Feature tensor (N, 20, F)
            y: Target vector (N,)
            split_name: Name of split for logging

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(X, np.ndarray):
            raise ValueError(f"{split_name} X must be numpy array")

        if X.dtype not in [np.float32, np.float64]:
            raise ValueError(
                f"{split_name} X dtype must be float32 or float64, got {X.dtype}"
            )

        if X.ndim != 3:
            raise ValueError(
                f"{split_name} X must be 3D (N, 20, F), got shape {X.shape}"
            )

        if X.shape[1] != 20:
            raise ValueError(f"{split_name} X must have timesteps=20, got {X.shape[1]}")

        if X.shape[2] < 1:
            raise ValueError(
                f"{split_name} X must have at least 1 feature, got {X.shape[2]}"
            )

        if np.isnan(X).any():
            raise ValueError(f"{split_name} X contains NaN values")

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise ValueError(f"{split_name} y must be numpy array")

            if y.dtype not in [np.float32, np.float64]:
                raise ValueError(f"{split_name} y dtype must be float32 or float64")

            if y.ndim != 1:
                raise ValueError(f"{split_name} y must be 1D (N,), got shape {y.shape}")

            if len(y) != len(X):
                raise ValueError(
                    f"{split_name} X and y length mismatch: {len(X)} vs {len(y)}"
                )

            if np.isnan(y).any():
                raise ValueError(f"{split_name} y contains NaN values")

    def __str__(self) -> str:

        es_cfg = self._resolve_early_stopping()

        model = self.lstm_model.model
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        else:
            total_params = 0
            trainable_params = 0

        payload = {
            "device": self.device,
            "seed": self.seed,
            "optimizer": self.config.get("optimizer", "adam"),
            "loss": self.config.get("loss", "mse"),
            "learning_rate": self.config.get("learning_rate"),
            "batch_size": self.config.get("batch_size"),
            "epochs": self.config.get("epochs"),
            "shuffle": self.config.get("shuffle", False),
            "grad_clip": self.config.get("grad_clip", 0.0),
            "use_amp": self.config.get("use_amp", False),
            "accumulation_steps": self.config.get("accumulation_steps", 1),
            "early_stopping": {
                "enabled": es_cfg["enabled"],
                "monitor": es_cfg["monitor"],
                "patience": es_cfg["patience"],
                "restore_best_weights": es_cfg["restore_best_weights"],
            },
            "model": {
                "total_params": total_params,
                "trainable_params": trainable_params,
            },
        }

        return json.dumps(payload, indent=2)

    def __repr__(self) -> str:
        return self.__str__()
