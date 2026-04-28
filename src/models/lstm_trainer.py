"""
LSTM Training Module

Unified, production-grade training pipeline for LSTM models with:
- Config-driven hyperparameter injection
- Early stopping on validation loss
- Gradient clipping
- Automatic Mixed Precision (AMP)
- Gradient accumulation
- Deterministic training
- Temporal integrity enforcement
- GPU/CPU automatic selection

TRD Compliance:
- No shuffling (shuffle=False mandatory)
- Temporal validation split
- MSE loss, Adam optimizer
- Early stopping (patience configurable)

Author: System Architect
Version: 2.1.0 CONFIG-COMPLIANT
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .lstm_model import LSTMModel
from .utils import set_seeds

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

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[LSTMModel, Dict[str, List[float]]]:
        """
        Train LSTM model with early stopping.

        Training Protocol:
        - Fit on training set
        - Monitor validation loss
        - Early stop if no improvement for `patience` epochs
        - Restore best weights (if configured)
        - No temporal shuffling
        - Gradient clipping (if configured)
        - AMP (if configured and CUDA available)
        - Gradient accumulation (if configured)

        Args:
            X_train: Training features (N_train, 20, F)
            y_train: Training targets (N_train,)
            X_val: Validation features (N_val, 20, F)
            y_val: Validation targets (N_val,)

        Returns:
            Tuple of (trained_model, training_history)

        Raises:
            ValueError: If input shapes invalid
        """
        # Validate inputs
        self._validate_inputs(X_train, y_train, "train")
        self._validate_inputs(X_val, y_val, "val")

        # Extract hyperparameters from config
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

        logger.info("=" * 80)
        logger.info("STARTING LSTM TRAINING")
        logger.info("=" * 80)
        logger.info(f"LSTM Model {self.lstm_model}")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Shuffle: {shuffle}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Optimizer: {self.config.get('optimizer', 'adam')}")
        logger.info(f"Loss: {self.config.get('loss', 'mse')}")
        logger.info(
            f"Grad clip: {grad_clip}, AMP: {use_amp}, Accum steps: {accumulation_steps}"
        )
        logger.info(
            f"Early stopping: enabled={es_cfg['enabled']}, patience={patience}, restore_best={restore_best}"
        )

        # Build model wrapper and network
        model = self.lstm_model.model
        if model is None:
            raise RuntimeError("Failed to build model: model_wrapper.model is None")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model: {total_params:,} total params, {trainable_params:,} trainable"
        )

        # Convert to tensors
        X_train_t = torch.from_numpy(X_train.astype(np.float32))
        y_train_t = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
        X_val_t = torch.from_numpy(X_val.astype(np.float32))
        y_val_t = torch.from_numpy(y_val.astype(np.float32)).view(-1, 1)

        # Create DataLoaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        # Optimizer and loss
        optimizer = self._resolve_optimizer(model, learning_rate)
        criterion = self._resolve_loss()

        # AMP scaler
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        # Training loop with early stopping
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_weights = None
        best_epoch = 0
        epochs_no_improve = 0

        logger.info("=" * 80)
        logger.info("TRAINING LOOP")
        logger.info("=" * 80)

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_losses = []
            train_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{epochs} [Train]",
                leave=False,
            )

            for batch_idx, (batch_X, batch_y) in enumerate(train_bar):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass with optional AMP
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss = loss / accumulation_steps
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss = loss / accumulation_steps

                # Backward pass
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Optimizer step (with accumulation)
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(
                    train_loader
                ):
                    if grad_clip > 0:
                        if use_amp and scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                    if use_amp and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()

                train_losses.append(loss.item() * accumulation_steps)

            avg_train_loss = np.mean(train_losses)

            # Validation phase
            model.eval()
            val_losses = []
            val_bar = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{epochs} [Val]",
                leave=False,
            )

            with torch.no_grad():
                for batch_X, batch_y in val_bar:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                    else:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)

                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)

            # Record history
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            # Early stopping check
            if es_cfg["enabled"]:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    best_weights = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                # Always save last epoch if early stopping disabled
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

            logger.info(
                f"Epoch {epoch+1:3d}/{epochs}: "
                f"train_loss={avg_train_loss:.6f}, "
                f"val_loss={avg_val_loss:.6f}, "
                f"best={best_val_loss:.6f} @ epoch {best_epoch+1}"
            )

            # Early stopping trigger
            if es_cfg["enabled"] and epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Restore best weights
        if restore_best and best_weights is not None:
            model.load_state_dict(best_weights)
            logger.info(
                f"Restored best weights from epoch {best_epoch+1} "
                f"(val_loss={best_val_loss:.6f})"
            )
        elif not restore_best and best_weights is not None:
            logger.info(
                f"Keeping final weights (restore_best_weights=False). "
                f"Best val_loss={best_val_loss:.6f} @ epoch {best_epoch+1}"
            )

        # ============================
        # HARD FREEZE (CRITICAL)
        # ============================
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        logger.info("MODEL HARD-FROZEN (requires_grad=False)")

        # ============================

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total epochs: {len(history['train_loss'])}")
        logger.info(f"Best epoch: {best_epoch + 1}")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Final train loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"Final val loss: {history['val_loss'][-1]:.6f}")

        return self.lstm_model, history

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
