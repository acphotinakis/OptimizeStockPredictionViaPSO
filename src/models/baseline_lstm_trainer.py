"""
LSTM Training Module

Production-grade training pipeline for LSTM models with:
- Early stopping
- Deterministic training
- Temporal integrity enforcement
- Validation-based model selection

TRD Compliance:
- No shuffling (shuffle=False mandatory)
- Temporal validation split
- MSE loss, Adam optimizer
- Early stopping (patience=10)

Author: System Architect
Version: 1.0.0 UNIFIED
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .baseline_lstm_model import LSTMNetwork, set_seeds

logger = logging.getLogger(__name__)


class LSTMTrainer:
    """
    Training pipeline for LSTM models with TRD-compliant constraints.

    Features:
    - Early stopping on validation loss
    - Best model checkpoint restoration
    - No temporal shuffling
    - Deterministic training
    - GPU/CPU automatic selection

    Example:
        >>> trainer = LSTMTrainer(config, seed=42)
        >>> model, history = trainer.train(X_train, y_train, X_val, y_val)
        >>> test_metrics = trainer.evaluate(model, X_test, y_test)
    """

    def __init__(self, config: Dict, seed: int = 42, device: Optional[str] = None):
        """
        Initialize LSTM trainer.

        Args:
            config: Configuration dictionary with:
                - lstm_units_1: int (50-300)
                - lstm_units_2: int (20-200)
                - dropout_rate: float (0.0-0.5)
                - learning_rate: float
                - epochs: int (50-300)
                - batch_size: int (32 or 64)
                - early_stopping: dict with patience, etc.
            seed: Random seed for reproducibility
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.config = config
        self.seed = seed
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Set seeds for reproducibility
        set_seeds(seed)

        logger.info(f"LSTMTrainer initialized on device: {self.device}")
        logger.info(f"Config: {config}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        lstm_units_1: int,
        lstm_units_2: int,
        dropout_rate: float,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        patience: int,
        shuffle: bool = False,
    ) -> Tuple[LSTMNetwork, Dict[str, List[float]]]:
        """
        Train LSTM model with early stopping.

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

        input_size = X_train.shape[2]

        logger.info("=" * 80)
        logger.info("STARTING LSTM TRAINING")
        logger.info("=" * 80)
        logger.info(f"Architecture: {lstm_units_1} → {lstm_units_2} → 1")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Input features: {input_size}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}, Dropout: {dropout_rate}")
        logger.info(f"Early stopping patience: {patience}")

        # Build model
        model = LSTMNetwork(
            input_size=input_size,
            hidden_size_1=lstm_units_1,
            hidden_size_2=lstm_units_2,
            dropout_rate=dropout_rate,
        ).to(self.device)

        # Log model parameters
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

        # Create DataLoaders (shuffle=False for temporal integrity)
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # MANDATORY: No shuffling
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
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_weights = None
        best_epoch = 0
        epochs_no_improve = 0

        logger.info("=" * 80)
        logger.info("STARTING LSTM TRAINING")
        logger.info("=" * 80)

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_losses = []
            train_bar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False
            )

            for batch_X, batch_y in train_bar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation phase
            model.eval()
            val_losses = []
            val_bar = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False
            )

            with torch.no_grad():
                for batch_X, batch_y in val_bar:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)

            # Record history
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Logging
            # if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs}: "
                f"train_loss={avg_train_loss:.6f}, "
                f"val_loss={avg_val_loss:.6f}, "
                f"best={best_val_loss:.6f} @ epoch {best_epoch+1}"
            )

            # Early stopping trigger
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Restore best weights
        if best_weights is not None:
            model.load_state_dict(best_weights)
            logger.info(
                f"Restored best weights from epoch {best_epoch+1} "
                f"(val_loss={best_val_loss:.6f})"
            )

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total epochs: {len(history['train_loss'])}")
        logger.info(f"Best epoch: {best_epoch + 1}")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Final train loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"Final val loss: {history['val_loss'][-1]:.6f}")

        return model, history

    def evaluate(
        self,
        model: LSTMNetwork,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_scaler=None,
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        CRITICAL: If target_scaler provided, metrics computed on ORIGINAL scale.
        
        Args:
            model: Trained LSTM model
            X_test: Test features (N_test, 20, F)
            y_test: Test targets (N_test,) - can be scaled or original
            target_scaler: Optional scaler to inverse-transform predictions
        
        Returns:
            Dictionary of metrics (mse, mae, rmse, r2)
        """
        self._validate_inputs(X_test, y_test, "test")
        
        model.eval()
        
        X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(self.device)
        
        with torch.no_grad():
            predictions = model(X_test_t)
        
        y_pred_scaled = predictions.cpu().numpy().flatten()
        
        # Inverse transform if scaler provided
        if target_scaler is not None:
            y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            metric_space = "original"
            logger.info("✓ Inverse transform applied (metrics on ORIGINAL scale)")
        else:
            y_pred = y_pred_scaled
            y_true = y_test
            metric_space = "scaled"
            logger.info("⚠️  No inverse transform (metrics on SCALED space)")
        
        # Compute metrics
        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(mse))
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
        
        # Directional accuracy
        correct_dir = np.sum(np.sign(y_true) == np.sign(y_pred))
        directional_accuracy = float(correct_dir / len(y_true))
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "directional_accuracy": directional_accuracy,
            "n_samples": len(y_true),
            "metric_space": metric_space,
        }
        
        logger.info("=" * 80)
        logger.info(f"TEST EVALUATION ({metric_space.upper()} SCALE)")
        logger.info("=" * 80)
        logger.info(f"MSE:  {mse:.6f}")
        logger.info(f"MAE:  {mae:.6f}")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"R²:   {r2:.4f}")
        logger.info(f"DA:   {directional_accuracy:.2%}")
        logger.info("=" * 80)
        
        return metrics

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
