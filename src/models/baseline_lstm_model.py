"""
LSTM Modeling Module for Financial Time-Series Forecasting

Production-grade implementation of a 2-layer LSTM for stock return prediction.
Guarantees deterministic training, strict temporal integrity, and PSO hyperparameter
compatibility.

Paper Alignment:
- Ji et al. 2021: Input shape (N, 20, F), PSO structure for hyperparameters
- Zeng et al. 2025: ReLU activation, 2-layer architecture constraints
- Deng & Peng 2025: Dropout regularization for overfitting prevention

Author: System Architect
Version: 1.0.0
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for deterministic behavior.

    Args:
        seed: Random seed value (default: 42)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA if available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LSTMNetwork(nn.Module):
    """
    PyTorch LSTM network for financial time-series regression.

    Architecture:
        Input(20, F) → LSTM_1 → Dropout → LSTM_2 → Dropout → Dense(1)

    Constraints:
    - Fixed 2-layer LSTM (unless explicitly configured otherwise by PSO)
    - ReLU activation (per Zeng et al. 2025)
    - Linear output for return prediction
    """

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int,
        hidden_size_2: int,
        dropout_rate: float,
    ):
        """
        Initialize LSTM network.

        Args:
            input_size: Number of features (F)
            hidden_size_1: Units in first LSTM layer (50-300)
            hidden_size_2: Units in second LSTM layer (20-200)
            dropout_rate: Dropout rate (0.0-0.5)
        """
        super(LSTMNetwork, self).__init__()

        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.input_size = input_size

        # Layer 1: LSTM with return sequences
        # Input shape: (batch, 20, F)
        # Output shape: (batch, 20, hidden_size_1)
        self.lstm_1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # No dropout between layers in single LSTM
        )

        # Dropout after LSTM 1 (per Deng & Peng 2025)
        self.dropout_1 = (
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

        # Layer 2: LSTM without return sequences
        # Input shape: (batch, 20, hidden_size_1)
        # Output shape: (batch, hidden_size_2)
        self.lstm_2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # Dropout after LSTM 2 (per Deng & Peng 2025)
        self.dropout_2 = (
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

        # Output layer: Linear for regression
        # Predicting: y[t] = (Close_{t+1} - Close_t) / Close_t
        self.fc = nn.Linear(hidden_size_2, 1)

        # ReLU activation for LSTM outputs (per Zeng et al. 2025)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: Input tensor, shape (batch, 20, F)

        Returns:
            Output tensor, shape (batch, 1)
        """
        # LSTM 1: (batch, 20, F) → (batch, 20, hidden_1)
        out, _ = self.lstm_1(x)
        out = self.relu(out)  # ReLU activation per Zeng et al. 2025
        out = self.dropout_1(out)

        # LSTM 2: (batch, 20, hidden_1) → (batch, hidden_2)
        out, _ = self.lstm_2(out)
        out = self.relu(out)  # ReLU activation per Zeng et al. 2025
        out = self.dropout_2(out)

        # Take only last timestep output (return_sequences=False behavior)
        # out shape: (batch, hidden_2)
        out = out[:, -1, :]  # Last timestep

        # Dense: (batch, hidden_2) → (batch, 1)
        out = self.fc(out)

        return out

    def save(self, filepath: str) -> None:
        """
        Save model weights to disk.

        Args:
            filepath: Path to save the state dict
        """
        torch.save(self.state_dict(), filepath)
        logger.info(f"Model weights saved to {filepath}")


class LSTMModel:
    """
    Production LSTM model wrapper for financial time-series.

    Handles training loop, early stopping, and deterministic inference.
    """

    def __init__(self, seed: int = 42, device: Optional[str] = None):
        """
        Initialize LSTM model wrapper.

        Args:
            seed: Random seed for reproducibility
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.seed = seed
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[LSTMNetwork] = None
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.training_config: Optional[Dict[str, Any]] = None
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch: int = 0

        # Set seeds immediately
        set_seeds(seed)
        logger.info(f"LSTMModel initialized on device: {self.device}")

    def _validate_inputs(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, split_name: str = "train"
    ) -> None:
        """
        Validate input tensor shapes and dtypes.

        Args:
            X: Feature tensor (N, 20, F)
            y: Target vector (N,) or None
            split_name: Name of split for logging

        Raises:
            AssertionError: If validation fails
            ValueError: If shape/dtype incorrect
        """
        # Check type
        if not isinstance(X, np.ndarray):
            raise ValueError(f"{split_name} X must be numpy array")

        # Check dtype - allow float64 but warn, require float32 ideally
        if X.dtype not in [np.float32, np.float64]:
            raise ValueError(
                f"{split_name} X dtype must be float32 or float64, got {X.dtype}"
            )

        # Check dimensions
        if X.ndim != 3:
            raise ValueError(
                f"{split_name} X must be 3D (N, 20, F), got shape {X.shape}"
            )

        # Check look-back window (fixed at 20)
        if X.shape[1] != 20:
            raise ValueError(f"{split_name} X must have timesteps=20, got {X.shape[1]}")

        # Check feature dimension (should be >= 1, typically 12-23)
        if X.shape[2] < 1:
            raise ValueError(
                f"{split_name} X must have at least 1 feature, got {X.shape[2]}"
            )

        # Check NaNs
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

        logger.info(f"Validation passed for {split_name}: shape {X.shape}")

    def build_model(self, config: Dict[str, Any], input_size: int) -> LSTMNetwork:
        """
        Build 2-layer LSTM architecture from PSO configuration.

        Args:
            config: Dictionary containing:
                - lstm_units_1: int (50-300), units in first LSTM layer
                - lstm_units_2: int (20-200), units in second LSTM layer
                - dropout_rate: float (0.0-0.5), dropout rate after each LSTM
            input_size: Number of features (F)

        Returns:
            Initialized LSTMNetwork

        Raises:
            KeyError: If required config keys missing
            ValueError: If config values out of allowed ranges
        """
        required_keys = ["lstm_units_1", "lstm_units_2", "dropout_rate"]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Required config key missing: {key}")

        # Extract parameters with validation
        units_1 = int(config["lstm_units_1"])
        units_2 = int(config["lstm_units_2"])
        dropout_rate = float(config["dropout_rate"])

        # Validate ranges (per specification)
        if not (50 <= units_1 <= 300):
            raise ValueError(f"lstm_units_1 must be in [50, 300], got {units_1}")

        if not (20 <= units_2 <= 200):
            raise ValueError(f"lstm_units_2 must be in [20, 200], got {units_2}")

        if not (0.0 <= dropout_rate <= 0.5):
            raise ValueError(f"dropout_rate must be in [0.0, 0.5], got {dropout_rate}")

        logger.info(
            f"Building LSTM model: LSTM_1={units_1}, LSTM_2={units_2}, "
            f"dropout={dropout_rate}, input_size={input_size}"
        )

        # Build model
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size_1=units_1,
            hidden_size_2=units_2,
            dropout_rate=dropout_rate,
        ).to(self.device)

        # Log parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            f"Model built: {total_params:,} total params, {trainable_params:,} trainable"
        )

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
    ) -> Tuple[LSTMNetwork, Dict[str, List[float]], Dict[str, Any]]:
        """
        Train LSTM model with strict temporal integrity.

        Training protocol:
        - No shuffling (shuffle=False, mandatory)
        - Early stopping on validation loss (patience=10)
        - Restore best weights
        - Batch size from config (32 or 64)
        - Epochs from config (50-300)

        Args:
            X_train: Training features, shape (N_train, 20, F)
            y_train: Training targets, shape (N_train,)
            X_val: Validation features, shape (N_val, 20, F)
            y_val: Validation targets, shape (N_val,)
            config: Training configuration dict

        Returns:
            Tuple of (trained_model, history, training_config)

        Raises:
            AssertionError: If validation checks fail
        """
        # Store config
        self.training_config = config.copy()

        # Validate all inputs
        self._validate_inputs(X_train, y_train, "train")
        self._validate_inputs(X_val, y_val, "val")

        # if self.model is None:
        #     raise RuntimeError("Model not initialized")

        # Convert to float32 tensors
        X_train_t = torch.from_numpy(X_train.astype(np.float32))
        y_train_t = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
        X_val_t = torch.from_numpy(X_val.astype(np.float32))
        y_val_t = torch.from_numpy(y_val.astype(np.float32)).view(-1, 1)

        # Infer input size from data
        input_size = X_train.shape[2]

        # Build model if not already built
        if self.model is None:
            self.build_model(config, input_size)

        # Extract training parameters
        batch_size = int(config.get("batch_size", 32))
        epochs = int(config.get("epochs", 100))
        learning_rate = float(config.get("learning_rate", 0.001))

        if batch_size not in [32, 64]:
            logger.warning(
                f"Batch size {batch_size} not in {{32, 64}}, proceeding anyway"
            )

        if not (50 <= epochs <= 300):
            logger.warning(f"Epochs {epochs} not in [50, 300], proceeding anyway")

        logger.info(
            f"Starting training: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}"
        )
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Validation samples: {len(X_val)}")

        # Create DataLoaders
        # CRITICAL: shuffle=False to prevent temporal leakage
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # MANDATORY: No shuffling of time series
            drop_last=False,
        )

        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffling for validation
            drop_last=False,
        )

        # Optimizer: Adam (per Ji et al. 2021)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Loss: MSE (Mean Squared Error)
        criterion = nn.MSELoss()

        # Early stopping setup
        patience = 10
        best_val_loss = float("inf")
        epochs_no_improve = 0
        self.best_weights = None
        self.best_epoch = 0

        # Training loop
        self.history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation phase
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)

            # Record history
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.best_epoch = epoch
                # Save best weights
                self.best_weights = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Log every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"train_loss={avg_train_loss:.6f}, "
                    f"val_loss={avg_val_loss:.6f}"
                )

            # Early stopping trigger
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Restore best weights
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)
            logger.info(
                f"Restored best weights from epoch {self.best_epoch+1} "
                f"(val_loss={best_val_loss:.6f})"
            )

        final_epoch = len(self.history["train_loss"])
        final_train_loss = self.history["train_loss"][-1]
        final_val_loss = self.history["val_loss"][-1]

        logger.info(f"Training complete at epoch {final_epoch}")
        logger.info(f"  Final train_loss: {final_train_loss:.6f}")
        logger.info(f"  Final val_loss: {final_val_loss:.6f}")

        # Validation check: Model should have trained at least a few epochs
        if final_epoch < 10:
            logger.warning(
                f"Model stopped early at epoch {final_epoch}, possible convergence issues"
            )

        return self.model, self.history, self.training_config

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input features.

        Args:
            X: Feature tensor, shape (N, 20, F) or (20, F) for single sample

        Returns:
            Predictions, shape (N,) or scalar if single sample

        Raises:
            RuntimeError: If model not trained
            ValueError: If input shape invalid
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Handle single sample (add batch dimension)
        single_sample = False
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
            single_sample = True

        # Validate
        self._validate_inputs(X, split_name="predict")

        # Convert to tensor
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_t)

        # Convert to numpy and flatten
        predictions = predictions.cpu().numpy().flatten()

        if single_sample:
            return float(predictions[0])

        return predictions

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics (loss, mae, rmse)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self._validate_inputs(X_test, y_test, "test")

        # Predict
        y_pred = self.predict(X_test)

        # Compute metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(mse)

        metrics = {"loss": float(mse), "mae": float(mae), "rmse": float(rmse)}

        logger.info(f"Test evaluation: loss={mse:.6f}, mae={mae:.6f}, rmse={rmse:.6f}")

        return metrics

    def save_weights(self, filepath: str) -> None:
        """Save model weights to disk."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model weights saved to {filepath}")

    def load_weights(self, filepath: str) -> None:
        """Load model weights from disk."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        logger.info(f"Model weights loaded from {filepath}")


def create_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    seed: int = 42,
) -> Tuple[LSTMModel, Dict[str, List[float]]]:
    """
    Factory function to create and train LSTM model in one call.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        config: Model configuration dict
        seed: Random seed

    Returns:
        Tuple of (LSTMModel instance, training history)
    """
    # Set seeds for reproducibility
    set_seeds(seed)

    # Create and train model
    model = LSTMModel(seed=seed)
    model.train(X_train, y_train, X_val, y_val, config)

    return model, model.history
