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
Version: 2.0.0 REFACTORED
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn

from src.utils.config_loader import Config

from .utils import set_seeds

logger = logging.getLogger(__name__)


class LSTMNetwork(nn.Module):
    """
    Initialize LSTM network.

    Args:
        input_size: Number of features (F)
        hidden_size_1: Units in first LSTM layer (50-300)
        hidden_size_2: Units in second LSTM layer (20-200)
        dropout_rate: Dropout rate (0.0-0.5)
        output_units: Output dimension (default: 1)
        activation: Hidden activation name — 'relu' | 'tanh' | 'leaky_relu'
        output_activation: Output activation name — 'linear' | 'relu' | 'tanh'
    """

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int,
        hidden_size_2: int,
        dropout_rate: float,
        activation: str,
        output_units: int,
        output_activation: str,
    ):
        """
        PyTorch LSTM network for financial time-series regression.

        Architecture:
            Input(T, F) → LSTM_1 → Act → Dropout → LSTM_2 → Act → Dropout → Dense(H)

        Notes:
        - Supports configurable activation + output activation
        - Supports multi-step prediction via prediction_horizon
        """
        super(LSTMNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.dropout_rate = dropout_rate

        # -------------------------
        # Activation selection
        # -------------------------
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # -------------------------
        # Output activation
        # -------------------------
        if output_activation in [None, "linear"]:
            self.output_activation = nn.Identity()
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation == "relu":
            self.output_activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported output_activation: {output_activation}")

        # -------------------------
        # LSTM layers
        # -------------------------
        self.lstm_1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        self.dropout_1 = (
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

        self.lstm_2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        self.dropout_2 = (
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        # -------------------------
        # Output layer
        # -------------------------
        # If prediction_horizon > 1 → multi-step forecast
        final_output_dim = output_units
        self.fc = nn.Linear(hidden_size_2, final_output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, F)

        Returns:
            (batch, H) where H = prediction_horizon or output_units
        """
        # Optional safety check (remove for performance if needed)
        if x.shape[1] != self.lookback:
            raise ValueError(f"Expected lookback={self.lookback}, got {x.shape[1]}")

        # LSTM 1
        out, _ = self.lstm_1(x)
        out = self.activation(out)
        out = self.dropout_1(out)

        # LSTM 2
        out, _ = self.lstm_2(out)
        out = self.activation(out)
        out = self.dropout_2(out)

        # Last timestep
        out = out[:, -1, :]

        # Dense + output activation
        out = self.fc(out)
        out = self.output_activation(out)

        return out


class LSTMModel:
    """
    Production LSTM model wrapper for financial time-series.

    Handles architecture construction, deterministic inference, and I/O.
    Training is owned exclusively by LSTMTrainer.
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
        self.feature_names: Optional[List[str]] = None

        set_seeds(seed)
        logger.info(f"LSTMModel initialized on device: {self.device}")

    def build_model(self, config: Dict[str, Any], input_size: int) -> LSTMNetwork:
        """
        Build 2-layer LSTM architecture from configuration.

        Args:
            config: Dictionary containing:
                - lstm_units_1: int (50-300)
                - lstm_units_2: int (20-200)
                - dropout_rate: float (0.0-0.5)
                - output_units: int (default 1)
                - activation: str (default 'relu')
                - output_activation: str (default 'linear')
            input_size: Number of features (F)

        Returns:
            Initialized LSTMNetwork
        """
        required_keys = [
            "lstm_units_1",
            "lstm_units_2",
            "dropout_rate",
            "output_units",
            "activation",
            "output_activation",
        ]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Required config key missing: {key}")

        units_1 = int(config["lstm_units_1"])
        units_2 = int(config["lstm_units_2"])
        dropout_rate = float(config["dropout_rate"])
        output_units = int(config.get("output_units", 1))
        activation = str(config.get("activation", "relu"))
        output_activation = str(config.get("output_activation", "linear"))

        if not (50 <= units_1 <= 300):
            raise ValueError(f"lstm_units_1 must be in [50, 300], got {units_1}")
        if not (20 <= units_2 <= 200):
            raise ValueError(f"lstm_units_2 must be in [20, 200], got {units_2}")
        if not (0.0 <= dropout_rate <= 0.5):
            raise ValueError(f"dropout_rate must be in [0.0, 0.5], got {dropout_rate}")

        logger.info(
            f"Building LSTM model: LSTM_1={units_1}, LSTM_2={units_2}, "
            f"dropout={dropout_rate}, input_size={input_size}, "
            f"output_units={output_units}, activation={activation}, "
            f"output_activation={output_activation}"
        )

        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size_1=units_1,
            hidden_size_2=units_2,
            dropout_rate=dropout_rate,
            output_units=output_units,
            activation=activation,
            output_activation=output_activation,
        ).to(self.device)

        self.config = config

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            f"Model built: {total_params:,} total params, {trainable_params:,} trainable"
        )

        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Feature tensor, shape (N, 20, F) or (20, F) for single sample

        Returns:
            Predictions, shape (N,) or scalar if single sample
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        single_sample = False
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
            single_sample = True

        self._validate_inputs(X, split_name="predict")

        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_t)

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
            Dictionary of metrics (mse, mae, rmse, r2, directional_accuracy)
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        self._validate_inputs(X_test, y_test, "test")

        y_pred = self.predict(X_test)

        mse = float(np.mean((y_test - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_test - y_pred)))
        rmse = float(np.sqrt(mse))

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

        correct_dir = np.sum(np.sign(y_test) == np.sign(y_pred))
        directional_accuracy = float(correct_dir / len(y_test))

        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "directional_accuracy": directional_accuracy,
            "n_samples": len(y_test),
        }

        logger.info("=" * 80)
        logger.info("TEST EVALUATION")
        logger.info("=" * 80)
        logger.info(f"MSE:  {mse:.6f}")
        logger.info(f"MAE:  {mae:.6f}")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"R²:   {r2:.4f}")
        logger.info(f"DA:   {directional_accuracy:.2%}")
        logger.info("=" * 80)

        return metrics

    def save(self, filepath: str) -> None:
        """Save model weights to disk."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model weights saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model weights from disk."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        logger.info(f"Model weights loaded from {filepath}")

    def _validate_inputs(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        split_name: str = "train",
    ) -> None:
        """
        Validate input tensor shapes and dtypes.

        Args:
            X: Feature tensor (N, 20, F)
            y: Target vector (N,) or None
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


def create_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    seed: int = 42,
) -> Tuple["LSTMModel", Dict[str, List[float]]]:
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
    from .lstm_trainer import LSTMTrainer

    set_seeds(seed)
    trainer = LSTMTrainer(config=config, seed=seed)
    model, history = trainer.train(X_train, y_train, X_val, y_val)
    return model, history
