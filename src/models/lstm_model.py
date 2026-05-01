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

import json
import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.evaluation.metrics import compute_and_log_all_statistical_metrics
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
            Input(T, F) --> LSTM_1 --> Dropout --> LSTM_2 --> Dropout --> Dense(H) --> OutputAct

        The configurable activation is applied only on the dense head's output
        (when ``output_activation`` is non-linear). It is not inserted between
        the stacked LSTM layers, since that would zero half of the recurrent
        hidden-state range and damage the gating dynamics of LSTM_2.

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
        # If prediction_horizon > 1 --> multi-step forecast
        final_output_dim = output_units
        self.fc = nn.Linear(hidden_size_2, final_output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, F)

        Returns:
            (batch, H) where H = prediction_horizon or output_units
        """
        # LSTM 1
        out, _ = self.lstm_1(x)
        out = self.dropout_1(out)

        # LSTM 2
        out, _ = self.lstm_2(out)
        out = self.dropout_2(out)

        # Last timestep
        out = out[:, -1, :]

        # Dense + output activation
        out = self.fc(out)
        out = self.output_activation(out)

        return out

    def __str__(self) -> str:
        total_params = sum(p.numel() for p in self.parameters())

        payload = {
            "input_size": self.input_size,
            "hidden_size_1": self.hidden_size_1,
            "hidden_size_2": self.hidden_size_2,
            "dropout_rate": self.dropout_rate,
            "activation": type(self.activation).__name__,
            "output_activation": type(self.output_activation).__name__,
            "total_params": total_params,
        }

        return json.dumps(payload, indent=2)

    def __repr__(self) -> str:
        return self.__str__()


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

    def build_model(self, config: Dict[str, Any]) -> Tuple[LSTMNetwork, "LSTMModel"]:
        """
        Build 2-layer LSTM architecture from configuration.

        Args:
            config: Dictionary containing:
                - input_size: int (X_train.shape[2])
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
            "input_size",
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

        self.input_size = int(config["input_size"])
        self.lookback = int(config.get("lookback", 20))
        self.units_1 = int(config["lstm_units_1"])
        self.units_2 = int(config["lstm_units_2"])
        self.dropout_rate = float(config["dropout_rate"])
        self.output_units = int(config.get("output_units", 1))
        self.activation = str(config.get("activation", "relu"))
        self.output_activation = str(config.get("output_activation", "linear"))

        if not (50 <= self.units_1 <= 300):
            raise ValueError(f"lstm_units_1 must be in [50, 300], got {self.units_1}")
        if not (20 <= self.units_2 <= 200):
            raise ValueError(f"lstm_units_2 must be in [20, 200], got {self.units_2}")
        if not (0.0 <= self.dropout_rate <= 0.5):
            raise ValueError(
                f"dropout_rate must be in [0.0, 0.5], got {self.dropout_rate}"
            )

        logger.info(
            f"Building LSTM model: LSTM_1={self.units_1}, LSTM_2={self.units_2}, "
            f"dropout={self.dropout_rate}, input_size={self.input_size}, "
            f"output_units={self.output_units}, activation={self.activation}, "
            f"output_activation={self.output_activation}"
        )

        self.model = LSTMNetwork(
            input_size=self.input_size,
            hidden_size_1=self.units_1,
            hidden_size_2=self.units_2,
            dropout_rate=self.dropout_rate,
            output_units=self.output_units,
            activation=self.activation,
            output_activation=self.output_activation,
        ).to(self.device)

        self.config = config

        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            f"Model built: {self.total_params:,} total params, {self.trainable_params:,} trainable"
        )

        return self.model, self

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Memory-safe batched inference.

        Args:
            X: (N, T, F)
            batch_size: number of samples per batch

        Returns:
            (N,) predictions
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        single_sample = False
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
            single_sample = True

        self._validate_inputs(X, split_name="predict")

        self.model.eval()

        preds = []
        N = X.shape[0]

        with torch.no_grad():
            for i in tqdm(
                range(0, N, batch_size),
                total=(N + batch_size - 1) // batch_size,
                desc="Predicting",
                unit="batch",
            ):
                X_batch = X[i : i + batch_size]

                X_t = torch.from_numpy(X_batch.astype(np.float32)).to(
                    self.device, non_blocking=True
                )

                y_batch = self.model(X_t)

                preds.append(y_batch.cpu().numpy())

                del X_t, y_batch
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        predictions = np.concatenate(preds, axis=0).flatten()

        if single_sample:
            return float(predictions[0])

        return predictions

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray]:
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

        metrics = compute_and_log_all_statistical_metrics(
            y_test, y_pred, "LSTM Baseline Evaluate Metrics"
        )

        metrics["n_samples"] = len(y_test)

        logger.info("=" * 80)
        logger.info("TEST EVALUATION")
        logger.info("=" * 80)

        logger.info(f"RMSE: {metrics['rmse']:.6f}")
        logger.info(f"MAE:  {metrics['mae']:.6f}")
        logger.info(f"MAPE: {metrics['mape']:.6f}")
        logger.info(f"R²:   {metrics['r2']:.6f}")
        logger.info(f"DA:   {metrics['directional_accuracy']:.4f}")
        logger.info(f"F1:   {metrics['f1_ternary']:.4f}")
        logger.info(f"AUC:  {metrics['auc_ternary']:.4f}")
        logger.info(f"N:    {metrics['n_samples']}")

        logger.info("=" * 80)
        return metrics, y_pred

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
        self.model.load_state_dict(
            torch.load(filepath, map_location=self.device, weights_only=True)
        )
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
            X: Feature tensor (N, T, F)
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
                f"{split_name} X must be 3D (N, T, F), got shape {X.shape}"
            )

        if X.shape[1] < 1:
            raise ValueError(
                f"{split_name} X must have at least 1 timestep, got {X.shape[1]}"
            )

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
        if self.model is None:
            return json.dumps({"LSTMModel": "uninitialized"}, indent=2)

        net = self.model

        payload = {
            "LSTMModel": {
                "device": self.device,
                "input_size": self.input_size,
                "lookback": self.lookback,
                "hidden_size_1": self.units_1,
                "hidden_size_2": self.units_2,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "output_activation": self.output_activation,
                "total_params": self.total_params,
                "trainable_params": self.trainable_params,
                "LSTMNetwork": {
                    "input_size": net.input_size,
                    "hidden_size_1": net.hidden_size_1,
                    "hidden_size_2": net.hidden_size_2,
                    "dropout_rate": net.dropout_rate,
                    "activation": type(net.activation).__name__,
                    "output_activation": type(net.output_activation).__name__,
                    "num_parameters": sum(p.numel() for p in net.parameters()),
                },
            }
        }

        return json.dumps(payload, indent=2)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns fully structured model metadata for experiment tracking,
        reproducibility, and BacktestResults serialization.

        This is the single source of truth for model identity.
        """

        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        net = self.model

        return {
            # Identity
            "model_name": "LSTMModel",
            "framework": "pytorch",
            "device": self.device,
            "seed": self.seed,
            # Architecture config
            "input_size": getattr(self, "input_size", None),
            "lookback": self.lookback,
            "hidden_size_1": self.units_1,
            "hidden_size_2": self.units_2,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "output_units": self.output_units,
            # Runtime config snapshot (frozen at build time)
            "config": getattr(self, "config", {}),
            # Parameter counts
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            # Network-level introspection
            "network": {
                "input_size": net.input_size,
                "hidden_size_1": net.hidden_size_1,
                "hidden_size_2": net.hidden_size_2,
                "dropout_rate": net.dropout_rate,
                "activation": type(net.activation).__name__,
                "output_activation": type(net.output_activation).__name__,
                "num_parameters": sum(p.numel() for p in net.parameters()),
            },
            # Inference constraints
            "inference": {
                "batch_inference": True,
                "deterministic": True,
                "mixed_precision": False,
            },
        }

    def __repr__(self) -> str:
        return self.__str__()
