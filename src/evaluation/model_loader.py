import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.utils.config_loader import Config

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ["pso_lstm", "lstm_baseline", "xgboost"]


class ModelAdapter(ABC):
    """Abstract base class for model adapters.

    All model adapters must implement:
    - predict(): Generate return predictions
    - get_metadata(): Return model information
    """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions from features.

        Args:
            X: Features (2D or 3D depending on model)

        Returns:
            Predictions as (N,) array of predicted returns
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict:
        """
        Return model metadata.

        Returns:
            Dictionary with model information
        """
        pass


class LSTMAdapter(ModelAdapter):
    """Adapter for LSTM models (baseline and PSO-LSTM).

    Handles:
    - 2D → 3D windowing if needed
    - Consistent prediction interface
    - Metadata extraction
    """

    def __init__(self, model: Any, model_type: str, lookback: int = 20):
        """
        Initialize LSTM adapter.

        Args:
            model: LSTMModel or PSOLSTMModel instance
            model_type: 'lstm_baseline' or 'pso_lstm'
            lookback: Window size (default: 20)
        """
        self.model = model
        self.model_type = model_type
        self.lookback = lookback

        logger.info(f"LSTMAdapter initialized: type={model_type}, lookback={lookback}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions, handling windowing if needed.

        Args:
            X: Features (N, F) 2D or (N, 20, F) 3D

        Returns:
            Predictions (N,) or (N-20,) depending on input
        """
        # Handle 2D → 3D windowing
        if X.ndim == 2:
            logger.info(
                f"Converting 2D features to 3D windows (lookback={self.lookback})"
            )
            from src.data.windowing import build_lstm_windows

            # Create dummy targets for windowing
            dummy_y = np.zeros(len(X))
            X_windowed, y_aligned = build_lstm_windows(X, dummy_y, self.lookback)

            logger.info(f"Windowed: {X.shape} → {X_windowed.shape}")

            predictions = self.model.predict(X_windowed)
        elif X.ndim == 3:
            # Already windowed
            logger.info(f"Using pre-windowed 3D features: {X.shape}")
            predictions = self.model.predict(X)
        else:
            raise ValueError(f"Invalid X shape for LSTM: {X.shape} (expected 2D or 3D)")

        # Ensure 1D output
        if predictions.ndim > 1:
            predictions = predictions.flatten()

        logger.info(f" Generated {len(predictions)} predictions")

        return predictions

    def get_metadata(self) -> Dict:
        """Return model metadata."""
        return {
            "model_type": self.model_type,
            "architecture": "2-layer LSTM (PyTorch)",
            "framework": "pytorch",
            "lookback": self.lookback,
            "output": "next-period return prediction",
        }


class XGBoostAdapter(ModelAdapter):
    """Adapter for XGBoost model.

    Handles:
    - 3D → 2D flattening if needed
    - Consistent prediction interface
    - Metadata extraction
    """

    def __init__(self, model: Any):
        """
        Initialize XGBoost adapter.

        Args:
            model: XGBoostModel instance
        """
        self.model = model

        logger.info("XGBoostAdapter initialized")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions, handling windowing if needed.

        Args:
            X: Features (N, F) 2D or (N, T, F) 3D

        Returns:
            Predictions (N,)
        """
        # Handle 3D → 2D flattening
        if X.ndim == 3:
            logger.info(f"Flattening 3D features to 2D for XGBoost: {X.shape}")
            # Flatten temporal dimension: (N, T, F) → (N, T*F)
            X_flat = X.reshape(X.shape[0], -1)
            logger.info(f"Flattened: {X.shape} → {X_flat.shape}")
            predictions = self.model.predict(X_flat)
        elif X.ndim == 2:
            # Already tabular
            logger.info(f"Using 2D tabular features: {X.shape}")
            predictions = self.model.predict(X)
        else:
            raise ValueError(
                f"Invalid X shape for XGBoost: {X.shape} (expected 2D or 3D)"
            )

        # Ensure 1D output
        if predictions.ndim > 1:
            predictions = predictions.flatten()

        logger.info(f" Generated {len(predictions)} predictions")

        return predictions

    def get_metadata(self) -> Dict:
        """Return model metadata."""
        return {
            "model_type": "xgboost",
            "architecture": "Gradient Boosted Trees",
            "framework": "xgboost",
            "output": "next-period return prediction",
        }


def load_model(
    model_type: str,
    model_path: Path,
    config: Config,
) -> ModelAdapter:
    """
    Load trained model and return unified adapter.

    Factory function that loads the appropriate model type and wraps it
    in a ModelAdapter for consistent prediction interface.

    Args:
        model_type: One of {'pso_lstm', 'lstm_baseline', 'xgboost'}
        model_path: Path to saved model file
        config: Configuration object

    Returns:
        ModelAdapter instance with unified predict() interface

    Raises:
        ValueError: If model_type unknown
        FileNotFoundError: If model_path doesn't exist
        RuntimeError: If model loading fails
    """
    # Validate model type
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Must be one of: {SUPPORTED_MODELS}"
        )

    # Validate model path
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info("=" * 80)
    logger.info(f"LOADING MODEL: {model_type.upper()}")
    logger.info("=" * 80)
    logger.info(f"Path: {model_path}")

    try:
        if model_type == "pso_lstm":
            import json
            from src.models import PSOLSTMModel

            # Load model config
            model_dir = model_path.parent
            config_file = model_dir / "model_config.json"

            if not config_file.exists():
                raise FileNotFoundError(
                    f"Model config file not found: {config_file}\n"
                    "PSO-LSTM models require model_config.json for loading."
                )

            with open(config_file, "r") as f:
                model_config = json.load(f)
            logger.info(f"Loaded model config from {config_file}")

            seed = config.lstm_baseline.random_seed
            model = PSOLSTMModel(seed=seed)

            # Build model with saved config
            input_size = model_config["input_shape"][1]
            model.build_model(model_config, input_size)

            # Load weights
            model.load_weights(str(model_path))

            adapter = LSTMAdapter(model, "pso_lstm", lookback=20)
            logger.info(" PSO-LSTM model loaded successfully")

            return adapter

        elif model_type == "lstm_baseline":
            import json
            from src.models import LSTMModel

            # Load model config
            model_dir = model_path.parent
            config_file = model_dir / "model_config.json"

            if not config_file.exists():
                raise FileNotFoundError(
                    f"Model config file not found: {config_file}\n"
                    "LSTM models require model_config.json for loading."
                )

            with open(config_file, "r") as f:
                model_config = json.load(f)
            logger.info(f"Loaded model config from {config_file}")

            seed = config.lstm_baseline.random_seed
            model = LSTMModel(seed=seed)

            # Build model with saved config
            input_size = model_config["input_shape"][1]
            model.build_model(model_config, input_size)

            # Load weights
            model.load_weights(str(model_path))

            adapter = LSTMAdapter(
                model, "lstm_baseline", lookback=config.lstm_baseline.lookback
            )
            logger.info(" Baseline LSTM model loaded successfully")

            return adapter

        elif model_type == "xgboost":
            from src.models import XGBoostModel

            xgb_config = config.xgboost

            xgb_params = {
                "objective": xgb_config.objective,
                "n_estimators": xgb_config.n_estimators,
                "max_depth": xgb_config.max_depth,
                "learning_rate": xgb_config.learning_rate,
                "subsample": xgb_config.subsample,
                "colsample_bytree": xgb_config.colsample_bytree,
                "min_child_weight": xgb_config.min_child_weight,
                "gamma": xgb_config.gamma,
                "reg_alpha": xgb_config.reg_alpha,
                "reg_lambda": xgb_config.reg_lambda,
                "early_stopping_rounds": xgb_config.early_stopping_rounds,
                "tree_method": xgb_config.tree_method,
                "max_bin": xgb_config.max_bin,
                "random_seed": xgb_config.random_seed,
            }

            model = XGBoostModel(config=xgb_params)
            model.load_model(str(model_path))

            adapter = XGBoostAdapter(model)
            logger.info(" XGBoost model loaded successfully")

            return adapter

    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load {model_type} model from {model_path}: {e}")

    logger.info("=" * 80)


def validate_model_compatibility(model_adapter: ModelAdapter, test_data: dict) -> None:
    """
    Validate that model is compatible with test data.

    Args:
        model_adapter: Loaded model adapter
        test_data: Test data dictionary

    Raises:
        ValueError: If incompatibility detected
    """

    X_test = test_data["X_test"]

    logger.info("Validating model-data compatibility...")

    # Check feature dimension compatibility
    if isinstance(model_adapter, LSTMAdapter):
        if X_test.ndim == 2:
            expected_features = X_test.shape[1]
        elif X_test.ndim == 3:
            expected_features = X_test.shape[2]
        else:
            raise ValueError(f"Invalid X_test shape: {X_test.shape}")

        logger.info(f"  LSTM expects {expected_features} features")

    elif isinstance(model_adapter, XGBoostAdapter):
        if X_test.ndim == 2:
            expected_features = X_test.shape[1]
        elif X_test.ndim == 3:
            expected_features = X_test.shape[1] * X_test.shape[2]
        else:
            raise ValueError(f"Invalid X_test shape: {X_test.shape}")

        logger.info(f"  XGBoost expects {expected_features} features")

    logger.info(" Model-data compatibility validated")
