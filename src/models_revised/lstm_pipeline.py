import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class LSTMPipeline:
    """
    End-to-end production pipeline for LSTM financial forecasting.

    Responsibilities:
    - Orchestrates data validation → training → evaluation → inference
    - Ensures strict temporal integrity (no leakage enforcement at pipeline level)
    - Encapsulates model lifecycle management
    - Provides single entry point for production execution
    """

    def __init__(
        self, config: Dict[str, Any], seed: int = 42, device: Optional[str] = None
    ):
        """
        Initialize pipeline.

        Args:
            config: Full runtime configuration (model + training + data)
            seed: Global deterministic seed
            device: compute device override
        """
        self.config = config
        self.seed = seed
        self.device = device

        self.model_wrapper = None
        self.is_fitted = False

        logger.info("LSTMPipeline initialized")

    # -----------------------------
    # DATA CONTRACT VALIDATION
    # -----------------------------
    def _validate_dataset_contract(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        Enforces dataset-level invariants before model training.
        """

        if not isinstance(X_train, np.ndarray):
            raise ValueError("X_train must be numpy array")

        if X_train.ndim != 3 or X_val.ndim != 3:
            raise ValueError("Inputs must be 3D tensors (N, 20, F)")

        if X_train.shape[1] != 20 or X_val.shape[1] != 20:
            raise ValueError("Timesteps must be fixed at 20")

        if X_train.shape[2] != X_val.shape[2]:
            raise ValueError("Feature dimension mismatch between train/val")

        if len(X_train) != len(y_train):
            raise ValueError("Train X/y length mismatch")

        if len(X_val) != len(y_val):
            raise ValueError("Val X/y length mismatch")

        if np.isnan(X_train).any() or np.isnan(X_val).any():
            raise ValueError("NaN detected in feature matrices")

        if np.isnan(y_train).any() or np.isnan(y_val).any():
            raise ValueError("NaN detected in target vectors")

        logger.info("Dataset contract validation passed")

    # -----------------------------
    # MODEL INITIALIZATION
    # -----------------------------
    def initialize_model(self):
        """
        Lazily initializes LSTMModel wrapper.
        """
        if self.model_wrapper is None:
            from lstm import (
                LSTMModel,
                set_seeds,
            )  # local import to avoid circular dependency

            set_seeds(self.seed)

            self.model_wrapper = LSTMModel(seed=self.seed, device=self.device)

        return self.model_wrapper

    # -----------------------------
    # TRAINING PIPELINE
    # -----------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full training pipeline execution.

        Steps:
        1. Validate dataset
        2. Initialize model
        3. Merge configs
        4. Train model
        5. Store artifacts
        """

        self._validate_dataset_contract(X_train, y_train, X_val, y_val)

        model = self.initialize_model()

        # merge config
        config = dict(self.config)
        if config_override:
            config.update(config_override)

        model, history, training_config = model.train(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, config=config
        )

        self.is_fitted = True

        return {
            "history": history,
            "config": training_config,
            "best_epoch": model.best_epoch if hasattr(model, "best_epoch") else None,
        }

    # -----------------------------
    # PREDICTION PIPELINE
    # -----------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Production inference entry point.
        """

        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted")

        model = self.initialize_model()
        return model.predict(X)

    # -----------------------------
    # EVALUATION PIPELINE
    # -----------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Full evaluation stage.
        """

        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted")

        model = self.initialize_model()
        return model.evaluate(X_test, y_test)

    # -----------------------------
    # PERSISTENCE
    # -----------------------------
    def save(self, path: str) -> None:
        """
        Save model weights.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted")

        model = self.initialize_model()
        model.save_weights(path)

    def load(self, path: str) -> None:
        """
        Load model weights into pipeline.
        """
        model = self.initialize_model()
        model.load_weights(path)
        self.is_fitted = True

    # -----------------------------
    # END-TO-END RUN
    # -----------------------------
    def run(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test=None,
        y_test=None,
        config_override=None,
    ) -> Dict[str, Any]:
        """
        Single-call full pipeline execution.

        Optional test evaluation included if provided.
        """

        result = self.fit(
            X_train, y_train, X_val, y_val, config_override=config_override
        )

        if X_test is not None and y_test is not None:
            result["test_metrics"] = self.evaluate(X_test, y_test)

        return result
