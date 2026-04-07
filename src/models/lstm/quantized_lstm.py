"""
src/models/quantized_lstm.py

Post-training quantization wrapper for LSTM models.
Reduces model size by 75% (float32 --> int8) with <1% accuracy loss.

Phase 2: Model Quantization (from QUANTIZATION.md)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .lstm_model import LSTMModel

logger = logging.getLogger(__name__)


class QuantizedLSTMModel:
    """Wrapper for quantized LSTM inference.

    Applies dynamic quantization to a trained LSTMModel, converting
    weights from float32 to int8. This reduces model size by ~75%
    with minimal accuracy loss (<1% RMSE increase).

    Args:
        model: Trained LSTMModel instance.
        device: Device for inference ('cpu' or 'cuda').
    """

    def __init__(self, model: LSTMModel, device: Optional[str] = None) -> None:
        self.model = model
        self.quantized_model: Optional[nn.Module] = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._is_quantized = False

    def quantize(self) -> nn.Module:
        """Apply dynamic quantization to the trained model.

        Quantizes LSTM and Linear layers to int8.

        Returns:
            Quantized model.
        """
        if self._is_quantized:
            logger.warning("Model is already quantized")
            return self.quantized_model

        # Move to CPU for quantization (required by PyTorch)
        self.model.cpu()
        self.model.eval()

        logger.info("Applying dynamic quantization (float32 --> int8)...")

        # Dynamic quantization - quantizes weights, activations stay float
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.LSTM, torch.nn.Linear},  # Layers to quantize
            dtype=torch.qint8,  # 8-bit integers
        )

        self._is_quantized = True

        # Log model sizes
        original_size = self._get_model_size(self.model)
        quantized_size = self._get_model_size(self.quantized_model)
        reduction = (1 - quantized_size / original_size) * 100

        logger.info(
            "Quantization complete: %.1f MB --> %.1f MB (%.1f%% reduction)",
            original_size,
            quantized_size,
            reduction,
        )

        return self.quantized_model

    def predict(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Inference with quantized model.

        Args:
            X: [N, T, F] input array.
            device: Device for inference (quantized models work best on CPU).

        Returns:
            [N] prediction array.
        """
        if not self._is_quantized:
            raise RuntimeError("Call quantize() before predict()")

        # Quantized models typically run on CPU
        self.quantized_model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(device)
            preds = self.quantized_model(x_tensor).squeeze(-1)

        return preds.cpu().numpy()

    @staticmethod
    def _get_model_size(model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    def save(self, path: str) -> None:
        """Save quantized model to disk."""
        if not self._is_quantized:
            raise RuntimeError("Call quantize() before save()")
        torch.save(self.quantized_model.state_dict(), path)
        logger.info("Saved quantized model to %s", path)

    def load(self, path: str) -> None:
        """Load quantized model from disk."""
        if not self._is_quantized:
            self.quantize()
        try:
            self.quantized_model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
            logger.info("Loaded quantized model from %s", path)
        except Exception as e:
            logger.error(f"Failed to load quantized model from {path}: {e}")
            raise


class QATLSTMTrainer:
    """Quantization-Aware Training for better accuracy.

    Simulates quantization during training to minimize accuracy loss.
    Optional - use only if post-training quantization shows >1% degradation.

    Args:
        model: LSTMModel instance.
        lr: Learning rate.
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM).
        **kwargs: Additional arguments passed to LSTMTrainer.
    """

    def __init__(
        self, model: LSTMModel, lr: float, backend: str = "fbgemm", **kwargs
    ) -> None:
        from .lstm_model import LSTMTrainer

        # Prepare model for QAT
        model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        torch.quantization.prepare_qat(model, inplace=True)

        # Initialize trainer with QAT-prepared model
        self.trainer = LSTMTrainer(model=model, lr=lr, **kwargs)
        self.backend = backend
        logger.info("Initialized QAT trainer with backend=%s", backend)

    def fit(self, X_train, y_train, X_val, y_val):
        """Train with quantization-aware training."""
        return self.trainer.fit(X_train, y_train, X_val, y_val)

    def finalize(self) -> nn.Module:
        """Convert to fully quantized model after training.

        Returns:
            Quantized model ready for inference.
        """
        self.trainer.model.eval()
        quantized_model = torch.quantization.convert(self.trainer.model, inplace=False)
        logger.info("QAT finalized - model converted to int8")
        return quantized_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the QAT model."""
        return self.trainer.predict(X)
