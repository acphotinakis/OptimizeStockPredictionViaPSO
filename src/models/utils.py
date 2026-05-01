"""
Model Utilities

Helper functions for data preparation and model I/O.

Author: System Architect
Version: 1.0.0 UNIFIED
"""

import logging
import random
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for deterministic behavior.

    Seeds the Python ``random`` module, NumPy, PyTorch (CPU and CUDA), enables
    deterministic cuDNN kernels, and requests deterministic algorithms across
    PyTorch (warn-only so ops without a deterministic implementation degrade
    gracefully rather than crashing).

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_model_weights(model: torch.nn.Module, filepath: str | Path) -> None:
    """
    Save model weights to disk.

    Args:
        model: PyTorch model
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    logger.info(f"Model weights saved to {filepath}")


def load_model_weights(
    model: torch.nn.Module, filepath: str | Path, device: str = "cpu"
) -> torch.nn.Module:
    """
    Load model weights from disk.

    Args:
        model: PyTorch model (must be initialized with correct architecture)
        filepath: Path to saved weights
        device: Device to load model onto

    Returns:
        Model with loaded weights
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model weights not found: {filepath}")

    model.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"Model weights loaded from {filepath}")

    return model
