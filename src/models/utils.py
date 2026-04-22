"""
Model Utilities

Helper functions for data preparation and model I/O.

Author: System Architect
Version: 1.0.0 UNIFIED
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def build_lstm_windows(
    X: np.ndarray, y: np.ndarray, lookback: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build temporal windows for LSTM input from tabular features.

    Converts:
        X: (N, F) tabular features
        y: (N,) target vector
    Into:
        X_windowed: (N-lookback, lookback, F) 3D sequences
        y_windowed: (N-lookback,) aligned targets

    Each window contains the past `lookback` timesteps of features,
    with the target aligned to the end of the window.

    Args:
        X: Feature matrix, shape (N, F)
        y: Target vector, shape (N,)
        lookback: Number of timesteps per window (default: 20 per TRD)

    Returns:
        Tuple of (X_windowed, y_windowed)

    Raises:
        ValueError: If insufficient samples or shape mismatch

    Example:
        >>> X = np.random.randn(1000, 15)  # 1000 samples, 15 features
        >>> y = np.random.randn(1000)
        >>> X_seq, y_seq = build_lstm_windows(X, y, lookback=20)
        >>> print(X_seq.shape)  # (980, 20, 15)
        >>> print(y_seq.shape)  # (980,)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, F), got shape {X.shape}")

    if y.ndim != 1:
        raise ValueError(f"y must be 1D (N,), got shape {y.shape}")

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    N, F = X.shape

    if N <= lookback:
        raise ValueError(
            f"Not enough samples ({N}) for lookback window ({lookback}). "
            f"Need at least {lookback + 1} samples."
        )

    n_windows = N - lookback
    X_windowed = np.zeros((n_windows, lookback, F), dtype=np.float32)

    for i in range(n_windows):
        X_windowed[i] = X[i : i + lookback]

    # Target aligned with end of window
    y_windowed = y[lookback:].astype(np.float32)

    logger.info(
        f"Built {n_windows} windows: "
        f"X {X.shape} → {X_windowed.shape}, "
        f"y {y.shape} → {y_windowed.shape}"
    )

    return X_windowed, y_windowed


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

    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    logger.info(f"Model weights loaded from {filepath}")

    return model
