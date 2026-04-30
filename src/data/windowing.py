"""
Temporal Windowing and Lag Feature Construction

Data transformation utilities for preparing time series data
for different model architectures (LSTM, XGBoost).

Relocated from src/models/ to src/data/ per MODEL_SIMPLIFICATION_AUDIT.md.

Author: System Architect
Version: 2.0.0 - MODEL SIMPLIFICATION
"""

import logging
from typing import Tuple

import numpy as np

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

    Each window contains the past `lookback` timesteps of features. The target
    is the 1-bar-ahead forward return at the most-recent feature bar: for
    window ``i`` covering rows ``[i, i+lookback-1]``, the target is
    ``y[i+lookback-1] = log(close[i+lookback] / close[i+lookback-1])``. This
    matches `build_xgboost_lag_features` so both models predict the same
    quantity.

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

    # 1-bar-ahead alignment: window i covers rows [i, i+lookback-1] (most recent
    # feature at index i+lookback-1) and the target is the forward return at the
    # most-recent feature bar, y[i+lookback-1] = log(close[i+lookback]/close[i+lookback-1]).
    # The final row is dropped because y[N-1] is NaN under the canonical target.
    y_windowed = y[lookback - 1 : -1].astype(np.float32)

    logger.info(
        f"Built {n_windows} windows: "
        f"X {X.shape} --> {X_windowed.shape}, "
        f"y {y.shape} --> {y_windowed.shape}"
    )

    return X_windowed, y_windowed


def build_xgboost_lag_features(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build lag-based features for XGBoost from tabular data.

    XGBoost Feature Representation Strategy (TRD-Aligned):
    - Uses LAG-BASED representation (NOT flattened sequences)
    - Each sample contains current values + L previous lags
    - More interpretable for tree-based models
    - Lower dimensionality than flattened sequences

    Converts:
        X: (N, F) tabular features
        y: (N,) target vector
    Into:
        X_lagged: (N-lookback, F * (lookback+1)) with lag features
        y_aligned: (N-lookback,) aligned targets

    For each sample i at time t:
        X_lagged[i] = [X[t], X[t-1], X[t-2], ..., X[t-lookback]]
        y_aligned[i] = y[t]

    Args:
        X: Feature matrix (N, F)
        y: Target vector (N,)
        lookback: Number of lags to create (default: 20)

    Returns:
        Tuple of (X_lagged, y_aligned)
        - X_lagged: (N-lookback, F*(lookback+1))
        - y_aligned: (N-lookback,)

    Raises:
        ValueError: If insufficient samples or shape mismatch

    Example:
        >>> X = np.random.randn(1000, 10)  # 1000 samples, 10 features
        >>> y = np.random.randn(1000)
        >>> X_lag, y_lag = build_xgboost_lag_features(X, y, lookback=20)
        >>> print(X_lag.shape)  # (980, 210) = 980 samples, 10*21 features
        >>> print(y_lag.shape)  # (980,)
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
            f"Not enough samples ({N}) for lookback ({lookback}). "
            f"Need at least {lookback + 1} samples."
        )

    # Create lag features
    lag_arrays = []

    for lag in range(lookback + 1):
        # lag=0 is current, lag=1 is previous, etc.
        if lag == 0:
            lag_arrays.append(X[lookback:])
        else:
            lag_arrays.append(X[lookback - lag : -lag])

    # Concatenate all lags horizontally
    X_lagged = np.concatenate(lag_arrays, axis=1).astype(np.float32)

    # Align targets
    y_aligned = y[lookback:].astype(np.float32)

    logger.info(
        f"Built {len(y_aligned)} lagged samples: "
        f"X {X.shape} --> {X_lagged.shape} ({lookback+1} lags), "
        f"y {y.shape} --> {y_aligned.shape}"
    )

    return X_lagged, y_aligned
