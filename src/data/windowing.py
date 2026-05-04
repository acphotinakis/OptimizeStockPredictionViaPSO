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
    X: np.ndarray, y: np.ndarray, lookback: int = 20, horizon: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Build temporal windows for LSTM input from tabular features.

    Converts ``X`` of shape ``(N, F)`` and ``y`` of shape ``(N,)`` into
    sequences ``X_seq`` of shape ``(M, lookback, F)`` and aligned targets
    ``y_seq`` of shape ``(M,)``, where ``M = N - lookback - horizon - 1``.

    The final row is dropped so that ``y[N-1]`` (which is NaN under the
    canonical target ``y[t] = log(close[t+1]/close[t])`` because the next
    close is unknown) is never selected as a target. This also makes the
    output sample count match ``build_xgboost_lag_features`` so the two
    representations cover the same number of samples and can be compared
    directly.

    Args:
        X: Feature matrix, shape ``(N, F)``.
        y: Target vector, shape ``(N,)``. Assumes the canonical target
            ``y[t] = log(close[t+1]/close[t])`` (forward return at bar ``t``).
        lookback: Number of timesteps per window (default 20).
        horizon: Steps further ahead than the most-recent feature bar.
            ``horizon=0`` (default) yields 1-bar-ahead alignment: window ``i``
            covers rows ``[i, i+lookback-1]`` and the target is
            ``y[i+lookback-1]``, the forward return realised at the
            most-recent feature bar.

    Returns:
        Tuple ``(X_seq, y_seq)`` where ``X_seq`` has shape
        ``(N - lookback - horizon - 1, lookback, F)`` and ``y_seq`` has
        shape ``(N - lookback - horizon - 1,)``.

    Raises:
        ValueError: If shapes are inconsistent or there is too little data.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, F), got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (N,), got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    N = len(X)
    n_windows = N - lookback - horizon - 1
    if n_windows <= 0:
        raise ValueError(
            f"Not enough samples ({N}) for lookback={lookback} + horizon={horizon}"
        )

    X_seq = np.array(
        [X[i : i + lookback] for i in range(n_windows)],
        dtype=np.float32,
    )
    y_seq = np.array(
        [y[i + lookback + horizon - 1] for i in range(n_windows)],
        dtype=np.float32,
    )

    logger.info(
        "Built %d windows: X %s --> %s, y %s --> %s (lookback=%d, horizon=%d)",
        n_windows,
        X.shape,
        X_seq.shape,
        y.shape,
        y_seq.shape,
        lookback,
        horizon,
    )

    return X_seq, y_seq


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
        X_lagged: (N-lookback-1, F * (lookback+1)) with lag features
        y_aligned: (N-lookback-1,) aligned targets

    For each sample i at time t:
        X_lagged[i] = [X[t], X[t-1], X[t-2], ..., X[t-lookback]]
        y_aligned[i] = y[t]

    The final row is dropped so that ``y[N-1]`` (which is NaN under the
    canonical target ``y[t] = log(close[t+1]/close[t])`` because the next
    close is unknown) is never selected as a target. The resulting sample
    count matches ``build_lstm_windows`` (with the same ``lookback`` and
    ``horizon=0``), so the two representations can be compared directly.

    Args:
        X: Feature matrix (N, F)
        y: Target vector (N,)
        lookback: Number of lags to create (default: 20)

    Returns:
        Tuple of (X_lagged, y_aligned)
        - X_lagged: (N-lookback-1, F*(lookback+1))
        - y_aligned: (N-lookback-1,)

    Raises:
        ValueError: If insufficient samples or shape mismatch

    Example:
        >>> X = np.random.randn(1000, 10)  # 1000 samples, 10 features
        >>> y = np.random.randn(1000)
        >>> X_lag, y_lag = build_xgboost_lag_features(X, y, lookback=20)
        >>> print(X_lag.shape)  # (979, 210) = 979 samples, 10*21 features
        >>> print(y_lag.shape)  # (979,)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, F), got shape {X.shape}")

    if y.ndim != 1:
        raise ValueError(f"y must be 1D (N,), got shape {y.shape}")

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    N, F = X.shape

    if N <= lookback + 1:
        raise ValueError(
            f"Not enough samples ({N}) for lookback ({lookback}). "
            f"Need at least {lookback + 2} samples."
        )

    # Create lag features. Each base array has length ``N - lookback``.
    # We trim the trailing element from each so the final row (whose target
    # would be y[N-1]) is dropped, leaving ``N - lookback - 1`` rows.
    lag_arrays = []

    for lag in range(lookback + 1):
        # lag=0 is current, lag=1 is previous, etc.
        if lag == 0:
            lag_arrays.append(X[lookback:-1])
        else:
            lag_arrays.append(X[lookback - lag : -lag - 1])

    # Concatenate all lags horizontally
    X_lagged = np.concatenate(lag_arrays, axis=1).astype(np.float32)

    # Align targets, dropping the final NaN target y[N-1].
    y_aligned = y[lookback:-1].astype(np.float32)

    logger.info(
        f"Built {len(y_aligned)} lagged samples: "
        f"X {X.shape} --> {X_lagged.shape} ({lookback+1} lags), "
        f"y {y.shape} --> {y_aligned.shape}"
    )

    return X_lagged, y_aligned
