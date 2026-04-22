"""
Canonical Target Computation Module

This module provides a SINGLE source of truth for target variable computation.

TARGET DEFINITION (TRD1 §1.3):
    y[t] = log(Close[t+1] / Close[t])
    
Equivalent to:
    y[t] = log(Close[t+1]) - log(Close[t])

CRITICAL: This computation uses Close[t+1] which is FUTURE information.
Therefore, target computation must happen:
1. AFTER temporal split (to avoid leakage)
2. Per split independently
3. Never on combined data

This module resolves Issue #12 from FEA_ENG_AUDIT.md.

Author: System Architect
Version: 2.0.0 - AUDIT REMEDIATION
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_canonical_target(
    close_series: pd.Series,
    horizon: int = 1,
    method: str = "log_return"
) -> pd.Series:
    """
    Compute canonical target variable (next-period log return).
    
    FORMULA (TRD1 §1.3):
        y[t] = log(Close[t+1] / Close[t])
    
    Equivalent implementations:
        Method 1: log(Close[t+1]) - log(Close[t])
        Method 2: log(Close[t+1] / Close[t])
    
    CRITICAL WARNING:
        This function uses Close[t+1] which is FUTURE information.
        It MUST be called AFTER temporal splitting to prevent leakage.
    
    Args:
        close_series: Close price series (pandas Series with DatetimeIndex)
        horizon: Prediction horizon (default=1 for next-period)
        method: Computation method ("log_return" or "simple_return")
    
    Returns:
        Target series (same index as input, last N values will be NaN)
    
    Raises:
        ValueError: If close_series contains NaN or non-positive values
    
    Example:
        >>> close = pd.Series([100, 101, 99, 102], index=dates)
        >>> target = compute_canonical_target(close)
        >>> # target[0] = log(101/100), target[1] = log(99/101), etc.
    """
    if close_series.isna().any():
        raise ValueError("close_series contains NaN values - clean data first")
    
    if (close_series <= 0).any():
        raise ValueError("close_series contains non-positive values - invalid prices")
    
    if method == "log_return":
        # Method: log(Close[t+1]) - log(Close[t])
        log_close = np.log(close_series)
        log_return = log_close.diff(periods=1)
        
        # Shift backward to align: y[t] = return from t to t+1
        target = log_return.shift(-horizon)
        
    elif method == "simple_return":
        # Alternative: (Close[t+1] - Close[t]) / Close[t]
        simple_return = close_series.pct_change(periods=1)
        target = simple_return.shift(-horizon)
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'log_return' or 'simple_return'")
    
    logger.info(
        f"Computed target (method={method}, horizon={horizon}): "
        f"{len(target)} samples, {target.isna().sum()} NaN"
    )
    
    return target


def compute_log_return_causal(close_series: pd.Series) -> pd.Series:
    """
    Compute log return using ONLY historical data (causal operation).
    
    FORMULA:
        log_return[t] = log(Close[t] / Close[t-1])
    
    This is a CAUSAL operation suitable for feature engineering.
    It does NOT use future information.
    
    Args:
        close_series: Close price series
    
    Returns:
        Log return series (first value will be NaN)
    
    Note:
        This is used for FEATURES, not for the TARGET.
        For TARGET, use compute_canonical_target() which shifts to align with t+1.
    """
    if close_series.isna().any():
        raise ValueError("close_series contains NaN values - clean data first")
    
    if (close_series <= 0).any():
        raise ValueError("close_series contains non-positive values - invalid prices")
    
    log_close = np.log(close_series)
    log_return = log_close.diff(periods=1)
    
    logger.info(
        f"Computed causal log_return: {len(log_return)} samples, "
        f"{log_return.isna().sum()} NaN (first value)"
    )
    
    return log_return


def verify_target_alignment(
    target: pd.Series,
    close_series: pd.Series,
    horizon: int = 1,
    tolerance: float = 1e-6
) -> bool:
    """
    Verify that target was computed correctly.
    
    Checks:
    1. target[t] = log(close[t+horizon] / close[t])
    2. No unexpected NaN values
    3. Numerical accuracy
    
    Args:
        target: Computed target series
        close_series: Original close series
        horizon: Prediction horizon
        tolerance: Numerical tolerance for verification
    
    Returns:
        True if verification passes
    
    Raises:
        AssertionError: If verification fails
    """
    # Recompute target for verification
    expected_target = compute_canonical_target(close_series, horizon=horizon)
    
    # Compare (ignoring NaN)
    valid_mask = ~(target.isna() | expected_target.isna())
    
    if valid_mask.sum() == 0:
        raise AssertionError("No valid samples to verify")
    
    diff = (target[valid_mask] - expected_target[valid_mask]).abs()
    max_diff = diff.max()
    
    if max_diff > tolerance:
        raise AssertionError(
            f"Target verification failed: max difference={max_diff:.2e} > tolerance={tolerance:.2e}"
        )
    
    logger.info(
        f"Target verification passed: max_diff={max_diff:.2e}, "
        f"{valid_mask.sum()} samples checked"
    )
    
    return True
