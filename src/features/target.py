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

# def compute_canonical_target(
#     close_series: pd.Series, horizon: int = 1, method: str = "log_return"
# ) -> pd.Series:
#     """
#     Compute canonical target variable (next-period log return).

#     FORMULA:
#         y[t] = log(Close[t+1] / Close[t])
#     """
#     target = np.log(close_series.shift(-horizon) / close_series)

#     logger.info(
#         f"Computed target (method={method}, horizon={horizon}): "
#         f"{len(target)} samples, {target.isna().sum()} NaN"
#     )
#     return target


# def compute_canonical_target(
#     close_series: pd.Series, horizon: int = 1, method: str = "next_close"
# ) -> pd.Series:
#     """
#     Compute canonical target variable (future closing price).

#     FORMULA:
#         y[t] = Close[t + horizon]

#     This is a direct price prediction target (not returns).
#     """

#     target = close_series.shift(-horizon)

#     logger.info(
#         f"Computed target (method={method}, horizon={horizon}): "
#         f"{len(target)} samples, {target.isna().sum()} NaN"
#     )

#     return target


def compute_canonical_target(
    close_series: pd.Series,
    horizon: int = 1,
    method: str = "log_return",
) -> pd.Series:
    """
    Compute canonical target variable.

    Supported methods:
    - "log_return": y[t] = log(Close[t + h] / Close[t])
    - "next_close": y[t] = Close[t + h]
    """

    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    if method == "log_return":
        target = np.log(close_series.shift(-horizon) / close_series)

    elif method == "next_close":
        target = close_series.shift(-horizon)

    else:
        raise ValueError(
            f"Unsupported method '{method}'. Use 'log_return' or 'next_close'."
        )

    logger.info(
        f"Computed target (method={method}, horizon={horizon}): "
        f"{len(target)} samples, {target.isna().sum()} NaN"
    )

    return target


def verify_target_alignment(
    target: pd.Series,
    close_series: pd.Series,
    horizon: int = 1,
    tolerance: float = 1e-6,
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
