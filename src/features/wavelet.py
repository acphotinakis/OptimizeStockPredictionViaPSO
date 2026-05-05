"""
Wavelet Denoising Module for Financial Time-Series

Implements 3-level Haar wavelet transform with soft thresholding for noise reduction.
Strictly enforces leakage-free processing: threshold computed on training data only.

TRD Reference: TRD1 4.1, TRD2 4.1, TRD3 4.1
Paper Attribution: Zeng et al. 2025

Author: System Architect
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pywt

logger = logging.getLogger(__name__)


def _apply_wavelet(
    X: np.ndarray, names: List[str], fit_mode: bool, _wavelet_threshold: Optional[float]
) -> Tuple[np.ndarray, List[str], Optional[float]]:
    """
    Apply wavelet denoising to 'close' feature.

    Args:
        X: Feature matrix (N, F)
        names: List of feature names
        fit_mode: If True, compute threshold; if False, use fitted threshold

    Returns:
        Tuple of (X_updated, names_updated) with close_denoised replacing close
    """
    if "close" not in names:
        logger.warning("'close' feature not found, skipping wavelet denoising")
        return X, names, None

    close_idx = names.index("close")
    close_series = pd.Series(X[:, close_idx])

    if fit_mode:
        # FIT MODE: Compute threshold on training data
        close_denoised, threshold = apply_wavelet_denoising(
            close_series, threshold_train=None
        )
        _wavelet_threshold = threshold
    else:
        # TRANSFORM MODE: Use fitted threshold
        if _wavelet_threshold is None:
            raise RuntimeError("Wavelet threshold not fitted")
        close_denoised = apply_wavelet_denoising(
            close_series, threshold_train=_wavelet_threshold
        )

    # Replace close with close_denoised
    X[:, close_idx] = close_denoised.values
    names[close_idx] = "close_denoised"

    return X, names, _wavelet_threshold


def apply_wavelet_denoising(
    close_series: pd.Series, threshold_train: Optional[float] = None
) -> pd.Series:
    """
    Apply DWT denoising to close price series using Haar wavelet.

    Method:
    - Haar wavelet (symmetric boundary handling)
    - 3-level decomposition
    - Soft thresholding on detail coefficients D1, D2, D3
    - Approximation coefficient A3 preserved unchanged

    Threshold Estimation:
    - Universal threshold: σ × √(2 × log(N))
    - σ estimated via Median Absolute Deviation (MAD) on training data
    - Formula: σ = median(|D1|) / 0.6745

    Args:
        close_series: Raw closing price series (pandas Series with DatetimeIndex)
        threshold_train: Pre-computed threshold from training (for val/test).
                        If None, compute threshold from current series (training mode).

    Returns:
        Denoised closing price series (same index as input)

    Raises:
        ValueError: If input series is empty or contains NaN/Inf

    Example:
        >>> # Training phase
        >>> close_train = df_train['close']
        >>> close_denoised, threshold = apply_wavelet_denoising(close_train)
        >>>
        >>> # Validation/Test phase (reuse threshold)
        >>> close_val_denoised = apply_wavelet_denoising(df_val['close'], threshold_train=threshold)
    """
    # Validate input
    if len(close_series) == 0:
        raise ValueError("Cannot denoise empty series")

    if close_series.isna().any():
        raise ValueError("Input series contains NaN values. Clean data first.")

    if np.isinf(close_series.values).any():
        raise ValueError("Input series contains infinite values. Clean data first.")

    # Extract values and index
    close_values = close_series.values.astype(np.float64)
    original_index = close_series.index
    N = len(close_values)

    # Decompose: 3-level Haar DWT
    # coeffs[0] = Approximation A3 (low frequency)
    # coeffs[1] = Detail D3
    # coeffs[2] = Detail D2
    # coeffs[3] = Detail D1 (high frequency - noise)
    coeffs = pywt.wavedec(close_values, wavelet="haar", level=3, mode="symmetric")

    if threshold_train is None:
        # FIT MODE: Estimate threshold on current (training) data
        # Extract highest frequency detail coefficients (D1)
        D1 = coeffs[-1]

        # MAD estimator for noise standard deviation
        # 0.6745 is the 75th percentile of standard normal distribution
        median_abs_dev = np.median(np.abs(D1))

        if median_abs_dev == 0:
            logger.warning(
                "MAD is zero (constant D1 coefficients). Setting threshold to 0.0. "
                "This may indicate insufficient variation in the price series."
            )
            threshold = 0.0
        else:
            sigma = median_abs_dev / 0.6745

            # Universal threshold (Donoho-Johnstone 1994)
            threshold = sigma * np.sqrt(2 * np.log(N))

        logger.info(
            f"Wavelet threshold computed (training): σ={sigma:.6f}, "
            f"threshold={threshold:.6f} (N={N})"
        )
    else:
        # TRANSFORM MODE: Use pre-computed threshold (val/test)
        threshold = threshold_train
        logger.info(
            f"Wavelet denoising (transform): applying threshold={threshold:.6f}"
        )

    # Apply soft thresholding to detail coefficients (D1, D2, D3)
    # Approximation A3 (coeffs[0]) is preserved unchanged
    coeffs_thresholded = [coeffs[0]]  # Keep approximation
    for detail_coeff in coeffs[1:]:  # Threshold details
        thresholded = pywt.threshold(detail_coeff, value=threshold, mode="soft")
        coeffs_thresholded.append(thresholded)

    # Reconstruct denoised signal
    denoised = pywt.waverec(coeffs_thresholded, wavelet="haar", mode="symmetric")

    # Trim to original length (waverec may return slightly longer array)
    denoised = denoised[:N]

    # Return as Series with original index
    denoised_series = pd.Series(denoised, index=original_index, dtype=np.float32)

    if threshold_train is None:
        # Return threshold for reuse in val/test
        return denoised_series, threshold
    else:
        return denoised_series


def compute_wavelet_threshold(close_train: pd.Series) -> float:
    """
    Compute universal threshold from training data for wavelet denoising.

    This is a convenience function when you want to separate threshold computation
    from denoising application.

    Args:
        close_train: Training close price series

    Returns:
        Universal threshold value (scalar float)

    Example:
        >>> threshold = compute_wavelet_threshold(df_train['close'])
        >>> df_train['close_denoised'] = apply_wavelet_denoising(df_train['close'], threshold)
        >>> df_val['close_denoised'] = apply_wavelet_denoising(df_val['close'], threshold)
    """
    _, threshold = apply_wavelet_denoising(close_train, threshold_train=None)
    return threshold


def denoise_pipeline(
    close_train: pd.Series, close_val: pd.Series, close_test: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, float]:
    """
    Full wavelet denoising pipeline for train/val/test splits.

    Guarantees leakage-free processing:
    - Threshold computed ONLY on training data
    - Same threshold applied to validation and test

    Args:
        close_train: Training close prices
        close_val: Validation close prices
        close_test: Test close prices

    Returns:
        Tuple of (train_denoised, val_denoised, test_denoised, threshold)

    Example:
        >>> train_dn, val_dn, test_dn, thr = denoise_pipeline(
        ...     df_train['close'], df_val['close'], df_test['close']
        ... )
    """
    logger.info("Wavelet denoising pipeline: processing train/val/test splits")

    # Fit: compute threshold on training data
    train_denoised, threshold = apply_wavelet_denoising(
        close_train, threshold_train=None
    )

    # Transform: apply threshold to validation and test
    val_denoised = apply_wavelet_denoising(close_val, threshold_train=threshold)
    test_denoised = apply_wavelet_denoising(close_test, threshold_train=threshold)

    logger.info(
        f"Wavelet denoising complete: threshold={threshold:.6f}, "
        f"train_len={len(train_denoised)}, val_len={len(val_denoised)}, "
        f"test_len={len(test_denoised)}"
    )

    return train_denoised, val_denoised, test_denoised, threshold
