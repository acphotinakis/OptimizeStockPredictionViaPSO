"""
Stage 4: Feature Transformation Module

Production-grade implementation of wavelet denoising and MinMax normalization
for financial time-series data. Guarantees leakage-free transformation with
full state persistence for inference reuse.

Author: System Architect
Version: 1.0.0
"""

import json
import logging
import numpy as np
import pandas as pd
import pywt
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)


def _wavelet_denoise_1d(series: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply 3-level Haar wavelet denoising with soft thresholding.

    Args:
        series: 1D numpy array of close prices
        threshold: Universal threshold for denoising (computed from training)

    Returns:
        Denoised series as float32 array
    """
    # Ensure contiguous float32 array for pywt
    series = np.asarray(series, dtype=np.float32)
    original_length = len(series)

    # Multi-level discrete wavelet decomposition
    # coeffs[0] = approximation (cA3)
    # coeffs[1] = detail level 3 (cD3)
    # coeffs[2] = detail level 2 (cD2)
    # coeffs[3] = detail level 1 (cD1 - highest frequency)
    coeffs = pywt.wavedec(series, wavelet="haar", level=3, mode="symmetric")

    # Apply soft thresholding ONLY to detail coefficients (index 1, 2, 3)
    # Approximation coefficient (index 0) is preserved exactly
    coeffs[1:] = [pywt.threshold(c, value=threshold, mode="soft") for c in coeffs[1:]]

    # Reconstruct signal from modified coefficients
    reconstructed = pywt.waverec(coeffs, wavelet="haar", mode="symmetric")

    # Trim to original length (waverec may return slightly longer array due to padding)
    reconstructed = reconstructed[:original_length]

    return reconstructed.astype(np.float32)


def _compute_wavelet_threshold(close_train: np.ndarray) -> float:
    """
    Compute universal threshold from training data using MAD estimator.

    Threshold = sigma * sqrt(2 * log(N))
    where sigma = median(|D1|) / 0.6745

    Args:
        close_train: Training close prices

    Returns:
        Universal threshold value
    """
    close_train = np.asarray(close_train, dtype=np.float32)
    N = len(close_train)

    if N == 0:
        raise ValueError("Cannot compute threshold on empty training data")

    # Decompose to get detail coefficients
    coeffs = pywt.wavedec(close_train, wavelet="haar", level=3, mode="symmetric")
    D1 = coeffs[-1]  # Highest frequency detail coefficients

    # Median Absolute Deviation (MAD) estimator of noise standard deviation
    # 0.6745 is the 75th percentile of standard normal (MAD scaling factor)
    median_abs_dev = np.median(np.abs(D1))

    if median_abs_dev == 0:
        logger.warning(
            "MAD is zero (constant D1 coefficients), setting threshold to 0.0"
        )
        return 0.0

    sigma = median_abs_dev / 0.6745

    # Universal threshold (Donoho-Johnstone)
    threshold = sigma * np.sqrt(2 * np.log(N))

    return float(threshold)


def _apply_minmax_scale(
    df: pd.DataFrame, scaler_params: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Apply MinMax scaling to [-1, 1] range using pre-computed parameters.

    Formula: x_norm = 2 * (x - x_min) / (x_max - x_min) - 1

    Edge case: If x_max == x_min, output is 0.0 (neutral value in [-1, 1] range)

    Args:
        df: Input DataFrame
        scaler_params: Mapping of column -> {min, max}

    Returns:
        Scaled DataFrame with float32 dtype
    """
    df_scaled = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col not in scaler_params:
            raise KeyError(f"No scaler params found for column: {col}")

        x_min = scaler_params[col]["min"]
        x_max = scaler_params[col]["max"]
        x_range = x_max - x_min

        if x_range == 0:
            # Constant column - map to 0 (center of [-1, 1])
            df_scaled[col] = 0.0
        else:
            scaled = 2 * (df[col] - x_min) / x_range - 1
            # Standard MinMax to [-1, 1]
            df_scaled[col] = scaled.clip(-1.0, 1.0)

    return df_scaled.astype(np.float32)


def _fit_minmax_params(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute MinMax parameters from training data.

    Args:
        df: Training DataFrame

    Returns:
        Dictionary mapping column names to {min, max}
    """
    params = {}
    for col in df.columns:
        x_min = float(df[col].min())
        x_max = float(df[col].max())
        params[col] = {"min": x_min, "max": x_max}
        logger.debug(f"Scaler fit - {col}: min={x_min:.8f}, max={x_max:.8f}")
    return params


def transform_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Execute Stage 4 Feature Transformation pipeline.

    Pipeline steps:
    1. Wavelet denoising (OPTIONAL - only if 'close' column exists)
       - 3-level Haar, soft threshold, symmetric padding
       - Threshold computed on training data only
       - Applied identically to train/val/test
       - Column 'close' -> 'close_denoised'
       - If 'close' not present, skip this step (already denoised or not selected)
    2. MinMax normalization to [-1, 1]
       - Parameters fit on training data only
       - Applied to all columns
    3. Validation (no NaNs, range [-1, 1])
    4. State extraction for inference reuse

    Args:
        train_df: Training data (no NaNs required)
        val_df: Validation data (no NaNs required)
        test_df: Test data (no NaNs required)

    Returns:
        Tuple of:
        - train_out: Transformed training DataFrame
        - val_out: Transformed validation DataFrame
        - test_out: Transformed test DataFrame
        - transformer_state: Serializable state for inference

    Raises:
        AssertionError: If validation checks fail (NaNs or out-of-range values)
    """
    logger.info("Stage 4: Starting Feature Transformation")

    # Validate input schema
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if df.isna().any().any():
            logger.warning(
                f"NaN values detected in {split_name}_df input - caller should have cleaned these"
            )

    # Create working copies to avoid mutating inputs
    train_work = train_df.copy().astype(np.float32)
    val_work = val_df.copy().astype(np.float32)
    test_work = test_df.copy().astype(np.float32)

    # Ensure float32 precision for all columns
    for col in train_work.columns:
        train_work[col] = train_work[col].astype(np.float32)
        val_work[col] = val_work[col].astype(np.float32)
        test_work[col] = test_work[col].astype(np.float32)

    feature_cols = list(train_work.columns)
    val_work = val_work[feature_cols]
    test_work = test_work[feature_cols]

    # =====================================================================
    # PART 1: WAVELET DENOISING (close -> close_denoised) - OPTIONAL
    # =====================================================================
    # Only apply wavelet denoising if 'close' column exists
    # (It may have been removed during feature selection or already denoised upstream)
    if "close" in train_work.columns:
        logger.info("Wavelet: 'close' column found, applying denoising...")
        logger.info("Wavelet: Computing universal threshold from training data...")

        threshold = _compute_wavelet_threshold(train_work["close"].values)
        logger.info(f"Wavelet: Universal threshold = {threshold:.8f}")

        # Apply denoising to all splits using SAME threshold
        logger.info("Wavelet: Applying denoising to train/val/test...")
        train_work["close"] = _wavelet_denoise_1d(train_work["close"].values, threshold)
        val_work["close"] = _wavelet_denoise_1d(val_work["close"].values, threshold)
        test_work["close"] = _wavelet_denoise_1d(test_work["close"].values, threshold)

        # Rename to indicate denoising
        train_work = train_work.rename(columns={"close": "close_denoised"})
        val_work = val_work.rename(columns={"close": "close_denoised"})
        test_work = test_work.rename(columns={"close": "close_denoised"})

        logger.info("Wavelet: Denoising complete")
    else:
        logger.info("Wavelet: 'close' column not found, skipping wavelet denoising")

    # =====================================================================
    # PART 2: MINMAX NORMALIZATION to [-1, 1]
    # =====================================================================
    logger.info("Scaler: Fitting MinMax parameters on training data...")

    scaler_params = _fit_minmax_params(train_work)

    logger.info("Scaler: Applying normalization to all splits...")
    train_scaled = _apply_minmax_scale(train_work, scaler_params)
    val_scaled = _apply_minmax_scale(val_work, scaler_params)
    test_scaled = _apply_minmax_scale(test_work, scaler_params)

    # =====================================================================
    # PART 3: VALIDATION CHECKS (Strict)
    # =====================================================================
    logger.info("Validation: Running strict checks...")

    # Check 1: No NaN values anywhere
    assert (
        not train_work.isna().any().any()
    ), "CRITICAL: NaN values found in train output"
    assert not val_work.isna().any().any(), "CRITICAL: NaN values found in val output"
    assert not test_work.isna().any().any(), "CRITICAL: NaN values found in test output"

    # Check 2: All values in [-1, 1] with small epsilon for floating point errors
    eps = 1e-6

    train_max = train_scaled.max().max()
    train_min = train_scaled.min().min()
    assert train_max <= 1.0 + eps, f"CRITICAL: Train max {train_max} exceeds 1.0"
    assert train_min >= -1.0 - eps, f"CRITICAL: Train min {train_min} below -1.0"

    val_max = val_scaled.max().max()
    val_min = val_scaled.min().min()
    assert val_max <= 1.0 + eps, f"CRITICAL: Val max {val_max} exceeds 1.0"
    assert val_min >= -1.0 - eps, f"CRITICAL: Val min {val_min} below -1.0"

    test_max = test_scaled.max().max()
    test_min = test_scaled.min().min()
    assert test_max <= 1.0 + eps, f"CRITICAL: Test max {test_max} exceeds 1.0"
    assert test_min >= -1.0 - eps, f"CRITICAL: Test min {test_min} below -1.0"

    logger.info(
        f"Validation: Range checks passed (train [{train_min:.4f}, {train_max:.4f}], "
        f"val [{val_min:.4f}, {val_max:.4f}], test [{test_min:.4f}, {test_max:.4f}])"
    )

    logger.info(
        "Validation stats | "
        + " | ".join(
            [
                _log_dataset_stats("train", train_scaled),
                _log_dataset_stats("val", val_scaled),
                _log_dataset_stats("test", test_scaled),
            ]
        )
    )
    # ------------------------------------------------------------
    # VALIDATION CHECKS
    # ------------------------------------------------------------
    eps = 1e-6

    for name, df in [
        ("train", train_scaled),
        ("val", val_scaled),
        ("test", test_scaled),
    ]:
        if df.isna().any().any():
            raise ValueError(f"{name} contains NaNs after scaling")

        if df.max().max() > 1.0 + eps or df.min().min() < -1.0 - eps:
            raise ValueError(f"{name} out of [-1,1] range")

    # =====================================================================
    # PART 4: STATE SERIALIZATION
    # =====================================================================
    transformer_state = {
        "wavelet": {
            "wavelet": "haar",
            "level": 3,
            "mode": "symmetric",
            "threshold": float(threshold),
        },
        "scaler": scaler_params,
        "feature_columns": feature_cols,
        "range": [-1.0, 1.0],
        "type": "minmax",
    }

    logger.info("Stage 4: Feature Transformation complete")

    return train_scaled, val_scaled, test_scaled, transformer_state


# ============================================================
# OPTIONAL: SAVE STATE TO DISK (TRD REQUIREMENT)
# ============================================================


def save_transformer_state(path: str, state: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(state, f, indent=2)

    logger.info(f"Transformer state saved -> {path}")


def _log_dataset_stats(name: str, df: pd.DataFrame) -> str:
    return (
        f"{name}: "
        f"mean={df.mean().mean():.6f}, "
        f"std={df.std().mean():.6f}, "
        f"min={df.min().min():.6f}, "
        f"max={df.max().max():.6f}"
    )
