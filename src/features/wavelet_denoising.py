import logging
import numpy as np
import pywt
import pandas as pd

logger = logging.getLogger(__name__)


def wavelet_denoise(series, wavelet="haar", level=1):
    """
    Applies Discrete Wavelet Transform (DWT) to denoise a financial time-series.

    This technique decomposes the signal into time and frequency domains,
    allowing the removal of stochastic noise while preserving the underlying
    trend of the stock price.

    Args:
        series (pd.Series or np.array): The input signal (e.g., closing price).
        wavelet (str): The type of wavelet to use (default: "haar").
        level (int): Decomposition level.

    Returns:
        np.array: The reconstructed, denoised signal.
    """
    # 1. Decompose the signal into wavelet coefficients
    coeffs = pywt.wavedec(series, wavelet, mode="per", level=level)

    # 2. Calculate a universal threshold for noise removal
    # Using the Median Absolute Deviation (MAD) of the finest detail coefficients
    sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-1] - np.median(coeffs[-1])))
    threshold = sigma * np.sqrt(2 * np.log(len(series)))

    # 3. Apply Soft Thresholding to detail coefficients (exclude the approximation)
    denoised_coeffs = [coeffs[0]]  # Keep the low-frequency approximation
    for i in range(1, len(coeffs)):
        denoised_coeffs.append(pywt.threshold(coeffs[i], value=threshold, mode="soft"))

    # 4. Reconstruct the signal
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet, mode="per")

    # Handle potential length mismatch due to padding/mode
    if len(denoised_signal) > len(series):
        denoised_signal = denoised_signal[: len(series)]

    return denoised_signal


def apply_denoising_pipeline(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Orchestrates the denoising process as part of the feature engineering pipeline.

    Args:
        df (pd.DataFrame): Dataframe containing the 'close' price.
        cfg: Hydra configuration (cfg.features.denoising).

    Returns:
        pd.DataFrame: Dataframe with an additional 'close_denoised' column.
    """
    if not cfg.enabled:
        logger.info("Wavelet denoising is disabled in config. Skipping...")
        return df

    logger.info(
        f"Applying {cfg.wavelet_type} wavelet denoising (Level: {cfg.decomposition_level})..."
    )

    try:
        # Apply to the 'close' column as specified in requirements
        df["close_denoised"] = wavelet_denoise(
            df["close"].values, wavelet=cfg.wavelet_type, level=cfg.decomposition_level
        )

        # In a production system, we often use the denoised signal
        # as the base for further indicators (RSI, MACD) to ensure consistency.
        logger.info("Signal denoising successful.")

    except Exception as e:
        logger.error(f"Failed to apply wavelet denoising: {str(e)}")
        # Fallback: create the column from the original so downstream modules don't break
        df["close_denoised"] = df["close"]

    return df
