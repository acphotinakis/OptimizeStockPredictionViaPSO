"""
Unit tests for wavelet denoising module.
"""
import pytest
import numpy as np
import pandas as pd
from src.features.wavelet_denoising import wavelet_denoise, apply_denoising_pipeline


def test_wavelet_denoise_basic():
    """Test basic wavelet denoising."""
    # Create a noisy signal
    np.random.seed(42)
    t = np.linspace(0, 1, 200)
    clean_signal = np.sin(2 * np.pi * 5 * t)
    noise = np.random.randn(200) * 0.1
    noisy_signal = clean_signal + noise
    
    # Denoise
    denoised = wavelet_denoise(noisy_signal, wavelet="haar", level=1)
    
    # Check output properties
    assert len(denoised) == len(noisy_signal)
    assert not np.isnan(denoised).any()
    
    # Denoised signal should be closer to clean signal than noisy signal
    noise_rmse = np.sqrt(np.mean((noisy_signal - clean_signal) ** 2))
    denoised_rmse = np.sqrt(np.mean((denoised - clean_signal) ** 2))
    
    assert denoised_rmse < noise_rmse


def test_wavelet_denoise_different_wavelets():
    """Test denoising with different wavelet types."""
    np.random.seed(42)
    signal = np.random.randn(100)
    
    for wavelet in ["haar", "db4", "sym5"]:
        denoised = wavelet_denoise(signal, wavelet=wavelet, level=1)
        assert len(denoised) == len(signal)
        assert not np.isnan(denoised).any()


def test_apply_denoising_pipeline_enabled(sample_ohlcv_data, mock_config):
    """Test denoising pipeline when enabled."""
    mock_config.features.denoising.enabled = True
    
    result = apply_denoising_pipeline(sample_ohlcv_data, mock_config.features.denoising)
    
    assert "close_denoised" in result.columns
    assert len(result) == len(sample_ohlcv_data)
    assert not result["close_denoised"].isna().any()


def test_apply_denoising_pipeline_disabled(sample_ohlcv_data, mock_config):
    """Test denoising pipeline when disabled."""
    mock_config.features.denoising.enabled = False
    
    result = apply_denoising_pipeline(sample_ohlcv_data, mock_config.features.denoising)
    
    # Should return original dataframe without close_denoised column
    assert "close_denoised" not in result.columns
    pd.testing.assert_frame_equal(result, sample_ohlcv_data)


def test_apply_denoising_pipeline_error_handling(sample_ohlcv_data, mock_config):
    """Test that denoising pipeline handles errors gracefully."""
    mock_config.features.denoising.enabled = True
    mock_config.features.denoising.wavelet_type = "invalid_wavelet"
    
    # Should not raise error, but fall back to original close
    result = apply_denoising_pipeline(sample_ohlcv_data, mock_config.features.denoising)
    
    assert "close_denoised" in result.columns
    # Should have fallen back to original close
    pd.testing.assert_series_equal(result["close_denoised"], result["close"])
