"""
Unit tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
from src.features.build_features import (
    add_rsi,
    add_macd,
    add_bollinger_bands,
    calculate_indicators,
)


def test_add_rsi(sample_ohlcv_data):
    """Test RSI calculation."""
    result = add_rsi(sample_ohlcv_data.copy(), window=14)
    
    assert "rsi" in result.columns
    # RSI should be between 0 and 100
    assert result["rsi"].dropna().min() >= 0
    assert result["rsi"].dropna().max() <= 100


def test_add_macd(sample_ohlcv_data):
    """Test MACD calculation."""
    result = add_macd(sample_ohlcv_data.copy(), fast=12, slow=26, signal=9)
    
    assert "macd" in result.columns
    assert "macd_signal" in result.columns
    assert "macd_hist" in result.columns
    
    # MACD histogram should be macd - signal
    np.testing.assert_array_almost_equal(
        result["macd_hist"].dropna().values,
        (result["macd"] - result["macd_signal"]).dropna().values,
        decimal=6,
    )


def test_add_bollinger_bands(sample_ohlcv_data):
    """Test Bollinger Bands calculation."""
    result = add_bollinger_bands(sample_ohlcv_data.copy(), window=20, std_dev=2)
    
    assert "bb_mid" in result.columns
    assert "bb_high" in result.columns
    assert "bb_low" in result.columns
    
    # High should be >= mid >= low
    valid_data = result.dropna()
    assert (valid_data["bb_high"] >= valid_data["bb_mid"]).all()
    assert (valid_data["bb_mid"] >= valid_data["bb_low"]).all()


def test_calculate_indicators(sample_ohlcv_data, mock_config):
    """Test full indicator calculation pipeline."""
    result = calculate_indicators(sample_ohlcv_data, mock_config.features)
    
    # Check that all expected features are present
    expected_features = [
        "rsi",
        "macd", "macd_signal", "macd_hist",
        "bb_mid", "bb_high", "bb_low",
        "ema_9", "ema_21",
        "sma_50",
        "rolling_mean", "rolling_std",
    ]
    
    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"
    
    # Result should have no NaN (dropped during processing)
    assert result.notna().all().all()
    
    # Should be smaller than original due to rolling window NaN removal
    assert len(result) < len(sample_ohlcv_data)


def test_calculate_indicators_preserves_ohlcv(sample_ohlcv_data, mock_config):
    """Test that original OHLCV columns are preserved."""
    result = calculate_indicators(sample_ohlcv_data, mock_config.features)
    
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in result.columns
