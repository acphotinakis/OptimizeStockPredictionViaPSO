"""
Unit tests for data validation module.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.data_validation import (
    check_missing_timestamps,
    check_duplicates,
    check_missing_values,
    detect_outliers,
    run_validation_suite,
)


def test_check_missing_timestamps(sample_ohlcv_data):
    """Test missing timestamp detection on clean data."""
    result = check_missing_timestamps(sample_ohlcv_data, freq="1min")
    
    assert "missing_count" in result
    assert "missing_pct" in result
    assert result["missing_count"] == 0
    assert result["missing_pct"] == 0.0


def test_check_missing_timestamps_with_gaps(sample_ohlcv_with_gaps):
    """Test missing timestamp detection with intentional gaps."""
    result = check_missing_timestamps(sample_ohlcv_with_gaps, freq="1min")
    
    assert result["missing_count"] == 0  # No missing timestamps in index
    assert isinstance(result["missing_pct"], float)


def test_check_duplicates_clean(sample_ohlcv_data):
    """Test duplicate detection on clean data."""
    result = check_duplicates(sample_ohlcv_data)
    
    assert result["duplicate_count"] == 0
    assert result["is_valid"] is True


def test_check_duplicates_with_dupes():
    """Test duplicate detection with intentional duplicates."""
    dates = pd.date_range(start="2023-01-01 09:30", periods=50, freq="1min")
    df = pd.DataFrame({"close": np.random.randn(50)}, index=dates)
    
    # Add duplicate timestamp
    df = pd.concat([df, pd.DataFrame({"close": [1.0]}, index=[dates[10]])])
    
    result = check_duplicates(df)
    
    assert result["duplicate_count"] > 0
    assert result["is_valid"] is False


def test_check_missing_values_clean(sample_ohlcv_data):
    """Test missing value detection on clean data."""
    result = check_missing_values(sample_ohlcv_data)
    
    assert result["total_nulls"] == 0
    assert result["is_valid"] is True


def test_check_missing_values_with_nans(sample_ohlcv_with_gaps):
    """Test missing value detection with NaN values."""
    result = check_missing_values(sample_ohlcv_with_gaps)
    
    assert result["total_nulls"] > 0
    assert result["is_valid"] is False
    assert "close" in result["null_counts"]


def test_detect_outliers_iqr(sample_ohlcv_with_outliers):
    """Test IQR-based outlier detection."""
    result = detect_outliers(sample_ohlcv_with_outliers, method="iqr")
    
    assert isinstance(result, dict)
    assert "close" in result
    assert result["close"] > 0  # Should detect the extreme values


def test_detect_outliers_zscore(sample_ohlcv_with_outliers):
    """Test Z-score based outlier detection."""
    result = detect_outliers(sample_ohlcv_with_outliers, method="zscore")
    
    assert isinstance(result, dict)
    assert "close" in result


def test_run_validation_suite_pass(sample_ohlcv_data, mock_config):
    """Test full validation suite on clean data."""
    result = run_validation_suite(sample_ohlcv_data, mock_config)
    
    assert "is_fit_for_training" in result
    assert "failed_reasons" in result
    assert "timestamp_integrity" in result
    assert "duplicates" in result
    assert "null_values" in result
    assert "outliers" in result
    
    assert result["is_fit_for_training"] is True
    assert len(result["failed_reasons"]) == 0


def test_run_validation_suite_fail_duplicates(mock_config):
    """Test validation suite failure on duplicate timestamps."""
    dates = pd.date_range(start="2023-01-01 09:30", periods=50, freq="1min")
    df = pd.DataFrame({
        "open": np.random.randn(50) + 100,
        "high": np.random.randn(50) + 101,
        "low": np.random.randn(50) + 99,
        "close": np.random.randn(50) + 100,
        "volume": np.random.randint(1000, 10000, 50),
    }, index=dates)
    
    # Add duplicate
    df = pd.concat([df, pd.DataFrame({
        "open": [100.0],
        "high": [101.0],
        "low": [99.0],
        "close": [100.0],
        "volume": [5000],
    }, index=[dates[10]])])
    
    result = run_validation_suite(df, mock_config)
    
    assert result["is_fit_for_training"] is False
    assert len(result["failed_reasons"]) > 0
