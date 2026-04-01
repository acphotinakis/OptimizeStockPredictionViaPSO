"""
Unit tests for data cleaning module.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.data_cleaning import (
    handle_missing_data,
    remove_duplicates,
    handle_outliers,
    run_cleaning_pipeline,
)


def test_remove_duplicates_clean(sample_ohlcv_data):
    """Test duplicate removal on clean data."""
    result = remove_duplicates(sample_ohlcv_data)
    
    assert len(result) == len(sample_ohlcv_data)
    assert not result.index.duplicated().any()


def test_remove_duplicates_with_dupes():
    """Test duplicate removal with intentional duplicates."""
    dates = pd.date_range(start="2023-01-01 09:30", periods=50, freq="1min")
    df = pd.DataFrame({"close": np.random.randn(50)}, index=dates)
    
    # Add duplicate timestamp
    df = pd.concat([df, pd.DataFrame({"close": [1.0]}, index=[dates[10]])])
    
    assert df.index.duplicated().any()
    
    result = remove_duplicates(df)
    
    assert not result.index.duplicated().any()
    assert len(result) == 50  # Original length


def test_handle_missing_data_short_gaps(sample_ohlcv_with_gaps):
    """Test that short gaps are forward-filled."""
    result = handle_missing_data(sample_ohlcv_with_gaps, short_gap=5, long_gap=15)
    
    # Should have filled the 3-minute gap but dropped the 16-minute gap
    assert result["close"].notna().all()
    assert len(result) < len(sample_ohlcv_with_gaps)


def test_handle_missing_data_empty_df():
    """Test handling of empty dataframe."""
    df = pd.DataFrame()
    result = handle_missing_data(df)
    
    assert result.empty


def test_handle_outliers_winsorize(sample_ohlcv_with_outliers):
    """Test winsorization of outliers."""
    original_max = sample_ohlcv_with_outliers["close"].max()
    original_min = sample_ohlcv_with_outliers["close"].min()
    
    result = handle_outliers(sample_ohlcv_with_outliers, method="winsorize")
    
    # After winsorization, extreme values should be clipped
    assert result["close"].max() < original_max
    assert result["close"].min() > original_min
    assert len(result) == len(sample_ohlcv_with_outliers)


def test_handle_outliers_unknown_method(sample_ohlcv_data):
    """Test that unknown method returns original data."""
    result = handle_outliers(sample_ohlcv_data, method="unknown")
    
    pd.testing.assert_frame_equal(result, sample_ohlcv_data)


def test_run_cleaning_pipeline(sample_ohlcv_with_gaps, mock_config):
    """Test full cleaning pipeline."""
    result = run_cleaning_pipeline(sample_ohlcv_with_gaps, mock_config)
    
    # Should have no duplicates
    assert not result.index.duplicated().any()
    
    # Should have no NaN values (either filled or dropped)
    assert result.notna().all().all()
    
    # Should be smaller than original due to long gap removal
    assert len(result) < len(sample_ohlcv_with_gaps)
