"""
Unit tests for target variable creation.
"""
import pytest
import pandas as pd
import numpy as np
from src.features.target import create_target_variable, validate_target


def test_create_target_mid_price_return(sample_ohlcv_data):
    """Test mid-price return target creation."""
    result = create_target_variable(sample_ohlcv_data, method="mid_price_return", horizon=1)
    
    assert "target" in result.columns
    assert len(result) == len(sample_ohlcv_data) - 1  # Lost last row due to shift
    
    # Verify calculation manually for first few rows
    mid_price = (sample_ohlcv_data["high"] + sample_ohlcv_data["low"]) / 2
    expected_return = mid_price.pct_change(periods=1).shift(-1)
    
    # Compare first valid value
    np.testing.assert_almost_equal(
        result["target"].iloc[1],
        expected_return.iloc[1],
        decimal=6
    )


def test_create_target_close_return(sample_ohlcv_data):
    """Test close return target creation."""
    result = create_target_variable(sample_ohlcv_data, method="close_return", horizon=1)
    
    assert "target" in result.columns
    
    # Verify calculation
    expected_return = sample_ohlcv_data["close"].pct_change(periods=1).shift(-1)
    
    # Compare valid values
    np.testing.assert_array_almost_equal(
        result["target"].iloc[1:10].values,
        expected_return.iloc[1:10].values,
        decimal=6
    )


def test_create_target_close_price(sample_ohlcv_data):
    """Test raw close price target creation."""
    result = create_target_variable(sample_ohlcv_data, method="close_price", horizon=1)
    
    assert "target" in result.columns
    
    # Verify: target[t] should equal close[t+1]
    np.testing.assert_array_almost_equal(
        result["target"].iloc[:-1].values,
        sample_ohlcv_data["close"].iloc[1:].values,
        decimal=6
    )


def test_create_target_multi_step_horizon(sample_ohlcv_data):
    """Test target creation with multi-step horizon."""
    horizon = 5
    result = create_target_variable(sample_ohlcv_data, method="close_return", horizon=horizon)
    
    assert "target" in result.columns
    
    # Should lose 'horizon' rows at the end
    assert len(result) <= len(sample_ohlcv_data) - horizon


def test_create_target_invalid_method(sample_ohlcv_data):
    """Test that invalid method raises error."""
    with pytest.raises(ValueError, match="Unknown target method"):
        create_target_variable(sample_ohlcv_data, method="invalid_method")


def test_validate_target_valid(sample_ohlcv_data):
    """Test target validation on valid target."""
    df = create_target_variable(sample_ohlcv_data, method="mid_price_return")
    
    is_valid = validate_target(df)
    
    assert is_valid is True


def test_validate_target_missing():
    """Test target validation when target column missing."""
    df = pd.DataFrame({"close": [1, 2, 3]})
    
    is_valid = validate_target(df)
    
    assert is_valid is False


def test_validate_target_with_nan():
    """Test target validation with NaN values."""
    df = pd.DataFrame({
        "close": [1, 2, 3, 4, 5],
        "target": [0.1, 0.2, np.nan, 0.4, 0.5],
    })
    
    is_valid = validate_target(df)
    
    assert is_valid is False


def test_validate_target_with_inf():
    """Test target validation with infinite values."""
    df = pd.DataFrame({
        "close": [1, 2, 3, 4, 5],
        "target": [0.1, 0.2, np.inf, 0.4, 0.5],
    })
    
    is_valid = validate_target(df)
    
    assert is_valid is False


def test_target_statistics(sample_ohlcv_data):
    """Test that target statistics are reasonable."""
    result = create_target_variable(sample_ohlcv_data, method="mid_price_return")
    
    # Returns should have mean close to 0 for random walk
    assert abs(result["target"].mean()) < 0.1
    
    # Standard deviation should be positive
    assert result["target"].std() > 0
