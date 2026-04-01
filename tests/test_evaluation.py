"""
Unit tests for evaluation metrics module.
"""
import pytest
import numpy as np
import pandas as pd
from src.evaluation.metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_directional_accuracy,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    evaluate_performance,
)


def test_calculate_rmse():
    """Test RMSE calculation with known values."""
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    rmse = calculate_rmse(actual, predicted)
    
    # Calculate expected RMSE manually
    expected = np.sqrt(np.mean((actual - predicted) ** 2))
    
    assert np.isclose(rmse, expected, rtol=1e-6)
    assert rmse > 0


def test_calculate_mae():
    """Test MAE calculation with known values."""
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    mae = calculate_mae(actual, predicted)
    
    # Calculate expected MAE manually
    expected = np.mean(np.abs(actual - predicted))
    
    assert np.isclose(mae, expected, rtol=1e-6)


def test_calculate_rmse_perfect_prediction():
    """Test RMSE is zero for perfect predictions."""
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = actual.copy()
    
    rmse = calculate_rmse(actual, predicted)
    
    assert np.isclose(rmse, 0.0, atol=1e-10)


def test_calculate_directional_accuracy():
    """Test directional accuracy calculation."""
    actual_returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    predicted_returns = np.array([0.015, -0.015, 0.025, 0.005, 0.018])
    
    accuracy = calculate_directional_accuracy(actual_returns, predicted_returns)
    
    # Check signs: +/-, -/-, +/+, -/+, +/+
    # Correct: 4 out of 5 = 0.8
    assert np.isclose(accuracy, 0.8, rtol=1e-6)


def test_calculate_directional_accuracy_perfect():
    """Test directional accuracy with perfect predictions."""
    actual_returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    predicted_returns = np.array([0.02, -0.03, 0.04, -0.02, 0.03])
    
    accuracy = calculate_directional_accuracy(actual_returns, predicted_returns)
    
    assert accuracy == 1.0


def test_calculate_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    # Create synthetic returns with positive mean
    np.random.seed(42)
    strategy_returns = pd.Series(np.random.randn(1000) * 0.001 + 0.0005)
    
    sharpe = calculate_sharpe_ratio(strategy_returns, risk_free_rate=0.0)
    
    # Sharpe should be positive for positive mean returns
    assert sharpe > 0
    assert not np.isnan(sharpe)


def test_calculate_sharpe_ratio_zero_std():
    """Test Sharpe ratio with zero standard deviation."""
    strategy_returns = pd.Series([0.001] * 100)
    
    sharpe = calculate_sharpe_ratio(strategy_returns)
    
    # Should return 0 when std is 0
    assert sharpe == 0.0


def test_calculate_sharpe_ratio_insufficient_data():
    """Test Sharpe ratio with insufficient data."""
    strategy_returns = pd.Series([0.001])
    
    sharpe = calculate_sharpe_ratio(strategy_returns)
    
    assert sharpe == 0.0


def test_calculate_max_drawdown():
    """Test maximum drawdown calculation."""
    # Create cumulative returns with a known drawdown
    cumulative_returns = pd.Series([1.0, 1.1, 1.2, 0.9, 0.8, 1.0, 1.3])
    
    mdd = calculate_max_drawdown(cumulative_returns)
    
    # Max drawdown from 1.2 to 0.8 = (0.8 - 1.2) / 1.2 = -0.333...
    expected_mdd = (0.8 - 1.2) / 1.2
    
    assert np.isclose(mdd, expected_mdd, rtol=1e-2)
    assert mdd < 0  # Drawdown is negative


def test_calculate_max_drawdown_no_drawdown():
    """Test max drawdown with monotonically increasing returns."""
    cumulative_returns = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
    
    mdd = calculate_max_drawdown(cumulative_returns)
    
    assert mdd == 0.0


def test_evaluate_performance():
    """Test full performance evaluation."""
    # Create synthetic results
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1min")
    
    np.random.seed(42)
    actual = np.random.randn(100) * 0.01
    predicted = actual + np.random.randn(100) * 0.005
    
    df = pd.DataFrame({
        "actual": actual,
        "predicted": predicted,
    }, index=dates)
    
    report = evaluate_performance(df, actual_col="actual", pred_col="predicted")
    
    # Check all metrics are present
    assert "rmse" in report
    assert "mae" in report
    assert "directional_accuracy" in report
    assert "sharpe_ratio" in report
    assert "max_drawdown" in report
    assert "total_return" in report
    
    # Check all metrics are numeric
    for key, value in report.items():
        assert isinstance(value, (int, float))
        assert not np.isnan(value)
