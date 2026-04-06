"""
tests/test_features.py

Unit tests for feature engineering modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.technical import compute_technical_features
from src.features.statistical import compute_statistical_features
from src.features.volume import compute_volume_features
from src.features.selector import FeatureSelector


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 1000
    
    dates = pd.date_range("2023-01-01 09:30", periods=n, freq="1min", tz="UTC")
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.abs(np.random.randn(n) * 1000000 + 5000000)
    
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    
    df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    df["session_minute"] = np.arange(n) % 390
    df["session_start"] = (df["session_minute"] == 0)
    
    return df


class TestTechnicalFeatures:
    """Test technical indicator computation."""

    def test_compute_technical_features(self, sample_ohlcv_data):
        """Test that technical features are computed."""
        features = compute_technical_features(sample_ohlcv_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv_data)
        
        # Check some expected columns
        expected_cols = ["sma_5", "sma_20", "rsi_14", "macd", "bb_upper", "atr_14"]
        for col in expected_cols:
            assert col in features.columns

    def test_no_nan_in_output(self, sample_ohlcv_data):
        """Test that output has no NaN values."""
        features = compute_technical_features(sample_ohlcv_data)
        
        # Should be filled with 0 or forward-filled
        assert not features.isna().any().any()

    def test_rsi_bounds(self, sample_ohlcv_data):
        """Test that RSI is bounded [0, 100]."""
        features = compute_technical_features(sample_ohlcv_data)
        
        assert (features["rsi_14"] >= 0).all()
        assert (features["rsi_14"] <= 100).all()


class TestStatisticalFeatures:
    """Test statistical feature computation."""

    def test_compute_statistical_features(self, sample_ohlcv_data):
        """Test that statistical features are computed."""
        features = compute_statistical_features(sample_ohlcv_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv_data)
        
        # Check expected columns
        expected_cols = ["ret_mean_10", "ret_var_20", "ret_skew_20", "rv_10"]
        for col in expected_cols:
            assert col in features.columns

    def test_no_nan_in_output(self, sample_ohlcv_data):
        """Test that output has no NaN values."""
        features = compute_statistical_features(sample_ohlcv_data)
        assert not features.isna().any().any()


class TestVolumeFeatures:
    """Test volume feature computation."""

    def test_compute_volume_features(self, sample_ohlcv_data):
        """Test that volume features are computed."""
        features = compute_volume_features(sample_ohlcv_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv_data)
        
        # Check expected columns
        expected_cols = ["vwap", "rvol_20", "obv", "cmf_20", "tod_sin", "dow_cos"]
        for col in expected_cols:
            assert col in features.columns

    def test_cyclical_encoding_bounds(self, sample_ohlcv_data):
        """Test that cyclical encodings are bounded [-1, 1]."""
        features = compute_volume_features(sample_ohlcv_data)
        
        assert (features["tod_sin"] >= -1).all()
        assert (features["tod_sin"] <= 1).all()
        assert (features["tod_cos"] >= -1).all()
        assert (features["tod_cos"] <= 1).all()


class TestFeatureSelector:
    """Test feature selection."""

    def test_selector_initialization(self):
        """Test selector can be initialized."""
        selector = FeatureSelector(
            variance_threshold=1e-6,
            correlation_threshold=0.98,
            importance_cumulative=0.70,
        )
        
        assert selector.variance_threshold == 1e-6
        assert selector.correlation_threshold == 0.98
        assert selector.importance_cumulative == 0.70

    def test_variance_filtering(self):
        """Test that low-variance features are removed."""
        selector = FeatureSelector(variance_threshold=0.01)
        
        # Create data with one constant feature
        X = np.random.randn(100, 10)
        X[:, 5] = 1.0  # Constant feature
        y = np.random.randn(100)
        
        feature_names = [f"feat_{i}" for i in range(10)]
        
        selector.fit(X, y, feature_names)
        
        # Constant feature should be removed
        assert "feat_5" not in selector.selected_features_

    def test_fit_transform(self):
        """Test fit_transform returns selected features."""
        selector = FeatureSelector(importance_cumulative=0.70)
        
        X = np.random.randn(200, 20)
        y = np.random.randn(200)
        feature_names = [f"feat_{i}" for i in range(20)]
        
        X_selected, selected_names = selector.fit_transform(X, y, feature_names)
        
        assert X_selected.shape[0] == 200
        assert X_selected.shape[1] < 20  # Some features should be removed
        assert len(selected_names) == X_selected.shape[1]

    def test_transform_consistency(self):
        """Test that transform uses fitted selection."""
        selector = FeatureSelector()
        
        X_train = np.random.randn(100, 15)
        y_train = np.random.randn(100)
        feature_names = [f"feat_{i}" for i in range(15)]
        
        X_train_sel, _ = selector.fit_transform(X_train, y_train, feature_names)
        
        X_test = np.random.randn(20, 15)
        X_test_sel, _ = selector.transform(X_test, feature_names)
        
        # Test set should have same number of features as train
        assert X_test_sel.shape[1] == X_train_sel.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
