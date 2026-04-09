"""
Integration test: FeaturePipeline with SymbolUniverseBuilder.

Tests the full flow:
  1. Load config
  2. Build universe for target
  3. Fit pipeline on training data
  4. Transform validation data
  5. Verify feature matrix shape and peer consistency
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.pipeline import FeaturePipeline
from src.features.universe_builder import SymbolUniverseBuilder


@pytest.fixture
def sample_config(tmp_path):
    """Create minimal config for integration test."""
    config_content = """
market_context: [SPY, QQQ]
sector_etfs:
  technology:
    etf: XLK
    stocks: [AAPL, MSFT]
market_internals: [UVXY, GLD]
peer_selection:
  max_peers: 2
  method: "pearson"
  min_correlation: 0.3
  lookback_days: 252
prediction_targets: [AAPL]
context_only: [SPY, QQQ, XLK, UVXY, GLD]
"""
    config_path = tmp_path / "symbol_universe.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_data():
    """Create realistic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01 09:30", periods=2000, freq="1min")
    
    def make_df(ticker):
        returns = np.random.randn(len(dates)) * 0.001
        close = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            "open": close * (1 + np.random.randn(len(dates)) * 0.0005),
            "high": close * (1 + np.abs(np.random.randn(len(dates))) * 0.001),
            "low": close * (1 - np.abs(np.random.randn(len(dates))) * 0.001),
            "close": close,
            "volume": np.random.randint(1000, 10000, len(dates)),
            "log_return": returns,
        }, index=dates)
        return df
    
    return {
        "AAPL": make_df("AAPL"),
        "MSFT": make_df("MSFT"),
        "GOOGL": make_df("GOOGL"),
        "SPY": make_df("SPY"),
        "QQQ": make_df("QQQ"),
        "XLK": make_df("XLK"),
        "UVXY": make_df("UVXY"),
        "GLD": make_df("GLD"),
    }


def test_full_pipeline_with_universe_builder(sample_config, sample_data):
    """Test complete feature pipeline with universe builder."""
    # Initialize builder
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    # Split data
    dfs_train = {k: v.iloc[:1000] for k, v in sample_data.items()}
    dfs_val = {k: v.iloc[1000:1500] for k, v in sample_data.items()}
    dfs_test = {k: v.iloc[1500:] for k, v in sample_data.items()}
    
    # Initialize pipeline
    pipeline = FeaturePipeline(
        target_ticker="AAPL",
        universe_builder=builder,
    )
    
    # Fit on training data
    X_train, y_train, feature_names = pipeline.fit_transform(dfs_train)
    
    # Check output shapes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[1] == len(feature_names)
    assert len(feature_names) > 0
    
    # Transform validation data
    X_val, y_val = pipeline.transform(dfs_val)
    
    # Check consistency
    assert X_val.shape[1] == X_train.shape[1]  # same number of features
    
    # Transform test data
    X_test, y_test = pipeline.transform(dfs_test)
    assert X_test.shape[1] == X_train.shape[1]
    
    # Verify peers were selected and frozen
    peers = builder.get_fitted_peers("AAPL")
    assert len(peers) <= 2
    assert all(p in ["MSFT", "GOOGL"] for p in peers)


def test_pipeline_legacy_mode(sample_data):
    """Test pipeline works in legacy mode without universe builder."""
    # Split data
    dfs_train = {k: v.iloc[:1000] for k, v in sample_data.items()}
    dfs_val = {k: v.iloc[1000:1500] for k, v in sample_data.items()}
    
    # Initialize pipeline without universe builder (legacy mode)
    pipeline = FeaturePipeline(
        target_ticker="AAPL",
        universe_tickers=list(dfs_train.keys()),
    )
    
    # Fit on training data
    X_train, y_train, feature_names = pipeline.fit_transform(dfs_train)
    
    # Check output shapes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[1] == len(feature_names)
    
    # Transform validation data
    X_val, y_val = pipeline.transform(dfs_val)
    
    # Check consistency
    assert X_val.shape[1] == X_train.shape[1]


def test_pipeline_requires_universe_or_tickers():
    """Test that pipeline requires either universe_builder or universe_tickers."""
    with pytest.raises(ValueError, match="Must provide either universe_builder or universe_tickers"):
        FeaturePipeline(target_ticker="AAPL")


def test_pipeline_transform_before_fit_raises_error(sample_config, sample_data):
    """Test that transform before fit raises error."""
    builder = SymbolUniverseBuilder.from_config(sample_config)
    pipeline = FeaturePipeline(
        target_ticker="AAPL",
        universe_builder=builder,
    )
    
    dfs_val = {k: v.iloc[1000:1500] for k, v in sample_data.items()}
    
    with pytest.raises(RuntimeError, match="Pipeline has not been fitted"):
        pipeline.transform(dfs_val)


def test_universe_builder_fit_false_before_fit_raises_error(sample_config, sample_data):
    """Test that get_universe with fit=False before fit=True raises error."""
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    dfs_val = {k: v.iloc[1000:1500] for k, v in sample_data.items()}
    
    with pytest.raises(RuntimeError, match="Peers for AAPL not fitted"):
        builder.get_universe("AAPL", dfs_val, fit=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
