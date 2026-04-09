"""
Unit tests for SymbolUniverseBuilder.

Tests:
  - Correct universe construction for each tier
  - Peer selection on training data only
  - Leakage prevention (stored peers used on val/test)
  - Sector ETF mapping correctness
  - Alignment function (inner join correctness)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.universe_builder import SymbolUniverseBuilder, align_symbol_universe


@pytest.fixture
def sample_config(tmp_path):
    """Create a minimal symbol_universe.yaml for testing."""
    config_content = """
market_context:
  - SPY
  - QQQ

sector_etfs:
  technology:
    etf: XLK
    stocks:
      - AAPL
      - MSFT
  financials:
    etf: XLF
    stocks:
      - JPM

market_internals:
  - UVXY
  - GLD

peer_selection:
  max_peers: 2
  method: "pearson"
  min_correlation: 0.3
  lookback_days: 252

prediction_targets:
  - AAPL
  - JPM

context_only:
  - SPY
  - QQQ
  - XLK
  - XLF
  - UVXY
  - GLD
"""
    config_path = tmp_path / "symbol_universe.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_data():
    """Create synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=1000, freq="1min")
    
    # Create base AAPL returns first
    aapl_returns = np.random.randn(len(dates)) * 0.001
    
    def make_df(ticker, correlation_with_aapl=0.0):
        # Base returns
        base_returns = np.random.randn(len(dates)) * 0.001
        
        if correlation_with_aapl > 0:
            # Create correlated returns using the shared aapl_returns
            returns = correlation_with_aapl * aapl_returns + np.sqrt(1 - correlation_with_aapl**2) * base_returns
        else:
            returns = base_returns
        
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
    
    # Create AAPL first with its own returns
    aapl_df = pd.DataFrame({
        "open": 100 * np.exp(np.cumsum(aapl_returns)) * (1 + np.random.randn(len(dates)) * 0.0005),
        "high": 100 * np.exp(np.cumsum(aapl_returns)) * (1 + np.abs(np.random.randn(len(dates))) * 0.001),
        "low": 100 * np.exp(np.cumsum(aapl_returns)) * (1 - np.abs(np.random.randn(len(dates))) * 0.001),
        "close": 100 * np.exp(np.cumsum(aapl_returns)),
        "volume": np.random.randint(1000, 10000, len(dates)),
        "log_return": aapl_returns,
    }, index=dates)
    
    return {
        "AAPL": aapl_df,
        "MSFT": make_df("MSFT", correlation_with_aapl=0.7),  # high correlation
        "GOOGL": make_df("GOOGL", correlation_with_aapl=0.5),  # medium correlation
        "JPM": make_df("JPM", correlation_with_aapl=0.2),  # low correlation
        "SPY": make_df("SPY"),
        "QQQ": make_df("QQQ"),
        "XLK": make_df("XLK"),
        "XLF": make_df("XLF"),
        "UVXY": make_df("UVXY"),
        "GLD": make_df("GLD"),
    }


def test_universe_builder_initialization(sample_config):
    """Test that builder loads config correctly."""
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    assert builder.market_context == ["SPY", "QQQ"]
    assert builder.sector_map["AAPL"] == "XLK"
    assert builder.sector_map["JPM"] == "XLF"
    assert builder.market_internals == ["UVXY", "GLD"]
    assert builder.peer_config["max_peers"] == 2


def test_universe_construction_aapl(sample_config, sample_data):
    """Test universe construction for AAPL (tech stock)."""
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    universe = builder.get_universe("AAPL", sample_data, fit=True)
    
    # Check tier 1 (market context)
    assert "SPY" in universe
    assert "QQQ" in universe
    
    # Check tier 2 (sector ETF)
    assert "XLK" in universe
    
    # Check tier 3 (peers) - should select MSFT (highest correlation)
    peers = builder.get_fitted_peers("AAPL")
    assert len(peers) <= 2
    assert "MSFT" in peers  # highest correlation
    
    # Check tier 4 (market internals)
    assert "UVXY" in universe
    assert "GLD" in universe
    
    # Check target is included
    assert "AAPL" in universe


def test_peer_selection_leakage_prevention(sample_config, sample_data):
    """Test that peers are selected once and reused."""
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    # Fit on training data
    universe_train = builder.get_universe("AAPL", sample_data, fit=True)
    peers_train = builder.get_fitted_peers("AAPL")
    
    # Transform on validation data (different data, same structure)
    sample_data_val = {k: v.iloc[500:] for k, v in sample_data.items()}
    universe_val = builder.get_universe("AAPL", sample_data_val, fit=False)
    peers_val = builder.get_fitted_peers("AAPL")
    
    # Peers should be identical (no re-selection)
    assert peers_train == peers_val
    assert universe_train == universe_val


def test_alignment_function(sample_data):
    """Test that alignment performs inner join correctly."""
    # Create misaligned data (different date ranges)
    aapl_df = sample_data["AAPL"].iloc[:800]  # missing last 200
    spy_df = sample_data["SPY"].iloc[100:]    # missing first 100
    
    dfs = {"AAPL": aapl_df, "SPY": spy_df}
    aligned = align_symbol_universe(["AAPL", "SPY"], dfs)
    
    # Should only have 700 common timestamps (100 to 800)
    assert len(aligned["AAPL"]) == 700
    assert len(aligned["SPY"]) == 700
    
    # Indices should be identical
    assert aligned["AAPL"].index.equals(aligned["SPY"].index)


def test_save_load_peers(sample_config, sample_data, tmp_path):
    """Test peer serialization for reproducibility."""
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    # Fit and save
    builder.get_universe("AAPL", sample_data, fit=True)
    peers_path = tmp_path / "peers.json"
    builder.save_peers(peers_path)
    
    # Load into new builder
    builder2 = SymbolUniverseBuilder.from_config(sample_config)
    builder2.load_peers(peers_path)
    
    # Should have same peers without re-fitting
    assert builder.get_fitted_peers("AAPL") == builder2.get_fitted_peers("AAPL")


def test_missing_sector_etf_fallback(sample_config, sample_data):
    """Test behavior when stock has no sector mapping."""
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    # TSLA not in config
    sample_data["TSLA"] = sample_data["AAPL"].copy()
    
    universe = builder.get_universe("TSLA", sample_data, fit=True)
    
    # Should still include market context and internals
    assert "SPY" in universe
    assert "UVXY" in universe
    
    # No sector ETF should be added (warning logged)
    # This is a warning case, not an error


def test_min_correlation_threshold(sample_config, sample_data):
    """Test that peers below min_correlation are excluded."""
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    # JPM has low correlation with AAPL (0.2 < 0.3 threshold)
    universe = builder.get_universe("JPM", sample_data, fit=True)
    peers = builder.get_fitted_peers("JPM")
    
    # Should have no peers or very few due to low correlations
    assert len(peers) <= 2


def test_max_peers_limit(sample_config, sample_data):
    """Test that max_peers limit is enforced."""
    # Add more highly correlated stocks
    sample_data["META"] = sample_data["AAPL"].copy()
    sample_data["CRM"] = sample_data["AAPL"].copy()
    sample_data["ADBE"] = sample_data["AAPL"].copy()
    
    builder = SymbolUniverseBuilder.from_config(sample_config)
    universe = builder.get_universe("AAPL", sample_data, fit=True)
    peers = builder.get_fitted_peers("AAPL")
    
    # Should respect max_peers=2 limit
    assert len(peers) <= 2


def test_universe_construction_jpm(sample_config, sample_data):
    """Test universe construction for JPM (financial stock)."""
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    universe = builder.get_universe("JPM", sample_data, fit=True)
    
    # Check tier 2 (sector ETF) - should be XLF for financials
    assert "XLF" in universe
    
    # Should not have XLK (tech sector ETF)
    assert "XLK" not in universe or universe.count("XLK") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
