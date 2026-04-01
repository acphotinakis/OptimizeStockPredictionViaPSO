"""
Unit tests for data splitting module.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.split import time_series_split


def test_time_series_split_basic(mock_config):
    """Test basic time series splitting."""
    # Create 5 years of data
    dates = pd.date_range(start="2021-01-01", end="2025-12-31", freq="1D")
    df = pd.DataFrame({
        "close": np.random.randn(len(dates)) + 100,
        "volume": np.random.randint(1000, 10000, len(dates)),
    }, index=dates)
    
    train_df, val_df, test_df = time_series_split(df, mock_config)
    
    # Check that all splits are non-empty
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    
    # Check chronological ordering
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()


def test_time_series_split_no_overlap(mock_config):
    """Test that splits have no temporal overlap."""
    dates = pd.date_range(start="2021-01-01", end="2025-12-31", freq="1D")
    df = pd.DataFrame({
        "close": np.random.randn(len(dates)) + 100,
    }, index=dates)
    
    train_df, val_df, test_df = time_series_split(df, mock_config)
    
    # Check no index overlap
    train_indices = set(train_df.index)
    val_indices = set(val_df.index)
    test_indices = set(test_df.index)
    
    assert len(train_indices & val_indices) == 0
    assert len(val_indices & test_indices) == 0
    assert len(train_indices & test_indices) == 0


def test_time_series_split_preserves_data(mock_config):
    """Test that splitting preserves all data (no loss)."""
    dates = pd.date_range(start="2021-01-01", end="2025-12-31", freq="1D")
    df = pd.DataFrame({
        "close": np.random.randn(len(dates)) + 100,
    }, index=dates)
    
    train_df, val_df, test_df = time_series_split(df, mock_config)
    
    # Total samples should be close to original (minus a few boundary samples)
    total_split_samples = len(train_df) + len(val_df) + len(test_df)
    
    # Allow for 2 samples lost due to boundary exclusion (iloc[:-1] in split logic)
    assert total_split_samples >= len(df) - 2


def test_time_series_split_insufficient_data(mock_config):
    """Test that split raises error with insufficient data."""
    # Create only 1 year of data (insufficient for 3+1+1 split)
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="1D")
    df = pd.DataFrame({
        "close": np.random.randn(len(dates)) + 100,
    }, index=dates)
    
    with pytest.raises(ValueError, match="insufficient"):
        time_series_split(df, mock_config)


def test_time_series_split_sorted(mock_config):
    """Test that split handles unsorted data."""
    dates = pd.date_range(start="2021-01-01", end="2025-12-31", freq="1D")
    df = pd.DataFrame({
        "close": np.random.randn(len(dates)) + 100,
    }, index=dates)
    
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=42)
    
    # Split should still work (it sorts internally)
    train_df, val_df, test_df = time_series_split(df_shuffled, mock_config)
    
    # All splits should be sorted
    assert train_df.index.is_monotonic_increasing
    assert val_df.index.is_monotonic_increasing
    assert test_df.index.is_monotonic_increasing
