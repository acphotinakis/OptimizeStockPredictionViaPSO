"""
Unit tests for PyTorch dataset module.
"""
import pytest
import torch
import pandas as pd
import numpy as np
from src.data.dataset import TimeSeriesDataset


def test_dataset_initialization(sample_features_df):
    """Test dataset initialization."""
    features = sample_features_df.drop(columns=["target"])
    target = sample_features_df["target"]
    
    dataset = TimeSeriesDataset(features, target, lookback=10)
    
    assert dataset.lookback == 10
    assert isinstance(dataset.X, torch.Tensor)
    assert isinstance(dataset.y, torch.Tensor)
    assert dataset.X.dtype == torch.float32
    assert dataset.y.dtype == torch.float32


def test_dataset_length(sample_features_df):
    """Test dataset length calculation."""
    features = sample_features_df.drop(columns=["target"])
    target = sample_features_df["target"]
    lookback = 10
    
    dataset = TimeSeriesDataset(features, target, lookback)
    
    # Length should be N - lookback
    expected_length = len(features) - lookback
    assert len(dataset) == expected_length


def test_dataset_getitem_shape(sample_features_df):
    """Test that __getitem__ returns correct shapes."""
    features = sample_features_df.drop(columns=["target"])
    target = sample_features_df["target"]
    lookback = 10
    
    dataset = TimeSeriesDataset(features, target, lookback)
    
    x_window, y_label = dataset[0]
    
    # X should be (lookback, num_features)
    assert x_window.shape == (lookback, features.shape[1])
    
    # Y should be a scalar
    assert y_label.shape == torch.Size([])


def test_dataset_sequence_correctness(sample_features_df):
    """Test that sequences are created correctly."""
    features = sample_features_df.drop(columns=["target"])
    target = sample_features_df["target"]
    lookback = 5
    
    dataset = TimeSeriesDataset(features, target, lookback)
    
    # Get first sequence
    x_window, y_label = dataset[0]
    
    # Verify the window corresponds to indices 0:5
    expected_x = torch.tensor(features.iloc[0:5].values, dtype=torch.float32)
    torch.testing.assert_close(x_window, expected_x)
    
    # Verify target corresponds to index 5
    expected_y = torch.tensor(target.iloc[5], dtype=torch.float32)
    torch.testing.assert_close(y_label, expected_y)


def test_dataset_last_sequence(sample_features_df):
    """Test that last sequence is correctly bounded."""
    features = sample_features_df.drop(columns=["target"])
    target = sample_features_df["target"]
    lookback = 10
    
    dataset = TimeSeriesDataset(features, target, lookback)
    
    # Get last sequence
    last_idx = len(dataset) - 1
    x_window, y_label = dataset[last_idx]
    
    # Last target should be the last value in the target array
    expected_y = torch.tensor(target.iloc[-1], dtype=torch.float32)
    torch.testing.assert_close(y_label, expected_y)


def test_dataset_with_numpy_arrays():
    """Test dataset works with numpy arrays."""
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    lookback = 10
    
    dataset = TimeSeriesDataset(X, y, lookback)
    
    assert len(dataset) == 90
    x_window, y_label = dataset[0]
    assert x_window.shape == (10, 5)
