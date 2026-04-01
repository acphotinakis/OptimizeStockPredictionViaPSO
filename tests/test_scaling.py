"""
Unit tests for scaling module.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from src.features.scaling import (
    fit_scaler,
    transform_data,
    inverse_transform,
    validate_inverse_transform,
    save_scaler,
    load_scaler,
)


def test_fit_scaler(sample_features_df):
    """Test scaler fitting."""
    features = sample_features_df.drop(columns=["target"])
    scaler = fit_scaler(features)
    
    assert scaler is not None
    assert hasattr(scaler, "data_min_")
    assert hasattr(scaler, "data_max_")
    assert scaler.feature_range == (-1, 1)


def test_transform_data(sample_features_df):
    """Test data transformation."""
    features = sample_features_df.drop(columns=["target"])
    scaler = fit_scaler(features)
    transformed = transform_data(features, scaler)
    
    # Check shape preservation
    assert transformed.shape == features.shape
    
    # Check that values are in range [-1, 1]
    assert transformed.min().min() >= -1.0
    assert transformed.max().max() <= 1.0
    
    # Check index and columns preserved
    pd.testing.assert_index_equal(transformed.index, features.index)
    assert list(transformed.columns) == list(features.columns)


def test_inverse_transform(sample_features_df):
    """Test inverse transformation."""
    features = sample_features_df.drop(columns=["target"])
    scaler = fit_scaler(features)
    transformed = transform_data(features, scaler)
    recovered = inverse_transform(transformed, scaler)
    
    # Check shape preservation
    assert recovered.shape == features.shape
    
    # Check values are close to original (within floating point tolerance)
    np.testing.assert_allclose(
        recovered.values,
        features.values,
        rtol=1e-5,
        atol=1e-8,
    )


def test_validate_inverse_transform_pass(sample_features_df):
    """Test inverse transform validation passes on clean data."""
    features = sample_features_df.drop(columns=["target"])
    scaler = fit_scaler(features)
    
    is_valid = validate_inverse_transform(features, scaler, tolerance=1e-6)
    
    assert is_valid is True


def test_save_and_load_scaler(sample_features_df):
    """Test scaler persistence."""
    features = sample_features_df.drop(columns=["target"])
    scaler = fit_scaler(features)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_scaler.joblib"
        save_scaler(scaler, str(save_path))
        
        assert save_path.exists()
        
        loaded_scaler = load_scaler(str(save_path))
        
        # Verify loaded scaler produces same results
        original_transform = transform_data(features, scaler)
        loaded_transform = transform_data(features, loaded_scaler)
        
        pd.testing.assert_frame_equal(original_transform, loaded_transform)


def test_load_scaler_not_found():
    """Test that loading non-existent scaler raises error."""
    with pytest.raises(FileNotFoundError):
        load_scaler("/nonexistent/path/scaler.joblib")


def test_scaler_prevents_data_leakage(sample_features_df):
    """Test that scaler fit on train doesn't leak to validation."""
    features = sample_features_df.drop(columns=["target"])
    
    # Split into train and validation
    train = features.iloc[:150]
    val = features.iloc[150:]
    
    # Fit scaler ONLY on training data
    scaler = fit_scaler(train)
    
    # Transform both sets
    train_scaled = transform_data(train, scaler)
    val_scaled = transform_data(val, scaler)
    
    # Validation set statistics should NOT match training set statistics
    # (because scaler was fit only on training data)
    assert train_scaled.mean().mean() != val_scaled.mean().mean()
