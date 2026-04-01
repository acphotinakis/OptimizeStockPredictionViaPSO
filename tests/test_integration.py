"""
Integration tests for end-to-end pipeline validation.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from omegaconf import OmegaConf

from src.data.data_validation import run_validation_suite
from src.data.data_cleaning import run_cleaning_pipeline
from src.features.wavelet_denoising import apply_denoising_pipeline
from src.features.build_features import calculate_indicators
from src.features.selection import run_selection_pipeline
from src.features.scaling import fit_scaler, transform_data, inverse_transform
from src.data.split import time_series_split


@pytest.fixture
def integration_data():
    """Create realistic multi-year dataset for integration testing."""
    # Create 5 years of daily data
    dates = pd.date_range(start="2021-01-01", end="2025-12-31", freq="1D")
    
    np.random.seed(42)
    base_price = 100.0
    
    # Generate realistic price movements
    returns = np.random.randn(len(dates)) * 0.02  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = {
        "open": prices + np.random.randn(len(dates)) * 0.5,
        "high": prices + np.abs(np.random.randn(len(dates))) * 1.0,
        "low": prices - np.abs(np.random.randn(len(dates))) * 1.0,
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, len(dates)),
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Ensure OHLC relationships
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    
    return df


@pytest.fixture
def integration_config():
    """Create configuration for integration testing."""
    return OmegaConf.create({
        "seed": 42,
        "data": {
            "tickers": ["TEST"],
            "timeframe": "1D",
            "start_date": "2021-01-01",
            "end_date": "2025-12-31",
            "splits": {
                "train_years": 3,
                "val_years": 1,
                "test_years": 1,
            },
            "validation": {
                "max_missing_pct": 0.05,
                "drop_duplicates": True,
                "handle_outliers": "winsorize",
            },
            "cleaning": {
                "short_gap_threshold": 5,
                "long_gap_threshold": 15,
            },
        },
        "features": {
            "indicators": {
                "rsi": {"window": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bollinger_bands": {"window": 20, "std_dev": 2},
                "ema": {"windows": [9, 21]},
                "sma": {"windows": [50]},
            },
            "statistics": {
                "rolling_mean": 30,
                "rolling_std": 30,
            },
            "denoising": {
                "enabled": True,
                "wavelet_type": "haar",
                "decomposition_level": 1,
            },
            "selection": {
                "pearson_threshold": 0.95,
                "use_xgboost_importance": False,
            },
        },
    })


def test_full_preprocessing_pipeline(integration_data, integration_config):
    """Test complete preprocessing pipeline from raw data to scaled features."""
    
    # 1. Validation
    val_report = run_validation_suite(integration_data, integration_config)
    assert val_report["is_fit_for_training"] is True
    
    # 2. Cleaning
    df_clean = run_cleaning_pipeline(integration_data, integration_config)
    assert len(df_clean) > 0
    assert df_clean.notna().all().all()
    
    # 3. Denoising
    df_denoised = apply_denoising_pipeline(df_clean, integration_config.features.denoising)
    assert "close_denoised" in df_denoised.columns
    
    # 4. Feature Engineering
    df_features = calculate_indicators(df_denoised, integration_config.features)
    assert len(df_features.columns) > len(df_clean.columns)
    
    # 5. Split (BEFORE feature selection to prevent leakage)
    train_df, val_df, test_df = time_series_split(df_features, integration_config)
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    
    # 6. Feature Selection (on training data only)
    train_selected = run_selection_pipeline(
        train_df, 
        integration_config.features.selection, 
        selected_features=None
    )
    
    # Get selected feature names
    selected_features = [col for col in train_selected.columns if col != "target"]
    
    # Apply to validation set
    val_selected = run_selection_pipeline(
        val_df,
        integration_config.features.selection,
        selected_features=selected_features
    )
    
    # Should have same features
    assert set(train_selected.columns) == set(val_selected.columns)
    
    # 7. Scaling (fit on training only)
    train_features = train_selected.drop(columns=["target"]) if "target" in train_selected.columns else train_selected
    scaler = fit_scaler(train_features)
    
    train_scaled = transform_data(train_features, scaler)
    
    # Verify scaled data is in correct range
    assert train_scaled.min().min() >= -1.0
    assert train_scaled.max().max() <= 1.0
    
    # 8. Inverse transform validation
    recovered = inverse_transform(train_scaled, scaler)
    
    # Should recover original values
    np.testing.assert_allclose(
        recovered.values,
        train_features.values,
        rtol=1e-5,
        atol=1e-8,
    )


def test_pipeline_preserves_chronological_order(integration_data, integration_config):
    """Test that pipeline maintains chronological ordering throughout."""
    
    df_clean = run_cleaning_pipeline(integration_data, integration_config)
    assert df_clean.index.is_monotonic_increasing
    
    df_denoised = apply_denoising_pipeline(df_clean, integration_config.features.denoising)
    assert df_denoised.index.is_monotonic_increasing
    
    df_features = calculate_indicators(df_denoised, integration_config.features)
    assert df_features.index.is_monotonic_increasing
    
    train_df, val_df, test_df = time_series_split(df_features, integration_config)
    assert train_df.index.is_monotonic_increasing
    assert val_df.index.is_monotonic_increasing
    assert test_df.index.is_monotonic_increasing


def test_pipeline_no_data_leakage(integration_data, integration_config):
    """Test that pipeline prevents data leakage between splits."""
    
    # Process through feature engineering
    df_clean = run_cleaning_pipeline(integration_data, integration_config)
    df_denoised = apply_denoising_pipeline(df_clean, integration_config.features.denoising)
    df_features = calculate_indicators(df_denoised, integration_config.features)
    
    # Split BEFORE feature selection
    train_df, val_df, test_df = time_series_split(df_features, integration_config)
    
    # Verify no temporal overlap
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()
    
    # Select features on training only
    train_selected = run_selection_pipeline(
        train_df,
        integration_config.features.selection,
        selected_features=None
    )
    
    selected_features = [col for col in train_selected.columns if col != "target"]
    
    # Fit scaler on training only
    train_features = train_selected.drop(columns=["target"]) if "target" in train_selected.columns else train_selected
    scaler = fit_scaler(train_features)
    
    # Apply to validation (should use training-derived parameters)
    val_selected = run_selection_pipeline(val_df, integration_config.features.selection, selected_features=selected_features)
    val_features = val_selected.drop(columns=["target"]) if "target" in val_selected.columns else val_selected
    val_scaled = transform_data(val_features, scaler)
    
    # Validation scaling should use training statistics
    # This is correct behavior - no leakage
    assert val_scaled.shape[1] == train_features.shape[1]
