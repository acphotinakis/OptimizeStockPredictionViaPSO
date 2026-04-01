"""
Unit tests for feature selection module.
"""
import pytest
import pandas as pd
import numpy as np
from src.features.selection import (
    correlation_filter,
    xgboost_importance_filter,
    run_selection_pipeline,
)


def test_correlation_filter_removes_correlated(sample_features_df):
    """Test that correlation filter removes highly correlated features."""
    # feature1 and feature2 are highly correlated (r > 0.95)
    result = correlation_filter(sample_features_df, threshold=0.95)
    
    # Should have removed at least one feature
    assert len(result.columns) < len(sample_features_df.columns)


def test_correlation_filter_preserves_independent():
    """Test that correlation filter preserves independent features."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1min")
    
    np.random.seed(42)
    # Create completely independent features
    data = {
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
    }
    
    df = pd.DataFrame(data, index=dates)
    
    result = correlation_filter(df, threshold=0.95)
    
    # Should keep all features
    assert len(result.columns) == len(df.columns)


def test_xgboost_importance_filter():
    """Test XGBoost importance-based feature selection."""
    dates = pd.date_range(start="2023-01-01", periods=200, freq="1min")
    
    np.random.seed(42)
    # Create features with varying importance
    feature1 = np.random.randn(200)
    feature2 = np.random.randn(200)
    feature3 = np.random.randn(200)
    target = 2 * feature1 + 0.1 * feature2 + np.random.randn(200) * 0.1  # feature1 is most important
    
    df = pd.DataFrame({
        "feature1": feature1,
        "feature2": feature2,
        "feature3": feature3,
    }, index=dates)
    
    result = xgboost_importance_filter(df, pd.Series(target), top_n=2)
    
    # Should return top 2 features
    assert len(result.columns) == 2
    # feature1 should be selected (most important)
    assert "feature1" in result.columns


def test_run_selection_pipeline_with_preselected_features(sample_features_df, mock_config):
    """Test that pipeline applies pre-selected features (for val/test sets)."""
    selected_features = ["feature1", "feature3"]
    
    result = run_selection_pipeline(
        sample_features_df,
        mock_config.features.selection,
        selected_features=selected_features,
    )
    
    # Should only have the selected features plus target
    assert "feature1" in result.columns
    assert "feature3" in result.columns
    assert "target" in result.columns
    assert len(result.columns) == 3


def test_run_selection_pipeline_training_mode(sample_features_df, mock_config):
    """Test feature selection on training data."""
    mock_config.features.selection.use_xgboost_importance = False
    
    result = run_selection_pipeline(
        sample_features_df,
        mock_config.features.selection,
        selected_features=None,
    )
    
    # Should have performed correlation filtering
    assert len(result.columns) <= len(sample_features_df.columns)


def test_run_selection_pipeline_prevents_leakage():
    """Test that selection pipeline can be used to prevent data leakage."""
    dates = pd.date_range(start="2023-01-01", periods=300, freq="1min")
    
    np.random.seed(42)
    data = {
        "feature1": np.random.randn(300),
        "feature2": np.random.randn(300),
        "feature3": np.random.randn(300),
        "target": np.random.randn(300),
    }
    
    train_df = pd.DataFrame(data, index=dates[:200])
    val_df = pd.DataFrame(data, index=dates[200:])
    
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "pearson_threshold": 0.95,
        "use_xgboost_importance": False,
    })
    
    # Select features on training data only
    train_selected = run_selection_pipeline(train_df, cfg, selected_features=None)
    selected_feature_names = [col for col in train_selected.columns if col != "target"]
    
    # Apply same features to validation set
    val_selected = run_selection_pipeline(val_df, cfg, selected_features=selected_feature_names)
    
    # Validation should have same features as training (no leakage)
    train_features = [col for col in train_selected.columns if col != "target"]
    val_features = [col for col in val_selected.columns if col != "target"]
    
    assert train_features == val_features
