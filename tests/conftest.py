"""
Pytest configuration and shared fixtures for the PSO-LSTM test suite.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from omegaconf import OmegaConf


@pytest.fixture
def sample_ohlcv_data():
    """
    Creates a synthetic OHLCV dataset for testing.
    
    Returns:
        pd.DataFrame: 100 rows of synthetic stock data with DatetimeIndex.
    """
    dates = pd.date_range(start="2023-01-01 09:30", periods=100, freq="1min")
    
    np.random.seed(42)
    base_price = 100.0
    
    data = {
        "open": base_price + np.random.randn(100) * 2,
        "high": base_price + np.random.randn(100) * 2 + 1,
        "low": base_price + np.random.randn(100) * 2 - 1,
        "close": base_price + np.random.randn(100) * 2,
        "volume": np.random.randint(1000, 10000, 100),
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Ensure high >= low, high >= open, high >= close
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    
    return df


@pytest.fixture
def sample_ohlcv_with_gaps():
    """
    Creates synthetic data with intentional gaps for testing cleaning logic.
    """
    dates = pd.date_range(start="2023-01-01 09:30", periods=100, freq="1min")
    
    np.random.seed(42)
    base_price = 100.0
    
    data = {
        "open": base_price + np.random.randn(100) * 2,
        "high": base_price + np.random.randn(100) * 2 + 1,
        "low": base_price + np.random.randn(100) * 2 - 1,
        "close": base_price + np.random.randn(100) * 2,
        "volume": np.random.randint(1000, 10000, 100),
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Introduce gaps
    df.loc[df.index[10:13], "close"] = np.nan  # 3-minute gap (short)
    df.loc[df.index[50:66], "close"] = np.nan  # 16-minute gap (long)
    
    return df


@pytest.fixture
def sample_ohlcv_with_outliers():
    """
    Creates synthetic data with extreme outliers for testing outlier handling.
    """
    dates = pd.date_range(start="2023-01-01 09:30", periods=100, freq="1min")
    
    np.random.seed(42)
    base_price = 100.0
    
    data = {
        "open": base_price + np.random.randn(100) * 2,
        "high": base_price + np.random.randn(100) * 2 + 1,
        "low": base_price + np.random.randn(100) * 2 - 1,
        "close": base_price + np.random.randn(100) * 2,
        "volume": np.random.randint(1000, 10000, 100),
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Introduce extreme outliers
    df.loc[df.index[25], "close"] = 500.0  # Extreme spike
    df.loc[df.index[75], "close"] = 10.0   # Extreme drop
    
    return df


@pytest.fixture
def mock_config():
    """
    Creates a minimal Hydra configuration for testing.
    """
    config_dict = {
        "seed": 42,
        "device": "cpu",
        "experiment_name": "test_run",
        "data": {
            "tickers": ["AAPL", "MSFT"],
            "timeframe": "1Min",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
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
        "model": {
            "architecture": {
                "input_size": 10,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "bidirectional": False,
            },
            "training": {
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10,
                "weight_decay": 1e-5,
                "early_stopping_patience": 3,
                "gradient_clip_value": 1.0,
            },
        },
        "optimization": {
            "swarm": {
                "particles": 5,
                "iterations": 3,
                "inertia_weight": 0.9,
                "cognitive_coeff": 1.5,
                "social_coeff": 2.0,
            },
            "improvements": {
                "mutation_factor": 0.1,
            },
        },
        "paths": {
            "log_dir": "logs/test",
            "model_save_path": "models/checkpoints/test",
            "results_dir": "reports/results/test",
            "data_storage": {
                "raw": "data/raw",
                "interim": "data/interim",
                "processed": "data/processed",
            },
        },
    }
    
    return OmegaConf.create(config_dict)


@pytest.fixture
def sample_features_df():
    """
    Creates a synthetic feature dataframe for testing feature selection and scaling.
    """
    dates = pd.date_range(start="2023-01-01", periods=200, freq="1min")
    
    np.random.seed(42)
    
    # Create correlated features
    feature1 = np.random.randn(200)
    feature2 = feature1 + np.random.randn(200) * 0.1  # Highly correlated with feature1
    feature3 = np.random.randn(200)  # Independent
    feature4 = np.random.randn(200)  # Independent
    
    data = {
        "feature1": feature1,
        "feature2": feature2,
        "feature3": feature3,
        "feature4": feature4,
        "target": np.random.randn(200),
    }
    
    return pd.DataFrame(data, index=dates)
