#!/usr/bin/env python3
"""
Unified XGBoost Training Pipeline

Production pipeline for training XGBoost models on TRD-compliant features.
Integrates with the unified feature pipeline to consume pre-processed features.

TRD Compliance:
- Consumes features from unified_feature_pipeline.py
- No internal feature engineering
- No normalization (expects pre-normalized features)
- Temporal validation split
- Early stopping on validation RMSE

Usage:
    python pipelines/unified_train_xgboost.py --ticker AAPL --config config/default_config.yaml

Author: System Architect
Version: 1.0.0 UNIFIED
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import XGBoostTrainer, build_xgboost_lag_features
from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_processed_features(ticker: str, output_dir: Path) -> dict:
    """
    Load processed features from unified feature pipeline.
    
    Args:
        ticker: Target ticker symbol
        output_dir: Directory containing processed features
    
    Returns:
        Dictionary with train/val/test splits
    
    Raises:
        FileNotFoundError: If feature files not found
    """
    feature_dir = output_dir / ticker
    
    if not feature_dir.exists():
        raise FileNotFoundError(
            f"Feature directory not found: {feature_dir}\n"
            f"Run unified_feature_pipeline.py first."
        )
    
    logger.info(f"Loading features from {feature_dir}")
    
    # Load arrays
    data = {}
    
    for split in ["train", "val", "test"]:
        X_path = feature_dir / f"X_{split}.npy"
        y_path = feature_dir / f"y_{split}.npy"
        
        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Missing {split} data: {X_path} or {y_path}"
            )
        
        data[f"X_{split}"] = np.load(X_path)
        data[f"y_{split}"] = np.load(y_path)
        
        logger.info(
            f"Loaded {split}: X={data[f'X_{split}'].shape}, "
            f"y={data[f'y_{split}'].shape}"
        )
    
    # Load metadata
    metadata_path = feature_dir / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            data["metadata"] = yaml.safe_load(f)
    
    return data


def train_xgboost(
    ticker: str,
    config_path: Path,
    feature_dir: Path,
    output_dir: Path,
) -> None:
    """
    Train XGBoost model on pre-processed features.
    
    Pipeline:
    1. Load processed features (from unified_feature_pipeline.py)
    2. Build lag-based representation (for XGBoost)
    3. Train XGBoost with early stopping
    4. Evaluate on test set
    5. Save model and metrics
    
    Args:
        ticker: Target ticker symbol
        config_path: Path to config YAML
        feature_dir: Directory with processed features
        output_dir: Directory for model outputs
    """
    logger.info("=" * 80)
    logger.info("UNIFIED XGBOOST TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Ticker: {ticker}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Feature dir: {feature_dir}")
    logger.info(f"Output dir: {output_dir}")
    
    # Load configuration
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading Configuration")
    logger.info("=" * 80)
    config = load_config(config_path)
    xgb_config = config.xgboost
    
    # Convert to dict for model initialization
    xgb_params = {
        "objective": xgb_config.objective,
        "n_estimators": xgb_config.n_estimators,
        "max_depth": xgb_config.max_depth,
        "learning_rate": xgb_config.learning_rate,
        "subsample": xgb_config.subsample,
        "colsample_bytree": xgb_config.colsample_bytree,
        "min_child_weight": xgb_config.min_child_weight,
        "gamma": xgb_config.gamma,
        "reg_alpha": xgb_config.reg_alpha,
        "reg_lambda": xgb_config.reg_lambda,
        "early_stopping_rounds": xgb_config.early_stopping_rounds,
        "tree_method": xgb_config.tree_method,
        "max_bin": xgb_config.max_bin,
    }
    
    lookback = xgb_config.lookback
    seed = xgb_config.random_seed
    
    logger.info(f"XGBoost Configuration: {xgb_params}")
    logger.info(f"Lookback: {lookback}, Seed: {seed}")
    
    # Load processed features
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Loading Processed Features")
    logger.info("=" * 80)
    data = load_processed_features(ticker, feature_dir)
    
    # Extract splits (tabular format)
    X_train_tab = data["X_train"]
    y_train_tab = data["y_train"]
    X_val_tab = data["X_val"]
    y_val_tab = data["y_val"]
    X_test_tab = data["X_test"]
    y_test_tab = data["y_test"]
    
    logger.info("Loaded tabular features (pre-windowing):")
    logger.info(f"  Train: X={X_train_tab.shape}, y={y_train_tab.shape}")
    logger.info(f"  Val:   X={X_val_tab.shape}, y={y_val_tab.shape}")
    logger.info(f"  Test:  X={X_test_tab.shape}, y={y_test_tab.shape}")
    
    # Build lag-based features for XGBoost
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Building Lag-Based Features")
    logger.info("=" * 80)
    logger.info(f"Creating {lookback} lag features per timestep...")
    
    X_train, y_train = build_xgboost_lag_features(
        X_train_tab, y_train_tab, lookback=lookback
    )
    X_val, y_val = build_xgboost_lag_features(
        X_val_tab, y_val_tab, lookback=lookback
    )
    X_test, y_test = build_xgboost_lag_features(
        X_test_tab, y_test_tab, lookback=lookback
    )
    
    logger.info("Lag-based features created:")
    logger.info(f"  Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    logger.info(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    
    # Initialize trainer
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Training XGBoost Model")
    logger.info("=" * 80)
    trainer = XGBoostTrainer(xgb_params, seed=seed)
    
    # Train
    model, history = trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Evaluating on Test Set")
    logger.info("=" * 80)
    metrics = trainer.evaluate(model, X_test, y_test)
    
    # Save model and results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Saving Model and Results")
    logger.info("=" * 80)
    model_dir = output_dir / ticker
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "xgboost_model.json"
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = model_dir / "xgboost_metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save training history
    history_path = model_dir / "xgboost_history.yaml"
    with open(history_path, "w") as f:
        # Convert numpy arrays to lists for YAML serialization
        history_serializable = {}
        for key, value in history.items():
            if isinstance(value, dict):
                history_serializable[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                history_serializable[key] = value
        yaml.dump(history_serializable, f, default_flow_style=False)
    logger.info(f"History saved to {history_path}")
    
    # Save feature importance
    if model.feature_importance:
        importance_path = model_dir / "feature_importance.yaml"
        # Sort by importance
        sorted_importance = dict(
            sorted(
                model.feature_importance.items(),
                key=lambda x: -x[1]
            )
        )
        with open(importance_path, "w") as f:
            yaml.dump(sorted_importance, f, default_flow_style=False)
        logger.info(f"Feature importance saved to {importance_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost model on unified features"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker symbol (e.g., AAPL)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "features_unified",
        help="Directory with processed features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "models" / "xgboost",
        help="Directory for model outputs",
    )
    
    args = parser.parse_args()
    
    try:
        train_xgboost(
            ticker=args.ticker,
            config_path=args.config,
            feature_dir=args.feature_dir,
            output_dir=args.output_dir,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
