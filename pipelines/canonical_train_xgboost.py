#!/usr/bin/env python3
"""
Canonical XGBoost Training Script

Implements static (single-fit) XGBoost training as defined in FINAL_PLAN.md Section 4.3.

CRITICAL RULES:
- Train EXACTLY ONCE on 70% training data
- Use 10% validation for early stopping
- Fixed hyperparameters (or grid-searched once)
- Lag-based feature representation (NOT flattened sequences)
- Model is FROZEN after training
- NO retraining during walk-forward

Usage:
    python pipelines/canonical_train_xgboost.py \\
        --data-path data/processed/features_unified/AAPL \\
        --config config/canonical_config.yaml \\
        --output-dir results/canonical/models/xgboost

Author: System Architect
Version: CANONICAL 1.0
Source: FINAL_PLAN.md Section 4.3
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

from src.models import XGBoostModel, XGBoostTrainer, build_xgboost_lag_features
from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_preprocessed_data(data_path: Path) -> dict:
    """
    Load preprocessed features from canonical feature pipeline.
    
    Args:
        data_path: Directory containing X_train.npy, y_train.npy, etc.
    
    Returns:
        Dictionary with train/val/test splits
    """
    logger.info(f"Loading preprocessed data from {data_path}")
    
    data = {}
    for split in ["train", "val", "test"]:
        X_path = data_path / f"X_{split}.npy"
        y_path = data_path / f"y_{split}.npy"
        
        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Missing {split} data: {X_path} or {y_path}\n"
                f"Run canonical feature pipeline first."
            )
        
        data[f"X_{split}"] = np.load(X_path)
        data[f"y_{split}"] = np.load(y_path)
        
        logger.info(
            f"  {split}: X={data[f'X_{split}'].shape}, y={data[f'y_{split}'].shape}"
        )
    
    return data


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    output_dir: Path,
) -> XGBoostModel:
    """
    Train XGBoost with fixed hyperparameters.
    
    FINAL_PLAN.md Section 4.3: XGBoost Training Phase
    
    CRITICAL: This function executes EXACTLY ONCE. The model is then FROZEN.
    
    Args:
        X_train: Training features (N, F) tabular
        y_train: Training targets (N,)
        X_val: Validation features (M, F) tabular
        y_val: Validation targets (M,)
        config: Configuration dict
        output_dir: Directory to save model
    
    Returns:
        Trained and FROZEN XGBoostModel
    """
    logger.info("=" * 80)
    logger.info("XGBOOST TRAINING (FINAL_PLAN.md Section 4.3)")
    logger.info("=" * 80)
    logger.info("CRITICAL: Model will be trained EXACTLY ONCE and then FROZEN")
    logger.info("=" * 80)
    
    # Extract config
    xgb_config = config.xgboost
    lookback = xgb_config.lookback
    seed = xgb_config.random_state
    
    logger.info(f"Random seed: {seed}")
    logger.info(f"Lookback: {lookback} (lag-based features)")
    
    # Build lag-based features
    logger.info("=" * 80)
    logger.info("BUILDING LAG-BASED FEATURES")
    logger.info("=" * 80)
    logger.info("Using LAG-BASED representation (NOT flattened sequences)")
    logger.info(f"Each sample: [X[t], X[t-1], ..., X[t-{lookback}]]")
    
    X_train_lag = build_xgboost_lag_features(X_train, lookback=lookback)
    X_val_lag = build_xgboost_lag_features(X_val, lookback=lookback)
    
    # Align targets (remove first lookback samples)
    y_train_lag = y_train[lookback:].astype(np.float32)
    y_val_lag = y_val[lookback:].astype(np.float32)
    
    logger.info(f"Lag-based feature shapes:")
    logger.info(f"  X_train: {X_train_lag.shape}, y_train: {y_train_lag.shape}")
    logger.info(f"  X_val:   {X_val_lag.shape}, y_val:   {y_val_lag.shape}")
    logger.info(f"  Features per sample: {X_train_lag.shape[1]}")
    
    # Create XGBoost configuration
    logger.info("=" * 80)
    logger.info("XGBOOST CONFIGURATION (FIXED)")
    logger.info("=" * 80)
    logger.info(f"  Objective:        {xgb_config.objective}")
    logger.info(f"  Max depth:        {xgb_config.max_depth}")
    logger.info(f"  Learning rate:    {xgb_config.learning_rate}")
    logger.info(f"  N estimators:     {xgb_config.n_estimators}")
    logger.info(f"  Subsample:        {xgb_config.subsample}")
    logger.info(f"  Colsample bytree: {xgb_config.colsample_bytree}")
    logger.info(f"  Early stopping:   {xgb_config.early_stopping_rounds} rounds")
    
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
        "random_seed": seed,
    }
    
    # Create and train model
    logger.info("=" * 80)
    logger.info("TRAINING (SINGLE FIT - NO RETRAINING)")
    logger.info("=" * 80)
    
    trainer = XGBoostTrainer(xgb_params, seed=seed)
    model, history = trainer.train(X_train_lag, y_train_lag, X_val_lag, y_val_lag)
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE - MODEL NOW FROZEN")
    logger.info("=" * 80)
    logger.info("⚠️  This model will NEVER be retrained")
    logger.info("⚠️  Walk-forward evaluation will use THIS frozen model")
    logger.info("=" * 80)
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgboost_model.json"
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save training history
    history_path = output_dir / "training_history.yaml"
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
    logger.info(f"Training history saved to {history_path}")
    
    # Save model configuration
    config_dict = {
        "xgb_params": xgb_params,
        "lookback": lookback,
        "feature_type": "lag_based",
    }
    config_path = output_dir / "model_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    logger.info(f"Model config saved to {config_path}")
    
    # Save feature importance
    if model.feature_importance:
        importance_path = output_dir / "feature_importance.yaml"
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
    
    # Save metadata
    metadata = {
        "model_type": "xgboost",
        "protocol": "CANONICAL_1.0",
        "source": "FINAL_PLAN.md Section 4.3",
        "training_samples": len(X_train_lag),
        "validation_samples": len(X_val_lag),
        "features_original": X_train.shape[1],
        "features_lag_based": X_train_lag.shape[1],
        "lookback": lookback,
        "hyperparameters": xgb_params,
        "training_complete": True,
        "frozen": True,
        "retraining_allowed": False,
    }
    
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    logger.info(f"Metadata saved to {metadata_path}")
    
    return model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost (static, single-fit)"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to preprocessed features directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "canonical_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "canonical" / "models" / "xgboost",
        help="Directory to save trained model",
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Load preprocessed data
        data = load_preprocessed_data(args.data_path)
        
        # Train model (ONCE)
        model = train_xgboost(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
            config=config,
            output_dir=args.output_dir,
        )
        
        logger.info("=" * 80)
        logger.info("XGBOOST TRAINING SUCCESSFUL")
        logger.info("=" * 80)
        logger.info("✓ Model trained and frozen")
        logger.info("✓ Lag-based features created")
        logger.info("✓ Model saved to disk")
        logger.info("✓ Ready for walk-forward evaluation")
        logger.info("=" * 80)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
