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
    python pipelines/train_xgboost.py \
        --data-path data/processed/features_unified/AAPL \
        --config config/canonical_config.yaml \
        --output-dir results/canonical/models/xgboost

Author: System Architect
Version: 2.0.0 REFACTORED
Source: FINAL_PLAN.md Section 4.3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import XGBoostModel, XGBoostTrainer
from src.data.windowing import build_xgboost_lag_features
from src.utils.config_loader import Config, load_config
from src.utils.logger import LogFileMode, setup_logger

# Configure logging

logger = logging.getLogger(__name__)

MODEL_TYPE = "xgboost"


def load_feature_data(data_path: Path) -> dict:
    """
    Load preprocessed features from canonical feature pipeline with full diagnostics.

    Args:
        data_path: Directory containing X_train.npy, y_train.npy, etc.
        config: Optional config dict for dataset/version metadata.

    Returns:
        Dictionary with train/val/test splits.
    """
    logger.info(f"Resolving dataset path: {data_path.resolve()}")

    splits = ["train", "val", "test"]
    data = {}

    total_samples = 0
    feature_dim = None

    logger.info(f"Expected splits: {splits}")

    for split in splits:
        X_path = data_path / f"X_{split}.npy"
        y_path = data_path / f"y_{split}.npy"

        # ----------------------------
        # File existence + metadata
        # ----------------------------
        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Missing {split} data: {X_path} or {y_path}\n"
                f"Run canonical feature pipeline first."
            )

        x_size_mb = X_path.stat().st_size / 1e6
        y_size_mb = y_path.stat().st_size / 1e6

        logger.info(f"[{split}] Loading files:")
        logger.info(f"  X: {X_path} ({x_size_mb:.2f} MB)")
        logger.info(f"  y: {y_path} ({y_size_mb:.2f} MB)")

        # ----------------------------
        # Load data
        # ----------------------------
        X = np.load(X_path)
        y = np.load(y_path)

        data[f"X_{split}"] = X
        data[f"y_{split}"] = y

        # ----------------------------
        # Shape diagnostics
        # ----------------------------
        logger.info(
            f"[{split}] Shapes -> "
            f"X: {X.shape} (samples={X.shape[0]}, features={X.shape[1]}), "
            f"y: {y.shape}"
        )

        total_samples += X.shape[0]

        if feature_dim is None:
            feature_dim = X.shape[1]
        elif feature_dim != X.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch in {split}: "
                f"expected {feature_dim}, got {X.shape[1]}"
            )

        # ----------------------------
        # dtype checks
        # ----------------------------
        logger.info(f"[{split}] dtypes -> X: {X.dtype}, y: {y.dtype}")

        # ----------------------------
        # Basic stats
        # ----------------------------
        logger.info(
            f"[{split}] X stats -> "
            f"min={X.min():.6f}, max={X.max():.6f}, mean={X.mean():.6f}, std={X.std():.6f}"
        )

        logger.info(
            f"[{split}] y stats -> "
            f"min={y.min():.6f}, max={y.max():.6f}, mean={y.mean():.6f}, std={y.std():.6f}"
        )

        # ----------------------------
        # Missing / invalid values
        # ----------------------------
        nan_x = np.isnan(X).sum()
        nan_y = np.isnan(y).sum()
        inf_x = np.isinf(X).sum()
        inf_y = np.isinf(y).sum()

        logger.info(
            f"[{split}] NaN/Inf -> "
            f"NaN(X)={nan_x}, NaN(y)={nan_y}, Inf(X)={inf_x}, Inf(y)={inf_y}"
        )

        # ----------------------------
        # Label distribution / range
        # ----------------------------
        logger.info(f"[{split}] y range: [{y.min():.6f}, {y.max():.6f}]")

        # ----------------------------
        # Sequence sanity (time series assumption)
        # ----------------------------
        logger.info(f"[{split}] sequence length (timesteps): {X.shape[1]}")

    # ----------------------------
    # Global dataset summary
    # ----------------------------
    logger.info("==== Dataset Summary ====")
    logger.info(f"Total samples across splits: {total_samples}")
    logger.info(f"Feature dimension: {feature_dim}")
    logger.info(f"Splits loaded successfully: {list(data.keys())}")

    return data


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
    output_dir: Path,
) -> None:
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

    X_train_lag, y_train_lag = build_xgboost_lag_features(
        X_train, y_train, lookback=lookback
    )
    X_val_lag, y_val_lag = build_xgboost_lag_features(X_val, y_val, lookback=lookback)

    logger.info(f"Lag-based feature shapes:")
    logger.info(f"  X_train: {X_train_lag.shape}, y_train: {y_train_lag.shape}")
    logger.info(f"  X_val:   {X_val_lag.shape}, y_val:   {y_val_lag.shape}")
    logger.info(f"  Features per sample: {X_train_lag.shape[1]}")

    xgb_params = {
        "objective": xgb_config.objective,
        "max_depth": xgb_config.max_depth,
        "learning_rate": xgb_config.learning_rate,
        "n_estimators": xgb_config.n_estimators,
        "subsample": xgb_config.subsample,
        "colsample_bytree": xgb_config.colsample_bytree,
        "min_child_weight": xgb_config.min_child_weight,
        "gamma": xgb_config.gamma,
        "reg_alpha": xgb_config.reg_alpha,
        "reg_lambda": xgb_config.reg_lambda,
        "tree_method": xgb_config.tree_method,
        "max_bin": xgb_config.max_bin,
        "n_jobs": xgb_config.n_jobs,
        "verbosity": xgb_config.verbosity,
        "random_state": xgb_config.random_state,
    }

    early_stopping_rounds = xgb_config.early_stopping_rounds
    feature_type = xgb_config.feature_type

    trainer = XGBoostTrainer(
        xgboost_config=xgb_params,
        early_stopping_rounds=early_stopping_rounds,
    )

    feature_names = [f"f{i}" for i in range(X_train_lag.shape[1])]

    logger.info(feature_names)

    model, history = trainer.train(
        X_train_lag, y_train_lag, X_val_lag, y_val_lag, feature_names
    )

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgboost_model.json"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for key, value in history.items():
            if isinstance(value, dict):
                history_serializable[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                history_serializable[key] = value
        json.dump(history_serializable, f, indent=4)
    logger.info(f"Training history saved to {history_path}")

    # Save model configuration
    config_dict = {
        "xgb_params": xgb_params,
        "lookback": lookback,
        "feature_type": feature_type,
    }
    config_path = output_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    logger.info(f"Model config saved to {config_path}")

    # Save feature importance
    if model.feature_importance:
        importance_path = output_dir / "feature_importance.json"
        sorted_importance = {
            str(k): float(v)
            for k, v in sorted(model.feature_importance.items(), key=lambda x: -x[1])
        }
        with open(importance_path, "w") as f:
            json.dump(sorted_importance, f, indent=4)
        logger.info(f"Feature importance saved to {importance_path}")

    # Save metadata
    metadata = {
        "model_type": "xgboost",
        "protocol": "CANONICAL_2.0",
        "source": "FINAL_PLAN.md Section 4.3",
        "training_samples": int(len(X_train_lag)),
        "validation_samples": int(len(X_val_lag)),
        "features_original": int(X_train.shape[1]),
        "features_lag_based": int(X_train_lag.shape[1]),
        "lookback": int(lookback),
        "hyperparameters": xgb_params,
        "training_complete": True,
        "frozen": True,
        "retraining_allowed": False,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            metadata,
            f,
            indent=4,
            default=lambda x: (
                float(x)
                if isinstance(x, (np.float32, np.float64))
                else int(x) if isinstance(x, (np.int32, np.int64)) else str(x)
            ),
        )

    logger.info(f"Metadata saved to {metadata_path}")


def build_output_dir(ticker: str) -> Path:
    return PROJECT_ROOT / "results" / "train" / ticker / MODEL_TYPE / "v1"


def main():
    parser = argparse.ArgumentParser(
        description="Train Baseline LSTM (static, single-fit)"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default_config.yaml",
    )

    # -------------------------
    # Mode selection (STRICT)
    # -------------------------
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--train-all",
        action="store_true",
        help="Train all tickers",
    )

    group.add_argument(
        "--ticker",
        type=str,
        help="Single ticker to train",
    )

    # -------------------------
    # Paths
    # -------------------------
    parser.add_argument("--features-path", type=Path)
    parser.add_argument("--data-path", type=Path)

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # ==========================================================
        # SINGLE TICKER MODE
        # ==========================================================
        if args.ticker:
            if args.data_path is None:
                raise ValueError("--data-path required for single ticker mode")

            logger.info(f"Training single ticker: {args.ticker}")

            data = load_feature_data(args.data_path)

            output_dir = build_output_dir(args.ticker)

            setup_logger(
                log_file=f"{output_dir}/{args.ticker}_train.log",
                level="INFO",
                mode=LogFileMode.OVERWRITE,
            )

            train_xgboost(
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_val=data["X_val"],
                y_val=data["y_val"],
                config=config,
                output_dir=output_dir,
            )

        # ==========================================================
        # MULTI TICKER MODE
        # ==========================================================
        elif args.train_all:
            if args.features_path is None:
                raise ValueError("--features-path required for --train-all")

            tickers = [d.name for d in args.features_path.iterdir() if d.is_dir()]

            logger.info(f"Training ALL tickers: {tickers}")

            for ticker in tickers:
                logger.info("=" * 80)
                logger.info(f"Training ticker: {ticker}")
                logger.info("=" * 80)

                ticker_path = args.features_path / ticker
                data = load_feature_data(ticker_path)

                output_dir = build_output_dir(ticker)
                setup_logger(
                    log_file=f"{output_dir}/{ticker}_train.log",
                    level="INFO",
                    mode=LogFileMode.OVERWRITE,
                )
                train_xgboost(
                    X_train=data["X_train"],
                    y_train=data["y_train"],
                    X_val=data["X_val"],
                    y_val=data["y_val"],
                    config=config,
                    output_dir=output_dir,
                )

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
