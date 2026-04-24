#!/usr/bin/env python3
"""
Canonical Baseline LSTM Training Script

Implements static (single-fit) Baseline LSTM training as defined in FINAL_PLAN.md Section 4.1.

CRITICAL RULES:
- Train EXACTLY ONCE on 70% training data
- Use 10% validation for early stopping
- Fixed hyperparameters (NEVER tuned)
- NO shuffling (shuffle=False mandatory)
- Model is FROZEN after training
- NO retraining during walk-forward

Usage:
    python pipelines/canonical_train_baseline_lstm.py \\
        --data-path data/processed/features_unified/AAPL \\
        --config config/canonical_config.yaml \\
        --output-dir results/canonical/models/baseline_lstm

Author: System Architect
Version: CANONICAL 1.0
Source: FINAL_PLAN.md Section 4.1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import LSTMModel, LSTMTrainer, set_seeds
from src.data.windowing import build_lstm_windows
from src.utils.config_loader import Config, load_config
from src.utils.logger import LogFileMode, setup_logger

logger = logging.getLogger(__name__)

MODEL_TYPE = "lstm_baseline"


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


def train_baseline_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
    output_dir: Path,
) -> None:
    """
    Train Baseline LSTM with fixed hyperparameters.

    FINAL_PLAN.md Section 4.1: Baseline LSTM Training Phase

    CRITICAL: This function executes EXACTLY ONCE. The model is then FROZEN.

    Args:
        X_train: Training features (N, F)
        y_train: Training targets (N,)
        X_val: Validation features (M, F)
        y_val: Validation targets (M,)
        config: Configuration dict
        output_dir: Directory to save model

    Returns:
        Trained and FROZEN LSTMModel
    """
    logger.info("=" * 80)
    logger.info("BASELINE LSTM TRAINING (FINAL_PLAN.md Section 4.1)")
    logger.info("=" * 80)
    logger.info("CRITICAL: Model will be trained EXACTLY ONCE and then FROZEN")
    logger.info("=" * 80)

    # Extract config
    lstm_config = config.lstm_baseline
    lookback = config.lstm_baseline.lookback
    seed = config.lstm_baseline.random_seed

    # Set seeds for reproducibility
    set_seeds(seed)
    logger.info(f"Random seed: {seed}")

    # Build LSTM windows
    logger.info(f"Building LSTM windows (lookback={lookback})...")
    X_train_win, y_train_win = build_lstm_windows(X_train, y_train, lookback)
    X_val_win, y_val_win = build_lstm_windows(X_val, y_val, lookback)

    logger.info(f"Windowed shapes:")
    logger.info(f"  X_train: {X_train_win.shape}, y_train: {y_train_win.shape}")
    logger.info(f"  X_val:   {X_val_win.shape}, y_val:   {y_val_win.shape}")

    # Model configuration dict (for metadata persistence)
    model_config = {
        "input_size": X_train_win.shape[2],
        "lstm_units_1": lstm_config.lstm_units_1,
        "lstm_units_2": lstm_config.lstm_units_2,
        "dropout_rate": lstm_config.dropout_rate,
        "output_units": lstm_config.output_units,
        "activation": lstm_config.activation,
        "output_activation": lstm_config.output_activation,
    }

    # Create model
    lstm_model_wrapper = LSTMModel(seed=seed)
    lstm_network, lstm_model = lstm_model_wrapper.build_model(model_config)

    logger.info("========== LSTM MODEL BUILT ==========")
    logger.info(lstm_model)
    logger.info("======================================")

    # Build unified trainer config dict — ALL hyperparameters from YAML
    trainer_config = {
        # Training
        "optimizer": lstm_config.optimizer,
        "learning_rate": lstm_config.learning_rate,
        "loss": lstm_config.loss,
        "epochs": lstm_config.epochs,
        "batch_size": lstm_config.batch_size,
        "shuffle": lstm_config.shuffle,
        # Advanced training
        "grad_clip": lstm_config.grad_clip,
        "use_amp": lstm_config.use_amp,
        "accumulation_steps": lstm_config.accumulation_steps,
        # Early stopping
        "early_stopping": {
            "enabled": lstm_config.early_stopping.enabled,
            "monitor": lstm_config.early_stopping.monitor,
            "patience": lstm_config.early_stopping.patience,
            "restore_best_weights": lstm_config.early_stopping.restore_best_weights,
        },
    }

    # create trainer and run
    trainer = LSTMTrainer(lstm_model=lstm_model, config=trainer_config, seed=seed)

    logger.info("========== LSTM TRAINER BUILT ==========")
    logger.info(trainer)
    logger.info("======================================")

    # Train with early stopping
    logger.info("=" * 80)
    logger.info("TRAINING (SINGLE FIT - NO RETRAINING)")
    logger.info("=" * 80)

    model, history = trainer.train(
        X_train_win,
        y_train_win,
        X_val_win,
        y_val_win,
    )

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE - MODEL NOW FROZEN")
    logger.info("=" * 80)

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "baseline_lstm_model.pt"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    logger.info(f"Training history saved to {history_path}")

    # Save model configuration
    config_path = output_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=4)
    logger.info(f"Model config saved to {config_path}")

    # Save metadata
    metadata = {
        "model_type": "baseline_lstm",
        "protocol": "CANONICAL_1.0",
        "source": "FINAL_PLAN.md Section 4.1",
        "training_samples": len(X_train_win),
        "validation_samples": len(X_val_win),
        "features": X_train.shape[1],
        "lookback": lookback,
        "hyperparameters": {
            "lstm_units_1": lstm_config.lstm_units_1,
            "lstm_units_2": lstm_config.lstm_units_2,
            "dropout_rate": lstm_config.dropout_rate,
            "learning_rate": lstm_config.learning_rate,
            "batch_size": lstm_config.batch_size,
            "epochs": lstm_config.epochs,
        },
        "training_complete": True,
        "frozen": True,
        "retraining_allowed": False,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved to {metadata_path}")

    # return model


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

            train_baseline_lstm(
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
                train_baseline_lstm(
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
