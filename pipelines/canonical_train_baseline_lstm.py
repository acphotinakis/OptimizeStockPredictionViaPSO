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

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import FrozenPipelineState
from src.models import LSTMModel, LSTMTrainer, build_lstm_windows, set_seeds
from src.utils.config_loader import Config, load_config

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


def train_baseline_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
    output_dir: Path,
) -> LSTMModel:
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
    lookback = config.features.windowing.lookback
    seed = lstm_config.random_seed
    
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
    
    # Create model with FIXED hyperparameters
    logger.info("Creating Baseline LSTM with FIXED hyperparameters:")
    logger.info(f"  Units L1: {lstm_config.units_1}")
    logger.info(f"  Units L2: {lstm_config.units_2}")
    logger.info(f"  Dropout:  {lstm_config.dropout}")
    logger.info(f"  LR:       {lstm_config.learning_rate}")
    logger.info(f"  Batch:    {lstm_config.batch_size}")
    logger.info(f"  Epochs:   {lstm_config.epochs}")
    logger.info(f"  Shuffle:  {lstm_config.shuffle}  <-- MUST BE FALSE")
    
    # Model configuration dict
    model_config = {
        "input_shape": (lookback, X_train.shape[1]),
        "lstm_units_1": lstm_config.units_1,
        "lstm_units_2": lstm_config.units_2,
        "dropout_rate": lstm_config.dropout,
        "activation": lstm_config.activation,
        "output_units": lstm_config.output_units,
        "output_activation": lstm_config.output_activation,
        "learning_rate": lstm_config.learning_rate,
        "loss": lstm_config.loss,
    }
    
    # Create model
    model = LSTMModel(model_config, seed=seed)
    
    # Train with early stopping
    logger.info("=" * 80)
    logger.info("TRAINING (SINGLE FIT - NO RETRAINING)")
    logger.info("=" * 80)
    
    trainer = LSTMTrainer(model_config, seed=seed)
    
    model, history = trainer.train(
        X_train_win,
        y_train_win,
        X_val_win,
        y_val_win,
        epochs=lstm_config.epochs,
        batch_size=lstm_config.batch_size,
        patience=lstm_config.early_stopping.patience,
        shuffle=lstm_config.shuffle,  # MUST be False
    )
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE - MODEL NOW FROZEN")
    logger.info("=" * 80)
    logger.info("⚠️  This model will NEVER be retrained")
    logger.info("⚠️  Walk-forward evaluation will use THIS frozen model")
    logger.info("=" * 80)
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "baseline_lstm_model.h5"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save training history
    history_path = output_dir / "training_history.yaml"
    with open(history_path, "w") as f:
        yaml.dump(history, f, default_flow_style=False)
    logger.info(f"Training history saved to {history_path}")
    
    # Save model configuration
    config_path = output_dir / "model_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)
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
            "units_1": lstm_config.units_1,
            "units_2": lstm_config.units_2,
            "dropout": lstm_config.dropout,
            "learning_rate": lstm_config.learning_rate,
            "batch_size": lstm_config.batch_size,
            "epochs": lstm_config.epochs,
        },
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
        description="Train Baseline LSTM (static, single-fit)"
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
        default=PROJECT_ROOT / "results" / "canonical" / "models" / "baseline_lstm",
        help="Directory to save trained model",
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Load preprocessed data
        data = load_preprocessed_data(args.data_path)
        
        # Train model (ONCE)
        model = train_baseline_lstm(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
            config=config,
            output_dir=args.output_dir,
        )
        
        logger.info("=" * 80)
        logger.info("BASELINE LSTM TRAINING SUCCESSFUL")
        logger.info("=" * 80)
        logger.info("✓ Model trained and frozen")
        logger.info("✓ Model saved to disk")
        logger.info("✓ Ready for walk-forward evaluation")
        logger.info("=" * 80)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
