#!/usr/bin/env python3
"""
Unified LSTM Training Pipeline

Production-grade end-to-end training script for TRD-compliant LSTM system.

Integrates:
1. Unified feature pipeline output loading
2. Canonical LSTM model training
3. Early stopping and best model selection
4. Test set evaluation
5. Model persistence

Usage:
    # Train single ticker
    python pipelines/unified_train.py --ticker AAPL

    # Train with custom config
    python pipelines/unified_train.py --ticker AAPL --config config/my_config.yaml

    # Train multiple tickers
    python pipelines/unified_train.py --tickers AAPL MSFT GOOGL

Author: System Architect
Version: 1.0.0 UNIFIED
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# Project root setup
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import LSTMTrainer, save_model_weights
from src.utils.config_loader import Config, load_config
from src.utils.logger import LogFileMode, setup_logger

logger = logging.getLogger(__name__)


def load_windowed_data(ticker: str, features_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load pre-windowed LSTM data from unified feature pipeline.

    Args:
        ticker: Ticker symbol
        features_dir: Base features directory (e.g., data/features_unified)

    Returns:
        Dictionary with keys:
            - X_train, y_train, X_val, y_val, X_test, y_test
            - train_index, val_index, test_index

    Raises:
        FileNotFoundError: If required files missing
    """
    ticker_dir = features_dir / ticker

    if not ticker_dir.exists():
        raise FileNotFoundError(f"Ticker directory not found: {ticker_dir}")

    logger.info(f"[{ticker}] Loading windowed data from {ticker_dir}")

    # Required files
    required_files = {
        "X_train": f"{ticker}_X_train_seq.npy",
        "y_train": f"{ticker}_y_train_seq.npy",
        "X_val": f"{ticker}_X_val_seq.npy",
        "y_val": f"{ticker}_y_val_seq.npy",
        "X_test": f"{ticker}_X_test_seq.npy",
        "y_test": f"{ticker}_y_test_seq.npy",
        "train_index": f"{ticker}_train_index.npy",
        "val_index": f"{ticker}_val_index.npy",
        "test_index": f"{ticker}_test_index.npy",
    }

    data = {}
    for key, filename in required_files.items():
        filepath = ticker_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")
        data[key] = np.load(filepath)

    # Log shapes
    logger.info(f"[{ticker}] Data loaded:")
    logger.info(f"  Train: X={data['X_train'].shape}, y={data['y_train'].shape}")
    logger.info(f"  Val:   X={data['X_val'].shape}, y={data['y_val'].shape}")
    logger.info(f"  Test:  X={data['X_test'].shape}, y={data['y_test'].shape}")

    return data


def train_single_ticker(
    ticker: str,
    config: Config,
    features_dir: Path,
    output_dir: Path,
) -> Dict:
    """
    Train LSTM model for single ticker.

    Args:
        ticker: Ticker symbol
        config: Configuration object
        features_dir: Directory with windowed features
        output_dir: Directory for model artifacts

    Returns:
        Dictionary with training results
    """
    logger.info("=" * 80)
    logger.info(f"TRAINING LSTM FOR {ticker}")
    logger.info("=" * 80)

    # Load data
    try:
        data = load_windowed_data(ticker, features_dir)
    except FileNotFoundError as e:
        logger.error(f"[{ticker}] {e}")
        return {"status": "failed", "error": str(e)}

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Build LSTM config from unified config
    lstm_config = {
        "lstm_units_1": config.lstm.lstm_units_1,
        "lstm_units_2": config.lstm.lstm_units_2,
        "dropout_rate": config.lstm.dropout_rate,
        "learning_rate": config.lstm.learning_rate,
        "epochs": config.lstm.epochs,
        "batch_size": config.lstm.batch_size,
        "early_stopping": {
            "enabled": config.lstm.early_stopping.enabled,
            "patience": config.lstm.early_stopping.patience,
            "monitor": config.lstm.early_stopping.monitor,
            "restore_best_weights": config.lstm.early_stopping.restore_best_weights,
        },
    }

    # Initialize trainer
    trainer = LSTMTrainer(
        lstm_config,
        seed=config.lstm.random_seed,
    )

    # Train model
    logger.info(f"[{ticker}] Starting training...")
    try:
        model, history = trainer.train(X_train, y_train, X_val, y_val)
    except Exception as e:
        logger.error(f"[{ticker}] Training failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}

    # Evaluate on test set
    logger.info(f"[{ticker}] Evaluating on test set...")
    try:
        test_metrics = trainer.evaluate(model, X_test, y_test)
    except Exception as e:
        logger.error(f"[{ticker}] Evaluation failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}

    # Save model
    ticker_output_dir = output_dir / ticker
    ticker_output_dir.mkdir(parents=True, exist_ok=True)

    model_path = ticker_output_dir / f"{ticker}_lstm_best.pth"
    save_model_weights(model, model_path)

    # Save training history
    history_path = ticker_output_dir / f"{ticker}_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save test metrics
    metrics_path = ticker_output_dir / f"{ticker}_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save training metadata
    metadata = {
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat(),
        "config": lstm_config,
        "data_shapes": {
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        "training": {
            "total_epochs": len(history["train_loss"]),
            "best_epoch": int(np.argmin(history["val_loss"])) + 1,
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "best_val_loss": float(min(history["val_loss"])),
        },
        "test_metrics": test_metrics,
        "model_path": str(model_path),
    }

    metadata_path = ticker_output_dir / f"{ticker}_training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"[{ticker}] Training complete")
    logger.info(f"[{ticker}] Model saved: {model_path}")
    logger.info(f"[{ticker}] Test MSE: {test_metrics['mse']:.6f}")

    return {
        "status": "success",
        "ticker": ticker,
        "metrics": test_metrics,
        "model_path": str(model_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Unified LSTM Training Pipeline (TRD-Compliant)"
    )
    parser.add_argument(
        "--ticker", type=str, help="Single ticker to train (e.g., AAPL)"
    )
    parser.add_argument(
        "--tickers", nargs="+", help="Multiple tickers (e.g., AAPL MSFT GOOGL)"
    )
    parser.add_argument(
        "--config", default="config/default_config.yaml", help="Configuration file"
    )
    parser.add_argument(
        "--features-dir",
        default="data/features_unified",
        help="Directory with windowed features from unified pipeline",
    )
    parser.add_argument(
        "--output-dir",
        default="models/trained",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(
        log_file="logs/unified_train.log",
        level=args.log_level,
        mode=LogFileMode.OVERWRITE,
    )

    logger.info("=" * 80)
    logger.info("UNIFIED LSTM TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config: {args.config}")
    logger.info(f"  Model: {config.lstm.name}")
    logger.info(
        f"  Architecture: {config.lstm.lstm_units_1} → {config.lstm.lstm_units_2} → 1"
    )
    logger.info(f"  Lookback: {config.lstm.lookback}")

    # Determine tickers
    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = args.tickers
    else:
        logger.error("Must specify --ticker or --tickers")
        sys.exit(1)

    logger.info(f"Training {len(tickers)} ticker(s): {tickers}")

    # Paths
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)

    if not features_dir.exists():
        logger.error(f"Features directory not found: {features_dir}")
        logger.error("Run unified feature pipeline first:")
        logger.error("  python pipelines/run_unified_features.py --ticker AAPL")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Train each ticker
    results = []
    success_count = 0
    fail_count = 0

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"\n[{i}/{len(tickers)}] Processing {ticker}")

        try:
            result = train_single_ticker(
                ticker=ticker,
                config=config,
                features_dir=features_dir,
                output_dir=output_dir,
            )

            if result["status"] == "success":
                success_count += 1
            else:
                fail_count += 1

            results.append(result)

        except Exception as e:
            logger.error(f"[{ticker}] Unexpected error: {e}", exc_info=True)
            fail_count += 1
            results.append(
                {
                    "status": "failed",
                    "ticker": ticker,
                    "error": str(e),
                }
            )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tickers: {len(tickers)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")

    if success_count > 0:
        logger.info("\nSuccessful tickers:")
        for result in results:
            if result["status"] == "success":
                ticker = result["ticker"]
                mse = result["metrics"]["mse"]
                logger.info(f"  {ticker}: MSE={mse:.6f}")

    if fail_count > 0:
        logger.warning("\nFailed tickers:")
        for result in results:
            if result["status"] == "failed":
                ticker = result.get("ticker", "unknown")
                error = result.get("error", "unknown error")
                logger.warning(f"  {ticker}: {error}")

    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "config_file": args.config,
                "total_tickers": len(tickers),
                "successful": success_count,
                "failed": fail_count,
                "results": results,
            },
            f,
            indent=2,
        )

    logger.info(f"\nSummary saved: {summary_path}")
    logger.info("=" * 80)

    if fail_count == 0:
        logger.info("\n✓ All tickers trained successfully!")
        sys.exit(0)
    else:
        logger.warning(f"\n⚠ {fail_count} ticker(s) failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
