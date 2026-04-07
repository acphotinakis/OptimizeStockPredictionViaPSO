#!/usr/bin/env python3
"""
scripts/run_lstm_baseline.py

Train vanilla LSTM baseline (no PSO optimization) for stock return prediction.

Usage:
    # Training
    python scripts/run_lstm_baseline.py --ticker AAPL --mode train
    
    # Validation
    python scripts/run_lstm_baseline.py --ticker AAPL --mode val
    
    # Testing
    python scripts/run_lstm_baseline.py --ticker AAPL --mode test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baselines import VanillaLSTM
from src.data.splitter import build_windows
from src.evaluation import all_statistical_metrics
from src.utils import set_all_seeds, setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LSTM Baseline Training, Validation, Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Common arguments
    parser.add_argument("--ticker", required=True, help="Target ticker symbol")
    parser.add_argument(
        "--mode",
        choices=["train", "val", "test"],
        required=True,
        help="Execution mode: train, val, or test",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument(
        "--features-dir", default="data/features", help="Features directory"
    )
    parser.add_argument("--results-dir", default="results", help="Output directory")
    parser.add_argument(
        "--log-file", default="logs/run_lstm_baseline.log", help="Log file path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Config file path",
    )
    
    # Model hyperparameters (overrides)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--hidden-units", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, 
                       help="Device to use (cpu/cuda). Auto-detect if not specified.")
    
    # Validation-specific arguments
    parser.add_argument("--wfv-fold-size", type=int, default=252)
    parser.add_argument("--wfv-folds", type=int, default=10)
    
    # Testing-specific arguments
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--position-fraction", type=float, default=0.02)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--stop-loss", type=float, default=0.02)
    parser.add_argument("--daily-loss-limit", type=float, default=0.05)
    
    args = parser.parse_args()
    
    # Filter arguments based on mode
    if args.mode == "train":
        for attr in [
            "wfv_fold_size", "wfv_folds",
            "initial_capital", "position_fraction",
            "transaction_cost", "slippage",
            "stop_loss", "daily_loss_limit",
        ]:
            if hasattr(args, attr):
                delattr(args, attr)
    elif args.mode == "val":
        for attr in [
            "initial_capital", "position_fraction",
            "transaction_cost", "slippage",
            "stop_loss", "daily_loss_limit",
        ]:
            if hasattr(args, attr):
                delattr(args, attr)
    elif args.mode == "test":
        for attr in ["wfv_fold_size", "wfv_folds"]:
            if hasattr(args, attr):
                delattr(args, attr)
    
    return args


# ======================================================================
# Helper functions
# ======================================================================

def load_data(ticker_dir: Path, split: str):
    """Load feature arrays for given split."""
    X = np.load(ticker_dir / f"X_{split}.npy")
    y = np.load(ticker_dir / f"y_{split}.npy")
    return X, y


def log_memory_usage(label: str):
    """Log current memory usage."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logger.info(f"[{label}] Memory: {mem_info.rss / 1024**3:.2f} GB")
    except ImportError:
        pass


def _print_info_metrics(label: str, m: dict) -> None:
    """Pretty-print metrics dict."""
    logger.info(
        f"  {label} — "
        f"RMSE={m['rmse']:.6f}  "
        f"DA={m['directional_accuracy']:.4f}  "
        f"F1={m['f1_ternary']:.4f}  "
        f"R²={m['r2']:.4f}"
    )


# ======================================================================
# Mode handlers
# ======================================================================

def run_train(args, features_dir, results_dir, ticker, tag):
    """Training mode: train LSTM baseline and save model."""
    cfg = load_config(args.config)
    
    logger.info("\n[TRAIN] Loading data...")
    ticker_dir = features_dir / ticker
    
    X_train_flat, y_train = load_data(ticker_dir, "train")
    X_val_flat, y_val = load_data(ticker_dir, "val")
    
    log_memory_usage("After loading data")
    
    # Build windows
    # Get lookback from args or config
    if args.lookback is not None:
        lookback = args.lookback
    elif hasattr(cfg, 'lstm_baseline') and hasattr(cfg.lstm_baseline, 'lookback'):
        lookback = cfg.lstm_baseline.lookback
    else:
        lookback = 30  # Default
    
    session_starts_train = np.zeros(len(X_train_flat), dtype=bool)
    session_starts_val = np.zeros(len(X_val_flat), dtype=bool)
    
    X_train_windows, y_train_windows = build_windows(
        X_train_flat, y_train, session_starts_train, lookback
    )
    X_val_windows, y_val_windows = build_windows(
        X_val_flat, y_val, session_starts_val, lookback
    )
    
    logger.info("Windowed Shapes:")
    logger.info("  X_train: %s", X_train_windows.shape)
    logger.info("  X_val: %s", X_val_windows.shape)
    
    log_memory_usage("After windowing")
    
    # Build hyperparameters
    # Use VanillaLSTM.DEFAULT_PARAMS as base
    from src.models.baselines import VanillaLSTM
    base_params = VanillaLSTM.DEFAULT_PARAMS.copy()
    
    # Override from config if available
    if hasattr(cfg, 'lstm_baseline'):
        for key in base_params.keys():
            if hasattr(cfg.lstm_baseline, key):
                base_params[key] = getattr(cfg.lstm_baseline, key)
    
    # Override from command line
    hyperparams = {
        "num_layers": args.num_layers if args.num_layers is not None else base_params["num_layers"],
        "hidden_units": args.hidden_units if args.hidden_units is not None else base_params["hidden_units"],
        "dropout": args.dropout if args.dropout is not None else base_params["dropout"],
        "learning_rate": args.learning_rate if args.learning_rate is not None else base_params["learning_rate"],
        "lookback": lookback,
        "max_epochs": args.max_epochs if args.max_epochs is not None else base_params["max_epochs"],
        "patience": args.patience if args.patience is not None else base_params["patience"],
        "batch_size": args.batch_size if args.batch_size is not None else base_params["batch_size"],
    }
    
    logger.info("Hyperparameters: %s", hyperparams)
    
    # Initialize model
    device = args.device if args.device is not None else None
    model = VanillaLSTM(
        input_size=X_train_flat.shape[1],
        device=device,
        **hyperparams
    )
    
    # Train
    logger.info("\nTraining model...")
    t0 = time.time()
    history = model.fit(X_train_windows, y_train_windows, X_val_windows, y_val_windows)
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s")
    
    log_memory_usage("After training")
    
    # Evaluate
    y_pred_val = model.predict(X_val_windows)
    val_metrics = all_statistical_metrics(y_val_windows, y_pred_val)
    _print_info_metrics("Validation", val_metrics)
    
    # Save model
    model_path = results_dir / f"lstm_baseline_model_{tag}.pth"
    torch.save(model._trainer.model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save params
    with open(results_dir / f"lstm_baseline_params_{tag}.json", "w") as f:
        json.dump({
            "ticker": ticker,
            "mode": "train",
            "seed": args.seed,
            "hyperparameters": hyperparams,
            "val_metrics": val_metrics,
            "runtime_seconds": elapsed,
        }, f, indent=2)
    
    # Save history
    with open(results_dir / f"lstm_baseline_history_{tag}.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Save predictions
    np.save(results_dir / f"lstm_baseline_predictions_{tag}.npy", y_pred_val)
    
    # Plot
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(16, 6))
    
    # Training history
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', alpha=0.7)
    plt.plot(history['val_loss'], label='Val Loss', alpha=0.7)
    plt.title(f'{ticker} LSTM Baseline Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Predictions
    plt.subplot(1, 2, 2)
    n_samples = min(500, len(y_val_windows))
    plt.plot(y_val_windows[:n_samples], label='True', alpha=0.7)
    plt.plot(y_pred_val[:n_samples], label='Predicted', alpha=0.7)
    plt.title(f'{ticker} LSTM Baseline Predictions')
    plt.xlabel('Sample')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = plots_dir / f"lstm_baseline_train_{tag}.png"
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Plot saved to {plot_path}")
    plt.close()
    
    logger.info("Training complete.")


def run_val(args, features_dir, results_dir, ticker, tag):
    """Validation mode: walk-forward validation."""
    logger.info("\n[VAL] Walk-forward validation...")
    
    # Load model
    model_path = results_dir / f"lstm_baseline_model_{ticker}_train_seed{args.seed}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")
    
    # Load data
    ticker_dir = features_dir / ticker
    X_val_flat, y_val = load_data(ticker_dir, "val")
    
    # Load params to get lookback and input_size
    params_path = results_dir / f"lstm_baseline_params_{ticker}_train_seed{args.seed}.json"
    with open(params_path) as f:
        params = json.load(f)
    lookback = params["hyperparameters"]["lookback"]
    
    # Build windows
    session_starts_val = np.zeros(len(X_val_flat), dtype=bool)
    X_val_windows, y_val_windows = build_windows(
        X_val_flat, y_val, session_starts_val, lookback
    )
    
    # Initialize model and load weights
    model = VanillaLSTM(input_size=X_val_flat.shape[1])
    model._trainer.model.load_state_dict(torch.load(model_path))
    
    # Walk-forward validation
    fold_size = args.wfv_fold_size
    max_folds = args.wfv_folds
    
    fold_metrics = []
    for fold_idx in range(max_folds):
        start_idx = fold_idx * fold_size
        end_idx = start_idx + fold_size
        
        if end_idx > len(X_val_windows):
            break
        
        X_fold = X_val_windows[start_idx:end_idx]
        y_fold = y_val_windows[start_idx:end_idx]
        
        y_pred_fold = model.predict(X_fold)
        metrics = all_statistical_metrics(y_fold, y_pred_fold)
        
        fold_metrics.append({
            "fold": fold_idx,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "metrics": metrics,
        })
        
        logger.info(f"Fold {fold_idx}: RMSE={metrics['rmse']:.6f}")
    
    # Aggregate
    avg_metrics = {}
    for metric_name in fold_metrics[0]["metrics"].keys():
        values = [f["metrics"][metric_name] for f in fold_metrics]
        avg_metrics[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    
    logger.info("\nAggregate Metrics:")
    logger.info(f"  RMSE: {avg_metrics['rmse']['mean']:.6f} ± {avg_metrics['rmse']['std']:.6f}")
    
    # Save
    with open(results_dir / f"lstm_baseline_wfv_{tag}.json", "w") as f:
        json.dump({
            "ticker": ticker,
            "fold_metrics": fold_metrics,
            "aggregate_metrics": avg_metrics,
        }, f, indent=2)
    
    logger.info("Validation complete.")


def run_test(args, features_dir, results_dir, ticker, tag):
    """Test mode: final evaluation on test set."""
    logger.info("\n[TEST] Final evaluation...")
    
    # Load model
    model_path = results_dir / f"lstm_baseline_model_{ticker}_train_seed{args.seed}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")
    
    # Load data
    ticker_dir = features_dir / ticker
    X_test_flat, y_test = load_data(ticker_dir, "test")
    
    # Load params
    params_path = results_dir / f"lstm_baseline_params_{ticker}_train_seed{args.seed}.json"
    with open(params_path) as f:
        params = json.load(f)
    lookback = params["hyperparameters"]["lookback"]
    
    # Build windows
    session_starts_test = np.zeros(len(X_test_flat), dtype=bool)
    X_test_windows, y_test_windows = build_windows(
        X_test_flat, y_test, session_starts_test, lookback
    )
    
    # Initialize model and load weights
    model = VanillaLSTM(input_size=X_test_flat.shape[1])
    model._trainer.model.load_state_dict(torch.load(model_path))
    
    # Predict
    y_pred_test = model.predict(X_test_windows)
    
    # Metrics
    test_metrics = all_statistical_metrics(y_test_windows, y_pred_test)
    _print_info_metrics("Test", test_metrics)
    
    # Save
    with open(results_dir / f"lstm_baseline_test_{tag}.json", "w") as f:
        json.dump({
            "ticker": ticker,
            "statistical_metrics": test_metrics,
        }, f, indent=2)
    
    np.save(results_dir / f"lstm_baseline_test_predictions_{tag}.npy", y_pred_test)
    
    # Plot
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(16, 6))
    n_samples = min(1000, len(y_test_windows))
    plt.plot(y_test_windows[:n_samples], label='True', alpha=0.7)
    plt.plot(y_pred_test[:n_samples], label='Predicted', alpha=0.7)
    plt.title(f'{ticker} LSTM Baseline Test Predictions')
    plt.xlabel('Sample')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = plots_dir / f"lstm_baseline_test_{tag}.png"
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Plot saved to {plot_path}")
    plt.close()
    
    logger.info("Testing complete.")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()
    
    # Setup
    setup_logger(args.log_file)
    set_all_seeds(args.seed)
    
    features_dir = Path(args.features_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    ticker = args.ticker
    tag = f"{ticker}_{args.mode}_seed{args.seed}"
    
    logger.info(f"{'='*60}")
    logger.info(
        f"LSTM Baseline Pipeline | mode={args.mode} | ticker={ticker} | seed={args.seed}"
    )
    logger.info(f"{'='*60}")
    
    # Mode dispatch
    if args.mode == "train":
        run_train(args, features_dir, results_dir, ticker, tag)
    elif args.mode == "val":
        run_val(args, features_dir, results_dir, ticker, tag)
    elif args.mode == "test":
        run_test(args, features_dir, results_dir, ticker, tag)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
