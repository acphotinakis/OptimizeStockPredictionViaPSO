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
from matplotlib.ticker import FuncFormatter

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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu/cuda). Auto-detect if not specified.",
    )

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
            "wfv_fold_size",
            "wfv_folds",
            "initial_capital",
            "position_fraction",
            "transaction_cost",
            "slippage",
            "stop_loss",
            "daily_loss_limit",
        ]:
            if hasattr(args, attr):
                delattr(args, attr)
    elif args.mode == "val":
        for attr in [
            "initial_capital",
            "position_fraction",
            "transaction_cost",
            "slippage",
            "stop_loss",
            "daily_loss_limit",
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
    elif hasattr(cfg, "lstm_baseline") and hasattr(cfg.lstm_baseline, "lookback"):
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
    if hasattr(cfg, "lstm_baseline"):
        for key in base_params.keys():
            if hasattr(cfg.lstm_baseline, key):
                base_params[key] = getattr(cfg.lstm_baseline, key)

    # Override from command line
    hyperparams = {
        "num_layers": (
            args.num_layers
            if args.num_layers is not None
            else base_params["num_layers"]
        ),
        "hidden_units": (
            args.hidden_units
            if args.hidden_units is not None
            else base_params["hidden_units"]
        ),
        "dropout": args.dropout if args.dropout is not None else base_params["dropout"],
        "learning_rate": (
            args.learning_rate
            if args.learning_rate is not None
            else base_params["learning_rate"]
        ),
        "lookback": lookback,
        "max_epochs": (
            args.max_epochs
            if args.max_epochs is not None
            else base_params["max_epochs"]
        ),
        "patience": (
            args.patience if args.patience is not None else base_params["patience"]
        ),
        "batch_size": (
            args.batch_size
            if args.batch_size is not None
            else base_params["batch_size"]
        ),
    }

    logger.info("Hyperparameters: %s", hyperparams)

    # Initialize model
    device = args.device if args.device is not None else None
    model = VanillaLSTM(input_size=X_train_flat.shape[1], device=device, **hyperparams)

    # Train
    logger.info("\nTraining model...")
    t0 = time.time()
    history = model.fit(X_train_windows, y_train_windows, X_val_windows, y_val_windows)
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s")

    log_memory_usage("After training")

    torch.cuda.empty_cache()

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
        json.dump(
            {
                "ticker": ticker,
                "mode": "train",
                "seed": args.seed,
                "hyperparameters": hyperparams,
                "val_metrics": val_metrics,
                "runtime_seconds": elapsed,
            },
            f,
            indent=2,
        )

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
    plt.plot(history["train_loss"], label="Train Loss", alpha=0.7)
    plt.plot(history["val_loss"], label="Val Loss", alpha=0.7)
    plt.title(f"{ticker} LSTM Baseline Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Predictions
    plt.subplot(1, 2, 2)
    n_samples = min(500, len(y_val_windows))
    plt.plot(y_val_windows[:n_samples], label="True", alpha=0.7)
    plt.plot(y_pred_val[:n_samples], label="Predicted", alpha=0.7)
    plt.title(f"{ticker} LSTM Baseline Predictions")
    plt.xlabel("Sample")
    plt.ylabel("Return")
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
    params_path = (
        results_dir / f"lstm_baseline_params_{ticker}_train_seed{args.seed}.json"
    )
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

        fold_metrics.append(
            {
                "fold": fold_idx,
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "metrics": metrics,
            }
        )

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
    logger.info(
        f"  RMSE: {avg_metrics['rmse']['mean']:.6f} ± {avg_metrics['rmse']['std']:.6f}"
    )

    # Save
    with open(results_dir / f"lstm_baseline_wfv_{tag}.json", "w") as f:
        json.dump(
            {
                "ticker": ticker,
                "fold_metrics": fold_metrics,
                "aggregate_metrics": avg_metrics,
            },
            f,
            indent=2,
        )

    logger.info("Validation complete.")


import pandas as pd


def run_test(args, features_dir, results_dir, ticker, tag):
    """Test mode: final evaluation on test set with signal generation."""
    logger.info("\n[TEST] Final evaluation...")

    # Load model
    model_path = results_dir / f"lstm_baseline_model_{ticker}_train_seed{args.seed}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")

    # Load data
    ticker_dir = features_dir / ticker
    X_test_flat, y_test = load_data(ticker_dir, "test")

    # Load params
    params_path = (
        results_dir / f"lstm_baseline_params_{ticker}_train_seed{args.seed}.json"
    )
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

    # Load OHLCV data for alignment
    try:
        # Try to load from processed data
        ohlcv_path = Path("data/processed") / f"{ticker}.parquet"
        if ohlcv_path.exists():
            ohlcv_df = pd.read_parquet(ohlcv_path)
            logger.info(f"Loaded OHLCV data from {ohlcv_path}")
        else:
            # Fallback to raw data
            ohlcv_path = Path("data/raw") / f"{ticker}.parquet"
            if ohlcv_path.exists():
                ohlcv_df = pd.read_parquet(ohlcv_path)
                logger.info(f"Loaded OHLCV data from {ohlcv_path}")
            else:
                logger.warning(
                    f"No OHLCV data found for {ticker}, creating minimal DataFrame"
                )
                ohlcv_df = None
    except Exception as e:
        logger.warning(f"Failed to load OHLCV data: {e}")
        ohlcv_df = None

    # Generate trading signals
    signal_threshold = 0.001  # 0.1% threshold
    signals = np.where(
        y_pred_test > signal_threshold,
        1,
        np.where(y_pred_test < -signal_threshold, -1, 0),
    )

    # Create aligned DataFrame
    if ohlcv_df is not None:
        # Get test period dates from config
        cfg = load_config(args.config)
        test_start = pd.Timestamp(cfg.data.val_end) + pd.Timedelta(days=1)

        # Handle timezone-aware vs timezone-naive comparison
        if ohlcv_df.index.tz is not None:
            # OHLCV has timezone, make test_start timezone-aware
            test_start = test_start.tz_localize(ohlcv_df.index.tz)

        test_ohlcv = ohlcv_df[ohlcv_df.index >= test_start].copy()

        # Align predictions with OHLCV (predictions start after lookback window)
        # The first prediction corresponds to bar at index = lookback
        pred_start_idx = lookback
        pred_end_idx = pred_start_idx + len(y_pred_test)

        if pred_end_idx <= len(test_ohlcv):
            aligned_dates = test_ohlcv.index[pred_start_idx:pred_end_idx]
            aligned_ohlcv = test_ohlcv.iloc[pred_start_idx:pred_end_idx].copy()

            aligned_df = pd.DataFrame(
                {
                    "date": aligned_dates,
                    "open": aligned_ohlcv["open"].values,
                    "high": aligned_ohlcv["high"].values,
                    "low": aligned_ohlcv["low"].values,
                    "close": aligned_ohlcv["close"].values,
                    "volume": aligned_ohlcv["volume"].values,
                    "actual_return": y_test_windows.flatten(),
                    "predicted_return": y_pred_test.flatten(),
                    "signal": signals.flatten(),
                }
            )
        else:
            logger.warning(
                "Prediction length exceeds OHLCV data, using minimal alignment"
            )
            aligned_df = pd.DataFrame(
                {
                    "actual_return": y_test_windows.flatten(),
                    "predicted_return": y_pred_test.flatten(),
                    "signal": signals.flatten(),
                }
            )
    else:
        # No OHLCV data available
        aligned_df = pd.DataFrame(
            {
                "actual_return": y_test_windows.flatten(),
                "predicted_return": y_pred_test.flatten(),
                "signal": signals.flatten(),
            }
        )

    # Calculate PnL from signals
    if "close" in aligned_df.columns and len(aligned_df) > 0:
        # Simple PnL calculation: signal[t] × actual_return[t+1]
        # This assumes we enter at close[t] and exit at close[t+1]
        position = signals.flatten()
        returns = aligned_df["actual_return"].values

        # Strategy returns: position[t] × return[t]
        strategy_returns = position * returns

        # Apply transaction costs when position changes
        position_changes = np.diff(position, prepend=0)
        transaction_costs = np.abs(position_changes) * args.transaction_cost
        strategy_returns_net = strategy_returns - transaction_costs

        # Cumulative PnL
        initial_capital = args.initial_capital
        cumulative_returns = np.cumprod(1 + strategy_returns_net)
        pnl = initial_capital * (cumulative_returns - 1)
        equity_curve = initial_capital * cumulative_returns

        # Add to DataFrame
        aligned_df["strategy_return"] = strategy_returns_net
        aligned_df["pnl"] = pnl
        aligned_df["equity"] = equity_curve

        # Calculate trading metrics
        total_return = equity_curve[-1] / initial_capital - 1
        n_trades = (np.abs(position_changes) > 0).sum()

        # Sharpe ratio (annualized)
        if len(strategy_returns_net) > 1:
            sharpe = (
                np.mean(strategy_returns_net)
                / (np.std(strategy_returns_net) + 1e-10)
                * np.sqrt(252 * 390)
            )  # 1-min bars
        else:
            sharpe = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_dd = np.min(drawdown)

        pnl_stats = {
            "initial_capital": float(initial_capital),
            "final_equity": float(equity_curve[-1]),
            "total_return": float(total_return),
            "total_pnl": float(pnl[-1]),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "n_trades": int(n_trades),
            "avg_trade_return": (
                float(np.mean(strategy_returns_net[position != 0]))
                if (position != 0).any()
                else 0.0
            ),
        }

        logger.info("\nPnL Statistics:")
        logger.info(f"  Initial Capital: ${initial_capital:,.2f}")
        logger.info(f"  Final Equity: ${equity_curve[-1]:,.2f}")
        logger.info(f"  Total Return: {total_return:.2%}")
        logger.info(f"  Total PnL: ${pnl[-1]:,.2f}")
        logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
        logger.info(f"  Max Drawdown: {max_dd:.2%}")
        logger.info(f"  Number of Trades: {n_trades}")
    else:
        pnl_stats = None
        logger.warning("Cannot calculate PnL: OHLCV data not available")

    # Save aligned predictions
    aligned_csv_path = results_dir / f"lstm_aligned_{tag}.csv"
    aligned_df.to_csv(aligned_csv_path, index=False)
    logger.info(f"Saved aligned predictions to {aligned_csv_path}")

    # Calculate signal statistics
    n_buy_signals = (signals == 1).sum()
    n_sell_signals = (signals == -1).sum()
    n_hold_signals = (signals == 0).sum()

    signal_stats = {
        "total_predictions": len(signals),
        "buy_signals": int(n_buy_signals),
        "sell_signals": int(n_sell_signals),
        "hold_signals": int(n_hold_signals),
        "buy_pct": float(n_buy_signals / len(signals)),
        "sell_pct": float(n_sell_signals / len(signals)),
        "hold_pct": float(n_hold_signals / len(signals)),
    }

    logger.info("\nSignal Statistics:")
    logger.info(f"  Buy signals: {n_buy_signals} ({signal_stats['buy_pct']:.2%})")
    logger.info(f"  Sell signals: {n_sell_signals} ({signal_stats['sell_pct']:.2%})")
    logger.info(f"  Hold signals: {n_hold_signals} ({signal_stats['hold_pct']:.2%})")

    # Save test results with signal stats and PnL
    results_dict = {
        "ticker": ticker,
        "statistical_metrics": test_metrics,
        "signal_statistics": signal_stats,
        "signal_threshold": signal_threshold,
    }

    if pnl_stats is not None:
        results_dict["pnl_statistics"] = pnl_stats

    with open(results_dir / f"lstm_baseline_test_{tag}.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    np.save(results_dir / f"lstm_baseline_test_predictions_{tag}.npy", y_pred_test)
    np.save(results_dir / f"lstm_baseline_test_signals_{tag}.npy", signals)

    # Create comprehensive plots
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of subplots based on PnL availability
    n_plots = 3 if pnl_stats is not None else 2

    # Plot 1: Predictions vs Actual (with optional PnL)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 5 * n_plots))

    n_samples = min(1000, len(y_test_windows))

    # Panel 1: Time series
    ax_idx = 0
    axes[ax_idx].plot(
        y_test_windows[:n_samples], label="Actual", alpha=0.7, linewidth=1.5
    )
    axes[ax_idx].plot(
        y_pred_test[:n_samples], label="Predicted", alpha=0.7, linewidth=1.5
    )
    axes[ax_idx].axhline(
        y=signal_threshold,
        color="g",
        linestyle="--",
        alpha=0.5,
        label=f"Buy Threshold ({signal_threshold})",
    )
    axes[ax_idx].axhline(
        y=-signal_threshold,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Sell Threshold ({-signal_threshold})",
    )
    axes[ax_idx].axhline(y=0, color="k", linestyle="-", alpha=0.3, linewidth=0.5)
    axes[ax_idx].set_title(
        f"{ticker} LSTM Baseline Test Predictions", fontsize=14, fontweight="bold"
    )
    axes[ax_idx].set_xlabel("Sample")
    axes[ax_idx].set_ylabel("Return")
    axes[ax_idx].legend(loc="upper left")
    axes[ax_idx].grid(True, alpha=0.3)

    # Panel 2: Scatter plot
    ax_idx = 1
    axes[ax_idx].scatter(
        y_test_windows[:n_samples], y_pred_test[:n_samples], alpha=0.3, s=10
    )
    axes[ax_idx].plot(
        [-0.05, 0.05], [-0.05, 0.05], "r--", label="Perfect Prediction", linewidth=2
    )
    axes[ax_idx].axhline(y=signal_threshold, color="g", linestyle="--", alpha=0.5)
    axes[ax_idx].axhline(y=-signal_threshold, color="r", linestyle="--", alpha=0.5)
    axes[ax_idx].axvline(x=signal_threshold, color="g", linestyle="--", alpha=0.5)
    axes[ax_idx].axvline(x=-signal_threshold, color="r", linestyle="--", alpha=0.5)
    axes[ax_idx].set_title(
        f"{ticker} Predicted vs Actual Returns", fontsize=14, fontweight="bold"
    )
    axes[ax_idx].set_xlabel("Actual Return")
    axes[ax_idx].set_ylabel("Predicted Return")
    axes[ax_idx].legend()
    axes[ax_idx].grid(True, alpha=0.3)

    # Panel 3: PnL / Equity Curve (if available)
    if pnl_stats is not None and "equity" in aligned_df.columns:
        ax_idx = 2
        equity = aligned_df["equity"].values

        # Plot equity curve
        axes[ax_idx].plot(equity, label="Equity Curve", linewidth=2, color="blue")
        axes[ax_idx].axhline(
            y=initial_capital,
            color="k",
            linestyle="--",
            alpha=0.5,
            label=f"Initial Capital (${initial_capital:,.0f})",
        )

        # Shade profitable/unprofitable regions
        axes[ax_idx].fill_between(
            range(len(equity)),
            initial_capital,
            equity,
            where=(equity >= initial_capital),
            color="green",
            alpha=0.2,
            label="Profit",
        )
        axes[ax_idx].fill_between(
            range(len(equity)),
            initial_capital,
            equity,
            where=(equity < initial_capital),
            color="red",
            alpha=0.2,
            label="Loss",
        )

        # Add final PnL annotation
        final_pnl = pnl_stats["total_pnl"]
        final_return = pnl_stats["total_return"]
        axes[ax_idx].annotate(
            f"Final PnL: ${final_pnl:,.2f} ({final_return:+.2%})",
            xy=(len(equity) - 1, equity[-1]),
            xytext=(len(equity) * 0.7, equity[-1]),
            fontsize=12,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="yellow" if final_pnl > 0 else "lightcoral",
                alpha=0.7,
            ),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )

        axes[ax_idx].set_title(
            f"{ticker} Equity Curve | Sharpe: {pnl_stats['sharpe_ratio']:.3f} | Max DD: {pnl_stats['max_drawdown']:.2%}",
            fontsize=14,
            fontweight="bold",
        )
        axes[ax_idx].set_xlabel("Sample")
        axes[ax_idx].set_ylabel("Equity ($)")
        axes[ax_idx].legend(loc="upper left")
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.0f}"))

    plt.tight_layout()
    plot_path = plots_dir / f"lstm_baseline_test_{tag}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved to {plot_path}")
    plt.close()

    # Plot 2: Signal distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Signal counts
    signal_labels = ["Sell", "Hold", "Buy"]
    signal_counts = [n_sell_signals, n_hold_signals, n_buy_signals]
    signal_colors = ["red", "gray", "green"]

    axes[0].bar(signal_labels, signal_counts, color=signal_colors, alpha=0.7)
    axes[0].set_title(f"{ticker} Signal Distribution")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Add percentage labels on bars
    for i, (label, count) in enumerate(zip(signal_labels, signal_counts)):
        pct = count / len(signals) * 100
        axes[0].text(i, count, f"{count}\n({pct:.1f}%)", ha="center", va="bottom")

    # Prediction distribution
    axes[1].hist(y_pred_test, bins=50, alpha=0.7, edgecolor="black")
    axes[1].axvline(
        x=signal_threshold,
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Buy Threshold",
    )
    axes[1].axvline(
        x=-signal_threshold,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Sell Threshold",
    )
    axes[1].axvline(x=0, color="k", linestyle="-", linewidth=1, alpha=0.5)
    axes[1].set_title(f"{ticker} Prediction Distribution")
    axes[1].set_xlabel("Predicted Return")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    signal_plot_path = plots_dir / f"lstm_signals_{tag}.png"
    plt.savefig(signal_plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Signal plot saved to {signal_plot_path}")
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
