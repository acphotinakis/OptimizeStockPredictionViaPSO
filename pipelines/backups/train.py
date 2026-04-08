#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.lstm.lstm_mdl import LSTMModel
from src.utils import set_all_seeds, setup_logger
from src.data.splitter import build_windows
from src.utils.config_loader import load_config
from src.utils.config_schema import Config
from src.models.xgboost.xgboost_mdl import (
    XGBoostModel,
    XGBoostTuner,
)

from src.utils.data_storage import load_windows, load_feature_names, load_training_data
from src.evaluation import (
    all_statistical_metrics,
    _print_stats_info_metrics,
    _print_trading_metrics,
    Backtester,
)
from utils_pipelines import (
    load_artefacts,
    load_prices,
    load_optimal_threshold,
    load_model,
    extract_hyperparameters,
    log_memory_usage,
)

from constants import (
    DEFAULT_WFV_FOLD_SIZE,
    DEFAULT_WFV_FOLDS,
    FEATURES_DIR,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)


def train_xgboost(
    cfg,
    args,
    is_tuner,
    X_train_windows,
    y_train_windows,
    X_val_windows,
    y_val_windows,
):
    logger.info("\nTraining XGBoost model...")
    xgb = XGBoostModel(cfg, 42)
    # Train the XGBoost model
    if is_tuner == "default":
        model, hyperparams = xgb.train_default(
            cfg, X_train_windows, y_train_windows, X_val_windows, y_val_windows
        )
    else:
        with tqdm(total=cfg.xgboost.optuna_trials, desc="Tuning") as pbar:
            model, hyperparams = xgb.train_tune(
                args,
                X_train_windows,
                y_train_windows,
                X_val_windows,
                y_val_windows,
                pbar,
                42,
            )
    return model, hyperparams


def train_baseline_lstm(
    cfg: Config,
    args: argparse.Namespace,
    X_train_windows: np.ndarray,
    y_train_windows: np.ndarray,
    X_val_windows: np.ndarray,
    y_val_windows: np.ndarray,
):
    """
    Train a baseline LSTM model.

    Args:
        cfg: Configuration object containing model and training params.
        args: Parsed command-line arguments.
        is_tuner: Whether to run hyperparameter tuning (currently unused for baseline).
        X_train_windows: [N_train, T, F] training input array.
        y_train_windows: [N_train] training targets.
        X_val_windows: [N_val, T, F] validation input array.
        y_val_windows: [N_val] validation targets.

    Returns:
        model: Trained LSTMModel instance.
        hyperparams: Dictionary of hyperparameters (from cfg).
    """
    logger.info("Initializing LSTM baseline model...")
    model = LSTMModel(cfg, seed=args.seed)

    logger.info("Training LSTM model...")
    result = model.fit(X_train_windows, y_train_windows, X_val_windows, y_val_windows)
    trained_model = result["model"]
    hyperparams = result["hyperparameters"]
    history = result["history"]
    logger.info("LSTM training complete. Best val loss: %.6f", min(history["val_loss"]))

    return trained_model, hyperparams


# ======================================================================
# Argument parsing
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run XGBoostModel Training, Validation, Testing for stock return prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -------------------------
    # Common arguments
    # -------------------------
    parser.add_argument("--ticker", required=True, help="Target ticker symbol")
    parser.add_argument(
        "--model",
        choices=["xgb", "lstm_baseline"],
        required=True,
        default="xgb",
        help="Execution mode: train, val, or test",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument(
        "--features-dir", default=FEATURES_DIR, help="Features directory"
    )
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Output directory")
    parser.add_argument(
        "--log-file", default="logs/run_xgboost.log", help="Log file path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Config file path",
    )

    # -------------------------
    # Training-specific arguments
    # -------------------------
    parser.add_argument(
        "--train-mode",
        choices=["default", "tune"],
        default="default",
        help="Training mode: default fixed or tune random search",
    )
    parser.add_argument("--lookback", type=int, default=30, help="Lookback window size")
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of random search trials"
    )
    parser.add_argument(
        "--retrain-on-trainval",
        action="store_true",
        default=True,
        help="Retrain on Train+Val after tuning",
    )

    # -------------------------
    # Validation-specific arguments
    # -------------------------
    parser.add_argument(
        "--wfv-fold-size",
        type=int,
        default=DEFAULT_WFV_FOLD_SIZE,
        help="Bars per walk-forward validation fold",
    )
    parser.add_argument(
        "--wfv-folds",
        type=int,
        default=DEFAULT_WFV_FOLDS,
        help="Max number of walk-forward folds",
    )

    # -------------------------
    # Testing-specific arguments
    # -------------------------
    # Parse all arguments first
    args = parser.parse_args()

    return args


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    args = parse_args()

    # --------------------------------------------------
    # Shared setup (runs for ALL modes)
    # --------------------------------------------------
    setup_logger(args.log_file)
    set_all_seeds(args.seed)

    features_dir = Path(args.features_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    ticker = args.ticker
    tag = f"{ticker}_{args.model}_seed{args.seed}"

    logger.info(f"{'='*60}")
    logger.info(
        f"Training Model Pipeline | model={args.model} | ticker={ticker} | seed={args.seed}"
    )
    logger.info(f"{'='*60}")

    import time

    cfg = load_config(args.config)

    if cfg is None:
        raise ValueError("Config is null")

    ticker_dir = Path(args.features_dir) / args.ticker
    if not ticker_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {ticker_dir}")

    # Load training data
    X_train_flat, y_train, X_val_flat, y_val = load_training_data(ticker_dir=ticker_dir)
    log_memory_usage("After loading data")

    # Load feature names
    logger.info("Loading features from %s", ticker_dir)
    feature_names = load_feature_names(features_dir, ticker)

    # Get lookback from config
    max_lookback = getattr(cfg.xgboost, "lookback", 30)

    session_starts_train = np.zeros(len(X_train_flat), dtype=bool)
    session_starts_val = np.zeros(len(X_val_flat), dtype=bool)

    # Build windows for training
    X_train_windows, y_train_windows = build_windows(
        X_train_flat, y_train, session_starts_train, max_lookback
    )
    X_val_windows, y_val_windows = build_windows(
        X_val_flat, y_val, session_starts_val, max_lookback
    )
    logger.info("Windowed Shapes:")
    logger.info("  X_train_windows: %s", X_train_windows.shape)
    logger.info("  y_train_windows: %s", y_train_windows.shape)
    logger.info("  X_val_windows  : %s", X_val_windows.shape)
    logger.info("  y_val_windows  : %s", y_val_windows.shape)
    log_memory_usage("After windowing")

    is_tuner = False
    t0 = time.time()
    if args.model == "xgb":
        model, hyperparams = train_xgboost(
            cfg,
            args,
            is_tuner,
            X_train_windows,
            y_train_windows,
            X_val_windows,
            y_val_windows,
        )
    elif args.model == "lstm_baseline":
        model, hyperparams = train_baseline_lstm(
            cfg,
            args,
            X_train_windows,
            y_train_windows,
            X_val_windows,
            y_val_windows,
        )
    else:
        raise ValueError("Model not properly selected")

    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s")
    log_memory_usage("After training")

    y_pred_val = model.predict(X_val_windows)
    val_metrics = all_statistical_metrics(y_val_windows, y_pred_val)
    _print_stats_info_metrics("Training", val_metrics)

    model.save(str(results_dir / f"{args.model}_model_{tag}.ubj"))

    with open(results_dir / f"{args.model}_params_{tag}.json", "w") as f:
        json.dump(
            {
                "ticker": ticker,
                "mode": args.train_mode,
                "seed": args.seed,
                "hyperparameters": hyperparams,
                "best_iteration": model._best_iteration,
                "val_metrics": val_metrics,
                "runtime_seconds": elapsed,
            },
            f,
            indent=2,
        )

    with open(results_dir / f"{args.model}_history_{tag}.json", "w") as f:
        json.dump(model.history, f, indent=2)

    if args.model == "xgb":
        np.save(
            results_dir / f"{args.model}_importances_{tag}.npy",
            model.get_feature_importances(),
        )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
