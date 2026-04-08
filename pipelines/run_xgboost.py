#!/usr/bin/env python3
"""
pipelines/run_xgboost.py

Main pipeline runner for XGBoost model training, validation, and testing.
"""

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

from src.utils import set_all_seeds, setup_logger
from src.data.splitter import build_windows
from src.utils.config_loader import load_config
from src.utils.config_schema import Config
from src.models.xgboost.xgboost_model import (
    XGBoostModel,
    XGBoostTuner,
)

from src.models.xgboost.xgboost_model import XGBoostModel
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


# ======================================================================
# Training
# ======================================================================


def train_default(
    cfg: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[XGBoostModel, Dict[str, Any]]:
    """Train one XGBoostModel with hyperparameters from the config."""
    hyperparams = extract_hyperparameters(cfg)
    logger.info(f"Hyperparameters: {hyperparams}")
    model = XGBoostModel(**hyperparams)
    model.fit(X_train, y_train, X_val, y_val)
    return model, hyperparams


def train_tune(
    args,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pbar: tqdm,
) -> Tuple[XGBoostModel, Dict[str, Any]]:
    """Run random hyperparameter search with tqdm, then optionally retrain winner on Train+Val."""
    tuner = XGBoostTuner(n_trials=args.n_trials, seed=args.seed)

    best_val_rmse = float("inf")
    best_model: Optional[XGBoostModel] = None
    best_params: Dict[str, Any] = {}

    for trial_idx in range(args.n_trials):
        model, params = tuner._run_single_trial(
            X_train, y_train, X_val, y_val, lookback=args.lookback, trial_idx=trial_idx
        )
        pbar.update(1)

        val_rmse = (
            min(model.history["val_rmse"])
            if model.history["val_rmse"]
            else float("inf")
        )
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = model
            best_params = params

    sorted_results = sorted(tuner.results_, key=lambda r: r["val_rmse"])
    logger.info(f"\nTuner results ({args.n_trials} trials):")
    for rank, r in enumerate(sorted_results[:5], 1):
        logger.info(
            f"  #{rank}  val_rmse={r['val_rmse']:.6f}  "
            f"max_depth={r['max_depth']}  lr={r['learning_rate']}  "
            f"n_est={r['n_estimators']}"
        )
    logger.info(f"\nBest params: {best_params}")

    if args.retrain_on_trainval and best_model is not None:
        logger.info("\nRetraining best model on Train+Val...")
        X_tv = np.concatenate([X_train, X_val], axis=0)
        y_tv = np.concatenate([y_train, y_val], axis=0)

        best_iter = best_model.best_iteration
        final_n_est = max(
            best_iter + 20,
            int(best_iter * 1.1),
            50,
        )
        logger.info(f"  Using n_estimators={final_n_est} (best_iteration={best_iter})")

        final_params = {
            **best_params,
            "n_estimators": final_n_est,
            "early_stopping_rounds": 50,
        }
        final_model = XGBoostModel(**final_params)

        n_pv = max(100, int(len(X_tv) * 0.01))
        final_model.fit(X_tv[:-n_pv], y_tv[:-n_pv], X_tv[-n_pv:], y_tv[-n_pv:])

        return final_model, final_params

    if best_model is None:
        raise RuntimeError("No successful trials in train_tune.")

    return best_model, best_params


def run_train(args, features_dir, results_dir, ticker, tag):
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

    # Train the XGBoost model
    logger.info("\nTraining model...")
    t0 = time.time()
    if args.train_mode == "default":
        model, hyperparams = train_default(
            cfg, X_train_windows, y_train_windows, X_val_windows, y_val_windows
        )
    else:
        with tqdm(total=args.n_trials, desc="Tuning") as pbar:
            model, hyperparams = train_tune(
                args,
                X_train_windows,
                y_train_windows,
                X_val_windows,
                y_val_windows,
                pbar,
            )
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s")
    log_memory_usage("After training")

    y_pred_val = model.predict(X_val_windows)
    val_metrics = all_statistical_metrics(y_val_windows, y_pred_val)
    _print_stats_info_metrics("Training", val_metrics)

    model.save(str(results_dir / f"xgb_model_{tag}.ubj"))

    with open(results_dir / f"xgb_params_{tag}.json", "w") as f:
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

    with open(results_dir / f"xgb_history_{tag}.json", "w") as f:
        json.dump(model.history, f, indent=2)

    np.save(results_dir / f"xgb_importances_{tag}.npy", model.get_feature_importances())

    logger.info("Training complete.")


# ======================================================================
# Validation
# ======================================================================


def run_walk_forward_validation(
    model_template: XGBoostModel,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold_size: int,
    max_folds: int,
) -> List[dict]:
    """Expanding-window walk-forward validation using the validation set."""
    hparams = {
        k: v
        for k, v in model_template.get_params().items()
        if k in XGBoostModel.__init__.__code__.co_varnames
    }
    hparams_notune = {**hparams, "early_stopping_rounds": 50}

    N_val = len(X_val)
    fold_starts = list(range(0, N_val - fold_size, fold_size))[:max_folds]

    fold_results = []
    for fold_idx, start in enumerate(fold_starts):
        end = min(start + fold_size, N_val)
        X_fold_test = X_val[start:end]
        y_fold_test = y_val[start:end]

        X_tr = (
            np.concatenate([X_train_full, X_val[:start]], axis=0)
            if start > 0
            else X_train_full
        )
        y_tr = (
            np.concatenate([y_train_full, y_val[:start]], axis=0)
            if start > 0
            else y_train_full
        )

        n_pv = max(100, int(len(X_tr) * 0.05))

        if len(X_tr) < n_pv + 100:
            raise ValueError(
                f"Insufficient training data: {len(X_tr)} samples. "
                f"Need at least {n_pv + 100} for pseudo-validation split."
            )

        fold_model = XGBoostModel(**hparams_notune)
        fold_model.fit(X_tr[:-n_pv], y_tr[:-n_pv], X_tr[-n_pv:], y_tr[-n_pv:])

        y_pred_fold = fold_model.predict(X_fold_test)
        metrics = all_statistical_metrics(y_fold_test, y_pred_fold)
        _print_stats_info_metrics("Validation", metrics)
        fold_results.append(
            {
                "fold": fold_idx + 1,
                "start_bar": start,
                "end_bar": end,
                **metrics,
            }
        )

    return fold_results


def run_val(args, features_dir, results_dir, ticker, tag):
    cfg = load_config(args.config)

    if cfg is None:
        raise ValueError("Config is null")

    logger.info("\n[VALIDATION] Loading artefacts...")
    model, meta, _ = load_artefacts(results_dir, ticker, "train", args.seed)

    ticker_dir = Path(args.features_dir) / args.ticker
    if not ticker_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {ticker_dir}")

    X_train_flat, y_train, X_val_flat, y_val = load_training_data(ticker_dir=ticker_dir)

    logger.info("Loading features from %s", ticker_dir)
    feature_names = load_feature_names(features_dir, ticker)
    F = X_train_flat.shape[1] if X_train_flat.ndim > 1 else 1

    log_memory_usage("After loading data")

    max_lookback = getattr(cfg.xgboost, "lookback", 30)

    session_starts_train = np.zeros(len(X_train_flat), dtype=bool)
    session_starts_val = np.zeros(len(X_val_flat), dtype=bool)

    X_val_windows, y_val_windows = build_windows(
        X_val_flat, y_val, session_starts_val, max_lookback
    )

    y_pred_val = model.predict(X_val_windows)
    val_metrics = all_statistical_metrics(y_val_windows, y_pred_val)

    from src.optimizer.fitness import generate_signals, sharpe_from_signals

    thresholds = [0.0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    results = []

    for theta in thresholds:
        sigs = generate_signals(y_pred_val, theta)
        sr = sharpe_from_signals(sigs, y_val_windows)
        results.append({"threshold": theta, "sharpe": sr})

    best = max(results, key=lambda x: x["sharpe"])

    flat = model.get_feature_importances()
    orig = model.get_per_original_feature_importances(F)

    json.dump(
        val_metrics, open(results_dir / f"xgb_val_metrics_{tag}.json", "w"), indent=2
    )

    json.dump(
        {"optimal_threshold": best["threshold"], "all_results": results},
        open(results_dir / f"xgb_val_threshold_{tag}.json", "w"),
        indent=2,
    )

    json.dump(
        {"flat": flat.tolist(), "original": orig.tolist(), "features": feature_names},
        open(results_dir / f"xgb_val_importances_{tag}.json", "w"),
        indent=2,
    )

    logger.info("Validation complete.")


# ======================================================================
# Testing
# ======================================================================


def run_test(args, features_dir, results_dir, ticker, tag):
    logger.info("\n[TEST] Loading model...")
    model, meta = load_model(results_dir, ticker, "train", args.seed)
    theta = load_optimal_threshold(results_dir, ticker, "val", args.seed)

    _, _, _, _, X_test, y_test = load_windows(features_dir, ticker)

    y_pred = model.predict(X_test)

    stat_metrics = all_statistical_metrics(y_test, y_pred)

    opens, closes, ts = load_prices(ticker, str(features_dir), len(y_test))

    bt = Backtester(
        initial_capital=args.initial_capital,
        position_fraction=args.position_fraction,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        stop_loss=args.stop_loss,
        daily_loss_limit=args.daily_loss_limit,
        signal_threshold=theta,
    )

    result = bt.run(y_pred.copy(), opens, closes, ts)

    trading_metrics = {
        "sharpe": result.sharpe,
        "max_drawdown": result.mdd,
        "cagr": result.cagr_,
        "n_trades": result.n_trades,
    }

    json.dump(
        {"statistical": stat_metrics, "trading": trading_metrics},
        open(results_dir / f"xgb_test_metrics_{tag}.json", "w"),
        indent=2,
    )

    logger.info("Test complete.")


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
        "--mode",
        choices=["train", "val", "test"],
        required=True,
        default="train",
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
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--position-fraction", type=float, default=0.02)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--stop-loss", type=float, default=0.02)
    parser.add_argument("--daily-loss-limit", type=float, default=0.05)

    # Parse all arguments first
    args = parser.parse_args()

    # -------------------------
    # Filter arguments based on mode
    # -------------------------
    if args.mode == "train":
        # Remove validation/test-only args
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
        # Remove train/test-only args
        for attr in [
            "train_mode",
            "lookback",
            "n_trials",
            "retrain_on_trainval",
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
        # Remove train/validation-only args
        for attr in [
            "train_mode",
            "lookback",
            "n_trials",
            "retrain_on_trainval",
            "wfv_fold_size",
            "wfv_folds",
        ]:
            if hasattr(args, attr):
                delattr(args, attr)
    else:
        raise ValueError("Mode for either ['train','val','test'] not set. ")

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
    tag = f"{ticker}_{args.mode}_seed{args.seed}"

    logger.info(f"{'='*60}")
    logger.info(
        f"XGBoost Pipeline | mode={args.mode} | ticker={ticker} | seed={args.seed}"
    )
    logger.info(f"{'='*60}")

    # --------------------------------------------------
    # Mode dispatch
    # --------------------------------------------------
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
