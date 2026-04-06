#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, Optional, Tuple, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.splitter import build_windows
from src.utils.config_loader import load_config
from src.models.xgboost_model import (
    XGBoostModel,
    XGBoostTuner,
    load_feature_names,
)
from src.evaluation import all_statistical_metrics
from src.utils.config_loader import Config
from helpers import (
    extract_hyperparameters,
)

logger = logging.getLogger(__name__)

from src.utils.config_loader import Config, load_config
from consts import *


import psutil
import os


def log_memory_usage(label: str):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"[{label}] Memory: {mem_info.rss / 1024**3:.2f} GB")


# Call at key points:
log_memory_usage("After loading data")
log_memory_usage("After windowing")
log_memory_usage("After training")

# ======================================================================
# Training routines
# ======================================================================


def train_default(
    cfg: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[XGBoostModel, Dict[str, Any]]:
    """Train one XGBoostModel with hyperparameters from the config."""
    # Pull hyperparameters directly from cfg.xgboost
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

    # Wrap the tuner trials in tqdm manually
    for trial_idx in range(args.n_trials):
        model, params = tuner._run_single_trial(
            X_train, y_train, X_val, y_val, lookback=args.lookback, trial_idx=trial_idx
        )
        pbar.update(1)

        # Track best model
        val_rmse = (
            min(model.history["val_rmse"])
            if model.history["val_rmse"]
            else float("inf")
        )
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = model
            best_params = params

    # Sort results and log top trials
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

        # Determine final number of estimators
        best_iter = best_model.best_iteration
        final_n_est = max(
            best_iter + FINAL_ESTIMATOR_BUFFER,
            int(best_iter * FINAL_ESTIMATOR_MULTIPLIER),
            MIN_FINAL_ESTIMATORS,
        )
        logger.info(f"  Using n_estimators={final_n_est} (best_iteration={best_iter})")

        # Final model hyperparameters
        final_params = {
            **best_params,
            "n_estimators": final_n_est,
            "early_stopping_rounds": 50,  # effectively no early stopping
        }
        final_model = XGBoostModel(**final_params)

        # Small pseudo-validation set
        n_pv = max(PSEUDO_VAL_MIN_SIZE, int(len(X_tv) * PSEUDO_VAL_FRACTION))
        final_model.fit(X_tv[:-n_pv], y_tv[:-n_pv], X_tv[-n_pv:], y_tv[-n_pv:])

        return final_model, final_params

    if best_model is None:
        raise RuntimeError("No successful trials in train_tune.")

    return best_model, best_params


# ======================================================================
# TRAIN
# ======================================================================


def run_train(args, features_dir, results_dir, ticker, tag):
    import time

    cfg = load_config(args.config)

    if cfg is None:
        raise ValueError("Config is null")

    logger.info("\n[TRAIN] Loading data...")
    # X_train, y_train, X_val, y_val, X_test, y_test = load_windows(features_dir, ticker)

    # Load feature data
    ticker_dir = Path(args.features_dir) / args.ticker
    if not ticker_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {ticker_dir}")

    logger.info("Loading features from %s", ticker_dir)

    # Use memory mapping for large files
    X_train_flat = np.load(ticker_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(ticker_dir / "y_train.npy", mmap_mode="r")
    X_val_flat = np.load(ticker_dir / "X_val.npy", mmap_mode="r")
    y_val = np.load(ticker_dir / "y_val.npy", mmap_mode="r")

    # Copy to writable arrays only when needed
    X_train_flat = np.array(X_train_flat)
    y_train = np.array(y_train)
    X_val_flat = np.array(X_val_flat)
    y_val = np.array(y_val)

    feature_names = load_feature_names(features_dir, ticker)

    log_memory_usage("After loading data")

    # ---------------------------
    # Raw arrays
    # ---------------------------
    logger.info("Raw Shapes:")
    logger.info("  X_train_flat: %s", X_train_flat.shape)
    logger.info("  y_train     : %s", y_train.shape)
    logger.info("  X_val_flat  : %s", X_val_flat.shape)
    logger.info("  y_val       : %s", y_val.shape)

    logger.info("Dtypes:")
    logger.info("  X_train_flat: %s", X_train_flat.dtype)
    logger.info("  y_train     : %s", y_train.dtype)
    logger.info("  X_val_flat  : %s", X_val_flat.dtype)
    logger.info("  y_val       : %s", y_val.dtype)

    # ---------------------------
    # Window setup
    # ---------------------------
    max_lookback = getattr(cfg.xgboost, "lookback", 30)

    session_starts_train = np.zeros(len(X_train_flat), dtype=bool)
    session_starts_val = np.zeros(len(X_val_flat), dtype=bool)

    logger.info("Session masks:")
    logger.info("  session_starts_train: %s", session_starts_train.shape)
    logger.info("  session_starts_val  : %s", session_starts_val.shape)

    logger.info("Lookback:")
    logger.info("  max_lookback: %d", max_lookback)

    # ---------------------------
    # Windowed arrays
    # ---------------------------
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

    # Optional: feature metadata
    logger.info("Feature Info:")
    logger.info(
        "  num_features: %d", X_train_flat.shape[1] if X_train_flat.ndim > 1 else 1
    )
    logger.info("  feature_names: %d", len(feature_names) if feature_names else 0)

    log_memory_usage("After windowing")

    # ---------------------------
    # Train
    # ---------------------------
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

    # ---------------------------
    # Validation quick check
    # ---------------------------
    y_pred_val = model.predict(X_val_windows)
    val_metrics = all_statistical_metrics(y_val, y_pred_val)
    _print_info_metrics("Validation", val_metrics)

    # ---------------------------
    # Save
    # ---------------------------
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


def _print_info_metrics(label: str, m: dict) -> None:
    """Pretty-logger.info a metrics dict in the same format as script 04."""
    logger.info(
        f"  {label} — "
        f"RMSE={m['rmse']:.6f}  "
        f"DA={m['directional_accuracy']:.4f}  "
        f"F1={m['f1_ternary']:.4f}  "
        f"R²={m['r2']:.4f}"
    )
