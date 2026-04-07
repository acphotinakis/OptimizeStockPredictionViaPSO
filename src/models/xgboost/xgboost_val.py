#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, Optional, Tuple, Any

from src.data.splitter import build_windows
from src.utils.config_loader import load_config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.xgboost.consts import PSEUDO_VAL_FRACTION, PSEUDO_VAL_MIN_SIZE
from src.models.xgboost.xgboost_model import (
    XGBoostModel,
    load_windows,
    load_feature_names,
)
from src.evaluation import all_statistical_metrics
from src.models.xgboost.helpers import (
    load_artefacts,
)

logger = logging.getLogger(__name__)

import psutil
import os


def log_memory_usage(label: str):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"[{label}] Memory: {mem_info.rss / 1024**3:.2f} GB")


# ======================================================================
# Validation routines
# ======================================================================
def run_walk_forward_validation(
    model_template: XGBoostModel,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold_size: int,
    max_folds: int,
) -> list:
    """Expanding-window walk-forward validation using the validation set.

    For each fold the model is retrained on (train + fold history) and
    evaluated on the next fold_size window of the validation set.

    Args:
        model_template: Fitted XGBoostModel (used only to copy hyperparams).
        X_train_full: Full training tensor [N_train, T, F].
        y_train_full: Full training targets [N_train].
        X_val:        Full validation tensor [N_val, T, F].
        y_val:        Full validation targets [N_val].
        fold_size:    Bars per test fold.
        max_folds:    Maximum number of folds to evaluate.

    Returns:
        List of per-fold metric dicts.
    """
    hparams = {
        k: v
        for k, v in model_template.get_params().items()
        if k in XGBoostModel.__init__.__code__.co_varnames
    }
    # Remove early_stopping_rounds for fold retraining (no separate val available)
    hparams_notune = {**hparams, "early_stopping_rounds": 50}

    N_val = len(X_val)
    fold_starts = list(range(0, N_val - fold_size, fold_size))[:max_folds]

    fold_results = []
    for fold_idx, start in enumerate(fold_starts):
        end = min(start + fold_size, N_val)
        X_fold_test = X_val[start:end]
        y_fold_test = y_val[start:end]

        # Training set: full original train + validation history up to this fold
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

        # Use a tiny slice at the end of X_tr as pseudo-val for early stopping
        # Calculate pseudo-validation size (5% of training, minimum 100 samples)
        n_pv = max(PSEUDO_VAL_MIN_SIZE, int(len(X_tr) * PSEUDO_VAL_FRACTION))

        # Ensure we have enough data
        if len(X_tr) < n_pv + 100:  # Need at least 100 training samples
            raise ValueError(
                f"Insufficient training data: {len(X_tr)} samples. "
                f"Need at least {n_pv + 100} for pseudo-validation split."
            )

        fold_model = XGBoostModel(**hparams_notune)
        fold_model.fit(X_tr[:-n_pv], y_tr[:-n_pv], X_tr[-n_pv:], y_tr[-n_pv:])

        y_pred_fold = fold_model.predict(X_fold_test)
        metrics = all_statistical_metrics(y_fold_test, y_pred_fold)
        fold_results.append(
            {
                "fold": fold_idx + 1,
                "start_bar": start,
                "end_bar": end,
                **metrics,
            }
        )
        logger.info(
            f"  Fold {fold_idx+1}/{len(fold_starts)} — "
            f"RMSE={metrics['rmse']:.6f}  DA={metrics['directional_accuracy']:.4f}"
        )

    return fold_results


# ======================================================================
# VALIDATION
# ======================================================================


def run_val(args, features_dir, results_dir, ticker, tag):
    cfg = load_config(args.config)

    if cfg is None:
        raise ValueError("Config is null")

    logger.info("\n[VALIDATION] Loading artefacts...")
    # Load model trained in 'train' mode, not 'val' mode
    model, meta, _ = load_artefacts(results_dir, ticker, "train", args.seed)

    # X_train, y_train, X_val, y_val, *_ = load_windows(features_dir, ticker)

    # # feature_names = load_feature_names(features_dir, ticker)
    # # _, _, F = X_train.shape

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

    # Get number of features
    F = X_train_flat.shape[1] if X_train_flat.ndim > 1 else 1

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
    X_val_windows, y_val_windows = build_windows(
        X_val_flat, y_val, session_starts_val, max_lookback
    )

    # ---------------------------
    # Metrics
    # ---------------------------
    y_pred_val = model.predict(X_val_windows)
    val_metrics = all_statistical_metrics(y_val_windows, y_pred_val)

    # ---------------------------
    # Threshold search
    # ---------------------------
    from src.optimizer.fitness import generate_signals, sharpe_from_signals

    thresholds = [0.0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    results = []

    for theta in thresholds:
        sigs = generate_signals(y_pred_val, theta)
        sr = sharpe_from_signals(sigs, y_val_windows)
        results.append({"threshold": theta, "sharpe": sr})

    best = max(results, key=lambda x: x["sharpe"])

    # ---------------------------
    # Importances
    # ---------------------------
    flat = model.get_feature_importances()
    orig = model.get_per_original_feature_importances(F)

    # ---------------------------
    # Save
    # ---------------------------
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
