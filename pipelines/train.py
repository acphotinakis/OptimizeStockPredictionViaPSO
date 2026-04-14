#!/usr/bin/env python3
"""
scripts/train.py

Train an LSTM or XGBoost model on a single ticker.

The script handles:
  - Feature loading (flat arrays from data/features/<ticker>/)
  - Sliding-window construction
  - Model construction from config + CLI overrides
  - Early-stopped training
  - Saving weights, hyperparameters, training history, and a quick plot

Usage:
    python scripts/train.py --model lstm    --ticker AAPL
    python scripts/train.py --model xgboost --ticker AAPL
    python scripts/train.py --model xgboost --ticker AAPL --train-mode tune --n-trials 30
    python scripts/train.py --model lstm    --ticker AAPL --hidden-units 256 --lookback 60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
import uuid

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path bootstrap (removed once the package is installed via pip install -e .)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from plots.lstm_history_plot import plot_training_history
from src.experiment.run_context import RuntimeContext
from src.data.splitter import build_windows
from src.evaluation.metrics import (
    compute_and_log_all_trading_metrics,
    compute_and_log_all_statistical_metrics,
)
from src.models.baselines import VanillaLSTM
from src.models.xgboost.xgboost_model import XGBoostModel, XGBoostTuner
from src.experiment.usage_enums import ModelType, Phase, RunMode, ArtifactType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LSTM training
# ---------------------------------------------------------------------------


def _lstm_hyperparams(args: argparse.Namespace, cfg) -> dict:
    """Merge config defaults with CLI overrides (CLI wins if not None)."""
    base = VanillaLSTM.DEFAULT_PARAMS.copy()
    lstm_cfg = getattr(cfg, "lstm_baseline", None)
    if lstm_cfg is not None:
        for k in base:
            if hasattr(lstm_cfg, k):
                base[k] = getattr(lstm_cfg, k)

    # CLI beats config
    overrides = {
        "num_layers": args.num_layers,
        "hidden_units": args.hidden_units,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "lookback": args.lookback,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
    }
    for k, v in overrides.items():
        if v is not None:
            base[k] = v

    return base


def train_lstm(ctx: RuntimeContext) -> None:
    logger.info("=== LSTM TRAINING  ticker=%s ===", ctx.ticker)

    X_train, y_train = ctx.tracker.load_split(Phase.TRAIN)
    X_val, y_val = ctx.tracker.load_split(Phase.VAL)

    hp = _lstm_hyperparams(ctx.args, ctx.cfg)
    logger.info("Hyperparameters: %s", hp)

    X_train, y_train_w = ctx.tracker._make_windows(X_train, y_train, hp["lookback"])
    X_val, y_val_w = ctx.tracker._make_windows(X_val, y_val, hp["lookback"])
    logger.info("Windowed — [X] train=%s  val=%s", X_train.shape, X_val.shape)
    logger.info("Windowed — [y] train=%s  val=%s", y_train_w.shape, y_val_w.shape)

    # sys.exit(0)
    model = VanillaLSTM(input_size=X_train.shape[2], device=ctx.args.device, **hp)

    t0 = time.time()
    history = model.fit(X_train, y_train_w, X_val, y_val_w)
    elapsed = time.time() - t0

    logger.info("Training complete in %.1fs", elapsed)

    ctx.tracker.save_history(Phase.TRAIN, history=history)

    metrics, y_true, y_pred = model.evaluate(X_val, y_val_w)

    ctx.tracker.save_metrics(Phase.VAL, metrics)
    ctx.tracker.save_predictions(Phase.VAL, y_pred)
    ctx.tracker.save_truth(Phase.VAL, y_true)

    ctx.tracker.save_torch(
        phase=Phase.TRAIN,
        obj=model._trainer.model.state_dict(),
        artifact=ArtifactType.MODEL,
    )

    ctx.tracker.save_json(
        ctx.tracker.artifact_path(f"lstm_params_{ctx.ticker}.json"),
        {
            "ticker": ctx.ticker,
            "model": "lstm",
            "seed": ctx.seed,
            "hyperparameters": hp,
            "val_metrics": metrics,
            "runtime_seconds": elapsed,
        },
    )

    logger.info("=== LSTM TRAINING DONE ===")


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------


def _xgb_hyperparams_from_cfg(cfg) -> dict:
    """Pull XGBoost hyperparameters from config, falling back to model defaults."""
    from src.models.xgboost.xgboost_model import _DEFAULT_PARAMS

    xgb_cfg = getattr(cfg, "xgboost", None)
    if xgb_cfg is None:
        return dict(_DEFAULT_PARAMS)

    hp = dict(_DEFAULT_PARAMS)
    for k in hp:
        if hasattr(xgb_cfg, k):
            hp[k] = getattr(xgb_cfg, k)
    return hp


def _xgb_train_default(
    cfg, X_train, y_train, X_val, y_val
) -> tuple[XGBoostModel, dict]:
    hp = _xgb_hyperparams_from_cfg(cfg)
    # lookback is structural, not an XGBRegressor kwarg — keep separate
    lookback = hp.pop("lookback", 30)
    model = XGBoostModel(lookback=lookback, **hp)
    model.fit(X_train, y_train, X_val, y_val)
    hp["lookback"] = lookback  # put it back for serialisation
    return model, hp


def _xgb_train_tune(args, X_train, y_train, X_val, y_val) -> tuple[XGBoostModel, dict]:
    """Random hyperparameter search using XGBoostTuner's public API."""
    tuner = XGBoostTuner(n_trials=args.n_trials, seed=args.seed)

    # Use the public fit() method — no private API access.
    # tqdm progress is handled inside the tuner.
    best_model, best_params = tuner.fit(
        X_train, y_train, X_val, y_val, lookback=args.lookback
    )

    # Log top-5 results
    sorted_results = sorted(tuner.results_, key=lambda r: r["val_rmse"])
    logger.info("Top-5 tuning results:")
    for rank, r in enumerate(sorted_results[:5], 1):
        logger.info(
            "  #%d  rmse=%.6f  depth=%s  lr=%s  n_est=%s",
            rank,
            r["val_rmse"],
            r["max_depth"],
            r["learning_rate"],
            r["n_estimators"],
        )

    # Optionally retrain best on Train+Val combined
    if args.retrain_on_trainval:
        logger.info("Retraining winner on Train+Val...")
        X_tv = np.concatenate([X_train, X_val], axis=0)
        y_tv = np.concatenate([y_train, y_val], axis=0)
        n_est = max(
            best_model.best_iteration + 20,
            int(best_model.best_iteration * 1.1),
            50,
        )
        # Small pseudo-val slice for early-stopping during combined fit
        n_pv = max(100, int(len(X_tv) * 0.01))
        final_hp = {**best_params, "n_estimators": n_est, "early_stopping_rounds": 50}
        lookback = final_hp.pop("lookback", args.lookback)
        final_model = XGBoostModel(lookback=lookback, **final_hp)
        final_model.fit(X_tv[:-n_pv], y_tv[:-n_pv], X_tv[-n_pv:], y_tv[-n_pv:])
        final_hp["lookback"] = lookback
        return final_model, final_hp

    return best_model, best_params


def train_xgboost(ctx: RuntimeContext) -> None:
    logger.info("=== XGBOOST TRAINING ticker=%s ===", ctx.ticker)

    X_train, y_train = ctx.tracker.load_split(Phase.TRAIN)
    X_val, y_val = ctx.tracker.load_split(Phase.VAL)

    hp = _xgb_hyperparams_from_cfg(ctx.cfg)
    logger.info("Hyperparameters: %s", hp)

    lookback = hp["lookback"]

    X_train, y_train_w = ctx.tracker._make_windows(X_train, y_train, lookback)
    X_val, y_val_w = ctx.tracker._make_windows(X_val, y_val, lookback)

    logger.info("Windowed — [X] train=%s val=%s", X_train.shape, X_val.shape)
    logger.info("Windowed — [y] train=%s val=%s", y_train_w.shape, y_val_w.shape)

    t0 = time.time()

    if ctx.args.train_mode == "default":
        model, hp = _xgb_train_default(ctx.cfg, X_train, y_train_w, X_val, y_val_w)
    else:
        model, hp = _xgb_train_tune(ctx.args, X_train, y_train_w, X_val, y_val_w)

    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)

    ctx.tracker.save_history(Phase.TRAIN, history=model.history)

    y_pred = model.predict(X_val)
    metrics = compute_and_log_all_statistical_metrics(y_val_w, y_pred)

    ctx.tracker.save_metrics(Phase.VAL, metrics)
    ctx.tracker.save_predictions(Phase.VAL, y_pred)

    # XGBoost uses its own binary format (.ubj), NOT torch.save()
    model_filename = ctx.tracker.make_filename(
        Phase.TRAIN, ArtifactType.MODEL, ext="ubj"
    )
    model_path = ctx.tracker.artifact_path(model_filename)
    model.save(str(model_path))
    logger.info("XGBoost model saved to %s", model_path)

    ctx.tracker.save_json(
        ctx.tracker.artifact_path(f"xgb_params_{ctx.ticker}.json"),
        {
            "ticker": ctx.ticker,
            "model": "xgboost",
            "seed": ctx.seed,
            "train_mode": ctx.args.train_mode,
            "hyperparameters": hp,
            "val_metrics": metrics,
            "runtime_seconds": elapsed,
        },
    )

    logger.info("=== XGBOOST TRAINING DONE ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LSTM or XGBoost on a single ticker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Required ----
    p.add_argument(
        "--model",
        required=True,
        choices=["lstm", "xgboost"],
        help="Model architecture to train",
    )
    p.add_argument(
        "--run-id",
        help="Model architecture to train",
    )

    p.add_argument("--ticker", required=True, help="Ticker symbol (e.g. AAPL)")
    # ---- Common ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default="config/default_config.yaml")
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--log-file", default="logs/train.log")

    # ---- LSTM hyperparameter overrides (all optional, config is baseline) ----
    lstm_grp = p.add_argument_group("LSTM overrides")
    lstm_grp.add_argument("--num-layers", type=int, default=None)
    lstm_grp.add_argument("--hidden-units", type=int, default=None)
    lstm_grp.add_argument("--dropout", type=float, default=None)
    lstm_grp.add_argument("--learning-rate", type=float, default=None)
    lstm_grp.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Lookback window (applies to both LSTM and XGBoost)",
    )
    lstm_grp.add_argument("--max-epochs", type=int, default=None)
    lstm_grp.add_argument("--patience", type=int, default=None)
    lstm_grp.add_argument("--batch-size", type=int, default=None)
    lstm_grp.add_argument(
        "--device", type=str, default=None, help="Torch device override (cpu/cuda)"
    )

    # ---- XGBoost-specific ----
    xgb_grp = p.add_argument_group("XGBoost options")
    xgb_grp.add_argument(
        "--train-mode",
        choices=["default", "tune"],
        default="default",
        help="'default' uses config params; 'tune' runs random search",
    )
    xgb_grp.add_argument(
        "--n-trials", type=int, default=20, help="Random search trials (tune mode only)"
    )
    xgb_grp.add_argument(
        "--retrain-on-trainval",
        action="store_true",
        default=True,
        help="After tuning, retrain winner on Train+Val",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------
    # Runtime initialization (single source of truth)
    # ---------------------------------------------------------
    ctx = RuntimeContext(
        args=args,
        model=ModelType(args.model),
        ticker=args.ticker,
        seed=args.seed,
        config_path=args.config,
        run_id=args.run_id,
        run_mode=RunMode.CREATE,
        log_file=args.log_file,
    )

    # ---------------------------------------------------------
    # Dispatch training
    # ---------------------------------------------------------
    if ctx.model == ModelType.LSTM:
        train_lstm(ctx)
    else:
        train_xgboost(ctx)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
