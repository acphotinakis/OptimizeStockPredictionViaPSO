"""
scripts/train.py

Thin training pipeline.  Loads data → builds model → trains → saves artifacts.
All business logic lives in src/; this file is pure orchestration.

Usage:
    python scripts/train.py --model lstm    --ticker AAPL
    python scripts/train.py --model xgboost --ticker AAPL
    python scripts/train.py --model xgboost --ticker AAPL --train-mode tune
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.splitter import build_windows
from src.evaluation.metrics import compute_and_log_all_statistical_metrics
from src.experiment.run_context import RuntimeContext
from src.models.lstm.inference import LSTMWrapper, DEFAULT_PARAMS as LSTM_DEFAULTS
from src.models.xgboost.model import XGBoostWrapper
from src.models.xgboost.trainer import XGBoostTuner

logger = logging.getLogger(__name__)


# ======================================================================
# Helpers
# ======================================================================


def _make_windows(X_flat, y, lookback):
    session_starts = np.zeros(len(X_flat), dtype=bool)
    return build_windows(X_flat, y, session_starts, lookback)


# ======================================================================
# LSTM
# ======================================================================


def train_lstm(ctx: RuntimeContext) -> None:
    logger.info("=== LSTM TRAINING  ticker=%s ===", ctx.ticker)

    X_train_flat, y_train = ctx.store.load_split("train")
    X_val_flat, y_val = ctx.store.load_split("val")

    # Merge config defaults with CLI overrides
    hp = {**LSTM_DEFAULTS}
    lstm_cfg = getattr(ctx.cfg, "lstm_baseline", None)
    if lstm_cfg:
        for k in hp:
            if hasattr(lstm_cfg, k):
                hp[k] = getattr(lstm_cfg, k)
    # CLI beats config
    args = ctx.args
    for attr, key in [
        ("num_layers", "num_layers"),
        ("hidden_units", "hidden_units"),
        ("dropout", "dropout"),
        ("learning_rate", "learning_rate"),
        ("lookback", "lookback"),
        ("max_epochs", "max_epochs"),
        ("patience", "patience"),
        ("batch_size", "batch_size"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            hp[key] = val

    lookback = hp["lookback"]
    X_train, y_train_w = _make_windows(X_train_flat, y_train, lookback)
    X_val, y_val_w = _make_windows(X_val_flat, y_val, lookback)
    logger.info("Windows — train=%s  val=%s", X_train.shape, X_val.shape)

    model = LSTMWrapper(
        input_size=X_train.shape[2],
        device=getattr(args, "device", None),
        **hp,
    )

    t0 = time.time()
    history = model.fit(X_train, y_train_w, X_val, y_val_w)
    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)

    y_pred_val = model.predict(X_val)
    val_metrics = compute_and_log_all_statistical_metrics(y_val_w, y_pred_val)

    # Save all artifacts
    ctx.store.save_model(model, ext="pth")
    ctx.store.save_params(
        hp, extra={"val_metrics": val_metrics, "runtime_seconds": elapsed}
    )
    ctx.store.save_train_history(history)
    ctx.store.save_train_predictions(y_pred_val, y_val_w)
    ctx.store.write_index()

    logger.info("=== LSTM TRAINING DONE  run_id=%s ===", ctx.run_id)


# ======================================================================
# XGBoost
# ======================================================================


def train_xgboost(ctx: RuntimeContext) -> None:
    logger.info("=== XGBOOST TRAINING  ticker=%s ===", ctx.ticker)

    X_train_flat, y_train = ctx.store.load_split("train")
    X_val_flat, y_val = ctx.store.load_split("val")

    # Hyperparameters from config
    xgb_cfg = getattr(ctx.cfg, "xgboost", None)
    base_cfg = getattr(ctx.cfg, "lstm_baseline", None)  # lookback lives here
    lookback = getattr(base_cfg, "lookback", 30)

    hp: dict = {}
    if xgb_cfg:
        from src.models.xgboost.model import DEFAULT_PARAMS as XGB_DEFAULTS

        for k in XGB_DEFAULTS:
            if hasattr(xgb_cfg, k):
                hp[k] = getattr(xgb_cfg, k)

    hp["lookback"] = lookback

    X_train, y_train_w = _make_windows(X_train_flat, y_train, lookback)
    X_val, y_val_w = _make_windows(X_val_flat, y_val, lookback)
    logger.info("Windows — train=%s  val=%s", X_train.shape, X_val.shape)

    args = ctx.args
    train_mode = getattr(args, "train_mode", "default")

    t0 = time.time()
    if train_mode == "tune":
        tuner = XGBoostTuner(
            n_trials=getattr(args, "n_trials", 20),
            seed=ctx.seed,
        )
        model, hp = tuner.fit(X_train, y_train_w, X_val, y_val_w, lookback=lookback)
    else:
        lookback_val = hp.pop("lookback", lookback)
        model = XGBoostWrapper(lookback=lookback_val, **hp)
        model.fit(X_train, y_train_w, X_val, y_val_w)
        hp["lookback"] = lookback_val

    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)

    y_pred_val = model.predict(X_val)
    val_metrics = compute_and_log_all_statistical_metrics(y_val_w, y_pred_val)

    ctx.store.save_model(model, ext="ubj")
    ctx.store.save_params(
        hp,
        extra={
            "val_metrics": val_metrics,
            "runtime_seconds": elapsed,
            "train_mode": train_mode,
        },
    )
    ctx.store.save_train_history(model.history)
    ctx.store.save_train_predictions(y_pred_val, y_val_w)
    ctx.store.write_index()

    logger.info("=== XGBOOST TRAINING DONE  run_id=%s ===", ctx.run_id)


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", required=True, choices=["lstm", "xgboost"])
    p.add_argument("--ticker", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default="config/default_config.yaml")
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--run-id", default=None)
    p.add_argument("--log-file", default="logs/train.log")
    p.add_argument("--device", default=None)

    lstm = p.add_argument_group("LSTM overrides")
    lstm.add_argument("--num-layers", type=int, default=None)
    lstm.add_argument("--hidden-units", type=int, default=None)
    lstm.add_argument("--dropout", type=float, default=None)
    lstm.add_argument("--learning-rate", type=float, default=None)
    lstm.add_argument("--lookback", type=int, default=None)
    lstm.add_argument("--max-epochs", type=int, default=None)
    lstm.add_argument("--patience", type=int, default=None)
    lstm.add_argument("--batch-size", type=int, default=None)

    xgb = p.add_argument_group("XGBoost options")
    xgb.add_argument("--train-mode", choices=["default", "tune"], default="default")
    xgb.add_argument("--n-trials", type=int, default=20)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    ctx = RuntimeContext(
        model=args.model,
        ticker=args.ticker,
        seed=args.seed,
        config_path=args.config,
        run_id=args.run_id,
        features_dir=args.features_dir,
        results_dir=args.results_dir,
        log_file=args.log_file,
        args=args,
    )
    logger.info("run_id=%s", ctx.run_id)

    if args.model == "lstm":
        train_lstm(ctx)
    else:
        train_xgboost(ctx)


if __name__ == "__main__":
    main()
