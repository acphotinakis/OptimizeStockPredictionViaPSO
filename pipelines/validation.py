#!/usr/bin/env python3
"""
scripts/validate.py

Walk-forward validation using RuntimeContext.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.splitter import build_windows
from src.evaluation.metrics import compute_and_log_all_statistical_metrics
from src.models.baselines import VanillaLSTM
from src.models.xgboost.xgboost_model import XGBoostModel
from src.optimizer.fitness import generate_signals, sharpe_from_signals
from src.experiment.run_context import RuntimeContext
from src.experiment.usage_enums import ModelType, Phase, RunMode, ArtifactType

logger = logging.getLogger(__name__)


def _make_windows(X_flat: np.ndarray, y: np.ndarray, lookback: int):
    session_starts = np.zeros(len(X_flat), dtype=bool)
    return build_windows(X_flat, y, session_starts, lookback)


def _make_tag(ticker: str, model: str, seed: int) -> str:
    return f"{ticker}_{model}_seed{seed}"


def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Compute mean ± std across folds."""
    if not fold_metrics:
        return {}
    keys = [
        k
        for k, v in fold_metrics[0].items()
        if isinstance(v, (int, float)) and k != "fold"
    ]
    agg = {}
    for k in keys:
        vals = np.array([f[k] for f in fold_metrics if k in f])
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return agg


def _threshold_sweep(
    fold_predictions: list[tuple[np.ndarray, np.ndarray]],
    thresholds: np.ndarray,
) -> dict:
    """Grid-search the signal threshold."""
    sweep = []
    for theta in thresholds:
        sharpes = [
            sharpe_from_signals(generate_signals(y_pred, theta), y_true)
            for y_pred, y_true in fold_predictions
        ]
        sweep.append({"threshold": float(theta), "sharpe": float(np.mean(sharpes))})

    best = max(sweep, key=lambda x: x["sharpe"])
    return {"optimal_threshold": best["threshold"], "sweep": sweep}


def _walk_forward(
    X_full: np.ndarray,
    y_full: np.ndarray,
    n_train: int,
    fold_size: int,
    max_folds: int,
    retrain_fn,
    predict_fn,
) -> tuple[list[dict], list[tuple]]:
    """Expanding-window walk-forward validation."""
    N = len(X_full)
    fold_starts = list(range(n_train, N - fold_size, fold_size))
    if max_folds > 0:
        fold_starts = fold_starts[:max_folds]

    if not fold_starts:
        raise ValueError(
            f"No folds possible: n_train={n_train}, N={N}, fold_size={fold_size}."
        )

    logger.info(
        "Walk-forward: %d folds  fold_size=%d  first_train_size=%d",
        len(fold_starts),
        fold_size,
        n_train,
    )

    fold_metrics: list[dict] = []
    fold_predictions: list[tuple] = []

    for fold_idx, start in enumerate(fold_starts):
        end = min(start + fold_size, N)

        X_tr, y_tr = X_full[:start], y_full[:start]
        X_te, y_te = X_full[start:end], y_full[start:end]

        logger.info(
            "Fold %d/%d  train=%d  test=%d  [%d:%d]",
            fold_idx + 1,
            len(fold_starts),
            len(X_tr),
            len(X_te),
            start,
            end,
        )

        model = retrain_fn(X_tr, y_tr)
        y_pred = predict_fn(model, X_te)

        m = compute_and_log_all_statistical_metrics(y_te, y_pred)
        m["fold"] = fold_idx + 1
        m["start"] = int(start)
        m["end"] = int(end)
        fold_metrics.append(m)
        fold_predictions.append((y_pred, y_te))

        logger.info(
            "  Fold %d — RMSE=%.6f  DA=%.4f  F1=%.4f",
            fold_idx + 1,
            m["rmse"],
            m["directional_accuracy"],
            m["f1_ternary"],
        )

    return fold_metrics, fold_predictions


def _load_lstm_checkpoint(ctx: RuntimeContext, ticker: str, seed: int, input_size: int):
    """Load saved LSTM weights + hyperparameters."""
    tag = _make_tag(ticker, "lstm", seed)
    # Load from artifacts (training saves model_train_{tag}.pth)
    model_path = ctx.tracker.artifact_path(f"model_train_{tag}.pth")
    params_path = ctx.tracker.artifact_path(f"lstm_params_{ticker}.json")

    if not model_path.exists():
        raise FileNotFoundError(
            f"LSTM checkpoint not found: {model_path}\n"
            f"Run train.py first for {ticker}"
        )

    params = json.loads(params_path.read_text())
    hp = params["hyperparameters"]

    model = VanillaLSTM(input_size=input_size, device=ctx.args.device, **hp)
    model._trainer.model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    return model, hp


def validate_lstm(ctx: RuntimeContext) -> None:
    """LSTM walk-forward validation."""
    logger.info("=== LSTM WALK-FORWARD VALIDATION  ticker=%s ===", ctx.ticker)
    tag = _make_tag(ctx.ticker, ctx.model.value, ctx.seed)

    # Load data via tracker (train + val)
    X_train_flat, y_train = ctx.tracker.load_split(Phase.TRAIN)
    X_val_flat, y_val = ctx.tracker.load_split(Phase.VAL)

    # Load checkpoint to get hyperparams
    _, hp = _load_lstm_checkpoint(ctx, ctx.ticker, ctx.seed, X_train_flat.shape[1])
    lookback = hp["lookback"]
    input_size = X_train_flat.shape[1]

    # Build windows
    X_train_w, y_train_w = _make_windows(X_train_flat, y_train, lookback)
    X_val_w, y_val_w = _make_windows(X_val_flat, y_val, lookback)

    # Concatenate for walk-forward
    X_full = np.concatenate([X_train_w, X_val_w], axis=0)
    y_full = np.concatenate([y_train_w, y_val_w], axis=0)
    n_train = len(X_train_w)

    def retrain_fn(X_tr: np.ndarray, y_tr: np.ndarray) -> VanillaLSTM:
        """Train fresh LSTM on expanding window."""
        m = VanillaLSTM(input_size=input_size, device=ctx.args.device, **hp)
        n_pv = max(200, int(len(X_tr) * 0.05))
        m.fit(X_tr[:-n_pv], y_tr[:-n_pv], X_tr[-n_pv:], y_tr[-n_pv:])
        return m

    def predict_fn(model: VanillaLSTM, X: np.ndarray) -> np.ndarray:
        return model.predict(X).flatten()

    # Run walk-forward
    fold_metrics, fold_predictions = _walk_forward(
        X_full,
        y_full,
        n_train,
        fold_size=ctx.args.fold_size,
        max_folds=ctx.args.max_folds,
        retrain_fn=retrain_fn,
        predict_fn=predict_fn,
    )

    agg = _aggregate_fold_metrics(fold_metrics)
    logger.info(
        "Aggregate — RMSE=%.6f±%.6f  DA=%.4f±%.4f",
        agg["rmse"]["mean"],
        agg["rmse"]["std"],
        agg["directional_accuracy"]["mean"],
        agg["directional_accuracy"]["std"],
    )

    thresholds = np.arange(0.00005, 0.00205, 0.00005)
    threshold_result = _threshold_sweep(fold_predictions, thresholds)
    logger.info(
        "Optimal threshold=%.5f  Sharpe=%.3f",
        threshold_result["optimal_threshold"],
        max(r["sharpe"] for r in threshold_result["sweep"]),
    )

    # Save results via tracker
    ctx.tracker.save_json(
        ctx.tracker.phase_dir(Phase.VAL) / f"lstm_val_fold_metrics_{tag}.json",
        fold_metrics,
    )
    ctx.tracker.save_json(
        ctx.tracker.phase_dir(Phase.VAL) / f"lstm_val_aggregate_{tag}.json", agg
    )
    ctx.tracker.save_json(
        ctx.tracker.phase_dir(Phase.VAL) / f"lstm_val_threshold_{tag}.json",
        threshold_result,
    )

    logger.info("=== LSTM VALIDATION DONE ===")


def _load_xgb_checkpoint(ctx: RuntimeContext, ticker: str, seed: int):
    """Load saved XGBoost booster."""
    tag = _make_tag(ticker, "xgboost", seed)
    model_path = ctx.tracker.artifact_path(f"model_train_{tag}.ubj")
    params_path = ctx.tracker.artifact_path(f"xgb_params_{ticker}.json")

    if not model_path.exists():
        raise FileNotFoundError(
            f"XGBoost checkpoint not found: {model_path}\n"
            f"Run train.py first for {ticker}"
        )

    saved = json.loads(params_path.read_text())
    hp = saved["hyperparameters"]

    lookback = hp.pop("lookback", 30)
    model = XGBoostModel(
        lookback=lookback, **{k: v for k, v in hp.items() if k != "lookback"}
    )
    model.load(str(model_path))
    hp["lookback"] = lookback
    return model, hp


def validate_xgboost(ctx: RuntimeContext) -> None:
    """XGBoost walk-forward validation."""
    logger.info("=== XGBOOST WALK-FORWARD VALIDATION  ticker=%s ===", ctx.ticker)
    tag = _make_tag(ctx.ticker, ctx.model.value, ctx.seed)

    # Load data via tracker
    X_train_flat, y_train = ctx.tracker.load_split(Phase.TRAIN)
    X_val_flat, y_val = ctx.tracker.load_split(Phase.VAL)

    _, hp = _load_xgb_checkpoint(ctx, ctx.ticker, ctx.seed)
    lookback = hp.get("lookback", 30)

    X_train_w, y_train_w = _make_windows(X_train_flat, y_train, lookback)
    X_val_w, y_val_w = _make_windows(X_val_flat, y_val, lookback)

    X_full = np.concatenate([X_train_w, X_val_w], axis=0)
    y_full = np.concatenate([y_train_w, y_val_w], axis=0)
    n_train = len(X_train_w)

    xgb_hp = {k: v for k, v in hp.items() if k != "lookback"}

    def retrain_fn(X_tr: np.ndarray, y_tr: np.ndarray) -> XGBoostModel:
        """Train fresh XGBoost."""
        n_pv = max(100, int(len(X_tr) * 0.05))
        m = XGBoostModel(lookback=lookback, **xgb_hp)
        m.fit(X_tr[:-n_pv], y_tr[:-n_pv], X_tr[-n_pv:], y_tr[-n_pv:])
        return m

    def predict_fn(model: XGBoostModel, X: np.ndarray) -> np.ndarray:
        return model.predict(X).flatten()

    fold_metrics, fold_predictions = _walk_forward(
        X_full,
        y_full,
        n_train,
        fold_size=ctx.args.fold_size,
        max_folds=ctx.args.max_folds,
        retrain_fn=retrain_fn,
        predict_fn=predict_fn,
    )

    agg = _aggregate_fold_metrics(fold_metrics)
    logger.info(
        "Aggregate — RMSE=%.6f±%.6f  DA=%.4f±%.4f",
        agg["rmse"]["mean"],
        agg["rmse"]["std"],
        agg["directional_accuracy"]["mean"],
        agg["directional_accuracy"]["std"],
    )

    thresholds = np.arange(0.00005, 0.00205, 0.00005)
    threshold_result = _threshold_sweep(fold_predictions, thresholds)
    logger.info(
        "Optimal threshold=%.5f  Sharpe=%.3f",
        threshold_result["optimal_threshold"],
        max(r["sharpe"] for r in threshold_result["sweep"]),
    )

    # Save results via tracker
    ctx.tracker.save_json(
        ctx.tracker.phase_dir(Phase.VAL) / f"xgb_val_fold_metrics_{tag}.json",
        fold_metrics,
    )
    ctx.tracker.save_json(
        ctx.tracker.phase_dir(Phase.VAL) / f"xgb_val_aggregate_{tag}.json", agg
    )
    ctx.tracker.save_json(
        ctx.tracker.phase_dir(Phase.VAL) / f"xgb_val_threshold_{tag}.json",
        threshold_result,
    )

    logger.info("=== XGBOOST VALIDATION DONE ===")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward validation for LSTM or XGBoost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", required=True, choices=["lstm", "xgboost"])
    p.add_argument("--ticker", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default="config/default_config.yaml")
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--log-file", default="logs/validate.log")
    p.add_argument("--run-id", help="Run ID to attach to (from training)")

    p.add_argument(
        "--fold-size",
        type=int,
        default=21 * 390,
        help="Bars per OOS fold",
    )
    p.add_argument(
        "--max-folds",
        type=int,
        default=0,
        help="Maximum folds to run (0 = unlimited)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Initialize RuntimeContext
    ctx = RuntimeContext(
        args=args,
        model=ModelType(args.model),
        ticker=args.ticker,
        seed=args.seed,
        config_path=args.config,
        run_id=args.run_id,
        run_mode=RunMode.ATTACH,  # Validation attaches to training run
        log_file=args.log_file,
    )

    logger.info(
        "tag=%s  fold_size=%d",
        _make_tag(ctx.ticker, ctx.model.value, ctx.seed),
        ctx.args.fold_size,
    )

    if ctx.model == ModelType.LSTM:
        validate_lstm(ctx)
    else:
        validate_xgboost(ctx)


if __name__ == "__main__":
    main()
