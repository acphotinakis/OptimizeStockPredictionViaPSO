#!/usr/bin/env python3
"""
scripts/validate.py

Walk-forward validation for a trained LSTM or XGBoost model.

The validation protocol is *expanding-window*:
  - The full val set is split into folds of `--fold-size` bars.
  - For each fold the model is retrained from scratch on all available
    data up to that fold's boundary (train split + any prior val folds).
  - Predictions on the held-out fold are collected and scored.
  - A signal-threshold sweep is run across all fold predictions to find
    the threshold that maximises mean Sharpe — this becomes the threshold
    used at test time (no lookahead: threshold is chosen inside the val
    period, not on test data).

Outputs (all written to --results-dir):
  <model>_val_fold_metrics_<tag>.json  — per-fold statistical metrics
  <model>_val_aggregate_<tag>.json     — mean ± std across folds
  <model>_val_threshold_<tag>.json     — optimal signal threshold

Usage:
    python scripts/validate.py --model lstm    --ticker AAPL
    python scripts/validate.py --model xgboost --ticker AAPL --fold-size 5000
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
from src.evaluation.metrics import all_statistical_metrics
from src.models.baselines import VanillaLSTM
from src.models.xgboost.xgboost_model import XGBoostModel
from src.optimizer.fitness import generate_signals, sharpe_from_signals
from src.utils.config_loader import load_config
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _setup_logging(log_file: str, level: str = "INFO") -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level),
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )


def _load_split(ticker_dir: Path, split: str):
    return (
        np.load(ticker_dir / f"X_{split}.npy"),
        np.load(ticker_dir / f"y_{split}.npy"),
    )


def _make_windows(X_flat: np.ndarray, y: np.ndarray, lookback: int):
    session_starts = np.zeros(len(X_flat), dtype=bool)
    return build_windows(X_flat, y, session_starts, lookback)


def _save_json(path: Path, obj) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=_default))


def _make_tag(ticker: str, model: str, seed: int) -> str:
    return f"{ticker}_{model}_seed{seed}"


def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Compute mean ± std across folds for every scalar metric."""
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
    """
    Grid-search the signal threshold that maximises mean Sharpe across folds.

    Args:
        fold_predictions: List of (y_pred, y_true) pairs, one per fold.
        thresholds: Candidate threshold values.

    Returns:
        Dict with 'optimal_threshold' and full 'sweep' results.
    """
    sweep = []
    for theta in thresholds:
        sharpes = [
            sharpe_from_signals(generate_signals(y_pred, theta), y_true)
            for y_pred, y_true in fold_predictions
        ]
        sweep.append({"threshold": float(theta), "sharpe": float(np.mean(sharpes))})

    best = max(sweep, key=lambda x: x["sharpe"])
    return {"optimal_threshold": best["threshold"], "sweep": sweep}


# ---------------------------------------------------------------------------
# Core walk-forward loop (model-agnostic)
# ---------------------------------------------------------------------------


def _walk_forward(
    X_full: np.ndarray,
    y_full: np.ndarray,
    n_train: int,
    fold_size: int,
    max_folds: int,
    retrain_fn,
    predict_fn,
) -> tuple[list[dict], list[tuple]]:
    """
    Expanding-window walk-forward validation.

    Args:
        X_full:      [N, T, F] windowed feature tensor (train + val concatenated).
        y_full:      [N] targets.
        n_train:     Number of original training samples — first fold starts here.
        fold_size:   Number of bars per OOS fold.
        max_folds:   Maximum folds to run (0 = unlimited).
        retrain_fn:  Callable(X_train, y_train) → fitted model.
        predict_fn:  Callable(model, X) → np.ndarray predictions.

    Returns:
        fold_metrics:     List of per-fold metric dicts.
        fold_predictions: List of (y_pred, y_true) tuples for threshold sweep.
    """
    N = len(X_full)
    fold_starts = list(range(n_train, N - fold_size, fold_size))
    if max_folds > 0:
        fold_starts = fold_starts[:max_folds]

    if not fold_starts:
        raise ValueError(
            f"No folds possible: n_train={n_train}, N={N}, fold_size={fold_size}. "
            "Reduce --fold-size or increase the validation set."
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

        m = all_statistical_metrics(y_te, y_pred)
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


# ---------------------------------------------------------------------------
# LSTM walk-forward
# ---------------------------------------------------------------------------


def _load_lstm_checkpoint(
    results_dir: Path,
    ticker: str,
    seed: int,
    input_size: int,
) -> tuple[VanillaLSTM, dict]:
    """Load saved LSTM weights + hyperparameters from train.py output."""
    tag = _make_tag(ticker, "lstm", seed)
    model_path = results_dir / f"lstm_model_{tag}.pth"
    params_path = results_dir / f"lstm_params_{tag}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"LSTM checkpoint not found: {model_path}\n"
            "Run  python scripts/train.py --model lstm --ticker {ticker}  first."
        )

    params = json.loads(params_path.read_text())
    hp = params["hyperparameters"]

    model = VanillaLSTM(input_size=input_size, **hp)
    model._trainer.model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    return model, hp


def validate_lstm(
    args: argparse.Namespace,
    cfg,
    ticker_dir: Path,
    results_dir: Path,
    tag: str,
) -> None:
    logger.info("=== LSTM WALK-FORWARD VALIDATION  ticker=%s ===", args.ticker)

    X_train_flat, y_train = _load_split(ticker_dir, "train")
    X_val_flat, y_val = _load_split(ticker_dir, "val")

    # Load checkpoint to get hyperparams (especially lookback)
    _, hp = _load_lstm_checkpoint(
        results_dir, args.ticker, args.seed, X_train_flat.shape[1]
    )
    lookback = hp["lookback"]
    input_size = X_train_flat.shape[1]

    # Build windowed arrays for both splits
    X_train_w, y_train_w = _make_windows(X_train_flat, y_train, lookback)
    X_val_w, y_val_w = _make_windows(X_val_flat, y_val, lookback)

    # Concatenate so walk-forward has access to the full timeline
    X_full = np.concatenate([X_train_w, X_val_w], axis=0)
    y_full = np.concatenate([y_train_w, y_val_w], axis=0)
    n_train = len(X_train_w)

    # ---------- retrain / predict closures ----------
    def retrain_fn(X_tr: np.ndarray, y_tr: np.ndarray) -> VanillaLSTM:
        """Train a fresh LSTM on the expanding window."""
        m = VanillaLSTM(input_size=input_size, **hp)
        # Small pseudo-val from the tail of training data for early-stopping
        n_pv = max(200, int(len(X_tr) * 0.05))
        m.fit(X_tr[:-n_pv], y_tr[:-n_pv], X_tr[-n_pv:], y_tr[-n_pv:])
        return m

    def predict_fn(model: VanillaLSTM, X: np.ndarray) -> np.ndarray:
        return model.predict(X).flatten()

    # ---------- run ----------
    fold_metrics, fold_predictions = _walk_forward(
        X_full,
        y_full,
        n_train,
        fold_size=args.fold_size,
        max_folds=args.max_folds,
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

    # ---- Persist ----
    _save_json(results_dir / f"lstm_val_fold_metrics_{tag}.json", fold_metrics)
    _save_json(results_dir / f"lstm_val_aggregate_{tag}.json", agg)
    _save_json(results_dir / f"lstm_val_threshold_{tag}.json", threshold_result)

    logger.info("=== LSTM VALIDATION DONE ===")


# ---------------------------------------------------------------------------
# XGBoost walk-forward
# ---------------------------------------------------------------------------


def _load_xgb_checkpoint(
    results_dir: Path,
    ticker: str,
    seed: int,
) -> tuple[XGBoostModel, dict]:
    """Load saved XGBoost booster + hyperparameters from train.py output."""
    tag = _make_tag(ticker, "xgboost", seed)
    model_path = results_dir / f"xgb_model_{tag}.ubj"
    params_path = results_dir / f"xgb_params_{tag}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"XGBoost checkpoint not found: {model_path}\n"
            "Run  python scripts/train.py --model xgboost --ticker {ticker}  first."
        )

    saved = json.loads(params_path.read_text())
    hp = saved["hyperparameters"]

    lookback = hp.pop("lookback", 30)
    model = XGBoostModel(
        lookback=lookback, **{k: v for k, v in hp.items() if k != "lookback"}
    )
    model.load(str(model_path))
    hp["lookback"] = lookback  # restore for callers
    return model, hp


def validate_xgboost(
    args: argparse.Namespace,
    cfg,
    ticker_dir: Path,
    results_dir: Path,
    tag: str,
) -> None:
    logger.info("=== XGBOOST WALK-FORWARD VALIDATION  ticker=%s ===", args.ticker)

    X_train_flat, y_train = _load_split(ticker_dir, "train")
    X_val_flat, y_val = _load_split(ticker_dir, "val")

    _, hp = _load_xgb_checkpoint(results_dir, args.ticker, args.seed)
    lookback = hp.get("lookback", 30)

    X_train_w, y_train_w = _make_windows(X_train_flat, y_train, lookback)
    X_val_w, y_val_w = _make_windows(X_val_flat, y_val, lookback)

    X_full = np.concatenate([X_train_w, X_val_w], axis=0)
    y_full = np.concatenate([y_train_w, y_val_w], axis=0)
    n_train = len(X_train_w)

    # Pull XGB-specific params (drop structural ones that aren't XGBRegressor kwargs)
    xgb_hp = {k: v for k, v in hp.items() if k != "lookback"}

    # ---------- retrain / predict closures ----------
    def retrain_fn(X_tr: np.ndarray, y_tr: np.ndarray) -> XGBoostModel:
        """Train a fresh XGBoost on the expanding window."""
        n_pv = max(100, int(len(X_tr) * 0.05))
        m = XGBoostModel(lookback=lookback, **xgb_hp)
        m.fit(X_tr[:-n_pv], y_tr[:-n_pv], X_tr[-n_pv:], y_tr[-n_pv:])
        return m

    def predict_fn(model: XGBoostModel, X: np.ndarray) -> np.ndarray:
        return model.predict(X).flatten()

    # ---------- run ----------
    fold_metrics, fold_predictions = _walk_forward(
        X_full,
        y_full,
        n_train,
        fold_size=args.fold_size,
        max_folds=args.max_folds,
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

    # ---- Persist ----
    _save_json(results_dir / f"xgb_val_fold_metrics_{tag}.json", fold_metrics)
    _save_json(results_dir / f"xgb_val_aggregate_{tag}.json", agg)
    _save_json(results_dir / f"xgb_val_threshold_{tag}.json", threshold_result)

    logger.info("=== XGBOOST VALIDATION DONE ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward validation for a trained LSTM or XGBoost model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", required=True, choices=["lstm", "xgboost"])
    p.add_argument("--ticker", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default="config/default_config.yaml")
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--log-file", default="logs/validate.log")

    p.add_argument(
        "--fold-size",
        type=int,
        default=21 * 390,
        help="Bars per OOS fold (default ≈ 1 trading month of 1-min bars)",
    )
    p.add_argument(
        "--max-folds",
        type=int,
        default=0,
        help="Maximum folds to run (0 = run all available folds)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_file)
    set_all_seeds(args.seed)

    cfg = load_config(args.config)

    ticker_dir = Path(args.features_dir) / args.ticker
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    if not ticker_dir.exists():
        logger.error("Feature directory not found: %s", ticker_dir)
        sys.exit(1)

    tag = _make_tag(args.ticker, args.model, args.seed)
    logger.info("tag=%s  fold_size=%d", tag, args.fold_size)

    if args.model == "lstm":
        validate_lstm(args, cfg, ticker_dir, results_dir, tag)
    else:
        validate_xgboost(args, cfg, ticker_dir, results_dir, tag)


if __name__ == "__main__":
    main()
