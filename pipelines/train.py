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

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path bootstrap (removed once the package is installed via pip install -e .)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from plots.lstm_history_plot import plot_training_history
from src.data.splitter import build_windows
from src.evaluation.metrics import all_statistical_metrics
from src.models.baselines import VanillaLSTM
from src.models.xgboost.xgboost_model import XGBoostModel, XGBoostTuner
from src.utils.config_loader import load_config
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _setup_logging(log_file: str, level: str = "INFO") -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ]
    logging.basicConfig(level=getattr(logging, level), format=fmt, handlers=handlers)


def _load_split(ticker_dir: Path, split: str):
    """Load a single X/y split from disk."""
    return (
        np.load(ticker_dir / f"X_{split}.npy"),
        np.load(ticker_dir / f"y_{split}.npy"),
    )


def _make_windows(X_flat: np.ndarray, y: np.ndarray, lookback: int):
    """Apply sliding-window construction with no session boundaries."""
    session_starts = np.zeros(len(X_flat), dtype=bool)
    return build_windows(X_flat, y, session_starts, lookback)


def _save_json(path: Path, obj: dict) -> None:
    """JSON-serialise an object that may contain numpy scalars."""

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


def _log_metrics(label: str, m: dict) -> None:
    logger.info(
        "%s  RMSE=%.6f  DA=%.4f  F1=%.4f  R²=%.4f",
        label,
        m["rmse"],
        m["directional_accuracy"],
        m["f1_ternary"],
        m["r2"],
    )


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


def train_lstm(
    args: argparse.Namespace,
    cfg,
    ticker_dir: Path,
    results_dir: Path,
    tag: str,
) -> None:
    logger.info("=== LSTM TRAINING  ticker=%s ===", args.ticker)

    X_train, y_train = _load_split(ticker_dir, "train")
    X_val, y_val = _load_split(ticker_dir, "val")
    logger.info("Train Data  — X_train=%s  X_val=%s", X_train.shape, X_val.shape)
    logger.info("Val Data  — X_val=%s  y_val=%s", X_val.shape, y_val.shape)

    hp = _lstm_hyperparams(args, cfg)
    logger.info("Hyperparameters: %s", hp)

    X_train, y_train_w = _make_windows(X_train, y_train, hp["lookback"])
    X_val, y_val_w = _make_windows(X_val, y_val, hp["lookback"])
    logger.info("Windowed — [X] train=%s  val=%s", X_train.shape, X_val.shape)
    logger.info("Windowed — [y] train=%s  val=%s", y_train_w.shape, y_val_w.shape)

    # sys.exit(0)
    model = VanillaLSTM(input_size=X_train.shape[2], device=args.device, **hp)

    t0 = time.time()
    history = model.fit(X_train, y_train_w, X_val, y_val_w)
    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)

    plot_training_history(
        history=history,
        output_path="outputs/lstm_training_history.png",
        title="LSTM Training Metrics",
    )

    y_pred_val = model.predict(X_val)
    val_metrics = all_statistical_metrics(y_val_w, y_pred_val)
    _log_metrics("Val", val_metrics)

    # ---- Persist ----
    model_path = results_dir / f"lstm_model_{tag}.pth"
    torch.save(model._trainer.model.state_dict(), model_path)
    logger.info("Weights saved → %s", model_path)

    _save_json(
        results_dir / f"lstm_params_{tag}.json",
        {
            "ticker": args.ticker,
            "model": "lstm",
            "seed": args.seed,
            "hyperparameters": hp,
            "val_metrics": val_metrics,
            "runtime_seconds": elapsed,
        },
    )
    _save_json(results_dir / f"lstm_history_{tag}.json", history)
    np.save(results_dir / f"lstm_val_predictions_{tag}.npy", y_pred_val)

    # ---- Quick diagnostic plot ----
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set(title=f"{args.ticker} LSTM  Training Loss", xlabel="Epoch", ylabel="MSE")
    ax1.legend()
    ax1.grid(alpha=0.3)

    n = min(500, len(y_val_w))
    ax2.plot(y_val_w[:n], label="True", alpha=0.7)
    ax2.plot(y_pred_val[:n], label="Predicted", alpha=0.7)
    ax2.set(
        title=f"{args.ticker} LSTM  Val Predictions", xlabel="Sample", ylabel="Return"
    )
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    plot_path = plots_dir / f"lstm_training_{tag}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Training plot saved → %s", plot_path)

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


def train_xgboost(
    args: argparse.Namespace,
    cfg,
    ticker_dir: Path,
    results_dir: Path,
    tag: str,
) -> None:
    logger.info(
        "=== XGBOOST TRAINING  ticker=%s  mode=%s ===", args.ticker, args.train_mode
    )

    X_train_flat, y_train = _load_split(ticker_dir, "train")
    X_val_flat, y_val = _load_split(ticker_dir, "val")
    logger.info("Flat  — train=%s  val=%s", X_train_flat.shape, X_val_flat.shape)

    lookback = args.lookback or getattr(getattr(cfg, "xgboost", None), "lookback", 30)
    X_train, y_train_w = _make_windows(X_train_flat, y_train, lookback)
    X_val, y_val_w = _make_windows(X_val_flat, y_val, lookback)
    logger.info("Windowed — train=%s  val=%s", X_train.shape, X_val.shape)

    t0 = time.time()
    if args.train_mode == "default":
        model, hp = _xgb_train_default(cfg, X_train, y_train_w, X_val, y_val_w)
    else:
        model, hp = _xgb_train_tune(args, X_train, y_train_w, X_val, y_val_w)
    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)

    y_pred_val = model.predict(X_val)
    val_metrics = all_statistical_metrics(y_val_w, y_pred_val)
    _log_metrics("Val", val_metrics)

    # ---- Persist ----
    model_path = results_dir / f"xgb_model_{tag}.ubj"
    model.save(str(model_path))
    logger.info("Booster saved → %s", model_path)

    _save_json(
        results_dir / f"xgb_params_{tag}.json",
        {
            "ticker": args.ticker,
            "model": "xgboost",
            "train_mode": args.train_mode,
            "seed": args.seed,
            "hyperparameters": hp,
            "best_iteration": model.best_iteration,
            "val_metrics": val_metrics,
            "runtime_seconds": elapsed,
        },
    )
    _save_json(results_dir / f"xgb_history_{tag}.json", model.history)
    np.save(
        results_dir / f"xgb_importances_{tag}.npy",
        model.get_feature_importances(),
    )
    np.save(results_dir / f"xgb_val_predictions_{tag}.npy", y_pred_val)

    # ---- Quick diagnostic plot ----
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    train_rmse = model.history.get("train_rmse", [])
    val_rmse = model.history.get("val_rmse", [])
    rounds = range(1, len(val_rmse) + 1)
    if train_rmse:
        ax1.plot(list(rounds)[: len(train_rmse)], train_rmse, label="Train")
    if val_rmse:
        ax1.plot(rounds, val_rmse, label="Val")
        if model.best_iteration:
            ax1.axvline(
                model.best_iteration,
                color="red",
                linestyle="--",
                label=f"Best iter {model.best_iteration}",
            )
    ax1.set(
        title=f"{args.ticker} XGBoost  RMSE Curves",
        xlabel="Boosting Round",
        ylabel="RMSE",
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    n = min(500, len(y_val_w))
    ax2.plot(y_val_w[:n], label="True", alpha=0.7)
    ax2.plot(y_pred_val[:n], label="Predicted", alpha=0.7)
    ax2.set(
        title=f"{args.ticker} XGBoost  Val Predictions",
        xlabel="Sample",
        ylabel="Return",
    )
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    plot_path = plots_dir / f"xgb_training_{tag}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Training plot saved → %s", plot_path)

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
    _setup_logging(args.log_file)
    set_all_seeds(args.seed)

    cfg = load_config(args.config)

    ticker_dir = Path(args.features_dir) / args.ticker
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    if not ticker_dir.exists():
        logger.error(
            "Feature directory not found: %s\n" "Run build_features.py first.",
            ticker_dir,
        )
        sys.exit(1)

    tag = _make_tag(args.ticker, args.model, args.seed)
    logger.info("tag=%s", tag)

    if args.model == "lstm":
        train_lstm(args, cfg, ticker_dir, results_dir, tag)
    else:
        train_xgboost(args, cfg, ticker_dir, results_dir, tag)


if __name__ == "__main__":
    main()
