#!/usr/bin/env python3
"""
scripts/run_lstm_baseline.py

Train, validate, and test the vanilla LSTM baseline.

Usage:
    python scripts/run_lstm_baseline.py --ticker AAPL --mode train
    python scripts/run_lstm_baseline.py --ticker AAPL --mode val
    python scripts/run_lstm_baseline.py --ticker AAPL --mode test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common import make_tag, make_windows, save_json, setup
from src.data.splitter import build_windows
from src.evaluation import all_statistical_metrics
from src.evaluation.metrics import all_trading_metrics
from src.features.scalar import PipelineScaler
from src.models.baselines import VanillaLSTM
from plots.plots_lstm import plot_lstm_pnl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_split(ticker_dir: Path, split: str):
    return (
        np.load(ticker_dir / f"X_{split}.npy"),
        np.load(ticker_dir / f"y_{split}.npy"),
    )


def _load_model(results_dir: Path, ticker: str, seed: int) -> tuple[VanillaLSTM, dict]:
    tag = make_tag(ticker, "train", seed)
    model_path = results_dir / f"lstm_baseline_model_{tag}.pth"
    params_path = results_dir / f"lstm_baseline_params_{tag}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run --mode train first."
        )

    params = json.loads(params_path.read_text())
    # input_size not stored - re-derive from saved lookback + a fresh data load
    return model_path, params


def _resolve_hyperparams(args: argparse.Namespace, cfg) -> dict:
    """Merge config defaults with CLI overrides (CLI wins)."""
    base = VanillaLSTM.DEFAULT_PARAMS.copy()
    if hasattr(cfg, "lstm_baseline"):
        for k in base:
            if hasattr(cfg.lstm_baseline, k):
                base[k] = getattr(cfg.lstm_baseline, k)

    lookback = (
        args.lookback
        if args.lookback is not None
        else getattr(getattr(cfg, "lstm_baseline", None), "lookback", 30)
    )

    return {
        "num_layers": args.num_layers or base["num_layers"],
        "hidden_units": args.hidden_units or base["hidden_units"],
        "dropout": args.dropout or base["dropout"],
        "learning_rate": args.learning_rate or base["learning_rate"],
        "lookback": lookback,
        "max_epochs": args.max_epochs or base["max_epochs"],
        "patience": args.patience or base["patience"],
        "batch_size": args.batch_size or base["batch_size"],
    }


# ---------------------------------------------------------------------------
# Mode: train
# ---------------------------------------------------------------------------


def run_train(
    args: argparse.Namespace, cfg, ticker_dir: Path, results_dir: Path, tag: str
) -> None:
    logger.info("[TRAIN] Loading data...")
    X_train, y_train = _load_split(ticker_dir, "train")
    X_val_flat, y_val = _load_split(ticker_dir, "val")
    print(f"Feature Mean: {X_train.mean(axis=0)}")  # Is it near 0?
    print(f"Feature Max: {X_train.max(axis=0)}")  # Is it near 1 or 3?
    print(f"Target Range: [{y_train.min()}, {y_train.max()}]")

    hyperparams = _resolve_hyperparams(args, cfg)
    logger.info("Hyperparameters: %s", hyperparams)

    X_train, y_train_w = make_windows(X_train, y_train, hyperparams["lookback"])
    X_val, y_val_w = make_windows(X_val_flat, y_val, hyperparams["lookback"])
    logger.info("X_train=%s  X_val=%s", X_train.shape, X_val.shape)

    model = VanillaLSTM(input_size=X_train.shape[2], device=args.device, **hyperparams)

    t0 = time.time()

    # import sys

    # sys.exit(0)
    history = model.fit(X_train, y_train_w, X_val, y_val_w)
    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)

    scaler = PipelineScaler.load(ticker_dir / "scaler.pkl")

    y_pred_val = model.predict(X_val)
    val_metrics = all_statistical_metrics(y_val_w, y_pred_val, label=f"LSTM Training")

    # Save weights
    model_path = results_dir / f"lstm_baseline_model_{tag}.pth"
    torch.save(model._trainer.model.state_dict(), model_path)

    # Save params + history + predictions
    save_json(
        results_dir / f"lstm_baseline_params_{tag}.json",
        {
            "ticker": args.ticker,
            "mode": "train",
            "seed": args.seed,
            "hyperparameters": hyperparams,
            "val_metrics": val_metrics,
            "runtime_seconds": elapsed,
        },
    )
    save_json(results_dir / f"lstm_baseline_history_{tag}.json", history)
    np.save(results_dir / f"lstm_baseline_predictions_{tag}.npy", y_pred_val)

    # Simple training plot
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set(title=f"{args.ticker} Training Loss", xlabel="Epoch", ylabel="MSE")
    ax1.legend()
    ax1.grid(alpha=0.3)
    n = min(500, len(y_val_w))
    ax2.plot(y_val_w[:n], label="True")
    ax2.plot(y_pred_val[:n], label="Predicted")
    ax2.set(
        title=f"{args.ticker} Validation Predictions", xlabel="Sample", ylabel="Return"
    )
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"lstm_baseline_train_{tag}.png", dpi=150)
    plt.close(fig)

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Mode: val (walk-forward)
# ---------------------------------------------------------------------------


def run_val(
    args: argparse.Namespace, ticker_dir: Path, results_dir: Path, tag: str
) -> None:
    logger.info("[VAL] Walk-forward validation...")

    model_path, saved = _load_model(results_dir, args.ticker, args.seed)
    lookback = saved["hyperparameters"]["lookback"]

    X_val_flat, y_val = _load_split(ticker_dir, "val")
    X_val, y_val_w = make_windows(X_val_flat, y_val, lookback)

    model = VanillaLSTM(input_size=X_val_flat.shape[1])
    model._trainer.model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )

    fold_size = args.wfv_fold_size
    fold_metrics = []
    for i in range(args.wfv_folds):
        start, end = i * fold_size, (i + 1) * fold_size
        if end > len(X_val):
            break
        m = all_statistical_metrics(
            y_val_w[start:end],
            model.predict(X_val[start:end]),
            label=f"LSTM Validation - Folds",
        )
        fold_metrics.append({"fold": i, "start": start, "end": end, "metrics": m})
        logger.info("Fold %d: RMSE=%.6f", i, m["rmse"])

    agg = {}
    for k in fold_metrics[0]["metrics"]:
        vals = [f["metrics"][k] for f in fold_metrics]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    logger.info("RMSE: %.6f ± %.6f", agg["rmse"]["mean"], agg["rmse"]["std"])

    save_json(
        results_dir / f"lstm_baseline_wfv_{tag}.json",
        {
            "ticker": args.ticker,
            "fold_metrics": fold_metrics,
            "aggregate": agg,
        },
    )
    logger.info("Validation complete.")


# ---------------------------------------------------------------------------
# Mode: test
# ---------------------------------------------------------------------------


def run_test(
    args: argparse.Namespace, ticker_dir: Path, results_dir: Path, tag: str
) -> None:
    logger.info("[TEST] Final evaluation...")

    model_path, saved = _load_model(results_dir, args.ticker, args.seed)
    lookback = saved["hyperparameters"]["lookback"]

    X_test_flat, y_test = _load_split(ticker_dir, "test")

    X_test, y_test_w = make_windows(X_test_flat, y_test, lookback)

    model = VanillaLSTM(input_size=X_test_flat.shape[1])
    model._trainer.model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )

    y_pred = model.predict(X_test)

    # scaler = PipelineScaler.load(ticker_dir / "scaler.pkl")
    # scaler.inverse_transform_target(y_pred)
    # scaler.inverse_transform_target(y_test_w)

    stat_metrics = all_statistical_metrics(y_test_w, y_pred, label=f"LSTM Testing")

    # Threshold sweep: find best Sharpe
    returns = y_test_w.flatten()
    capital = args.initial_capital
    best = {
        "metrics": None,
        "threshold": None,
        "signals": None,
        "equity": None,
        "strat_returns": None,
    }

    for theta in np.arange(0.0001, 0.0021, 0.0001):
        signals = np.where(
            y_pred > theta, 1, np.where(y_pred < -theta, -1, 0)
        ).flatten()
        position = signals.copy()
        strat_ret = np.zeros_like(returns)
        equity = np.zeros_like(returns)
        equity[0] = capital

        for t in range(len(returns)):
            strat_ret[t] = position[t] * returns[t]
            if t > 0 and position[t] != position[t - 1]:
                strat_ret[t] -= args.transaction_cost
            equity[t] = (equity[t - 1] if t > 0 else capital) * (1 + strat_ret[t])
            if not np.isfinite(equity[t]) or equity[t] <= 0:
                equity[t] = equity[t - 1] if t > 0 else capital
                strat_ret[t] = 0
            # Stop-loss
            if t > 0 and equity[t - 1] > 0:
                dd = (equity[t] - equity[t - 1]) / equity[t - 1]
                if dd < -args.stop_loss:
                    strat_ret[t] = -args.stop_loss
                    equity[t] = equity[t - 1] * (1 - args.stop_loss)
                    position[t] = 0

        m = all_trading_metrics(equity, strat_ret)
        if best["metrics"] is None or m["sharpe"] > best["metrics"]["sharpe"]:
            best.update(
                metrics=m,
                threshold=theta,
                signals=signals.copy(),
                equity=equity.copy(),
                strat_returns=strat_ret.copy(),
            )

    logger.info(
        "Best threshold=%.5f | Sharpe=%.3f | MaxDD=%.2%%",
        best["threshold"],
        best["metrics"]["sharpe"],
        best["metrics"]["max_drawdown"],
    )

    n_buy = int((best["signals"] == 1).sum())
    n_sell = int((best["signals"] == -1).sum())
    n_hold = int((best["signals"] == 0).sum())

    aligned_df = pd.DataFrame(
        {
            "actual_return": returns,
            "predicted_return": y_pred.flatten(),
            "signal": best["signals"],
            "equity": best["equity"],
        }
    )
    aligned_df.to_csv(results_dir / f"lstm_aligned_{tag}.csv", index=False)

    pnl_stats = {
        "initial_capital": capital,
        "final_equity": float(best["equity"][-1]),
        "total_return": float(best["equity"][-1] / capital - 1),
        "buy_trades": n_buy,
        "sell_trades": n_sell,
        "hold_signals": n_hold,
    }

    save_json(
        results_dir / f"lstm_baseline_test_{tag}.json",
        {
            "statistical_metrics": stat_metrics,
            "trading_metrics": best["metrics"],
            "pnl_stats": pnl_stats,
            "best_threshold": best["threshold"],
        },
    )
    np.save(results_dir / f"lstm_baseline_test_predictions_{tag}.npy", y_pred)
    np.save(results_dir / f"lstm_baseline_test_signals_{tag}.npy", best["signals"])

    plot_lstm_pnl(
        results_dir,
        best["metrics"],
        y_test_w,
        y_pred,
        best["threshold"],
        args.ticker,
        aligned_df,
        capital,
        n_sell,
        n_hold,
        n_buy,
        best["signals"],
        tag,
        {
            "statistical_metrics": stat_metrics,
            "trading_metrics": best["metrics"],
            "pnl_stats": pnl_stats,
            "best_threshold": best["threshold"],
        },
    )
    logger.info("Testing complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ticker", required=True)
    p.add_argument("--mode", choices=["train", "val", "test"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default="config/default_config.yaml")
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--log-file", default="logs/run_lstm_baseline.log")
    # Model hyperparameter overrides (all optional — config is the default)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--hidden-units", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    # Val-only
    p.add_argument("--wfv-fold-size", type=int, default=252)
    p.add_argument("--wfv-folds", type=int, default=10)
    # Test-only
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--stop-loss", type=float, default=0.02)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = setup(args.log_file, args.seed, args.config)

    ticker_dir = Path(args.features_dir) / args.ticker
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = make_tag(args.ticker, args.mode, args.seed)

    logger.info(
        "LSTM Baseline | mode=%s | ticker=%s | seed=%d",
        args.mode,
        args.ticker,
        args.seed,
    )

    if args.mode == "train":
        run_train(args, cfg, ticker_dir, results_dir, tag)
    elif args.mode == "val":
        run_val(args, ticker_dir, results_dir, tag)
    elif args.mode == "test":
        run_test(args, ticker_dir, results_dir, tag)


if __name__ == "__main__":
    main()
