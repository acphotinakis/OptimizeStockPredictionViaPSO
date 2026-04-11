#!/usr/bin/env python3
"""
scripts/run_xgboost.py

XGBoost training, validation, and testing pipeline.

Usage:
    python scripts/run_xgboost.py --ticker AAPL --mode train
    python scripts/run_xgboost.py --ticker AAPL --mode train --train-mode tune
    python scripts/run_xgboost.py --ticker AAPL --mode val
    python scripts/run_xgboost.py --ticker AAPL --mode test
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common import (
    make_tag,
    make_windows,
    load_prices,
    load_optimal_threshold,
    load_xgb_model,
    save_json,
    setup,
)
from src.evaluation import (
    all_statistical_metrics,
    Backtester,
)
from src.models.xgboost.xgboost_model import XGBoostModel, XGBoostTuner
from src.optimizer.fitness import generate_signals, sharpe_from_signals
from src.utils.data_storage import load_feature_names, load_training_data
from src.evaluation.walk_forward import WalkForwardValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _train_default(cfg, X_train, y_train, X_val, y_val) -> Tuple[XGBoostModel, dict]:
    hp = {
        k: getattr(cfg.xgboost, k)
        for k in XGBoostModel.__init__.__code__.co_varnames
        if hasattr(cfg.xgboost, k)
    }
    model = XGBoostModel(**hp)
    model.fit(X_train, y_train, X_val, y_val)
    return model, hp


def _train_tune(args, X_train, y_train, X_val, y_val) -> Tuple[XGBoostModel, dict]:
    tuner = XGBoostTuner(n_trials=args.n_trials, seed=args.seed)
    best_rmse = float("inf")
    best_model = None
    best_params: dict = {}

    with tqdm(total=args.n_trials, desc="Tuning") as pbar:
        for i in range(args.n_trials):
            model, params = tuner._run_single_trial(
                X_train, y_train, X_val, y_val, lookback=args.lookback, trial_idx=i
            )
            pbar.update(1)
            rmse = (
                min(model.history["val_rmse"])
                if model.history["val_rmse"]
                else float("inf")
            )
            if rmse < best_rmse:
                best_rmse, best_model, best_params = rmse, model, params

    for rank, r in enumerate(
        sorted(tuner.results_, key=lambda x: x["val_rmse"])[:5], 1
    ):
        logger.info(
            "#%d  rmse=%.6f  depth=%s  lr=%s",
            rank,
            r["val_rmse"],
            r["max_depth"],
            r["learning_rate"],
        )

    if args.retrain_on_trainval and best_model is not None:
        logger.info("Retraining on Train+Val...")
        X_tv = np.concatenate([X_train, X_val])
        y_tv = np.concatenate([y_train, y_val])
        n_est = max(
            best_model.best_iteration + 20, int(best_model.best_iteration * 1.1), 50
        )
        final = XGBoostModel(
            **{**best_params, "n_estimators": n_est, "early_stopping_rounds": 50}
        )
        n_pv = max(100, int(len(X_tv) * 0.01))
        final.fit(X_tv[:-n_pv], y_tv[:-n_pv], X_tv[-n_pv:], y_tv[-n_pv:])
        return final, best_params

    if best_model is None:
        raise RuntimeError("No successful tuning trials.")
    return best_model, best_params


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------


def run_train(args: argparse.Namespace, cfg, results_dir: Path, tag: str) -> None:
    ticker_dir = Path(args.features_dir) / args.ticker
    X_train_flat, y_train, X_val_flat, y_val = load_training_data(ticker_dir=ticker_dir)

    lookback = getattr(cfg.xgboost, "lookback", 30)
    X_train, y_train_w = make_windows(X_train_flat, y_train, lookback)
    X_val, y_val_w = make_windows(X_val_flat, y_val, lookback)
    logger.info("X_train=%s  X_val=%s", X_train.shape, X_val.shape)

    t0 = time.time()
    if args.train_mode == "default":
        model, hp = _train_default(cfg, X_train, y_train_w, X_val, y_val_w)
    else:
        model, hp = _train_tune(args, X_train, y_train_w, X_val, y_val_w)
    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)

    val_metrics = all_statistical_metrics(y_val_w, model.predict(X_val))
    

    model.save(str(results_dir / f"xgb_model_{tag}.ubj"))
    save_json(
        results_dir / f"xgb_params_{tag}.json",
        {
            "ticker": args.ticker,
            "mode": args.train_mode,
            "seed": args.seed,
            "hyperparameters": hp,
            "best_iteration": model._best_iteration,
            "val_metrics": val_metrics,
            "runtime_seconds": elapsed,
        },
    )
    save_json(results_dir / f"xgb_history_{tag}.json", model.history)
    np.save(results_dir / f"xgb_importances_{tag}.npy", model.get_feature_importances())
    logger.info("Training complete.")


def run_val(args: argparse.Namespace, cfg, results_dir: Path, tag: str) -> None:
    logger.info("[VAL] Loading artefacts...")
    model, meta = load_xgb_model(results_dir, args.ticker, "train", args.seed)

    ticker_dir = Path(args.features_dir) / args.ticker
    X_train_flat, y_train, X_val_flat, y_val = load_training_data(ticker_dir=ticker_dir)

    lookback = getattr(cfg.xgboost, "lookback", 30)

    # Build windows (IMPORTANT: WFV expects full sequence)
    X_train, y_train_w = make_windows(X_train_flat, y_train, lookback)
    X_val, y_val_w = make_windows(X_val_flat, y_val, lookback)

    # Combine train + val for full timeline
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train_w, y_val_w], axis=0)

    logger.info("X_full=%s", X_full.shape)

    # -------------------------
    # Extract hyperparameters
    # -------------------------
    hparams = {
        k: v
        for k, v in model.get_params().items()
        if k in XGBoostModel.__init__.__code__.co_varnames
    }
    hparams["early_stopping_rounds"] = 50

    # -------------------------
    # Define retrain + predict
    # -------------------------
    def retrain_fn(X_tr, y_tr, X_val_fold, y_val_fold):
        n_pv = max(100, int(len(X_tr) * 0.05))

        m = XGBoostModel(**hparams)
        m.fit(
            X_tr[:-n_pv],
            y_tr[:-n_pv],
            X_tr[-n_pv:],
            y_tr[-n_pv:],
        )
        return m

    def predict_fn(model, X):
        return model.predict(X)

    # -------------------------
    # Walk-forward validation
    # -------------------------
    validator = WalkForwardValidator(
        fold_size_bars=args.wfv_fold_size,
        min_train_bars=len(X_train),  # start after original train split
        retrain_fn=retrain_fn,
        predict_fn=predict_fn,
    )

    logger.info("\n[VALIDATION] Running walk-forward validation...")
    wfv_results = validator.validate(X_full, y_full)

    # Pretty print mean metrics
    logger.info("\n[WFV] Aggregate Metrics:")

    # -------------------------
    # Threshold tuning (NO leakage)
    # -------------------------
    thresholds = [0.0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]

    # Collect fold predictions again (needed for threshold tuning)
    fold_preds = []
    N = len(X_full)

    for start in range(len(X_train), N - args.wfv_fold_size, args.wfv_fold_size):
        end = start + args.wfv_fold_size

        X_tr, y_tr = X_full[:start], y_full[:start]
        X_te, y_te = X_full[start:end], y_full[start:end]

        model_fold = retrain_fn(X_tr, y_tr, X_te, y_te)
        y_pred = predict_fn(model_fold, X_te)

        fold_preds.append((y_pred, y_te))

    sweep = []
    for t in thresholds:
        sharpes = [
            sharpe_from_signals(generate_signals(y_pred, t), y_true)
            for y_pred, y_true in fold_preds
        ]
        sweep.append({"threshold": t, "sharpe": float(np.mean(sharpes))})

    best = max(sweep, key=lambda x: x["sharpe"])

    # -------------------------
    # Save outputs
    # -------------------------
    feature_names = load_feature_names(Path(args.features_dir), args.ticker)

    save_json(
        results_dir / f"xgb_val_wfv_metrics_{tag}.json",
        wfv_results,
    )

    save_json(
        results_dir / f"xgb_val_threshold_{tag}.json",
        {
            "optimal_threshold": best["threshold"],
            "sweep": sweep,
        },
    )

    save_json(
        results_dir / f"xgb_val_importances_{tag}.json",
        {
            "flat": model.get_feature_importances().tolist(),
            "original": model.get_per_original_feature_importances(
                X_train_flat.shape[1]
            ).tolist(),
            "features": feature_names,
        },
    )

    logger.info("Validation complete. Best threshold=%.5f", best["threshold"])


def run_test(args: argparse.Namespace, results_dir: Path, tag: str) -> None:
    logger.info("[TEST] Loading model...")
    model, _ = load_xgb_model(results_dir, args.ticker, "train", args.seed)
    theta = load_optimal_threshold(results_dir, args.ticker, "val", args.seed)

    ticker_dir = Path(args.features_dir) / args.ticker
    X_test = np.load(ticker_dir / "X_test.npy")
    y_test = np.load(ticker_dir / "y_test.npy")
    y_pred = model.predict(X_test)

    stat_metrics = all_statistical_metrics(y_test, y_pred)
    opens, closes, ts = load_prices(args.ticker, Path(args.features_dir), len(y_test))

    result = Backtester(
        initial_capital=args.initial_capital,
        position_fraction=args.position_fraction,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        stop_loss=args.stop_loss,
        daily_loss_limit=args.daily_loss_limit,
        signal_threshold=theta,
    ).run(y_pred.copy(), opens, closes, ts)

    save_json(
        results_dir / f"xgb_test_metrics_{tag}.json",
        {
            "statistical": stat_metrics,
            "trading": {
                "sharpe": result.sharpe,
                "max_drawdown": result.mdd,
                "cagr": result.cagr_,
                "n_trades": result.n_trades,
            },
        },
    )
    logger.info("Test complete.")


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
    p.add_argument("--log-file", default="logs/run_xgboost.log")
    # Train
    p.add_argument("--train-mode", choices=["default", "tune"], default="default")
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--retrain-on-trainval", action="store_true", default=True)
    # Val
    p.add_argument("--wfv-fold-size", type=int, default=252)
    p.add_argument("--wfv-folds", type=int, default=10)
    # Test
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--position-fraction", type=float, default=0.02)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--stop-loss", type=float, default=0.02)
    p.add_argument("--daily-loss-limit", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = setup(args.log_file, args.seed, args.config)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = make_tag(args.ticker, args.mode, args.seed)

    logger.info(
        "XGBoost | mode=%s | ticker=%s | seed=%d", args.mode, args.ticker, args.seed
    )

    if args.mode == "train":
        run_train(args, cfg, results_dir, tag)
    elif args.mode == "val":
        run_val(args, cfg, results_dir, tag)
    elif args.mode == "test":
        run_test(args, results_dir, tag)


if __name__ == "__main__":
    main()
