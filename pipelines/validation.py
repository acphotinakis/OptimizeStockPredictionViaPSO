"""
scripts/validation.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.splitter import build_windows
from src.experiment.run_context import RuntimeContext
from src.evaluation.walk_forward import WalkForwardValidator
from src.optimizer.fitness import generate_signals, sharpe_from_signals

from src.models.lstm.inference import LSTMWrapper
from src.models.xgboost.model import XGBoostWrapper

logger = logging.getLogger(__name__)


def _make_windows(X_flat, y, lookback):
    session_starts = np.zeros(len(X_flat), dtype=bool)
    return build_windows(X_flat, y, session_starts, lookback)


def _threshold_sweep(fold_results, thresholds):
    sweep = []

    for theta in thresholds:
        sharpes = []

        for f in fold_results:
            sig = generate_signals(f["y_pred"], theta)
            sharpes.append(sharpe_from_signals(sig, f["y_true"]))

        sweep.append(
            {
                "threshold": float(theta),
                "sharpe": float(np.mean(sharpes)),
            }
        )

    best = max(sweep, key=lambda x: x["sharpe"])
    return {"optimal_threshold": best["threshold"], "sweep": sweep}


# =========================================================
# LSTM VALIDATION
# =========================================================


def validate_lstm(ctx: RuntimeContext) -> None:
    logger.info("=== LSTM WALK-FORWARD VALIDATION ===")

    saved = ctx.store.load_params()
    hp = saved["hyperparameters"]
    lookback = hp["lookback"]

    X_train_flat, y_train = ctx.store.load_split("train")
    X_val_flat, y_val = ctx.store.load_split("val")

    X_train_w, y_train_w = _make_windows(X_train_flat, y_train, lookback)
    X_val_w, y_val_w = _make_windows(X_val_flat, y_val, lookback)

    X_full = np.concatenate([X_train_w, X_val_w])
    y_full = np.concatenate([y_train_w, y_val_w])

    input_size = X_full.shape[2]

    def retrain_fn(X_tr, y_tr, X_v, y_v):
        n_pv = max(200, int(len(X_tr) * 0.05))
        model = LSTMWrapper(input_size=input_size, device=ctx.args.device, **hp)
        model.fit(X_tr[:-n_pv], y_tr[:-n_pv], X_tr[-n_pv:], y_tr[-n_pv:])
        return model

    validator = WalkForwardValidator(
        fold_size_bars=getattr(ctx.args, "fold_size", 21 * 390),
        min_train_bars=len(X_train_w),
        retrain_fn=retrain_fn,
        predict_fn=lambda m, X: m.predict(X),
    )

    agg, folds = validator.validate(X_full, y_full)

    thresholds = np.arange(0.00005, 0.00205, 0.00005)
    threshold_result = _threshold_sweep(folds, thresholds)

    assert folds is not None

    ctx.store.save_val_fold_metrics(folds)
    ctx.store.save_val_aggregate(agg)
    ctx.store.save_val_threshold(threshold_result)

    logger.info("=== LSTM VALIDATION DONE ===")


# =========================================================
# XGBOOST VALIDATION
# =========================================================


def validate_xgboost(ctx: RuntimeContext) -> None:
    logger.info("=== XGBOOST WALK-FORWARD VALIDATION ===")

    saved = ctx.store.load_params()
    hp = saved["hyperparameters"]
    lookback = hp.get("lookback", 30)
    xgb_hp = {k: v for k, v in hp.items() if k != "lookback"}

    X_train_flat, y_train = ctx.store.load_split("train")
    X_val_flat, y_val = ctx.store.load_split("val")

    X_train_w, y_train_w = _make_windows(X_train_flat, y_train, lookback)
    X_val_w, y_val_w = _make_windows(X_val_flat, y_val, lookback)

    X_full = np.concatenate([X_train_w, X_val_w])
    y_full = np.concatenate([y_train_w, y_val_w])

    def retrain_fn(X_tr, y_tr, X_v, y_v):
        n_pv = max(100, int(len(X_tr) * 0.05))
        model = XGBoostWrapper(lookback=lookback, **xgb_hp)
        model.fit(X_tr[:-n_pv], y_tr[:-n_pv], X_tr[-n_pv:], y_tr[-n_pv:])
        return model

    validator = WalkForwardValidator(
        fold_size_bars=getattr(ctx.args, "fold_size", 21 * 390),
        min_train_bars=len(X_train_w),
        retrain_fn=retrain_fn,
        predict_fn=lambda m, X: m.predict(X),
    )

    agg, folds = validator.validate(X_full, y_full)

    thresholds = np.arange(0.00005, 0.00205, 0.00005)
    threshold_result = _threshold_sweep(folds, thresholds)

    ctx.store.save_val_fold_metrics(folds)
    ctx.store.save_val_aggregate(agg)
    ctx.store.save_val_threshold(threshold_result)

    logger.info("=== XGBOOST VALIDATION DONE ===")


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", required=True, choices=["lstm", "xgboost"])
    p.add_argument("--ticker", required=True)
    p.add_argument("--run-id", required=True, help="Run ID from training step")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default="config/default_config.yaml")
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--log-file", default="logs/validation.log")
    p.add_argument("--fold-size", type=int, default=21 * 390)
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--device", default=None)
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
    if args.model == "lstm":
        validate_lstm(ctx)
    else:
        validate_xgboost(ctx)


if __name__ == "__main__":
    main()
