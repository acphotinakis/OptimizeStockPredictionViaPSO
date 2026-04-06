#!/usr/bin/env python3
"""
scripts/06_train_xgboost.py

Train an XGBoostModel on the pre-built feature windows and persist the
fitted booster and training history to disk.

Two operating modes:
  --mode default : Train once with the project-default hyperparameters
                   (n_estimators=500, max_depth=6, lr=0.05, ...).
                   Fastest option; good for comparison against LSTM.

  --mode tune    : Run a random hyperparameter search (n_trials trials)
                   using the validation set as the objective, then
                   retrain the winner on Train+Val before final save.

Outputs
-------
results/xgb_model_<TICKER>_<MODE>_seed<SEED>.ubj   (XGBoost binary booster)
results/xgb_params_<TICKER>_<MODE>_seed<SEED>.json  (hyperparameter dict)
results/xgb_history_<TICKER>_<MODE>_seed<SEED>.json (per-round RMSE)
results/xgb_importances_<TICKER>_<MODE>_seed<SEED>.npy (feature importances)

Usage
-----
    # Default hyperparameters
    python scripts/06_train_xgboost.py --ticker AAPL --mode default

    # Random hyperparameter search (20 trials)
    python scripts/06_train_xgboost.py --ticker AAPL --mode tune --n-trials 20

    # Custom lookback and estimators
    python scripts/06_train_xgboost.py --ticker AAPL --lookback 30 --n-estimators 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pickle


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_config
from src.utils import set_all_seeds, setup_logger
from src.models.xgboost_model import XGBoostModel, XGBoostTuner
from src.utils.config_loader import Config

FEATURES_DIR = "data/features/"
RESULTS_DIR = "results/"


# ======================================================================
# Argument parsing
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoostModel for stock return prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Target ticker symbol (must exist in features dir).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode",
        choices=["default", "tune"],
        default="default",
        help="Training mode: 'default' uses fixed hyperparameters; "
        "'tune' runs random search.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    # Data
    parser.add_argument(
        "--features-dir", default=FEATURES_DIR, help="Directory produced by script 02."
    )
    parser.add_argument(
        "--results-dir", default=RESULTS_DIR, help="Directory for output artefacts."
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Timesteps T to feed into the model (sliced from windows).",
    )
    # Default-mode hyperparameters

    # Tune-mode settings
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of random trials (tune mode only).",
    )
    parser.add_argument(
        "--retrain-on-trainval",
        action="store_true",
        default=True,
        help="After selecting best hyperparameters, retrain on "
        "Train+Val before saving (tune mode only).",
    )
    parser.add_argument("--log-file", default="logs/06_train_xgboost.log")
    return parser.parse_args()


# ======================================================================
# Helpers
# ======================================================================


def load_windows(features_dir: Path, ticker: str):
    """Load pre-built numpy arrays from script 02."""
    prefix = features_dir / ticker

    file_map = {
        "X_train": prefix / "X_train.npy",
        "X_val": prefix / "X_val.npy",
        "X_test": prefix / "X_test.npy",
        "y_train": prefix / "y_train.npy",
        "y_val": prefix / "y_val.npy",
        "y_test": prefix / "y_test.npy",
    }

    arrays = {}

    for key, path in file_map.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run script 02 first:\n"
                f"  python scripts/02_build_features.py --target {ticker}"
            )
        arrays[key] = np.load(path)

    return (
        arrays["X_train"],
        arrays["y_train"],
        arrays["X_val"],
        arrays["y_val"],
        arrays["X_test"],
        arrays["y_test"],
    )


def load_feature_names(features_dir: Path, ticker: str) -> list:
    """Load feature names from metadata.pkl for a given ticker."""
    path = features_dir / ticker / "metadata.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read or parse {path}: {e}")

    if "feature_names" not in data:
        raise KeyError(f"'feature_names' key missing in {path}")

    feature_names = data["feature_names"]

    if not isinstance(feature_names, list):
        raise TypeError(f"'feature_names' in {path} is not a list")

    return feature_names


# ======================================================================
# Training routines
# ======================================================================


def train_default(
    cfg: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple["XGBoostModel", dict]:
    """Train one XGBoostModel with hyperparameters from the config."""

    # Pull hyperparameters directly from cfg.xgboost
    hp = cfg.xgboost

    hyperparams = {
        "objective": hp.get("objective", "multi:softprob"),
        "num_class": hp.get("num_class", 3),
        "n_estimators": hp.get("n_estimators", 200),
        "max_depth": hp.get("max_depth", 4),
        "learning_rate": hp.get("learning_rate", 1e-5),
        "subsample": hp.get("subsample", 0.7),
        "colsample_bytree": hp.get("colsample_bytree", 0.6),
        "min_child_weight": hp.get("min_child_weight", 5),
        "gamma": hp.get("gamma", 0.1),
        "reg_alpha": hp.get("reg_alpha", 0.0),
        "reg_lambda": hp.get("reg_lambda", 1.0),
        "early_stopping_rounds": hp.get("early_stopping_rounds", 50),
        "random_state": getattr(cfg, "seed", 42),
        "tree_method": hp.get("tree_method", "hist"),
        "max_bin": hp.get("max_bin", 128),
    }

    print(f"Hyperparameters: {hyperparams}")

    model = XGBoostModel(**hyperparams)
    model.fit(X_train, y_train, X_val, y_val)

    return model, hyperparams


def train_tune(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[XGBoostModel, dict]:
    """Run random hyperparameter search, then optionally retrain winner on Train+Val."""
    tuner = XGBoostTuner(n_trials=args.n_trials, seed=args.seed)
    best_model, best_params = tuner.fit(
        X_train, y_train, X_val, y_val, lookback=args.lookback
    )

    # Print search summary
    print(f"\nTuner results ({args.n_trials} trials):")
    sorted_results = sorted(tuner.results_, key=lambda r: r["val_rmse"])
    for rank, r in enumerate(sorted_results[:5], 1):
        print(
            f"  #{rank}  val_rmse={r['val_rmse']:.6f}  "
            f"max_depth={r['max_depth']}  lr={r['learning_rate']}  "
            f"n_est={r['n_estimators']}"
        )
    print(f"\nBest params: {best_params}")

    if args.retrain_on_trainval:
        print("\nRetraining best model on Train+Val...")
        X_tv = np.concatenate([X_train, X_val], axis=0)
        y_tv = np.concatenate([y_train, y_val], axis=0)

        # For the final model we cannot use early stopping (no hold-out left),
        # so set n_estimators to best_iteration from the tuning run, or a
        # fixed fraction above it for a small boost.
        best_iter = best_model._best_iteration
        final_n_est = max(best_iter + 20, int(best_iter * 1.1), 50)
        print(f"  Using n_estimators={final_n_est} (best_iteration={best_iter})")

        final_params = {
            **best_params,
            "n_estimators": final_n_est,
            "early_stopping_rounds": 9999,
        }  # effectively no early stopping
        final_model = XGBoostModel(**final_params)
        # Use a tiny pseudo-val (last 1 % of trainval) to satisfy the API
        n_pv = max(100, len(X_tv) // 100)
        final_model.fit(X_tv[:-n_pv], y_tv[:-n_pv], X_tv[-n_pv:], y_tv[-n_pv:])
        return final_model, final_params

    return best_model, best_params


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    setup_logger(args.log_file)
    set_all_seeds(args.seed)

    features_dir = Path(args.features_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    ticker = args.ticker
    tag = f"{ticker}_{args.mode}_seed{args.seed}"

    print(f"\n{'='*60}")
    print(f"XGBoost Training  |  ticker={ticker}  mode={args.mode}  seed={args.seed}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\nLoading feature windows from {features_dir}/...")
    (X_train, y_train, X_val, y_val, X_test, y_test) = load_windows(
        features_dir, ticker
    )

    print(f"X_train Data: {X_train.shape}")

    # import sys

    # sys.exit(0)

    feature_names = load_feature_names(features_dir, ticker)
    F = X_train.shape[1]

    print(f"  Train : {X_train.shape}  ({y_train.shape[0]} targets)")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")
    print(
        f"  Features (F) : {F}"
        + (f"  ({len(feature_names)} named)" if feature_names else "")
    )

    # ------------------------------------------------------------------
    # 2. Train
    # ------------------------------------------------------------------
    print(f"\n[{args.mode.upper()}] Starting training...")
    t0 = time.time()

    if args.mode == "default":
        model, hyperparams = train_default(cfg, X_train, y_train, X_val, y_val)
    else:
        model, hyperparams = train_tune(X_train, y_train, X_val, y_val)

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Best iteration : {model._best_iteration}")
    if model.history["val_rmse"]:
        print(f"  Best val RMSE  : {min(model.history['val_rmse']):.6f}")

    # ------------------------------------------------------------------
    # 3. Quick validation-set check
    # ------------------------------------------------------------------
    print("\nValidation-set quick check...")
    from src.evaluation import all_statistical_metrics

    y_pred_val = model.predict(X_val)
    val_metrics = all_statistical_metrics(y_val, y_pred_val)
    _print_metrics("Validation", val_metrics)

    # ------------------------------------------------------------------
    # 4. Save artefacts
    # ------------------------------------------------------------------
    model_path = results_dir / f"xgb_model_{tag}.ubj"
    params_path = results_dir / f"xgb_params_{tag}.json"
    history_path = results_dir / f"xgb_history_{tag}.json"
    importances_path = results_dir / f"xgb_importances_{tag}.npy"

    model.save(str(model_path))

    params_out = {
        "ticker": ticker,
        "mode": args.mode,
        "seed": args.seed,
        "hyperparameters": hyperparams,
        "best_iteration": model._best_iteration,
        "val_metrics": val_metrics,
        "runtime_seconds": elapsed,
    }
    with open(params_path, "w") as f:
        json.dump(params_out, f, indent=2)

    history_out = {
        "train_rmse": model.history.get("train_rmse", []),
        "val_rmse": model.history.get("val_rmse", []),
    }
    with open(history_path, "w") as f:
        json.dump(history_out, f, indent=2)

    importances = model.get_feature_importances()
    np.save(importances_path, importances)

    print(f"\nSaved booster    : {model_path}")
    print(f"Saved params     : {params_path}")
    print(f"Saved history    : {history_path}")
    print(f"Saved importances: {importances_path}")
    print("\nStep 6 (train_xgboost) complete.")


def _print_metrics(label: str, m: dict) -> None:
    """Pretty-print a metrics dict in the same format as script 04."""
    print(
        f"  {label} — "
        f"RMSE={m['rmse']:.6f}  "
        f"DA={m['directional_accuracy']:.4f}  "
        f"F1={m['f1_ternary']:.4f}  "
        f"R²={m['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
