#!/usr/bin/env python3
"""
scripts/07_validate_xgboost.py

Validation analysis for a trained XGBoostModel:

  1. Reload the saved booster from script 06.
  2. Evaluate all statistical metrics on the validation set.
  3. Plot / export the training RMSE learning curve.
  4. Grid-search the optimal signal threshold on the validation set.
  5. Run walk-forward validation to check metric stability across time.
  6. Export per-original-feature importances.

Outputs
-------
results/xgb_val_metrics_<TAG>.json
results/xgb_val_history_<TAG>.json   (already written by script 06, echoed here)
results/xgb_val_importances_<TAG>.json   (original-feature level)
results/xgb_val_threshold_<TAG>.json
results/xgb_wfv_<TAG>.json           (walk-forward fold results)

Usage
-----
    python scripts/07_validate_xgboost.py --ticker AAPL --mode default --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import set_all_seeds, setup_logger
from src.models.xgboost_model import XGBoostModel
from src.evaluation import all_statistical_metrics, Backtester
from src.evaluation.metrics import rmse as rmse_fn

FEATURES_DIR = "data/features/"
RESULTS_DIR = "results/"


# ======================================================================
# Argument parsing
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a trained XGBoostModel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--mode", choices=["default", "tune"], default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features-dir", default=FEATURES_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    # Walk-forward settings
    parser.add_argument(
        "--wfv-fold-size",
        type=int,
        default=21 * 390,
        help="Bars per walk-forward fold (~1 month of 1-min bars).",
    )
    parser.add_argument(
        "--wfv-folds", type=int, default=6, help="Maximum number of walk-forward folds."
    )
    parser.add_argument("--log-file", default="logs/07_validate_xgboost.log")
    return parser.parse_args()


# ======================================================================
# Helpers
# ======================================================================


def load_artefacts(results_dir: Path, ticker: str, mode: str, seed: int):
    """Load booster and metadata written by script 06."""
    tag = f"{ticker}_{mode}_seed{seed}"

    model_path = results_dir / f"xgb_model_{tag}.ubj"
    params_path = results_dir / f"xgb_params_{tag}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Booster not found: {model_path}\n"
            f"Run script 06 first:\n"
            f"  python scripts/06_train_xgboost.py --ticker {ticker} --mode {mode}"
        )

    with open(params_path) as f:
        meta = json.load(f)

    hparams = meta["hyperparameters"]
    # Rebuild shell model with same constructor params, then load weights
    model = XGBoostModel(
        **{
            k: v
            for k, v in hparams.items()
            if k in XGBoostModel.__init__.__code__.co_varnames
        }
    )
    model.load(str(model_path))
    return model, meta, tag


def load_windows(features_dir: Path, ticker: str):
    prefix = features_dir
    arrays = {}
    for split in ("train", "val", "test"):
        for kind in ("X", "y"):
            key = f"{kind}_{split}"
            path = prefix / f"{ticker}_{key}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}. Run script 02 first.")
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
    path = features_dir / f"{ticker}_feature_names.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


# ======================================================================
# Walk-forward validation (XGBoost-specific: fast enough to retrain each fold)
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
    hparams_notune = {**hparams, "early_stopping_rounds": 9999}

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
        n_pv = max(100, len(X_tr) // 20)
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
        print(
            f"  Fold {fold_idx+1}/{len(fold_starts)} — "
            f"RMSE={metrics['rmse']:.6f}  DA={metrics['directional_accuracy']:.4f}"
        )

    return fold_results


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    args = parse_args()
    setup_logger(args.log_file)
    set_all_seeds(args.seed)

    features_dir = Path(args.features_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    ticker = args.ticker
    tag = f"{ticker}_{args.mode}_seed{args.seed}"

    print(f"\n{'='*60}")
    print(f"XGBoost Validation  |  ticker={ticker}  mode={args.mode}  seed={args.seed}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load data and model
    # ------------------------------------------------------------------
    print("\nLoading artefacts...")
    model, meta, _ = load_artefacts(results_dir, ticker, args.mode, args.seed)

    (X_train, y_train, X_val, y_val, X_test, y_test) = load_windows(
        features_dir, ticker
    )

    feature_names = load_feature_names(features_dir, ticker)
    F = X_train.shape[2]
    print(f"  Feature dim: {F}  ({len(feature_names)} named)")

    # ------------------------------------------------------------------
    # 2. Validation-set statistical metrics
    # ------------------------------------------------------------------
    print("\n[1/4] Validation-set statistical metrics...")
    y_pred_val = model.predict(X_val)
    val_metrics = all_statistical_metrics(y_val, y_pred_val)

    print(f"  RMSE = {val_metrics['rmse']:.6f}")
    print(f"  MAE  = {val_metrics['mae']:.6f}")
    print(f"  MAPE = {val_metrics['mape']:.4f}%")
    print(f"  R²   = {val_metrics['r2']:.4f}")
    print(f"  DA   = {val_metrics['directional_accuracy']:.4f}")
    print(f"  F1   = {val_metrics['f1_ternary']:.4f}")
    print(f"  AUC  = {val_metrics['auc_ternary']:.4f}")

    # ------------------------------------------------------------------
    # 3. Signal threshold grid-search on validation set
    # ------------------------------------------------------------------
    print("\n[2/4] Signal threshold optimisation on validation set...")
    threshold_candidates = [0.0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    threshold_results = []
    for theta in threshold_candidates:
        from src.optimizer.fitness import generate_signals, sharpe_from_signals

        sigs = generate_signals(y_pred_val, threshold=theta)
        sr = sharpe_from_signals(sigs, y_val)
        threshold_results.append({"threshold": theta, "sharpe": sr})
        print(f"  θ={theta:.5f}  Sharpe={sr:.4f}")

    best_theta_row = max(threshold_results, key=lambda r: r["sharpe"])
    best_theta = best_theta_row["threshold"]
    print(
        f"  → Optimal threshold: {best_theta:.5f}  (Sharpe={best_theta_row['sharpe']:.4f})"
    )

    # ------------------------------------------------------------------
    # 4. Feature importances (per original feature)
    # ------------------------------------------------------------------
    print("\n[3/4] Feature importances...")
    flat_importances = model.get_feature_importances()
    if F > 0:
        orig_importances = model.get_per_original_feature_importances(F)
        top_n = min(20, len(orig_importances))
        top_idx = np.argsort(orig_importances)[::-1][:top_n]
        print(f"  Top {top_n} original features (by mean gain across timesteps):")
        for rank, idx in enumerate(top_idx, 1):
            name = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
            print(f"    #{rank:2d}  {name:<30}  {orig_importances[idx]:.4f}")
    else:
        orig_importances = flat_importances

    # ------------------------------------------------------------------
    # 5. Walk-forward validation
    # ------------------------------------------------------------------
    print(
        f"\n[4/4] Walk-forward validation ({args.wfv_folds} folds, "
        f"fold_size={args.wfv_fold_size} bars)..."
    )
    wfv_results = run_walk_forward_validation(
        model_template=model,
        X_train_full=X_train,
        y_train_full=y_train,
        X_val=X_val,
        y_val=y_val,
        fold_size=args.wfv_fold_size,
        max_folds=args.wfv_folds,
    )

    # Aggregate WFV
    if wfv_results:
        metric_keys = [
            k for k in wfv_results[0] if k not in ("fold", "start_bar", "end_bar")
        ]
        print("\n  Walk-forward summary (mean ± std):")
        wfv_summary = {}
        for k in metric_keys:
            vals = np.array([r[k] for r in wfv_results])
            wfv_summary[f"{k}_mean"] = float(vals.mean())
            wfv_summary[f"{k}_std"] = float(vals.std())
            print(f"    {k:<30}  {vals.mean():.4f} ± {vals.std():.4f}")
    else:
        wfv_summary = {}

    # ------------------------------------------------------------------
    # 6. Save all outputs
    # ------------------------------------------------------------------
    val_metrics_path = results_dir / f"xgb_val_metrics_{tag}.json"
    threshold_path = results_dir / f"xgb_val_threshold_{tag}.json"
    importances_path = results_dir / f"xgb_val_importances_{tag}.json"
    wfv_path = results_dir / f"xgb_wfv_{tag}.json"

    with open(val_metrics_path, "w") as f:
        json.dump(
            {
                "ticker": ticker,
                "mode": args.mode,
                "seed": args.seed,
                "val_metrics": val_metrics,
            },
            f,
            indent=2,
        )

    with open(threshold_path, "w") as f:
        json.dump(
            {
                "ticker": ticker,
                "mode": args.mode,
                "seed": args.seed,
                "optimal_threshold": best_theta,
                "all_results": threshold_results,
            },
            f,
            indent=2,
        )

    importances_out = {
        "ticker": ticker,
        "flat_importances": flat_importances.tolist(),
        "per_original_feature": orig_importances.tolist(),
        "feature_names": feature_names,
    }
    with open(importances_path, "w") as f:
        json.dump(importances_out, f, indent=2)

    with open(wfv_path, "w") as f:
        json.dump(
            {
                "ticker": ticker,
                "mode": args.mode,
                "seed": args.seed,
                "fold_results": wfv_results,
                "summary": wfv_summary,
            },
            f,
            indent=2,
        )

    print(f"\nSaved validation metrics : {val_metrics_path}")
    print(f"Saved threshold results  : {threshold_path}")
    print(f"Saved importances        : {importances_path}")
    print(f"Saved walk-forward       : {wfv_path}")
    print("\nStep 7 (validate_xgboost) complete.")


if __name__ == "__main__":
    main()
