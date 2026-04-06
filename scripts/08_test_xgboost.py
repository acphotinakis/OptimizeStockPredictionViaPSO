#!/usr/bin/env python3
"""
scripts/08_test_xgboost.py

Final held-out test evaluation for XGBoostModel.

This script is the XGBoost equivalent of scripts 04 + 05 combined:

  1. Load the trained booster from script 06.
  2. Load the optimal signal threshold from script 07.
  3. Evaluate all statistical metrics on the test set.
  4. Run the backtesting framework (same Backtester, same costs as script 05).
  5. Append XGBoost results to the existing model-comparison JSON produced
     by script 04 so that a single file holds all models side-by-side.
  6. Print the full cross-model comparison table (statistical + trading).

Outputs
-------
results/xgb_test_metrics_<TAG>.json
results/xgb_test_predictions_<TAG>.npz
results/xgb_test_backtest_<TAG>.json
results/model_comparison_<TICKER>_seed<SEED>.json   ← merged all-models file
results/trades_<TAG>_xgboost.csv

Usage
-----
    python scripts/08_test_xgboost.py --ticker AAPL --mode default --seed 42

    # With custom backtest costs
    python scripts/08_test_xgboost.py --ticker AAPL --mode tune --seed 42 \
        --transaction-cost 0.0015 --slippage 0.0008
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import set_all_seeds, setup_logger
from src.models.xgboost_model import XGBoostModel
from src.evaluation import all_statistical_metrics, all_trading_metrics, Backtester

FEATURES_DIR = "data/features/"
RESULTS_DIR = "results/"


# ======================================================================
# Argument parsing
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained XGBoostModel on the held-out test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--mode", choices=["default", "tune"], default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features-dir", default=FEATURES_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    # Backtester settings — kept identical to script 05 defaults
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--position-fraction", type=float, default=0.02)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--stop-loss", type=float, default=0.02)
    parser.add_argument("--daily-loss-limit", type=float, default=0.05)
    parser.add_argument("--log-file", default="logs/08_test_xgboost.log")
    return parser.parse_args()


# ======================================================================
# Helpers
# ======================================================================


def load_model(
    results_dir: Path, ticker: str, mode: str, seed: int
) -> tuple[XGBoostModel, dict]:
    tag = f"{ticker}_{mode}_seed{seed}"
    model_path = results_dir / f"xgb_model_{tag}.ubj"
    params_path = results_dir / f"xgb_params_{tag}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Booster not found: {model_path}\n"
            f"Run script 06 first: python scripts/06_train_xgboost.py "
            f"--ticker {ticker} --mode {mode}"
        )

    with open(params_path) as f:
        meta = json.load(f)

    hparams = meta["hyperparameters"]
    valid_keys = set(XGBoostModel.__init__.__code__.co_varnames)
    model = XGBoostModel(**{k: v for k, v in hparams.items() if k in valid_keys})
    model.load(str(model_path))
    return model, meta


def load_optimal_threshold(
    results_dir: Path, ticker: str, mode: str, seed: int
) -> float:
    """Load the threshold selected by script 07; fall back to 1e-4 if missing."""
    tag = f"{ticker}_{mode}_seed{seed}"
    path = results_dir / f"xgb_val_threshold_{tag}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        theta = float(data.get("optimal_threshold", 1e-4))
        print(f"  Optimal threshold (from script 07): {theta:.5f}")
        return theta
    print(
        "  WARNING: threshold file not found (run script 07 first). Falling back to θ=1e-4."
    )
    return 1e-4


def load_windows(features_dir: Path, ticker: str):
    arrays = {}
    for split in ("train", "val", "test"):
        for kind in ("X", "y"):
            key = f"{kind}_{split}"
            path = features_dir / f"{ticker}_{key}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}.")
            arrays[key] = np.load(path)
    return (
        arrays["X_train"],
        arrays["y_train"],
        arrays["X_val"],
        arrays["y_val"],
        arrays["X_test"],
        arrays["y_test"],
    )


def load_prices(
    ticker: str, features_dir: str, n_test: int
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load test-period prices; build synthetic fallback if raw is absent."""
    raw_path = Path("data/raw") / f"{ticker}.parquet"
    if not raw_path.exists():
        print("  WARNING: Raw price file not found — using synthetic prices.")
        y_test = np.load(Path(features_dir) / f"{ticker}_y_test.npy")
        prices = 100.0 * np.exp(np.cumsum(y_test))
        ts = pd.date_range(
            "2023-01-03 14:30", periods=len(prices), freq="1min", tz="UTC"
        )
        return prices, prices, ts

    from src.data import AlpacaIngestor, DataCleaner, DataSplitter

    raw = AlpacaIngestor.load_bars(raw_path)
    cleaner = DataCleaner()
    df = cleaner.clean(raw)
    splitter = DataSplitter(train_end="2022-01-03", val_end="2023-01-03")
    _, _, df_test = splitter.split(df)

    opens = df_test["open"].values.astype(np.float32)
    closes = df_test["close"].values.astype(np.float32)
    ts = df_test.index

    # Align lengths
    if len(opens) > n_test:
        opens = opens[-n_test:]
        closes = closes[-n_test:]
        ts = ts[-n_test:]

    return opens, closes, ts


# ======================================================================
# Cross-model comparison table helpers
# ======================================================================


def load_existing_results(results_dir: Path, ticker: str, seed: int) -> dict:
    """Load script-04 statistical metrics if available."""
    path = results_dir / f"metrics_{ticker}_seed{seed}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"ticker": ticker, "seed": seed, "models": {}}


def load_existing_backtest(results_dir: Path, ticker: str, seed: int) -> dict:
    """Load script-05 trading metrics if available."""
    path = results_dir / f"backtest_metrics_{ticker}_seed{seed}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _fmt(v, decimals: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    if isinstance(v, int):
        return str(v)
    return str(v)


def print_comparison_table(
    stat_results: dict,
    trading_results: dict,
    stat_cols: list,
    trade_cols: list,
) -> None:
    """Print a unified cross-model comparison table."""
    all_models = sorted(set(list(stat_results.keys()) + list(trading_results.keys())))
    col_w = 12

    # Header
    all_cols = stat_cols + trade_cols
    header = f"{'Model':<24}" + "".join(f"{c:>{col_w}}" for c in all_cols)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for model_name in all_models:
        row = f"{model_name:<24}"
        for col in stat_cols:
            v = stat_results.get(model_name, {}).get(col, float("nan"))
            row += f"{_fmt(v):>{col_w}}"
        for col in trade_cols:
            v = trading_results.get(model_name, {}).get(col, float("nan"))
            row += f"{_fmt(v):>{col_w}}"
        print(row)

    print("=" * len(header))


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
    model_tag = f"xgboost_{args.mode}"  # Key used in comparison tables

    print(f"\n{'='*60}")
    print(
        f"XGBoost Test Evaluation  |  ticker={ticker}  mode={args.mode}  seed={args.seed}"
    )
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load model, threshold, and data
    # ------------------------------------------------------------------
    print("\nLoading model and data...")
    model, meta = load_model(results_dir, ticker, args.mode, args.seed)
    theta = load_optimal_threshold(results_dir, ticker, args.mode, args.seed)

    (X_train, y_train, X_val, y_val, X_test, y_test) = load_windows(
        features_dir, ticker
    )

    print(f"  Test shape: {X_test.shape}")

    # ------------------------------------------------------------------
    # 2. Predictions
    # ------------------------------------------------------------------
    print("\nGenerating test-set predictions...")
    y_pred = model.predict(X_test)
    print(f"  Predictions shape: {y_pred.shape}")
    print(f"  Pred range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
    print(f"  True range: [{y_test.min():.6f}, {y_test.max():.6f}]")

    # ------------------------------------------------------------------
    # 3. Statistical metrics
    # ------------------------------------------------------------------
    print("\n[1/3] Statistical metrics (test set)...")
    stat_metrics = all_statistical_metrics(y_test, y_pred)

    print(f"  RMSE = {stat_metrics['rmse']:.6f}")
    print(f"  MAE  = {stat_metrics['mae']:.6f}")
    print(f"  MAPE = {stat_metrics['mape']:.4f}%")
    print(f"  R²   = {stat_metrics['r2']:.4f}")
    print(f"  DA   = {stat_metrics['directional_accuracy']:.4f}")
    print(f"  F1   = {stat_metrics['f1_ternary']:.4f}")
    print(f"  AUC  = {stat_metrics['auc_ternary']:.4f}")

    # ------------------------------------------------------------------
    # 4. Backtesting
    # ------------------------------------------------------------------
    print(
        f"\n[2/3] Backtesting (θ={theta:.5f}, TC={args.transaction_cost:.4f}, "
        f"slip={args.slippage:.4f})..."
    )

    opens, closes, timestamps = load_prices(ticker, str(features_dir), len(y_test))

    bt = Backtester(
        initial_capital=args.initial_capital,
        position_fraction=args.position_fraction,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        stop_loss=args.stop_loss,
        daily_loss_limit=args.daily_loss_limit,
        signal_threshold=theta,
    )
    bt_result = bt.run(y_pred.copy(), opens, closes, timestamps)

    trading_metrics = {
        "sharpe": bt_result.sharpe,
        "sortino": bt_result.sortino,
        "max_drawdown": bt_result.mdd,
        "cagr": bt_result.cagr_,
        "calmar": bt_result.calmar,
        "profit_factor": bt_result.profit_factor_,
        "win_rate": bt_result.win_rate_,
        "n_trades": bt_result.n_trades,
        "turnover": bt_result.turnover,
        "threshold_used": theta,
    }

    print(f"  Sharpe       = {bt_result.sharpe:.4f}")
    print(f"  Sortino      = {bt_result.sortino:.4f}")
    print(f"  Max Drawdown = {bt_result.mdd:.4f}")
    print(f"  CAGR         = {bt_result.cagr_:.4f}")
    print(f"  Calmar       = {bt_result.calmar:.4f}")
    print(f"  Profit Factor= {bt_result.profit_factor_:.4f}")
    print(f"  Win Rate     = {bt_result.win_rate_:.4f}")
    print(f"  Trades       = {bt_result.n_trades}")

    # ------------------------------------------------------------------
    # 5. Save XGBoost-specific outputs
    # ------------------------------------------------------------------
    test_metrics_path = results_dir / f"xgb_test_metrics_{tag}.json"
    predictions_path = results_dir / f"xgb_test_predictions_{tag}.npz"
    backtest_path = results_dir / f"xgb_test_backtest_{tag}.json"
    trades_path = results_dir / f"trades_{tag}_xgboost.csv"

    with open(test_metrics_path, "w") as f:
        json.dump(
            {
                "ticker": ticker,
                "mode": args.mode,
                "seed": args.seed,
                "statistical": stat_metrics,
                "trading": trading_metrics,
            },
            f,
            indent=2,
        )

    np.savez(
        predictions_path,
        y_test=y_test,
        y_pred=y_pred,
        equity_curve=bt_result.equity_curve,
    )

    with open(backtest_path, "w") as f:
        json.dump(
            {
                "ticker": ticker,
                "mode": args.mode,
                "seed": args.seed,
                **trading_metrics,
            },
            f,
            indent=2,
        )

    if not bt_result.trade_log.empty:
        bt_result.trade_log.to_csv(trades_path, index=False)
        print(f"  Trades saved: {trades_path}")

    # ------------------------------------------------------------------
    # 6. Merge into cross-model comparison file and print table
    # ------------------------------------------------------------------
    print("\n[3/3] Building cross-model comparison table...")

    existing_stat = load_existing_results(results_dir, ticker, args.seed)
    existing_trading = load_existing_backtest(results_dir, ticker, args.seed)

    # Merge XGBoost into existing dicts
    existing_stat.setdefault("models", {})[model_tag] = stat_metrics
    existing_trading[model_tag] = trading_metrics

    # Save merged comparison
    comparison_path = results_dir / f"model_comparison_{ticker}_seed{args.seed}.json"
    comparison_out = {
        "ticker": ticker,
        "seed": args.seed,
        "statistical": existing_stat.get("models", {}),
        "trading": existing_trading,
    }
    with open(comparison_path, "w") as f:
        json.dump(comparison_out, f, indent=2)

    # Print unified table
    stat_cols = ["rmse", "mae", "directional_accuracy", "f1_ternary", "r2"]
    trade_cols = ["sharpe", "max_drawdown", "cagr", "calmar", "n_trades"]

    print_comparison_table(
        stat_results=existing_stat.get("models", {}),
        trading_results=existing_trading,
        stat_cols=stat_cols,
        trade_cols=trade_cols,
    )

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\nSaved test metrics   : {test_metrics_path}")
    print(f"Saved predictions    : {predictions_path}")
    print(f"Saved backtest       : {backtest_path}")
    print(f"Saved comparison     : {comparison_path}")
    print("\nStep 8 (test_xgboost) complete.")


if __name__ == "__main__":
    main()
