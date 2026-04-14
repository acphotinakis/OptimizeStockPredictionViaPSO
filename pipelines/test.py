#!/usr/bin/env python3
"""
scripts/test.py

Final evaluation on the held-out test set for a trained LSTM or XGBoost model.

Steps:
  1. Load the trained model checkpoint (from train.py).
  2. Load the signal threshold chosen on the validation set (from validate.py).
  3. Generate predictions on the test set.
  4. Compute statistical metrics (RMSE, DA, F1, …).
  5. Run the full Backtester (fills at next-bar open, slippage, TC, stop-loss,
     daily loss limit, session-end forced flat).
  6. Save all metrics and a summary equity-curve plot.

Outputs (written to --results-dir):
  <model>_test_stat_metrics_<tag>.json   — prediction quality
  <model>_test_trading_metrics_<tag>.json — Sharpe, MDD, CAGR, …
  <model>_test_trade_log_<tag>.csv       — every trade entry/exit
  plots/<model>_test_equity_<tag>.png    — equity curve vs buy-and-hold

Usage:
    python scripts/test.py --model lstm    --ticker AAPL
    python scripts/test.py --model xgboost --ticker AAPL
    python scripts/test.py --model xgboost --ticker AAPL --initial-capital 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import _log_results
from src.data.splitter import build_windows
from src.evaluation.backtester import Backtester
from src.evaluation.metrics import (
    compute_and_log_all_statistical_metrics,
)
from src.models.baselines import VanillaLSTM
from src.models.xgboost.xgboost_model import XGBoostModel
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


def _make_tag(ticker: str, model: str, seed: int) -> str:
    return f"{ticker}_{model}_seed{seed}"


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


def _make_windows(X_flat: np.ndarray, y: np.ndarray, lookback: int):
    session_starts = np.zeros(len(X_flat), dtype=bool)
    return build_windows(X_flat, y, session_starts, lookback)


def _load_threshold(results_dir: Path, ticker: str, model: str, seed: int) -> float:
    """
    Load the optimal signal threshold chosen during validation.
    Falls back to 1e-4 with a warning if the file is absent (run validate.py first).
    """
    tag = _make_tag(ticker, model, seed)
    path = results_dir / f"{model}_val_threshold_{tag}.json"

    if path.exists():
        data = json.loads(path.read_text())
        theta = float(data.get("optimal_threshold", 1e-4))
        logger.info("Signal threshold loaded from validation: %.5f", theta)
        return theta

    logger.warning(
        "Threshold file not found: %s — falling back to θ=1e-4. "
        "Run  python scripts/validate.py --model %s --ticker %s  for a proper threshold.",
        path,
        model,
        ticker,
    )
    return 1e-4


def _load_prices(
    ticker: str, features_dir: Path, n_test: int
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Load open/close prices and UTC timestamps for the test period.

    Tries raw/<ticker>.parquet first; falls back to synthetic prices derived
    from the y_test log-return series if the raw file is absent.
    """
    raw_path = Path("data/raw") / f"{ticker}.parquet"

    if raw_path.exists():
        try:
            from src.data.alpaca_ingestor import AlpacaIngestor
            from src.data.cleaner import DataCleaner
            from src.data.splitter import DataSplitter

            raw = AlpacaIngestor.load_bars(raw_path)
            df = DataCleaner().clean(raw)
            cfg = load_config("config/default_config.yaml")
            _, _, df_test = DataSplitter(
                train_end=cfg.data.train_end,
                val_end=cfg.data.val_end,
            ).split(df)

            opens = df_test["open"].to_numpy(dtype=np.float32)
            closes = df_test["close"].to_numpy(dtype=np.float32)
            ts = df_test.index
            if not isinstance(ts, pd.DatetimeIndex):
                ts = pd.DatetimeIndex(ts)

            # Align to test window length
            if len(opens) > n_test:
                opens, closes, ts = opens[-n_test:], closes[-n_test:], ts[-n_test:]

            logger.info("Prices loaded from raw parquet: %d bars", len(opens))
            return opens, closes, ts
        except Exception as exc:
            logger.warning("Could not load raw prices (%s) — using synthetic.", exc)

    # --- Synthetic fallback ---
    logger.warning("Using synthetic prices derived from log-return targets.")
    y_test = np.load(features_dir / ticker / "y_test.npy")
    base = 100.0
    closes = (base * np.exp(np.cumsum(y_test))).astype(np.float32)
    rng = np.random.default_rng(42)
    opens = (np.roll(closes, 1) * (1 + rng.normal(0, 0.0001, len(closes)))).astype(
        np.float32
    )
    opens[0] = base
    ts = pd.DatetimeIndex(
        pd.date_range("2024-01-02 14:30", periods=len(closes), freq="1min", tz="UTC")
    )
    return opens[:n_test], closes[:n_test], ts[:n_test]


# ---------------------------------------------------------------------------
# Backtesting + plotting
# ---------------------------------------------------------------------------


def _run_backtest(
    y_pred: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    ts: pd.DatetimeIndex,
    args: argparse.Namespace,
    theta: float,
):
    """Instantiate and run the Backtester, return a BacktestResult."""
    bt = Backtester(
        initial_capital=args.initial_capital,
        position_fraction=args.position_fraction,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        stop_loss=args.stop_loss,
        daily_loss_limit=args.daily_loss_limit,
        signal_threshold=theta,
    )
    return bt.run(y_pred.copy(), opens, closes, ts)


def _plot_equity(
    result,
    ticker: str,
    model: str,
    closes: np.ndarray,
    initial_capital: float,
    save_path: Path,
) -> None:
    """
    3-panel summary plot:
      1. Strategy equity curve vs simple buy-and-hold
      2. Drawdown
      3. Per-bar return distribution
    """
    equity = result.equity_curve
    bar_ret = result.bar_returns
    bah_curve = initial_capital * np.exp(
        np.cumsum(np.log(closes[1:] / closes[:-1] + 1e-10))
    )
    bah_curve = np.concatenate([[initial_capital], bah_curve])[: len(equity)]

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1.5, 1.5]}
    )
    fig.suptitle(
        f"{ticker} — {model.upper()} Test Evaluation\n"
        f"Sharpe={result.sharpe:.3f}  MDD={result.mdd:.2%}  "
        f"CAGR={result.cagr_:.2%}  Trades={result.n_trades}",
        fontsize=13,
        fontweight="bold",
    )

    # ---- Panel 1: Equity ----
    ax = axes[0]
    x = np.arange(len(equity))
    ax.plot(x, equity, color="royalblue", linewidth=1.5, label="Strategy")
    ax.plot(
        x,
        bah_curve,
        color="grey",
        linewidth=1.0,
        linestyle="--",
        alpha=0.7,
        label="Buy-and-hold",
    )
    ax.axhline(initial_capital, color="black", linewidth=0.8, linestyle=":")
    ax.fill_between(
        x,
        initial_capital,
        equity,
        where=(equity >= initial_capital),
        color="green",
        alpha=0.15,
    )
    ax.fill_between(
        x,
        initial_capital,
        equity,
        where=(equity < initial_capital),
        color="red",
        alpha=0.15,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # ---- Panel 2: Drawdown ----
    ax = axes[1]
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-10) * 100
    ax.fill_between(x, 0, dd, color="red", alpha=0.5)
    ax.plot(x, dd, color="darkred", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)

    # ---- Panel 3: Return distribution ----
    ax = axes[2]
    nonzero = bar_ret[bar_ret != 0]
    ax.hist(nonzero * 100, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(
        nonzero.mean() * 100,
        color="green",
        linewidth=1.5,
        linestyle="--",
        label=f"Mean {nonzero.mean()*100:.3f}%",
    )
    ax.set_xlabel("Bar Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Equity plot saved --> %s", save_path)


# ---------------------------------------------------------------------------
# LSTM test
# ---------------------------------------------------------------------------


def test_lstm(
    args: argparse.Namespace,
    ticker_dir: Path,
    results_dir: Path,
    tag: str,
) -> None:
    logger.info("=== LSTM TEST  ticker=%s ===", args.ticker)

    # ---- Load checkpoint ----
    model_path = results_dir / f"lstm_model_{tag}.pth"
    params_path = results_dir / f"lstm_params_{tag}.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            "Run  python scripts/train.py --model lstm --ticker {args.ticker}"
        )
    hp = json.loads(params_path.read_text())["hyperparameters"]

    # ---- Load test data ----
    X_test_flat = np.load(ticker_dir / "X_test.npy")
    y_test = np.load(ticker_dir / "y_test.npy")

    X_test, y_test_w = _make_windows(X_test_flat, y_test, hp["lookback"])
    logger.info("Test windows: %s", X_test.shape)

    # ---- Reconstruct model and load weights ----
    model = VanillaLSTM(input_size=X_test_flat.shape[1], **hp)
    model._trainer.model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )

    # ---- Predictions ----
    y_pred = model.predict(X_test).flatten()

    # ---- Statistical metrics ----
    stat_metrics = compute_and_log_all_statistical_metrics(y_test_w, y_pred)

    # ---- Backtest ----
    theta = _load_threshold(results_dir, args.ticker, "lstm", args.seed)
    opens, closes, ts = _load_prices(args.ticker, ticker_dir.parent, len(y_pred))
    result = _run_backtest(y_pred, opens, closes, ts, args, theta)

    _log_results(stat_metrics, result, theta)

    # ---- Persist ----
    _save_json(results_dir / f"lstm_test_stat_metrics_{tag}.json", stat_metrics)
    _save_json(
        results_dir / f"lstm_test_trading_metrics_{tag}.json",
        {
            "signal_threshold": theta,
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "max_drawdown": result.mdd,
            "cagr": result.cagr_,
            "calmar": result.calmar,
            "profit_factor": result.profit_factor_,
            "win_rate": result.win_rate_,
            "n_trades": result.n_trades,
            "turnover": result.turnover,
        },
    )
    if not result.trade_log.empty:
        result.trade_log.to_csv(
            results_dir / f"lstm_test_trade_log_{tag}.csv", index=False
        )

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    _plot_equity(
        result,
        args.ticker,
        "lstm",
        closes,
        args.initial_capital,
        plots_dir / f"lstm_test_equity_{tag}.png",
    )

    logger.info("=== LSTM TEST DONE ===")


# ---------------------------------------------------------------------------
# XGBoost test
# ---------------------------------------------------------------------------


def test_xgboost(
    args: argparse.Namespace,
    ticker_dir: Path,
    results_dir: Path,
    tag: str,
) -> None:
    logger.info("=== XGBOOST TEST  ticker=%s ===", args.ticker)

    # ---- Load checkpoint ----
    model_path = results_dir / f"xgb_model_{tag}.ubj"
    params_path = results_dir / f"xgb_params_{tag}.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            "Run  python scripts/train.py --model xgboost --ticker {args.ticker}"
        )
    saved = json.loads(params_path.read_text())
    hp = dict(saved["hyperparameters"])
    lookback = hp.pop("lookback", 30)

    model = XGBoostModel(lookback=lookback, **{k: v for k, v in hp.items()})
    model.load(str(model_path))

    # ---- Load test data ----
    X_test_flat = np.load(ticker_dir / "X_test.npy")
    y_test = np.load(ticker_dir / "y_test.npy")

    X_test, y_test_w = _make_windows(X_test_flat, y_test, lookback)
    logger.info("Test windows: %s", X_test.shape)

    # ---- Predictions ----
    y_pred = model.predict(X_test).flatten()

    # ---- Statistical metrics ----
    stat_metrics = compute_and_log_all_statistical_metrics(y_test_w, y_pred)

    # ---- Backtest ----
    theta = _load_threshold(results_dir, args.ticker, "xgboost", args.seed)
    opens, closes, ts = _load_prices(args.ticker, ticker_dir.parent, len(y_pred))
    result = _run_backtest(y_pred, opens, closes, ts, args, theta)

    _log_results(stat_metrics, result, theta)

    # ---- Persist ----
    _save_json(results_dir / f"xgb_test_stat_metrics_{tag}.json", stat_metrics)
    _save_json(
        results_dir / f"xgb_test_trading_metrics_{tag}.json",
        {
            "signal_threshold": theta,
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "max_drawdown": result.mdd,
            "cagr": result.cagr_,
            "calmar": result.calmar,
            "profit_factor": result.profit_factor_,
            "win_rate": result.win_rate_,
            "n_trades": result.n_trades,
            "turnover": result.turnover,
        },
    )
    if not result.trade_log.empty:
        result.trade_log.to_csv(
            results_dir / f"xgb_test_trade_log_{tag}.csv", index=False
        )

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    _plot_equity(
        result,
        args.ticker,
        "xgboost",
        closes,
        args.initial_capital,
        plots_dir / f"xgb_test_equity_{tag}.png",
    )

    logger.info("=== XGBOOST TEST DONE ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Final test-set evaluation for LSTM or XGBoost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", required=True, choices=["lstm", "xgboost"])
    p.add_argument("--ticker", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default="config/default_config.yaml")
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--log-file", default="logs/test.log")

    bt = p.add_argument_group("Backtester settings")
    bt.add_argument("--initial-capital", type=float, default=100_000.0)
    bt.add_argument("--position-fraction", type=float, default=0.02)
    bt.add_argument("--transaction-cost", type=float, default=0.001)
    bt.add_argument("--slippage", type=float, default=0.0005)
    bt.add_argument("--stop-loss", type=float, default=0.02)
    bt.add_argument("--daily-loss-limit", type=float, default=0.05)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_file)
    set_all_seeds(args.seed)

    ticker_dir = Path(args.features_dir) / args.ticker
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    if not ticker_dir.exists():
        logger.error("Feature directory not found: %s", ticker_dir)
        sys.exit(1)

    tag = _make_tag(args.ticker, args.model, args.seed)
    logger.info("tag=%s", tag)

    if args.model == "lstm":
        test_lstm(args, ticker_dir, results_dir, tag)
    else:
        test_xgboost(args, ticker_dir, results_dir, tag)


if __name__ == "__main__":
    main()
