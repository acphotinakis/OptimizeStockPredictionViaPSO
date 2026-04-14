"""
scripts/test.py

Final test-set evaluation.
Loads saved model → predicts on test set → runs backtester → saves all outputs.

Usage:
    python scripts/test.py --model lstm    --ticker AAPL --run-id <id>
    python scripts/test.py --model xgboost --ticker AAPL --run-id <id>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.splitter import build_windows
from src.evaluation.backtester import Backtester
from src.evaluation.metrics import compute_and_log_all_statistical_metrics, _log_results
from src.experiment.run_context import RuntimeContext
from src.models.lstm.inference import LSTMWrapper
from src.models.xgboost.model import XGBoostWrapper

logger = logging.getLogger(__name__)


# ======================================================================
# Shared utilities
# ======================================================================


def _make_windows(X_flat, y, lookback):
    session_starts = np.zeros(len(X_flat), dtype=bool)
    return build_windows(X_flat, y, session_starts, lookback)


def _load_prices(ticker: str, n_test: int):
    """Load open/close prices for the test period."""
    raw_path = Path("data/raw") / f"{ticker}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw price data not found: {raw_path}")

    from src.data.alpaca_ingestor import AlpacaIngestor
    from src.data.cleaner import DataCleaner
    from src.data.splitter import DataSplitter

    raw = AlpacaIngestor._load_bars(raw_path)
    df = DataCleaner().clean(raw)
    _, _, df_test = DataSplitter().split(df)

    opens = df_test["open"].to_numpy(dtype=np.float32)
    closes = df_test["close"].to_numpy(dtype=np.float32)
    ts = df_test.index
    if not isinstance(ts, pd.DatetimeIndex):
        ts = pd.DatetimeIndex(ts)

    if len(opens) > n_test:
        opens, closes, ts = opens[-n_test:], closes[-n_test:], ts[-n_test:]

    return opens, closes, ts


def _run_backtest(ctx, y_pred, opens, closes, ts, theta):
    bt = Backtester(
        initial_capital=ctx.cfg.backtesting.initial_capital,
        position_fraction=ctx.cfg.backtesting.position_fraction,
        transaction_cost=ctx.cfg.backtesting.transaction_cost,
        slippage=ctx.cfg.backtesting.slippage,
        stop_loss=ctx.cfg.backtesting.stop_loss,
        daily_loss_limit=ctx.cfg.backtesting.daily_loss_limit,
        signal_threshold=theta,
    )
    return bt.run(y_pred.copy(), opens, closes, ts)


# ======================================================================
# LSTM test
# ======================================================================


def test_lstm(ctx: RuntimeContext) -> None:
    logger.info("=== LSTM TEST  ticker=%s ===", ctx.ticker)

    saved = ctx.store.load_params()
    hp = saved["hyperparameters"]
    lookback = hp["lookback"]

    X_test_flat, y_test = ctx.store.load_split("test")
    X_test, y_test_w = _make_windows(X_test_flat, y_test, lookback)
    logger.info("Test windows: %s", X_test.shape)

    model = LSTMWrapper.load(
        str(ctx.store.model_path("pth")),
        input_size=X_test.shape[2],
        device=getattr(ctx.args, "device", None),
        **hp,
    )
    y_pred = model.predict(X_test).flatten()

    stat_metrics = compute_and_log_all_statistical_metrics(y_test_w, y_pred)
    theta = ctx.store.load_val_threshold()
    opens, closes, ts = _load_prices(ctx.ticker, len(y_pred))
    result = _run_backtest(ctx, y_pred, opens, closes, ts, theta)

    _log_results(stat_metrics, result, theta)

    ctx.store.save_backtest_result(result, stat_metrics, theta)
    ctx.store.save_test_predictions(y_pred, y_test_w)
    ctx.store.write_index()
    logger.info("=== LSTM TEST DONE ===")


# ======================================================================
# XGBoost test
# ======================================================================


def test_xgboost(ctx: RuntimeContext) -> None:
    logger.info("=== XGBOOST TEST  ticker=%s ===", ctx.ticker)

    saved = ctx.store.load_params()
    hp = saved["hyperparameters"]
    lookback = hp.get("lookback", 30)

    X_test_flat, y_test = ctx.store.load_split("test")
    X_test, y_test_w = _make_windows(X_test_flat, y_test, lookback)
    logger.info("Test windows: %s", X_test.shape)

    xgb_hp = {k: v for k, v in hp.items() if k != "lookback"}
    model = XGBoostWrapper.load(
        str(ctx.store.model_path("ubj")),
        lookback=lookback,
        **xgb_hp,
    )
    y_pred = model.predict(X_test).flatten()

    stat_metrics = compute_and_log_all_statistical_metrics(y_test_w, y_pred)
    theta = ctx.store.load_val_threshold()
    opens, closes, ts = _load_prices(ctx.ticker, len(y_pred))
    result = _run_backtest(ctx, y_pred, opens, closes, ts, theta)

    _log_results(stat_metrics, result, theta)

    ctx.store.save_backtest_result(result, stat_metrics, theta)
    ctx.store.save_test_predictions(y_pred, y_test_w)
    ctx.store.write_index()
    logger.info("=== XGBOOST TEST DONE ===")


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
    p.add_argument("--log-file", default="logs/test.log")
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
        test_lstm(ctx)
    else:
        test_xgboost(ctx)


if __name__ == "__main__":
    main()
