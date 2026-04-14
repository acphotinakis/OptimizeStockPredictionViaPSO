#!/usr/bin/env python3
"""
scripts/test.py

Final evaluation on the held-out test set using RuntimeContext.
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
from src.evaluation.metrics import compute_and_log_all_statistical_metrics
from src.models.baselines import VanillaLSTM
from src.models.xgboost.xgboost_model import XGBoostModel
from src.experiment.run_context import RuntimeContext
from src.experiment.usage_enums import ModelType, Phase, RunMode, ArtifactType

logger = logging.getLogger(__name__)


def _make_tag(ticker: str, model: str, seed: int) -> str:
    return f"{ticker}_{model}_seed{seed}"


def _make_windows(X_flat: np.ndarray, y: np.ndarray, lookback: int):
    session_starts = np.zeros(len(X_flat), dtype=bool)
    return build_windows(X_flat, y, session_starts, lookback)


def _load_threshold(ctx: RuntimeContext, ticker: str, model: str, seed: int) -> float:
    """Load the optimal signal threshold from validation phase."""
    tag = _make_tag(ticker, model, seed)
    # Validation saves to val phase directory
    threshold_path = (
        ctx.tracker.phase_dir(Phase.VAL) / f"{model}_val_threshold_{tag}.json"
    )

    if threshold_path.exists():
        data = json.loads(threshold_path.read_text())
        theta = float(data.get("optimal_threshold", 1e-4))
        logger.info("Signal threshold loaded from validation: %.5f", theta)
        return theta

    logger.warning(
        "Threshold file not found: %s — falling back to θ=1e-4. "
        "Run validation first.",
        threshold_path,
    )
    return 1e-4


def _load_prices(
    ctx: RuntimeContext, ticker: str, n_test: int
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load open/close prices and UTC timestamps for the test period."""
    raw_path = Path("data/raw") / f"{ticker}.parquet"

    if raw_path.exists():
        try:
            from src.data.alpaca_ingestor import AlpacaIngestor
            from src.data.cleaner import DataCleaner
            from src.data.splitter import DataSplitter

            raw = AlpacaIngestor.load_bars(raw_path)
            df = DataCleaner().clean(raw)
            _, _, df_test = DataSplitter(
                train_end=ctx.cfg.data.train_end,
                val_end=ctx.cfg.data.val_end,
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
    else:
        raise ValueError("Can't find raw data, bruh.....")


def _run_backtest(
    y_pred: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    ts: pd.DatetimeIndex,
    ctx: RuntimeContext,
    theta: float,
):
    """Instantiate and run the Backtester."""
    bt = Backtester(
        initial_capital=ctx.args.initial_capital,
        position_fraction=ctx.args.position_fraction,
        transaction_cost=ctx.args.transaction_cost,
        slippage=ctx.args.slippage,
        stop_loss=ctx.args.stop_loss,
        daily_loss_limit=ctx.args.daily_loss_limit,
        signal_threshold=theta,
    )
    return bt.run(y_pred.copy(), opens, closes, ts)


def test_lstm(ctx: RuntimeContext) -> None:
    """Test LSTM model using RuntimeContext."""
    logger.info("=== LSTM TEST  ticker=%s ===", ctx.ticker)
    tag = _make_tag(ctx.ticker, ctx.model.value, ctx.seed)

    # Load from tracker artifacts
    model_path = ctx.tracker.artifact_path(f"model_train_{tag}.pth")
    params_path = ctx.tracker.artifact_path(f"lstm_params_{ctx.ticker}.json")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            f"Run train.py first for {ctx.ticker}"
        )

    hp = json.loads(params_path.read_text())["hyperparameters"]

    # Load test data via tracker
    X_test_split, y_test_split = ctx.tracker.load_split(Phase.TEST)

    # X_test, y_test_w = _make_windows(X_test_flat, y_test, hp["lookback"])
    session_starts = np.zeros(len(X_test_split), dtype=bool)

    lookback = ctx.cfg.lstm_baseline.lookback

    X_test, y_test_w = build_windows(
        X_test_split, y_test_split, session_starts[: len(X_test_split)], lookback
    )
    logger.info("Test windows: %s", X_test.shape)

    # Reconstruct model
    model = VanillaLSTM(input_size=X_test.shape[2], device=ctx.args.device, **hp)
    model._trainer.model.load_state_dict(
        torch.load(model_path, map_location="cuda", weights_only=True)
    )

    # Predictions
    y_pred = model.predict(X_test).flatten()

    # Statistical metrics
    stat_metrics = compute_and_log_all_statistical_metrics(y_test_w, y_pred)

    # Backtest
    theta = _load_threshold(ctx, ctx.ticker, ctx.model.value, ctx.seed)
    opens, closes, ts = _load_prices(ctx, ctx.ticker, len(y_pred))
    result = _run_backtest(y_pred, opens, closes, ts, ctx, theta)

    _log_results(stat_metrics, result, theta)

    # Persist via tracker
    ctx.tracker.save_json(
        ctx.tracker.phase_dir(Phase.TEST) / f"lstm_test_stat_metrics_{tag}.json",
        stat_metrics,
    )
    ctx.tracker.save_json(
        ctx.tracker.phase_dir(Phase.TEST) / f"lstm_test_trading_metrics_{tag}.json",
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
            ctx.tracker.phase_dir(Phase.TEST) / f"lstm_test_trade_log_{tag}.csv",
            index=False,
        )

    logger.info("=== LSTM TEST DONE ===")


def test_xgboost(ctx: RuntimeContext) -> None:
    """Test XGBoost model using RuntimeContext."""
    logger.info("=== XGBOOST TEST  ticker=%s ===", ctx.ticker)
    tag = _make_tag(ctx.ticker, ctx.model.value, ctx.seed)

    # Load from tracker artifacts
    model_path = ctx.tracker.artifact_path(f"model_train_{tag}.ubj")
    params_path = ctx.tracker.artifact_path(f"xgb_params_{ctx.ticker}.json")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            f"Run train.py first for {ctx.ticker}"
        )

    saved = json.loads(params_path.read_text())
    hp = dict(saved["hyperparameters"])
    lookback = hp.pop("lookback", 30)

    model = XGBoostModel(lookback=lookback, **{k: v for k, v in hp.items()})
    model.load(str(model_path))

    # Load test data via tracker
    X_test_split, y_test_split = ctx.tracker.load_split(Phase.TEST)

    # X_test, y_test_w = _make_windows(X_test_flat, y_test, lookback)
    session_starts = np.zeros(len(X_test_split), dtype=bool)

    lookback = ctx.cfg.xgboost.lookback

    X_test, y_test_w = build_windows(
        X_test_split, y_test_split, session_starts[: len(X_test_split)], lookback
    )
    logger.info("Test windows: %s", X_test.shape)
    logger.info("Test windows: %s", X_test.shape)

    # Predictions
    y_pred = model.predict(X_test).flatten()

    # Statistical metrics
    stat_metrics = compute_and_log_all_statistical_metrics(y_test_w, y_pred)

    # Backtest
    theta = _load_threshold(ctx, ctx.ticker, ctx.model.value, ctx.seed)
    opens, closes, ts = _load_prices(ctx, ctx.ticker, len(y_pred))
    result = _run_backtest(y_pred, opens, closes, ts, ctx, theta)

    _log_results(stat_metrics, result, theta)

    # Persist via tracker
    ctx.tracker.save_json(
        ctx.tracker.phase_dir(Phase.TEST) / f"xgb_test_stat_metrics_{tag}.json",
        stat_metrics,
    )
    ctx.tracker.save_json(
        ctx.tracker.phase_dir(Phase.TEST) / f"xgb_test_trading_metrics_{tag}.json",
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
            ctx.tracker.phase_dir(Phase.TEST) / f"xgb_test_trade_log_{tag}.csv",
            index=False,
        )

    logger.info("=== XGBOOST TEST DONE ===")


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
    p.add_argument("--run-id", help="Run ID to attach to (from training)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Initialize RuntimeContext
    ctx = RuntimeContext(
        args=args,
        model=ModelType(args.model),
        ticker=args.ticker,
        seed=args.seed,
        config_path=args.config,
        run_id=args.run_id,
        run_mode=RunMode.ATTACH,  # Test attaches to existing training run
        log_file=args.log_file,
    )

    logger.info("tag=%s", _make_tag(ctx.ticker, ctx.model.value, ctx.seed))

    if ctx.model == ModelType.LSTM:
        test_lstm(ctx)
    else:
        test_xgboost(ctx)


if __name__ == "__main__":
    main()
