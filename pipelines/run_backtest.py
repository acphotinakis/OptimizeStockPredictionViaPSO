import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# Add project root to path
# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_model import LSTMModel
from src.backtesting.backtester import BacktestResult, Backtester
from pipelines.run_lstm import (
    load_trained_model as load_trained_lstm_model,
    save_results,
)
from src.data.windowing import build_lstm_windows
from src.utils.config_loader import Config, load_config
from src.backtesting.backtest import CanonicalBacktest
from src.backtesting.backtest_results import BacktestResults, save_backtest_results
from src.evaluation.metrics import (
    compute_and_log_all_statistical_metrics,
    compute_and_log_all_trading_metrics,
)
from src.evaluation.model_loader import load_model, validate_model_compatibility
from src.evaluation.plotting import create_all_plots
from src.utils.logger import LogFileMode, setup_logger
from src.utils.data_storage import _load_parquet

logger = logging.getLogger(__name__)


def compute_all_metrics(
    backtest_result: BacktestResult,
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    config: Config,
) -> tuple:
    """
    Compute all statistical and trading metrics.

    Args:
        backtest_df: DataFrame from backtest engine
        predictions: Model predictions
        actual_returns: True returns
        config: Configuration object

    Returns:
        Tuple of (statistical_metrics, trading_metrics)
    """
    logger.info("=" * 80)
    logger.info("COMPUTING METRICS")
    logger.info("=" * 80)

    # Statistical metrics (prediction quality)
    statistical_metrics = compute_and_log_all_statistical_metrics(
        y_true=actual_returns, y_pred=predictions, label="Statistical"
    )

    # Trading metrics (portfolio performance)
    equity_curve = backtest_result.equity_curve
    strategy_returns = backtest_result.bar_returns

    trading_metrics = compute_and_log_all_trading_metrics(
        equity_curve=equity_curve,
        bar_returns=strategy_returns,
        benchmark_returns=actual_returns,
        label="Trading",
    )

    logger.info("=" * 80)

    return statistical_metrics, trading_metrics


def build_experiment_dir(
    ticker: str, timeframe: str, run_id: str, model_type: str
) -> Path:
    return (
        PROJECT_ROOT
        / "results"
        / "experiments"
        / f"{ticker}_{timeframe}_{model_type}_{run_id}"
    )


def build_experiment_dirs(base: Path) -> dict:
    return {
        "root": base,
        "train": base / "train",
        "val": base / "val",
        "test": base / "test",
        "model": base / "model",
        "logs": base / "logs",
        "plots": base / "plots",
        "backtest": base / "backtest",
        "backtest_plots": base / "backtest" / "plots",
    }


def load_backtest_data(
    feature_data_path: Path, processed_data_path: Path, timeframe: str, ticker: str
) -> dict:

    logger.info(f"[LOAD] Feature data: {feature_data_path}")

    X_path = feature_data_path / "X_test.npy"
    y_path = feature_data_path / "y_test.npy"
    idx_path = feature_data_path / "test_index.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing test data: {X_path} or {y_path}")

    # -----------------------------
    # Load feature arrays
    # -----------------------------
    X_test = np.load(X_path)
    y_test = np.load(y_path)
    test_index = np.load(idx_path)

    logger.info(f"[FEATURES] X={X_test.shape} y={y_test.shape} idx={test_index.shape}")
    logger.info(f"[IDX RAW] type={type(test_index)} dtype={test_index.dtype}")

    # -----------------------------
    # Normalize timestamps (SINGLE SOURCE OF TRUTH)
    # -----------------------------
    test_index = pd.to_datetime(test_index, utc=True).tz_convert(None)

    logger.info(
        f"[INDEX] range={test_index.min()} → {test_index.max()} | n={len(test_index)}"
    )

    # -----------------------------
    # Load price data
    # -----------------------------
    processed_path = processed_data_path / timeframe / f"{ticker}.parquet"
    df = pd.read_parquet(processed_path).sort_index()

    # force SAME format as test_index
    # df.index = df.index.tz_convert("UTC").tz_localize(None)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    logger.info(f"[PRICES] shape={df.shape} cols={df.columns.tolist()}")
    logger.info(
        f"[PRICES] range={df.index.min()} → {df.index.max()} | dtype={df.index.dtype}"
    )

    # -----------------------------
    # ALIGNMENT CHECK
    # -----------------------------
    missing = test_index.difference(df.index)
    logger.info(f"[ALIGN] missing={len(missing)}")

    if len(missing) > 0:
        logger.warning(f"[ALIGN] sample missing={missing[:5]}")

    # -----------------------------
    # SAFE ALIGNMENT
    # -----------------------------
    df_test = df.reindex(test_index)

    logger.info(f"[TEST] shape={df_test.shape}")

    opens = df_test["open"].values
    closes = df_test["close"].values

    logger.info(f"[PRICES] open(min={opens.min()}, max={opens.max()})")
    logger.info(f"[PRICES] close(min={closes.min()}, max={closes.max()})")

    idx = df_test.index

    if idx.tz is None:
        idx = idx.tz_localize("UTC")  # OR correct source timezone if known

    # timestamps = idx.tz_convert("America/New_York").to_numpy()
    timestamps = idx.tz_convert("America/New_York")

    # sys.exit(0)

    return {
        "X_test": X_test,
        "y_test": y_test,
        "dates": test_index,
        "open": opens,
        "close": closes,
        "timestamps": timestamps,
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Backtesting Pipeline for All Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default_config.yaml",
        help="Path to config file (default: config/default_config.yaml)",
    )

    # Required arguments
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["pso_lstm", "lstm_baseline", "xgboost"],
        help="Model type to backtest",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to trained model file",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker symbol to backtest (e.g., AAPL)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to features directory",
    )
    parser.add_argument("--run-id", type=str, default=None, required=True)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--timeframe", type=str, required=True)

    return parser.parse_args()


def backtest_baseline_lstm(
    ticker: str,
    model_dir: Path,
    feature_data_path: Path,
    processed_data_path: Path,
    config: Config,
    output_dir: Path,
    timeframe: str,
    device: Optional[str] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, BacktestResult, LSTMModel]:
    """
    End-to-end test pipeline using LSTMModel.evaluate() as the canonical path.
    """
    logger.info("=" * 80)
    logger.info(f"BASELINE LSTM INFERENCE: {ticker}")
    logger.info("=" * 80)

    data = load_backtest_data(
        feature_data_path, processed_data_path, timeframe=timeframe, ticker=ticker
    )

    X_test = data["X_test"]
    y_test = data["y_test"]
    dates = data["dates"]
    opens = data["open"]
    closes = data["close"]
    timestamps = data["timestamps"]

    model = load_trained_lstm_model(model_dir, config, device=device)
    lookback = config.lstm_baseline.lookback

    logger.info(f"Building LSTM windows (lookback={lookback})...")
    X_train_win, y_train_win = build_lstm_windows(X_test, y_test, lookback)

    y_test = y_test[lookback:]
    dates = dates[lookback:]
    opens = opens[lookback:]
    closes = closes[lookback:]
    timestamps = timestamps[lookback:]
    logger.info("Running canonical evaluation via model.evaluate()...")
    metrics, y_pred = model.evaluate(X_train_win, y_train_win)
    logger.info(
        f"[WINDOWS] X={X_train_win.shape} y={y_train_win.shape} || {y_pred.shape}"
    )

    # Ensure alignment
    if len(y_pred) != len(y_test):
        raise ValueError(
            f"y_pred len not equal to y_test --> len(y_pred)={len(y_pred)} || len(y_test)={len(y_test)}"
        )

    save_results(output_dir, y_test, y_pred, metrics, model_dir)

    logger.info(f" Generated {len(y_pred)} predictions")

    # Initialize canonical backtest engine
    bt_config = config.backtesting

    # # Run backtest
    # backtest_df = backtest_engine.run_backtest(y_pred, y_test, dates)
    bt = Backtester(
        initial_capital=bt_config.initial_capital,
        position_fraction=bt_config.position_fraction,
        transaction_cost=bt_config.transaction_cost,
        slippage=bt_config.slippage,
        stop_loss=bt_config.stop_loss,
        daily_loss_limit=bt_config.daily_loss_limit,
    )

    backtest_result: BacktestResult = bt.run(
        y_pred=y_pred,
        opens=opens,
        closes=closes,
        timestamps=timestamps,
    )

    logger.info("=" * 80)

    logger.info("=" * 80)
    logger.info("BACKTESTING COMPLETE")
    logger.info("=" * 80)

    return metrics, y_pred, y_test, backtest_result, model


def main():
    """Main execution function."""
    args = parse_args()

    logger.info(json.dumps(vars(args), indent=4, default=str))

    logger.info("=" * 80)
    logger.info("UNIFIED BACKTESTING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Ticker: {args.ticker}")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(args.config)

    run_id = args.run_id

    experiment_dir = build_experiment_dir(
        ticker=args.ticker,
        timeframe=args.timeframe,
        run_id=run_id,
        model_type=args.model_type,
    )

    dirs = build_experiment_dirs(experiment_dir)

    setup_logger(
        log_file=f"{dirs['logs']}/{args.ticker}_backtest.log",
        level="INFO",
        mode=LogFileMode.OVERWRITE,
    )

    logger.info(f"RUN ID: {run_id}")
    logger.info(f"EXPERIMENT: {experiment_dir}")

    processed_data_path = Path(args.processed_dir)

    if args.model_type == "lstm_baseline":
        metrics, y_pred, y_test, backtest_result, model = backtest_baseline_lstm(
            ticker=args.ticker,
            model_dir=args.model_path,
            feature_data_path=args.data_path,
            processed_data_path=processed_data_path,
            config=config,
            output_dir=dirs["backtest"],
            timeframe=args.timeframe,
        )
        logger.info("=" * 80)
        logger.info("BACKTEST RESULT (FULL DUMP)")
        logger.info("=" * 80)

        logger.info(f"Sharpe: {backtest_result.sharpe}")
        logger.info(f"Sortino: {backtest_result.sortino}")
        logger.info(f"MDD: {backtest_result.mdd}")
        logger.info(f"CAGR: {backtest_result.cagr_}")
        logger.info(f"Calmar: {backtest_result.calmar}")
        logger.info(f"Profit Factor: {backtest_result.profit_factor_}")
        logger.info(f"Win Rate: {backtest_result.win_rate_}")
        logger.info(f"Trades: {backtest_result.n_trades}")
        logger.info(f"Turnover: {backtest_result.turnover}")

        logger.info(f"Equity Curve Shape: {backtest_result.equity_curve.shape}")
        logger.info(f"Bar Returns Shape: {backtest_result.bar_returns.shape}")
        logger.info(f"Trade Log Shape: {backtest_result.trade_log.shape}")

        # Optional: head previews
        logger.info("Equity Curve (head): %s", backtest_result.equity_curve[:5])
        logger.info("Bar Returns (head): %s", backtest_result.bar_returns[:5])

        logger.info("Trade Log (head):")
        logger.info("\n%s", backtest_result.trade_log.head(10).to_string(index=False))

        logger.info("=" * 80)
        logger.info("STEP 4: Computing metrics...")
        # FIXED: adapt function to BacktestResult
        statistical_metrics, trading_metrics = compute_all_metrics(
            backtest_result, y_pred, y_test, config
        )

        results = BacktestResults.from_engine(
            model_type=args.model_type,
            ticker=args.ticker,
            predictions=y_pred,
            actual_returns=y_test,
            dates=None,  # replace if you have timestamps
            backtest_result=backtest_result,
            signals=backtest_result.trade_log.get("direction", np.zeros_like(y_pred)),
            strategy_returns=backtest_result.bar_returns,
            trade_costs=np.zeros_like(y_pred),  # replace if tracked separately
            model_metadata=model.get_metadata(),
            statistical_metrics=statistical_metrics,
            trading_metrics=trading_metrics,
            backtest_config={
                "transaction_cost": config.backtesting.transaction_cost,
                "initial_capital": config.backtesting.initial_capital,
                "position_fraction": config.backtesting.position_fraction,
                "slippage": config.backtesting.slippage,
                "stop_loss": config.backtesting.stop_loss,
                "daily_loss_limit": config.backtesting.daily_loss_limit,
            },
        )

        save_backtest_results(results=results, output_dir=dirs["backtest"])

        logger.info("STEP 6: Generating visualizations...")

        create_all_plots(
            backtest_result,
            y_pred,
            y_test,
            args.model_type,
            dirs["backtest_plots"],
        )

        logger.info("=" * 80)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Sharpe Ratio: {backtest_result.sharpe:.2f}")
        logger.info(f"Max Drawdown: {backtest_result.mdd:.2%}")
        logger.info(f"Win Rate: {backtest_result.win_rate_:.2%}")

    logger.info(" PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
