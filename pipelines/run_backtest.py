#!/usr/bin/env python3
"""
Unified Backtesting Pipeline - Production Entry Point

Single CLI tool for backtesting all model types (PSO-LSTM, Baseline LSTM, XGBoost)
with consistent metrics, evaluation, and visualization.

Usage:
    python pipelines/run_backtest.py \
        --model_type pso_lstm \
        --model_path results/pso_lstm/best_model.pt \
        --ticker AAPL \
        --config config/default_config.yaml

Supported Models:
    - pso_lstm: PSO-optimized LSTM
    - lstm_baseline: Baseline LSTM
    - xgboost: XGBoost model

Outputs:
    - Metrics (JSON)
    - Equity curve (CSV + PNG)
    - Drawdown chart (PNG)
    - Returns distribution (PNG)
    - Signal analysis (PNG)
    - Backtest report (Markdown)

Author: Production System
Version: 1.0
Source: BACKTEST_DESIGN.md
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import yaml

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.config_loader import Config, load_config
from src.evaluation.backtest import CanonicalBacktest
from src.evaluation.backtest_results import BacktestResults, save_backtest_results
from src.evaluation.metrics import (
    compute_and_log_all_statistical_metrics,
    compute_and_log_all_trading_metrics,
)
from src.evaluation.model_loader import load_model, validate_model_compatibility
from src.evaluation.plotting import create_all_plots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Backtesting Pipeline for All Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Path to preprocessed features directory",
    )

    # Optional arguments
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "config" / "default_config.yaml",
        help="Path to config file (default: config/default_config.yaml)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: results/backtest/{model_type}/{ticker})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to backtest on (default: test)",
    )

    return parser.parse_args()


def load_test_data(data_path: Path) -> dict:
    logger.info(f"Loading preprocessed data from {data_path}")

    X_path = data_path / f"X_test.npy"
    y_path = data_path / f"y_test.npy"
    dates = data_path / f"test_index.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing test data: {X_path} or {y_path}\n"
            f"Run canonical feature pipeline first."
        )

    X_test = np.load(X_path)
    y_test = np.load(y_path)
    dates = np.load(dates)

    return {
        "X_test": X_test,
        "y_test": y_test,
        "dates": dates,
    }


def run_unified_backtest(
    model_adapter: Any,
    test_data: dict,
    config: Config,
) -> tuple:
    """
    Run unified backtesting engine.

    Args:
        model_adapter: ModelAdapter instance
        test_data: Test data dictionary
        config: Configuration object

    Returns:
        Tuple of (backtest_results, predictions, actual_returns)
    """
    logger.info("=" * 80)
    logger.info("RUNNING UNIFIED BACKTEST")
    logger.info("=" * 80)

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    dates = test_data["dates"]

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = model_adapter.predict(X_test)

    # Ensure alignment
    if len(predictions) != len(y_test):
        logger.warning(
            f"Prediction length mismatch: {len(predictions)} vs {len(y_test)}"
        )
        # Truncate to shorter length
        min_len = min(len(predictions), len(y_test))
        predictions = predictions[:min_len]
        y_test = y_test[:min_len]
        dates = dates[:min_len]

    logger.info(f"✓ Generated {len(predictions)} predictions")

    # Initialize canonical backtest engine
    bt_config = config.backtesting
    backtest_engine = CanonicalBacktest(
        transaction_cost=bt_config.transaction_cost,
        initial_capital=bt_config.initial_capital,
    )

    # Run backtest
    backtest_df = backtest_engine.run_backtest(predictions, y_test, dates)

    logger.info("=" * 80)

    return backtest_df, predictions, y_test


def compute_all_metrics(
    backtest_df: pd.DataFrame,
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
    strategy_returns = backtest_df["strategy_return"].values
    equity_curve = backtest_df["capital"].values

    trading_metrics = compute_and_log_all_trading_metrics(
        equity_curve=equity_curve,
        bar_returns=strategy_returns,
        benchmark_returns=actual_returns,
        label="Trading",
    )

    logger.info("=" * 80)

    return statistical_metrics, trading_metrics


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("UNIFIED BACKTESTING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Split: {args.split}")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(str(args.config))

    # Set output directory
    if args.output_dir is None:
        output_dir = (
            ROOT_DIR
            / "results"
            / "backtest"
            / args.model_type
            / args.ticker
            / args.split
        )
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    # Load model
    logger.info("STEP 1: Loading model...")
    model_adapter = load_model(args.model_type, args.model_path, config)
    logger.info(f"✓ Model loaded: {model_adapter.get_metadata()}")

    # Load test data
    logger.info("STEP 2: Loading test data...")
    test_data = load_test_data(args.data_path)

    # Validate compatibility
    validate_model_compatibility(model_adapter, test_data)

    # Run backtest
    logger.info("STEP 3: Running backtest...")
    backtest_df, predictions, actual_returns = run_unified_backtest(
        model_adapter, test_data, config
    )

    # Compute metrics
    logger.info("STEP 4: Computing metrics...")
    statistical_metrics, trading_metrics = compute_all_metrics(
        backtest_df, predictions, actual_returns, config
    )

    # Save results
    logger.info("STEP 5: Saving results...")

    # Extract signals for results
    signals = backtest_df["signal"].values
    strategy_returns = backtest_df["strategy_return"].values
    equity_curve = backtest_df["capital"].values
    trade_costs = backtest_df["trade_cost"].values
    
    # Align dates with backtest results (in case of length mismatch)
    backtest_dates = backtest_df["date"].values if "date" in backtest_df.columns else None
    if backtest_dates is None and len(test_data["dates"]) >= len(predictions):
        backtest_dates = test_data["dates"][:len(predictions)]
    
    # Log lengths for debugging
    logger.info(f"Array lengths: predictions={len(predictions)}, actual_returns={len(actual_returns)}, "
                f"signals={len(signals)}, equity={len(equity_curve)}")

    results = BacktestResults(
        model_type=args.model_type,
        ticker=args.ticker,
        timestamp=datetime.now().isoformat(),
        predictions=predictions,
        actual_returns=actual_returns,
        dates=backtest_dates,
        signals=signals,
        strategy_returns=strategy_returns,
        equity_curve=equity_curve,
        trade_costs=trade_costs,
        statistical_metrics=statistical_metrics,
        trading_metrics=trading_metrics,
        model_metadata=model_adapter.get_metadata(),
        backtest_config={
            "split": args.split,
            "transaction_cost": config.backtesting.transaction_cost,
            "initial_capital": config.backtesting.initial_capital,
        },
    )

    save_backtest_results(results, output_dir)

    # Generate visualizations
    logger.info("STEP 6: Generating visualizations...")
    create_all_plots(
        backtest_df, predictions, actual_returns, args.model_type, output_dir
    )

    # Final summary
    logger.info("=" * 80)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total Return: {trading_metrics['cagr']:.2%}")
    logger.info(f"Sharpe Ratio: {trading_metrics['sharpe']:.2f}")
    logger.info(f"Max Drawdown: {trading_metrics['max_drawdown']:.2%}")
    logger.info(f"Win Rate: {trading_metrics['win_rate']:.2%}")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)

    logger.info("✓ PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
