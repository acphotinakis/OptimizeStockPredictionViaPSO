#!/usr/bin/env python3
"""
scripts/05_backtest.py

Runs realistic backtesting on the test set using the IPSO-optimized model.
Generates equity curves, trade logs, and performance metrics.

Usage:
    python scripts/05_backtest.py --ticker AAPL --config config/default_config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.data_storage import load_windows

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.lstm.lstm_model import LSTMModel, LSTMTrainer
from backup.quantized_lstm import QuantizedLSTMModel
from src.evaluation.backtester import Backtester
from src.data.splitter import build_windows
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.utils.seed import set_all_seeds
from src.utils.memory_profiler import MemoryProfiler

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backtest IPSO-LSTM model")
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker symbol to backtest",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="data/features",
        help="Directory containing feature matrices",
    )
    parser.add_argument(
        "--pso-results",
        type=str,
        default="results",
        help="Directory containing PSO results",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing aligned price data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/backtest",
        help="Output directory for backtest results",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    set_all_seeds(cfg.pso.seed)

    setup_logger(log_file=f"logs/05_backtest_{args.ticker}.log", level="INFO")
    logger.info("=" * 60)
    logger.info("Backtesting: %s", args.ticker)
    logger.info("=" * 60)

    # Load PSO results
    pso_file = Path(args.pso_results) / f"pso_results_{args.ticker}.json"
    with open(pso_file) as f:
        pso_results = json.load(f)

    best_params = pso_results["best_params"]
    logger.info("Using IPSO hyperparameters:")
    for key, value in best_params.items():
        logger.info("  %s: %s", key, value)

    # Load feature data
    X_train_flat, y_train, X_val_flat, y_val, X_test_flat, y_test = load_windows(
        data_dir=Path(args.features_dir), ticker=args.ticker
    )

    # Build sliding windows
    lookback = best_params["lookback"]
    session_starts = np.zeros(len(X_train_flat), dtype=bool)

    X_train, y_train_w = build_windows(
        X_train_flat, y_train, session_starts[: len(X_train_flat)], lookback
    )
    X_val, y_val_w = build_windows(
        X_val_flat, y_val, session_starts[: len(X_val_flat)], lookback
    )
    X_test, y_test_w = build_windows(
        X_test_flat, y_test, session_starts[: len(X_test_flat)], lookback
    )

    # Train final model on Train+Val
    logger.info("Training final model on Train+Val data...")
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train_w, y_val_w], axis=0)

    input_size = X_train.shape[2]
    use_checkpointing = cfg.lstm_baseline.use_checkpointing
    model = LSTMModel(
        input_size=input_size,
        num_layers=best_params["num_layers"],
        hidden_units=best_params["hidden_units"],
        dropout=best_params["dropout"],
        use_checkpointing=use_checkpointing,
    )

    use_amp = cfg.lstm_baseline.use_amp
    accumulation_steps = cfg.lstm_baseline.accumulation_steps

    trainer = LSTMTrainer(
        model=model,
        lr=best_params["learning_rate"],
        max_epochs=cfg.lstm.max_epochs,
        patience=cfg.lstm.early_stopping_patience,
        batch_size=cfg.lstm.batch_size,
        use_amp=use_amp,
        accumulation_steps=accumulation_steps,
    )

    # Use a small validation split from combined data for early stopping
    val_split = int(0.9 * len(X_combined))
    trainer.fit(
        X_combined[:val_split],
        y_combined[:val_split],
        X_combined[val_split:],
        y_combined[val_split:],
    )

    # Generate predictions on test set
    logger.info("Generating predictions on test set...")

    # Phase 2: Quantize if requested
    if args.quantize:
        logger.info("Quantizing model for backtesting inference...")
        quantized_wrapper = QuantizedLSTMModel(model)
        quantized_wrapper.quantize()
        y_pred = quantized_wrapper.predict(X_test)
    else:
        y_pred = trainer.predict(X_test)

    # Load price data for backtesting
    logger.info("Loading price data...")
    df_aligned = pd.read_parquet(Path(args.data_dir) / "aligned_universe.parquet")

    # Extract test period prices
    # Note: This is simplified - in production you'd need to properly align windows with timestamps
    ticker_data = df_aligned[args.ticker]
    test_start_idx = len(X_train_flat) + len(X_val_flat) + lookback
    test_end_idx = test_start_idx + len(y_test_w)

    opens_test = ticker_data["open"].iloc[test_start_idx:test_end_idx].values
    closes_test = ticker_data["close"].iloc[test_start_idx:test_end_idx].values
    timestamps_test = ticker_data.index[test_start_idx:test_end_idx]

    # Run backtest
    logger.info("Running backtest...")
    backtester = Backtester(
        initial_capital=100_000.0,
        position_fraction=cfg.backtesting.position_size,
        transaction_cost=cfg.backtesting.transaction_cost,
        slippage=cfg.backtesting.slippage,
        stop_loss=cfg.backtesting.stop_loss,
    )

    result = backtester.run(y_pred, opens_test, closes_test, timestamps_test)

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("Backtest Results:")
    logger.info("=" * 60)
    logger.info("Sharpe Ratio:     %.4f", result.sharpe)
    logger.info("Sortino Ratio:    %.4f", result.sortino)
    logger.info("Max Drawdown:     %.4f", result.mdd)
    logger.info("CAGR:             %.4f", result.cagr_)
    logger.info("Calmar Ratio:     %.4f", result.calmar)
    logger.info("Profit Factor:    %.4f", result.profit_factor_)
    logger.info("Win Rate:         %.4f", result.win_rate_)
    logger.info("Number of Trades: %d", result.n_trades)
    logger.info("Turnover:         %.4f", result.turnover)
    logger.info("Final Equity:     $%.2f", result.equity_curve[-1])
    logger.info("=" * 60)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics = {
        "ticker": args.ticker,
        "sharpe": float(result.sharpe),
        "sortino": float(result.sortino),
        "max_drawdown": float(result.mdd),
        "cagr": float(result.cagr_),
        "calmar": float(result.calmar),
        "profit_factor": float(result.profit_factor_),
        "win_rate": float(result.win_rate_),
        "n_trades": int(result.n_trades),
        "turnover": float(result.turnover),
        "final_equity": float(result.equity_curve[-1]),
    }

    with open(output_dir / f"backtest_{args.ticker}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save trade log
    if not result.trade_log.empty:
        result.trade_log.to_csv(output_dir / f"trades_{args.ticker}.csv", index=False)
        logger.info("Trade log saved to %s", output_dir / f"trades_{args.ticker}.csv")

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(result.equity_curve, linewidth=1.5)
    plt.title(f"Equity Curve - {args.ticker}")
    plt.xlabel("Bar")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = output_dir / f"equity_curve_{args.ticker}.png"
    plt.savefig(plot_file, dpi=150)
    logger.info("Equity curve plot saved to %s", plot_file)

    logger.info("\nBacktest complete!")


if __name__ == "__main__":
    main()
