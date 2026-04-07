#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.xgboost.xgboost_model import (
    load_windows,
)
from src.evaluation import all_statistical_metrics, Backtester
from src.models.xgboost.helpers import (
    load_model,
    load_prices,
    load_optimal_threshold,
)

logger = logging.getLogger(__name__)


# ======================================================================
# TEST
# ======================================================================


def run_test(args, features_dir, results_dir, ticker, tag):
    logger.info("\n[TEST] Loading model...")
    # Load model trained in 'train' mode, not 'test' mode
    model, meta = load_model(results_dir, ticker, "train", args.seed)
    # Load threshold from 'val' mode (determined during validation)
    theta = load_optimal_threshold(results_dir, ticker, "val", args.seed)

    _, _, _, _, X_test, y_test = load_windows(features_dir, ticker)

    # ---------------------------
    # Predictions
    # ---------------------------
    y_pred = model.predict(X_test)

    # ---------------------------
    # Metrics
    # ---------------------------
    stat_metrics = all_statistical_metrics(y_test, y_pred)

    # ---------------------------
    # Backtest
    # ---------------------------
    opens, closes, ts = load_prices(ticker, str(features_dir), len(y_test))

    bt = Backtester(
        initial_capital=args.initial_capital,
        position_fraction=args.position_fraction,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        stop_loss=args.stop_loss,
        daily_loss_limit=args.daily_loss_limit,
        signal_threshold=theta,
    )

    result = bt.run(y_pred.copy(), opens, closes, ts)

    trading_metrics = {
        "sharpe": result.sharpe,
        "max_drawdown": result.mdd,
        "cagr": result.cagr_,
        "n_trades": result.n_trades,
    }

    # ---------------------------
    # Save
    # ---------------------------
    json.dump(
        {"statistical": stat_metrics, "trading": trading_metrics},
        open(results_dir / f"xgb_test_metrics_{tag}.json", "w"),
        indent=2,
    )

    logger.info("Test complete.")
