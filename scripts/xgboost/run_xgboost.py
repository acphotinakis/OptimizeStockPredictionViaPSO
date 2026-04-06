#!/usr/bin/env python3
"""
scripts/run_xgboost.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.xgboost.xgboost_test import run_test
from scripts.xgboost.xgboost_val import run_val
from scripts.xgboost.xgboost_train import run_train
from src.utils import set_all_seeds, setup_logger
from consts import *

logger = logging.getLogger(__name__)


# ======================================================================
# Argument parsing
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run XGBoostModel Training, Validation, Testing for stock return prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -------------------------
    # Common arguments
    # -------------------------
    parser.add_argument("--ticker", required=True, help="Target ticker symbol")
    parser.add_argument(
        "--mode",
        choices=["train", "val", "test"],
        required=True,
        default="train",
        help="Execution mode: train, val, or test",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument(
        "--features-dir", default=FEATURES_DIR, help="Features directory"
    )
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Output directory")
    parser.add_argument(
        "--log-file", default="logs/run_xgboost.log", help="Log file path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Config file path",
    )

    # -------------------------
    # Training-specific arguments
    # -------------------------
    parser.add_argument(
        "--train-mode",
        choices=["default", "tune"],
        default="default",
        help="Training mode: default fixed or tune random search",
    )
    parser.add_argument("--lookback", type=int, default=30, help="Lookback window size")
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of random search trials"
    )
    parser.add_argument(
        "--retrain-on-trainval",
        action="store_true",
        default=True,
        help="Retrain on Train+Val after tuning",
    )

    # -------------------------
    # Validation-specific arguments
    # -------------------------
    parser.add_argument(
        "--wfv-fold-size",
        type=int,
        default=DEFAULT_WFV_FOLD_SIZE,
        help="Bars per walk-forward validation fold",
    )
    parser.add_argument(
        "--wfv-folds",
        type=int,
        default=DEFAULT_WFV_FOLDS,
        help="Max number of walk-forward folds",
    )

    # -------------------------
    # Testing-specific arguments
    # -------------------------
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--position-fraction", type=float, default=0.02)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--stop-loss", type=float, default=0.02)
    parser.add_argument("--daily-loss-limit", type=float, default=0.05)

    # Parse all arguments first
    args = parser.parse_args()

    # -------------------------
    # Filter arguments based on mode
    # -------------------------
    if args.mode == "train":
        # Remove validation/test-only args
        for attr in [
            "wfv_fold_size",
            "wfv_folds",
            "initial_capital",
            "position_fraction",
            "transaction_cost",
            "slippage",
            "stop_loss",
            "daily_loss_limit",
        ]:
            if hasattr(args, attr):
                delattr(args, attr)
    elif args.mode == "val":
        # Remove train/test-only args
        for attr in [
            "train_mode",
            "lookback",
            "n_trials",
            "retrain_on_trainval",
            "initial_capital",
            "position_fraction",
            "transaction_cost",
            "slippage",
            "stop_loss",
            "daily_loss_limit",
        ]:
            if hasattr(args, attr):
                delattr(args, attr)
    elif args.mode == "test":
        # Remove train/validation-only args
        for attr in [
            "train_mode",
            "lookback",
            "n_trials",
            "retrain_on_trainval",
            "wfv_fold_size",
            "wfv_folds",
        ]:
            if hasattr(args, attr):
                delattr(args, attr)
    else:
        raise ValueError("Mode for either ['train','val','test'] not set. ")

    return args


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    args = parse_args()

    # --------------------------------------------------
    # Shared setup (runs for ALL modes)
    # --------------------------------------------------
    setup_logger(args.log_file)
    set_all_seeds(args.seed)

    features_dir = Path(args.features_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    ticker = args.ticker
    tag = f"{ticker}_{args.mode}_seed{args.seed}"

    logger.info(f"{'='*60}")
    logger.info(
        f"XGBoost Pipeline | mode={args.mode} | ticker={ticker} | seed={args.seed}"
    )
    logger.info(f"{'='*60}")

    # --------------------------------------------------
    # Mode dispatch
    # --------------------------------------------------
    if args.mode == "train":
        run_train(args, features_dir, results_dir, ticker, tag)

    elif args.mode == "val":
        run_val(args, features_dir, results_dir, ticker, tag)

    elif args.mode == "test":
        run_test(args, features_dir, results_dir, ticker, tag)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
