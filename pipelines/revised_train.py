"""
scripts/train.py

Thin training pipeline.  Loads data → builds model → trains → saves artifacts.
All business logic lives in src/; this file is pure orchestration.

Usage:
    python scripts/train.py --model lstm    --ticker AAPL
    python scripts/train.py --model xgboost --ticker AAPL
    python scripts/train.py --model xgboost --ticker AAPL --train-mode tune
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.splitter import build_windows
from src.evaluation.metrics import compute_and_log_all_statistical_metrics
from src.experiment.run_context import RuntimeContext
from src.models.lstm.inference import LSTMWrapper, DEFAULT_PARAMS as LSTM_DEFAULTS
from src.experiment.run_context import RuntimeContext

logger = logging.getLogger(__name__)


# ======================================================================
# Helpers
# ======================================================================


def _make_windows(X_flat, y, lookback):
    session_starts = np.zeros(len(X_flat), dtype=bool)
    return build_windows(X_flat, y, session_starts, lookback)


import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml
import numpy as np

logger = logging.getLogger(__name__)



# ======================================================================
# CONFIG LOADER
# ======================================================================


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_overrides(base: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Deterministic override logic.
    Only applies non-None CLI overrides.
    """

    override_map = {
        "num_layers": args.num_layers,
        "hidden_units": args.hidden_units,
        "dropout_rate": args.dropout,
        "learning_rate": args.learning_rate,
        "lookback": args.lookback,
        "epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
    }

    for k, v in override_map.items():
        if v is not None:
            base[k] = v

    return base


# ======================================================================
# ARG PARSER
# ======================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # -------------------------
    # GLOBAL OPTIONS
    # -------------------------
    p.add_argument("--model", required=True, choices=["lstm", "xgboost"])
    p.add_argument("--ticker", required=True)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--config", default="config/default_config.yaml")
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--run-id", default=None)
    p.add_argument("--log-file", default="logs/train.log")
    p.add_argument("--device", default=None)

    # -------------------------
    # LSTM OVERRIDES
    # -------------------------
    lstm = p.add_argument_group("LSTM overrides")

    lstm.add_argument("--num-layers", type=int, default=None)
    lstm.add_argument("--hidden-units", type=int, default=None)
    lstm.add_argument("--dropout", type=float, default=None)
    lstm.add_argument("--learning-rate", type=float, default=None)
    lstm.add_argument("--lookback", type=int, default=None)
    lstm.add_argument("--max-epochs", type=int, default=None)
    lstm.add_argument("--patience", type=int, default=None)
    lstm.add_argument("--batch-size", type=int, default=None)

    # -------------------------
    # XGBOOST OPTIONS
    # -------------------------
    xgb = p.add_argument_group("XGBoost options")

    xgb.add_argument("--train-mode", choices=["default", "tune"], default="default")
    xgb.add_argument("--n-trials", type=int, default=20)

    return p.parse_args()


# ======================================================================
# LSTM TRAINING ENTRYPOINT
# ======================================================================


def train_lstm(ctx: RuntimeContext) -> None:
    from src.models_revised.lstm_pipeline import LSTMPipeline  # your pipeline class

    config = load_config(ctx.config_path)
    config = merge_overrides(config, ctx.args)

    logger.info("Loaded LSTM config: %s", config)

    # ------------------------------------------------------------
    # LOAD FEATURES
    # ------------------------------------------------------------
    X_train = np.load(f"{ctx.features_dir}/{ctx.ticker}/{ctx.ticker}_X_train.npy")
    y_train = np.load(f"{ctx.features_dir}/{ctx.ticker}/{ctx.ticker}_y_train.npy")
    X_val = np.load(f"{ctx.features_dir}/{ctx.ticker}/{ctx.ticker}_X_val.npy")
    y_val = np.load(f"{ctx.features_dir}/{ctx.ticker}/{ctx.ticker}_y_val.npy")

    X_test = None
    y_test = None

    try:
        X_test = np.load(f"{ctx.features_dir}/{ctx.ticker}_X_test.npy")
        y_test = np.load(f"{ctx.features_dir}/{ctx.ticker}_y_test.npy")
    except FileNotFoundError:
        logger.warning("Test set not found, skipping evaluation")

    # ------------------------------------------------------------
    # PIPELINE EXECUTION
    # ------------------------------------------------------------
    pipeline = LSTMPipeline(config=config, seed=ctx.seed, device=ctx.device)

    result = pipeline.run(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        config_override=config,
    )

    logger.info("LSTM training complete")

    # ------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------
    import json
    import os

    os.makedirs(ctx.results_dir, exist_ok=True)

    out_path = f"{ctx.results_dir}/{ctx.ticker}_lstm_results.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Results saved to %s", out_path)


# ======================================================================
# XGBOOST PLACEHOLDER
# ======================================================================


def train_xgboost(ctx: RuntimeContext) -> None:
    logger.info("XGBoost pipeline not implemented in this module yet")


# ======================================================================
# MAIN
# ======================================================================


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

    logger.info("run_id=%s", ctx.run_id)

    if args.model == "lstm":
        train_lstm(ctx)
    else:
        train_xgboost(ctx)


if __name__ == "__main__":
    main()
