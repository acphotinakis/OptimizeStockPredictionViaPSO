#!/usr/bin/env python3
"""
scripts/evaluate.py

Evaluate IPSO-optimised LSTM against baselines on the test set.

Usage:
    python scripts/evaluate.py --ticker AAPL
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common import make_tag, make_windows, save_json, setup
from src.data.splitter import build_windows
from src.evaluation.metrics import (
    compute_and_log_all_statistical_metrics,
)
from src.models.lstm.lstm_model import LSTMModel, LSTMTrainer
from src.models.baselines import PersistenceModel, VanillaLSTM, XGBoostBaseline

logger = logging.getLogger(__name__)


def _train_and_predict(name: str, model, X_tr, y_tr, X_va, y_va, X_te, y_te):
    logger.info("Training %s...", name)
    if name == "Persistence":
        model.fit(X_tr, y_tr)
        y_pred = model.predict_from_returns(y_te)
    else:
        model.fit(X_tr, y_tr, X_va, y_va)
        y_pred = model.predict(X_te)
    metrics = compute_and_log_all_statistical_metrics(
        y_te, y_pred, label=f"[{name}] Prediction Stats"
    )

    return y_pred, metrics


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ticker", required=True)
    p.add_argument("--config", default="config/default_config.yaml")
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--pso-results", default="results")
    p.add_argument("--output-dir", default="results/evaluation")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = setup(f"logs/evaluate_{args.ticker}.log", args.seed, args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PSO results
    import json

    pso_file = Path(args.pso_results) / f"pso_results_{args.ticker}.json"
    if not pso_file.exists():
        raise FileNotFoundError(
            f"PSO results not found: {pso_file}. Run run_pso.py first."
        )
    best_params = json.loads(pso_file.read_text())["best_params"]
    logger.info("Best PSO params: %s", best_params)

    # Load data
    ticker_dir = Path(args.features_dir) / args.ticker
    X_train_flat = np.load(ticker_dir / "X_train.npy")
    y_train = np.load(ticker_dir / "y_train.npy")
    X_val_flat = np.load(ticker_dir / "X_val.npy")
    y_val = np.load(ticker_dir / "y_val.npy")
    X_test_flat = np.load(ticker_dir / "X_test.npy")
    y_test = np.load(ticker_dir / "y_test.npy")

    lookback = best_params["lookback"]
    X_train, y_tr_w = make_windows(X_train_flat, y_train, lookback)
    X_val, y_va_w = make_windows(X_val_flat, y_val, lookback)
    X_test, y_te_w = make_windows(X_test_flat, y_test, lookback)
    logger.info(
        "Windows — Train=%s  Val=%s  Test=%s", X_train.shape, X_val.shape, X_test.shape
    )

    input_size = X_train.shape[2]
    results = {}

    # 1. IPSO-LSTM
    ipso_model = LSTMModel(
        input_size=input_size,
        num_layers=best_params["num_layers"],
        hidden_units=best_params["hidden_units"],
        dropout=best_params["dropout"],
    )
    ipso_trainer = LSTMTrainer(
        model=ipso_model,
        lr=best_params["learning_rate"],
        max_epochs=cfg.lstm.max_epochs,
        patience=cfg.lstm.early_stopping_patience,
        batch_size=cfg.lstm.batch_size,
        use_amp=cfg.lstm.use_amp,
        accumulation_steps=cfg.lstm.accumulation_steps,
    )
    _, results["IPSO-LSTM"] = _train_and_predict(
        "IPSO-LSTM", ipso_trainer, X_train, y_tr_w, X_val, y_va_w, X_test, y_te_w
    )
    _, results["Vanilla-LSTM"] = _train_and_predict(
        "Vanilla-LSTM",
        VanillaLSTM(input_size),
        X_train,
        y_tr_w,
        X_val,
        y_va_w,
        X_test,
        y_te_w,
    )
    _, results["XGBoost"] = _train_and_predict(
        "XGBoost",
        XGBoostBaseline(lookback),
        X_train,
        y_tr_w,
        X_val,
        y_va_w,
        X_test,
        y_te_w,
    )
    _, results["Persistence"] = _train_and_predict(
        "Persistence",
        PersistenceModel(),
        X_train,
        y_tr_w,
        X_val,
        y_va_w,
        X_test,
        y_te_w,
    )

    save_json(output_dir / f"evaluation_{args.ticker}.json", results)

    logger.info("\n%-20s %10s %10s %10s %10s", "Model", "RMSE", "DA", "F1", "R²")
    logger.info("-" * 60)
    for name, m in results.items():
        logger.info(
            "%-20s %10.6f %10.4f %10.4f %10.4f",
            name,
            m["rmse"],
            m["directional_accuracy"],
            m["f1_ternary"],
            m["r2"],
        )
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
