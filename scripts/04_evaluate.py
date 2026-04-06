#!/usr/bin/env python3
"""
scripts/04_evaluate.py

Evaluates the IPSO-optimized LSTM model on the test set.
Compares against baseline models (Persistence, Vanilla LSTM, XGBoost).

Usage:
    python scripts/04_evaluate.py --ticker AAPL --config config/default_config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.memory_profiler import MemoryMonitor

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.lstm_model import LSTMModel, LSTMTrainer
from src.models.baselines import PersistenceModel, VanillaLSTM, XGBoostBaseline
from src.evaluation.metrics import all_statistical_metrics
from src.data.splitter import build_windows
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


def train_and_evaluate_model(model_name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train a model and evaluate on test set."""
    logger.info("Training %s...", model_name)
    
    if hasattr(model, 'fit'):
        if model_name == "Persistence":
            model.fit(X_train, y_train)
            y_pred = model.predict_from_returns(y_test)
        else:
            model.fit(X_train, y_train, X_val, y_val)
            y_pred = model.predict(X_test)
    else:
        raise ValueError(f"Model {model_name} has no fit method")
    
    metrics = all_statistical_metrics(y_test, y_pred)
    logger.info("%s metrics:", model_name)
    for key, value in metrics.items():
        logger.info("  %s: %.6f", key, value)
    
    return y_pred, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate IPSO-LSTM and baselines")
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker symbol to evaluate",
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
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for evaluation results",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    set_all_seeds(cfg.pso.seed)
    
    setup_logger(log_file=f"logs/04_evaluate_{args.ticker}.log", level="INFO")
    logger.info("=" * 60)
    logger.info("Model Evaluation: %s", args.ticker)
    logger.info("=" * 60)
    
    if args.quantize:
        logger.info("✓ Quantization enabled (int8)")
    if args.profile_memory:
        logger.info("✓ Memory profiling enabled")
        MemoryProfiler.log_memory("Initial")

    # Load PSO results
    pso_file = Path(args.pso_results) / f"pso_results_{args.ticker}.json"
    if not pso_file.exists():
        raise FileNotFoundError(f"PSO results not found: {pso_file}")
    
    with open(pso_file) as f:
        pso_results = json.load(f)
    
    best_params = pso_results["best_params"]
    logger.info("Loaded IPSO best hyperparameters:")
    for key, value in best_params.items():
        logger.info("  %s: %s", key, value)

    # Load feature data
    ticker_dir = Path(args.features_dir) / args.ticker
    X_train_flat = np.load(ticker_dir / "X_train.npy")
    y_train = np.load(ticker_dir / "y_train.npy")
    X_val_flat = np.load(ticker_dir / "X_val.npy")
    y_val = np.load(ticker_dir / "y_val.npy")
    X_test_flat = np.load(ticker_dir / "X_test.npy")
    y_test = np.load(ticker_dir / "y_test.npy")

    # Build sliding windows
    lookback = best_params["lookback"]
    session_starts = np.zeros(len(X_train_flat), dtype=bool)
    
    X_train, y_train_w = build_windows(X_train_flat, y_train, session_starts[:len(X_train_flat)], lookback)
    X_val, y_val_w = build_windows(X_val_flat, y_val, session_starts[:len(X_val_flat)], lookback)
    X_test, y_test_w = build_windows(X_test_flat, y_test, session_starts[:len(X_test_flat)], lookback)
    
    logger.info("Window shapes - Train: %s, Val: %s, Test: %s", X_train.shape, X_val.shape, X_test.shape)

    input_size = X_train.shape[2]
    results = {}

    # 1. IPSO-LSTM
    logger.info("\n" + "=" * 60)
    logger.info("1. IPSO-Optimized LSTM")
    logger.info("=" * 60)
    
    use_checkpointing = cfg.lstm.get("use_checkpointing", False)
    ipso_model = LSTMModel(
        input_size=input_size,
        num_layers=best_params["num_layers"],
        hidden_units=best_params["hidden_units"],
        dropout=best_params["dropout"],
        use_checkpointing=use_checkpointing,
    )
    
    use_amp = cfg.lstm.get("use_amp", True)
    accumulation_steps = cfg.lstm.get("accumulation_steps", 1)
    
    ipso_trainer = LSTMTrainer(
        model=ipso_model,
        lr=best_params["learning_rate"],
        max_epochs=cfg.lstm.max_epochs,
        patience=cfg.lstm.early_stopping_patience,
        batch_size=cfg.lstm.batch_size,
        use_amp=use_amp,
        accumulation_steps=accumulation_steps,
    )
    
    if args.profile_memory:
        with MemoryMonitor("Training IPSO-LSTM"):
            y_pred_ipso, metrics_ipso = train_and_evaluate_model(
                "IPSO-LSTM", ipso_trainer, X_train, y_train_w, X_val, y_val_w, X_test, y_test_w
            )
    else:
        y_pred_ipso, metrics_ipso = train_and_evaluate_model(
            "IPSO-LSTM", ipso_trainer, X_train, y_train_w, X_val, y_val_w, X_test, y_test_w
        )
    
    # Phase 2: Quantize if requested
    if args.quantize:
        logger.info("Quantizing IPSO-LSTM model...")
        quantized_wrapper = QuantizedLSTMModel(ipso_model)
        quantized_wrapper.quantize()
        y_pred_ipso = quantized_wrapper.predict(X_test)
        
        # Save quantized model
        quantized_path = output_dir / f"quantized_ipso_lstm_{args.ticker}.pt"
        quantized_wrapper.save(str(quantized_path))
        logger.info("Saved quantized model to %s", quantized_path)
    
    results["IPSO-LSTM"] = metrics_ipso

    # 2. Vanilla LSTM
    logger.info("\n" + "=" * 60)
    logger.info("2. Vanilla LSTM (Fixed Hyperparameters)")
    logger.info("=" * 60)
    
    vanilla_model = VanillaLSTM(input_size=input_size)
    y_pred_vanilla, metrics_vanilla = train_and_evaluate_model(
        "Vanilla-LSTM", vanilla_model, X_train, y_train_w, X_val, y_val_w, X_test, y_test_w
    )
    results["Vanilla-LSTM"] = metrics_vanilla

    # 3. XGBoost
    logger.info("\n" + "=" * 60)
    logger.info("3. XGBoost Baseline")
    logger.info("=" * 60)
    
    xgb_model = XGBoostBaseline(lookback=lookback)
    y_pred_xgb, metrics_xgb = train_and_evaluate_model(
        "XGBoost", xgb_model, X_train, y_train_w, X_val, y_val_w, X_test, y_test_w
    )
    results["XGBoost"] = metrics_xgb

    # 4. Persistence
    logger.info("\n" + "=" * 60)
    logger.info("4. Persistence (Naive) Baseline")
    logger.info("=" * 60)
    
    persist_model = PersistenceModel()
    y_pred_persist, metrics_persist = train_and_evaluate_model(
        "Persistence", persist_model, X_train, y_train_w, X_val, y_val_w, X_test, y_test_w
    )
    results["Persistence"] = metrics_persist

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"evaluation_{args.ticker}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete! Results saved to %s", output_file)
    logger.info("=" * 60)

    # Print comparison table
    logger.info("\nModel Comparison:")
    logger.info("-" * 80)
    logger.info("%-20s %10s %10s %10s %10s", "Model", "RMSE", "DA", "F1", "R²")
    logger.info("-" * 80)
    for model_name, metrics in results.items():
        logger.info(
            "%-20s %10.6f %10.4f %10.4f %10.4f",
            model_name,
            metrics["rmse"],
            metrics["directional_accuracy"],
            metrics["f1_ternary"],
            metrics["r2"],
        )
    logger.info("-" * 80)


if __name__ == "__main__":
    main()
