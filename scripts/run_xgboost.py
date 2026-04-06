#!/usr/bin/env python3
"""
scripts/run_xgboost.py

Runs IPSO hyperparameter optimization for a given ticker.
Outputs best hyperparameters and fitness history to results/.

Usage:
    python scripts/run_xgboost.py --ticker AAPL --config config/default_config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimizer.ipso import IPSO
from src.optimizer.fitness import CompositeFitness
from src.models.lstm_model import LSTMModel, LSTMTrainer
from src.data.splitter import build_windows
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.utils.seed import set_all_seeds
from src.utils.memory_profiler import MemoryProfiler, MemoryMonitor
from src.utils.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


def model_builder(params: dict, X_train, y_train, X_val, y_val, cfg=None, args=None):
    """
    Build and train an XGBoost model with given hyperparameters.

    Returns validation predictions for fitness evaluation.
    """
    input_size = X_train.shape[2]

    # Phase 3: Apply memory optimizations
    use_checkpointing = cfg.xgboost.get("use_checkpointing", False) if cfg else False
    if args and args.no_checkpointing:
        use_checkpointing = False

    model = LSTMModel(
        input_size=input_size,
        num_layers=params["num_layers"],
        hidden_units=params["hidden_units"],
        dropout=params["dropout"],
        use_checkpointing=use_checkpointing,
    )

    # Phase 3: Apply training optimizations
    use_amp = cfg.lstm.get("use_amp", True) if cfg else True
    if args and args.no_amp:
        use_amp = False

    accumulation_steps = cfg.lstm.get("accumulation_steps", 1) if cfg else 1
    batch_size = cfg.lstm.get("batch_size", 256) if cfg else 256

    trainer = LSTMTrainer(
        model=model,
        lr=params["learning_rate"],
        max_epochs=cfg.lstm.max_epochs if cfg else 100,
        patience=cfg.lstm.early_stopping_patience if cfg else 10,
        batch_size=batch_size,
        use_amp=use_amp,
        accumulation_steps=accumulation_steps,
    )

    trainer.fit(X_train, y_train, X_val, y_val)
    y_pred = trainer.predict(X_val)

    return y_pred


def main():
    parser = argparse.ArgumentParser(description="Run IPSO hyperparameter optimization")
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker symbol to optimize",
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
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Use memory-optimized settings (smaller batch, AMP, checkpointing)",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Enable detailed memory profiling",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--no-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    args = parser.parse_args()

    # Load config (use memory-optimized if --low-memory flag set)
    if args.low_memory and "memory_optimized" not in args.config:
        logger.info("--low-memory flag set, using memory_optimized.yaml")
        cfg = load_config("config/memory_optimized.yaml")
    else:
        cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else 42
    set_all_seeds(seed)

    setup_logger(log_file=f"logs/run_xgboost_{args.ticker}.log", level="INFO")

    # Memory profiling setup
    if args.profile_memory:
        logger.info("Memory profiling enabled")
        MemoryProfiler.log_memory("Initial")
    logger.info("=" * 60)
    logger.info("XGBoost Hyperparameter Optimization: %s", args.ticker)
    logger.info("=" * 60)

    # Load feature data
    ticker_dir = Path(args.features_dir) / args.ticker
    if not ticker_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {ticker_dir}")

    logger.info("Loading features from %s", ticker_dir)

    if args.profile_memory:
        with MemoryMonitor("Loading feature data"):
            X_train_flat = np.load(ticker_dir / "X_train.npy")
            y_train = np.load(ticker_dir / "y_train.npy")
            X_val_flat = np.load(ticker_dir / "X_val.npy")
            y_val = np.load(ticker_dir / "y_val.npy")
    else:
        X_train_flat = np.load(ticker_dir / "X_train.npy")
        y_train = np.load(ticker_dir / "y_train.npy")
        X_val_flat = np.load(ticker_dir / "X_val.npy")
        y_val = np.load(ticker_dir / "y_val.npy")

    logger.info("Train: %s, Val: %s", X_train_flat.shape, X_val_flat.shape)
    logger.info("Data types: X_train=%s, y_train=%s", X_train_flat.dtype, y_train.dtype)

    # Build sliding windows for maximum lookback
    lookback_choices = getattr(getattr(cfg, "lstm", {}), "lookback", {}).get(
        "choices", [512]
    )
    max_lookback = max(lookback_choices) if lookback_choices else 512
    session_starts_train = np.zeros(len(X_train_flat), dtype=bool)
    session_starts_val = np.zeros(len(X_val_flat), dtype=bool)

    X_train_windows, y_train_windows = build_windows(
        X_train_flat, y_train, session_starts_train, max_lookback
    )
    X_val_windows, y_val_windows = build_windows(
        X_val_flat, y_val, session_starts_val, max_lookback
    )

    logger.info(
        "Windows - Train: %s, Val: %s", X_train_windows.shape, X_val_windows.shape
    )

    # Initialize fitness function
    # fitness_fn = CompositeFitness(
    #     weights={
    #         "rmse": cfg.fitness.rmse_weight,
    #         "sharpe": cfg.fitness.sharpe_weight,
    #         "mdd": cfg.fitness.drawdown_weight,
    #     },
    #     signal_threshold=cfg.fitness.signal_threshold,
    #     transaction_cost=cfg.fitness.transaction_cost,
    # )

    logger.info("  Seed: %d", seed)

    # Wrap model_builder to pass config and args
    def model_builder_wrapper(params, X_train, y_train, X_val, y_val):
        return model_builder(params, X_train, y_train, X_val, y_val, cfg, args)

    # Run optimization
    logger.info("Starting IPSO optimization...")

    if args.profile_memory:
        MemoryProfiler.reset_peak_memory()
        with MemoryMonitor("IPSO optimization"):
            best_params, best_fitness = optimizer.run(
                X_train_windows, y_train_windows, X_val_windows, y_val_windows
            )
        MemoryManager.log_memory_stats("Final")
    else:
        best_params, best_fitness = optimizer.run(
            X_train_windows, y_train_windows, X_val_windows, y_val_windows
        )

    logger.info("=" * 60)
    logger.info("Optimization complete!")
    logger.info("Best fitness: %.6f", best_fitness)
    logger.info("Best hyperparameters:")
    for key, value in best_params.items():
        logger.info("  %s: %s", key, value)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "ticker": args.ticker,
        "best_params": best_params,
        "best_fitness": float(best_fitness),
        "fitness_history": [float(f) for f in optimizer.fitness_history],
        "diversity_history": [float(d) for d in optimizer.diversity_history],
        "config": {
            "n_particles": n_particles,
            "n_iterations": n_iterations,
            "seed": seed,
        },
    }

    output_file = output_dir / f"pso_results_{args.ticker}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", output_file)


if __name__ == "__main__":
    main()
