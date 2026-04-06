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
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        default="lstm_baseline",
        help="Model to train.",
    )
    args = parser.parse_args()

    # Load config (use memory-optimized if --low-memory flag set)
    if args.low_memory and "memory_optimized" not in args.config:
        logger.info("--low-memory flag set, using memory_optimized.yaml")
        cfg = load_config("config/memory_optimized.yaml")
    else:
        cfg = load_config(args.config)

    seed = (
        args.seed
        if args.seed is not None
        else getattr(getattr(cfg, "pso", {}), "seed", 42)
    )
    set_all_seeds(seed)

    setup_logger(log_file=f"logs/03_run_pso_{args.ticker}.log", level="INFO")

    # Memory profiling setup
    if args.profile_memory:
        logger.info("Memory profiling enabled")
        MemoryProfiler.log_memory("Initial")
    logger.info("=" * 60)
    logger.info("IPSO Hyperparameter Optimization: %s", args.ticker)
    logger.info("=" * 60)

    # Log memory optimizations
    lstm_cfg = getattr(cfg, "lstm", {})
    if lstm_cfg.get("use_amp") and not args.no_amp:
        logger.info("✓ Mixed precision training enabled (FP16)")
    if lstm_cfg.get("use_checkpointing") and not args.no_checkpointing:
        logger.info("✓ Gradient checkpointing enabled")
    if lstm_cfg.get("accumulation_steps", 1) > 1:
        logger.info(
            "✓ Gradient accumulation: %d steps (effective batch=%d)",
            lstm_cfg.get("accumulation_steps"),
            lstm_cfg.get("batch_size") * lstm_cfg.get("accumulation_steps"),
        )

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
