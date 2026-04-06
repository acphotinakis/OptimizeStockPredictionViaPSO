#!/usr/bin/env python3
"""
scripts/02_build_features.py

Builds feature matrices for each ticker using the aligned OHLCV data.
Loads ALL tickers to compute proper cross-ticker features (SPY correlation, etc.).
Applies feature selection (XGBoost importance) and saves to disk.

GPU acceleration: Automatically uses GPU for XGBoost if available.

Usage:
    python scripts/02_build_features.py --config config/default_config.yaml
    python scripts/02_build_features.py --config config/default_config.yaml --n-jobs 4
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.pipeline import FeaturePipeline
from src.data.splitter import DataSplitter
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build and select features")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/aligned_universe.parquet",
        help="Path to aligned universe parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/features",
        help="Output directory for feature matrices",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="config/tickers.txt",
        help="Path to ticker list file",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (for future use)",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    setup_logger(log_file="logs/02_build_features.log", level="INFO")

    # Check GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("🚀 GPU detected: %s (%.1f GB)", gpu_name, gpu_memory)
            logger.info("XGBoost will automatically use GPU acceleration")
        else:
            logger.info("No GPU detected - using CPU only")
    except ImportError:
        gpu_available = False
        logger.info("PyTorch not available - using CPU only")

    # Read ticker list
    with open(args.tickers) as f:
        tickers = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    logger.info("Loaded %d tickers", len(tickers))

    # Load aligned data
    logger.info("Loading aligned data from %s", args.input)
    df_aligned = pd.read_parquet(args.input)
    logger.info("Loaded data shape: %s", df_aligned.shape)

    # Split into train/val/test
    splitter = DataSplitter(
        train_end=cfg.data.train_end,
        val_end=cfg.data.val_end,
    )

    # Split the aligned data
    df_train_all, df_val_all, df_test_all = splitter.split(df_aligned)
    logger.info("Split complete - Train: %d, Val: %d, Test: %d", 
                len(df_train_all), len(df_val_all), len(df_test_all))

    # Convert MultiIndex DataFrames to dict of single-ticker DataFrames
    def extract_ticker_dfs(df_multi):
        """Extract individual ticker DataFrames from MultiIndex DataFrame."""
        ticker_dfs = {}
        for ticker in tickers:
            if ticker not in df_multi.columns.get_level_values(0):
                continue
            ticker_dfs[ticker] = df_multi[ticker].copy()
        return ticker_dfs

    dfs_train = extract_ticker_dfs(df_train_all)
    dfs_val = extract_ticker_dfs(df_val_all)
    dfs_test = extract_ticker_dfs(df_test_all)
    
    logger.info("Extracted %d tickers for feature engineering", len(dfs_train))

    # Build features for each ticker
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, ticker in enumerate(tickers, 1):
        if ticker not in dfs_train:
            logger.warning("Skipping %s (not in training data)", ticker)
            continue

        logger.info("=" * 60)
        logger.info("[%d/%d] Processing ticker: %s", idx, len(tickers), ticker)

        # Initialize feature pipeline with full universe for cross-ticker features
        pipeline = FeaturePipeline(
            target_ticker=ticker,
            universe_tickers=list(dfs_train.keys()),
            selector_kwargs={
                "importance_cumulative": cfg.features.selector.importance_threshold,
            },
        )

        # Fit on training data (pass ALL tickers for cross-ticker features)
        logger.info("  Computing and selecting features...")
        X_train, y_train, feature_names = pipeline.fit_transform(dfs_train)
        logger.info("  Train features: %d samples × %d features", *X_train.shape)

        # Transform val and test (pass ALL tickers for cross-ticker features)
        X_val, y_val = pipeline.transform(dfs_val)
        X_test, y_test = pipeline.transform(dfs_test)
        logger.info("  Val: %d samples, Test: %d samples", len(X_val), len(X_test))

        # Normalize features
        logger.info("  Normalizing features...")
        X_train_norm = splitter.fit_transform(X_train, feature_names)
        X_val_norm = splitter.transform(X_val)
        X_test_norm = splitter.transform(X_test)

        # Save to disk
        ticker_dir = output_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        np.save(ticker_dir / "X_train.npy", X_train_norm)
        np.save(ticker_dir / "y_train.npy", y_train)
        np.save(ticker_dir / "X_val.npy", X_val_norm)
        np.save(ticker_dir / "y_val.npy", y_val)
        np.save(ticker_dir / "X_test.npy", X_test_norm)
        np.save(ticker_dir / "y_test.npy", y_test)

        # Save metadata
        metadata = {
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        }
        with open(ticker_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info("  ✓ Saved features to %s", ticker_dir)

    logger.info("=" * 60)
    logger.info("✓ Feature engineering complete!")


if __name__ == "__main__":
    main()


