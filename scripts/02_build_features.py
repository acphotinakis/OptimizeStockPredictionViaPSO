#!/usr/bin/env python3
"""
scripts/02_build_features.py

Builds feature matrices for each ticker using the aligned OHLCV data.
Applies feature selection (XGBoost importance) and saves to disk.

Usage:
    python scripts/02_build_features.py --config config/default_config.yaml
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
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    setup_logger(log_file="logs/02_build_features.log", level="INFO")

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
        train_end="2022-01-03",
        val_end="2023-01-03",
    )

    # Convert MultiIndex DataFrame to dict of single-ticker DataFrames
    dfs_all = {}
    for ticker in tickers:
        if ticker not in df_aligned.columns.get_level_values(0):
            logger.warning("Ticker %s not in aligned data", ticker)
            continue
        df_ticker = df_aligned[ticker].copy()
        dfs_all[ticker] = df_ticker

    df_train_all, df_val_all, df_test_all = splitter.split(df_aligned)

    # Build features for each ticker
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        if ticker not in dfs_all:
            continue

        logger.info("=" * 60)
        logger.info("Processing ticker: %s", ticker)

        # Extract single-ticker DataFrames for train/val/test
        dfs_train = {t: df_train_all[t] for t in dfs_all if t in df_train_all.columns.get_level_values(0)}
        dfs_val = {t: df_val_all[t] for t in dfs_all if t in df_val_all.columns.get_level_values(0)}
        dfs_test = {t: df_test_all[t] for t in dfs_all if t in df_test_all.columns.get_level_values(0)}

        # Initialize feature pipeline
        pipeline = FeaturePipeline(
            target_ticker=ticker,
            universe_tickers=list(dfs_all.keys()),
            selector_kwargs={
                "importance_cumulative": cfg.features.selector.importance_threshold,
            },
        )

        # Fit on training data
        X_train, y_train, feature_names = pipeline.fit_transform(dfs_train)
        logger.info("Train features: %d samples × %d features", *X_train.shape)

        # Transform val and test
        X_val, y_val = pipeline.transform(dfs_val)
        X_test, y_test = pipeline.transform(dfs_test)
        logger.info("Val features: %d samples", len(X_val))
        logger.info("Test features: %d samples", len(X_test))

        # Normalize features
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

        logger.info("Saved features to %s", ticker_dir)

    logger.info("=" * 60)
    logger.info("Feature engineering complete!")


if __name__ == "__main__":
    main()
