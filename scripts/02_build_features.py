#!/usr/bin/env python3
"""
scripts/02_build_features.py

Builds feature matrices for each ticker using the individually aligned OHLCV data.
Applies feature selection (XGBoost importance) and saves to disk.

Memory-efficient: handles one ticker at a time.
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
    parser = argparse.ArgumentParser(description="Build and select features per ticker")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed",
        help="Directory containing individual aligned ticker Parquets",
    )
    parser.add_argument(
        "--output_dir",
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

    # Load config and logger
    cfg = load_config(args.config)
    setup_logger(log_file="logs/02_build_features.log", level="INFO")

    # Read tickers
    with open(args.tickers) as f:
        tickers = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    logger.info("Loaded %d tickers", len(tickers))

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train/val/test split setup
    train_years = cfg.data.train_years
    val_years = cfg.data.val_years
    test_years = cfg.data.test_years
    logger.info(
        "Using train/val/test split: %d years / %d years / %d years",
        train_years,
        val_years,
        test_years,
    )
    splitter = DataSplitter(
        train_end=cfg.data.train_end,
        val_end=cfg.data.val_end,
    )

    # Process each ticker individually
    for ticker in tickers:
        ticker_file = input_dir / f"{ticker}.parquet"
        if not ticker_file.exists():
            logger.warning("Skipping %s (no aligned data found)", ticker)
            continue

        logger.info("=" * 60)
        logger.info("Processing ticker: %s", ticker)

        # Load aligned ticker data
        df_ticker = pd.read_parquet(ticker_file)
        df_ticker.index = pd.to_datetime(df_ticker.index)

        # Filter by config dates
        df_ticker = df_ticker.loc[cfg.data.start_date : cfg.data.end_date]

        # Split train/val/test
        df_train, df_val, df_test = splitter.split(df_ticker)

        # Initialize feature pipeline
        pipeline = FeaturePipeline(
            target_ticker=ticker,
            universe_tickers=[ticker],  # only using this ticker individually
            selector_kwargs={
                "importance_cumulative": cfg.features.selector.importance_threshold,
            },
        )

        # Fit on training data
        X_train, y_train, feature_names = pipeline.fit_transform({ticker: df_train})
        logger.info("Train features: %d samples × %d features", *X_train.shape)

        # Transform val and test
        X_val, y_val = pipeline.transform({ticker: df_val})
        X_test, y_test = pipeline.transform({ticker: df_test})
        logger.info(
            "Val features: %d samples, Test features: %d samples",
            len(X_val),
            len(X_test),
        )

        # Normalize features
        X_train_norm = splitter.fit_transform(X_train, feature_names)
        X_val_norm = splitter.transform(X_val)
        X_test_norm = splitter.transform(X_test)

        # Save features per ticker
        ticker_output_dir = output_dir / ticker
        ticker_output_dir.mkdir(parents=True, exist_ok=True)

        np.save(ticker_output_dir / "X_train.npy", X_train_norm)
        np.save(ticker_output_dir / "y_train.npy", y_train)
        np.save(ticker_output_dir / "X_val.npy", X_val_norm)
        np.save(ticker_output_dir / "y_val.npy", y_val)
        np.save(ticker_output_dir / "X_test.npy", X_test_norm)
        np.save(ticker_output_dir / "y_test.npy", y_test)

        # Save metadata
        metadata = {
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        }
        with open(ticker_output_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info("Saved features for %s to %s", ticker, ticker_output_dir)

    logger.info("=" * 60)
    logger.info("Feature engineering complete!")


if __name__ == "__main__":
    main()
