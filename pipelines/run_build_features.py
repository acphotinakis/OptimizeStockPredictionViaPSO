#!/usr/bin/env python3
"""
Builds feature matrices for each ticker using aligned OHLCV data.
Optimized for memory, parallelization, and GPU-aware XGBoost.
"""


import argparse
import logging
from pathlib import Path
import pickle
import sys
import gc
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.pipeline import FeaturePipeline
from src.features.universe_builder import SymbolUniverseBuilder
from src.data.splitter import DataSplitter
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.features.scalar import PipelineScaler
import yaml

logger = logging.getLogger(__name__)


def process_ticker(
    ticker: str,
    dfs_train,
    dfs_val,
    dfs_test,
    cfg,
    output_dir,
    splitter: DataSplitter,
    universe_builder: SymbolUniverseBuilder,
):
    if ticker not in dfs_train:
        logger.info("Skipping %s (not in training data)", ticker)
        return

    pipeline = FeaturePipeline(
        target_ticker=ticker,
        universe_builder=universe_builder,
        selector_kwargs={
            # "importance_cumulative": cfg.features.selector.importance_threshold
        },
    )

    # --- Fit feature pipeline on TRAIN ---
    X_train, y_train, feature_names = pipeline.fit_transform(dfs_train)

    # --- Transform VAL and TEST ---
    X_val, y_val = pipeline.transform(dfs_val)
    X_test, y_test = pipeline.transform(dfs_test)

    # --- Scaling with PipelineScaler ---
    scaler = PipelineScaler()

    # Combine features + target into DataFrames
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["log_return"] = y_train
    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df["log_return"] = y_val
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["log_return"] = y_test

    # Fit scaler on TRAIN only
    scaler.fit(train_df, target_col="log_return", feature_cols=feature_names)

    # Transform all splits
    train_scaled = scaler.transform(train_df, target_col="log_return")
    val_scaled = scaler.transform(val_df, target_col="log_return")
    test_scaled = scaler.transform(test_df, target_col="log_return")

    # Extract scaled NumPy arrays
    X_train_scaled = train_scaled[feature_names].to_numpy(dtype=np.float32)
    y_train_scaled = train_scaled["log_return"].to_numpy(dtype=np.float32)
    X_val_scaled = val_scaled[feature_names].to_numpy(dtype=np.float32)
    y_val_scaled = val_scaled["log_return"].to_numpy(dtype=np.float32)
    X_test_scaled = test_scaled[feature_names].to_numpy(dtype=np.float32)
    y_test_scaled = test_scaled["log_return"].to_numpy(dtype=np.float32)

    assert "log_return" not in feature_names, "Target column found in X!"

    # --- Save scaled arrays ---
    ticker_dir = Path(output_dir) / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in zip(
        ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"],
        [
            X_train_scaled,
            y_train_scaled,
            X_val_scaled,
            y_val_scaled,
            X_test_scaled,
            y_test_scaled,
        ],
    ):
        np.save(ticker_dir / f"{name}.npy", arr, allow_pickle=False)

    # --- Save scaler for later use ---
    scaler.save(ticker_dir / "scaler.pkl")

    # --- Save metadata ---
    metadata = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "train_samples": len(X_train_scaled),
        "val_samples": len(X_val_scaled),
        "test_samples": len(X_test_scaled),
    }
    with open(ticker_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    logger.info("✓ %s complete (%d features)", ticker, len(feature_names))

    # --- Clean memory ---
    del (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        train_df,
        val_df,
        test_df,
        train_scaled,
        val_scaled,
        test_scaled,
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        scaler,
    )
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Build and select features")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument(
        "--input", type=str, default="data/processed/aligned_universe.parquet"
    )
    parser.add_argument("--output", type=str, default="data/features")
    parser.add_argument("--tickers", type=str, default="config/tickers.txt")
    parser.add_argument(
        "--universe-config", type=str, default="config/symbol_universe.yaml"
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logger(log_file="logs/02_build_features.log", level="INFO")

    # Load symbol universe builder
    universe_builder = None
    if Path(args.universe_config).exists():
        logger.info(f"Loading symbol universe builder from {args.universe_config}")
        universe_builder = SymbolUniverseBuilder.from_config(args.universe_config)
    else:
        logger.warning(
            f"Symbol universe config not found at {args.universe_config}. "
            f"Using legacy mode (all tickers as universe)."
        )

    # GPU info
    try:
        import torch

        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("GPU detected: %s (%.1f GB)", gpu_name, gpu_memory)
    except ImportError:
        gpu_available = False
        logger.info("PyTorch not available - using CPU only")

    # --- Load tickers ---
    with open(args.tickers) as f:
        tickers = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
    logger.info("Loaded %d tickers", len(tickers))

    # --- Load aligned data ---
    # df_aligned = cudf.read_parquet(args.input)
    # logger.info("Loaded aligned data: %s", df_aligned.shape)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.info(
            "Input Parquet %s not found, merging individual tickers...", input_path
        )
        # Load SPY first
        raw_dir = Path("data/processed")
        spy_path = raw_dir / "SPY.parquet"
        if not spy_path.exists():
            raise FileNotFoundError(f"SPY file not found at {spy_path}")
        df_spy = pd.read_parquet(spy_path)
        logger.info(
            "SPY: shape=%s, columns=%s, rows=%d",
            df_spy.shape,
            df_spy.columns.tolist(),
            len(df_spy),
        )

        merged_dfs = {"SPY": df_spy}

        # Load other tickers
        for ticker in tickers:
            if ticker == "SPY":
                continue
            ticker_path = raw_dir / f"{ticker}.parquet"
            if not ticker_path.exists():
                logger.info("Ticker %s not found at %s, skipping", ticker, ticker_path)
                continue
            df_ticker = pd.read_parquet(ticker_path)
            logger.info(
                "%s: shape=%s, columns=%s, rows=%d",
                ticker,
                df_ticker.shape,
                df_ticker.columns.tolist(),
                len(df_ticker),
            )
            merged_dfs[ticker] = df_ticker

        # Merge into MultiIndex columns: ticker × fields
        df_aligned = pd.concat(
            merged_dfs, axis=1, names=["Ticker", "Field"]
        ).sort_index(axis=1)
        # Save for future runs
        input_path.parent.mkdir(parents=True, exist_ok=True)
        df_aligned.to_parquet(input_path)
        logger.info("Merged aligned universe saved to %s", input_path)
    else:
        df_aligned = pd.read_parquet(input_path)
        logger.info("Loaded aligned data: %s", df_aligned.shape)

    # --- Split data ---
    splitter = DataSplitter()
    df_train_all, df_val_all, df_test_all = splitter.split(df_aligned)

    # --- Extract per-ticker DataFrames ---
    def extract_ticker_dfs(df_multi, tickers):
        """Return a dict of {ticker: DataFrame} for all tickers in df_multi"""
        return {
            ticker: df_multi.xs(ticker, axis=1, level=0)
            for ticker in tickers
            if ticker in df_multi.columns.get_level_values(0)
        }

    dfs_train = extract_ticker_dfs(df_train_all, tickers)
    dfs_val = extract_ticker_dfs(df_val_all, tickers)
    dfs_test = extract_ticker_dfs(df_test_all, tickers)

    # Ensure correct index types
    for df_dict in [dfs_train, dfs_val, dfs_test]:
        for ticker, df in df_dict.items():
            df.index = pd.to_datetime(df.index).tz_localize(None)

    logger.info("Extracted %d tickers for feature engineering", len(dfs_train))

    # --- Clean memory ---
    del df_aligned, df_train_all, df_val_all, df_test_all
    gc.collect()

    # --- Output directory ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which tickers to process
    if universe_builder is not None:
        # Load prediction targets from universe config
        with open(args.universe_config) as f:
            universe_cfg = yaml.safe_load(f)
        prediction_targets = universe_cfg.get("prediction_targets", [])

        if prediction_targets:
            logger.info(
                f"Processing {len(prediction_targets)} prediction targets from universe config"
            )
            tickers_to_process = prediction_targets
        else:
            logger.warning(
                "No prediction_targets in universe config, processing all tickers"
            )
            tickers_to_process = tickers
    else:
        # Legacy mode: process all tickers
        tickers_to_process = tickers

    # --- Parallel feature construction ---
    logger.info(
        "Starting feature generation for %d tickers with %d jobs",
        len(tickers_to_process),
        args.n_jobs,
    )
    for ticker in tickers_to_process:
        process_ticker(
            ticker,
            dfs_train,
            dfs_val,
            dfs_test,
            cfg,
            output_dir,
            splitter,
            universe_builder,
        )

    # Save fitted peers for reproducibility
    if universe_builder is not None:
        peers_path = Path(output_dir) / "fitted_peers.json"
        universe_builder.save_peers(peers_path)
        logger.info(f"Saved fitted peers to {peers_path}")

    logger.info("=" * 60)
    logger.info("✓ Feature engineering complete!")


if __name__ == "__main__":
    main()
