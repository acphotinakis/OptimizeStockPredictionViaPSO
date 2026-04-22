#!/usr/bin/env python3
"""
Unified Feature Engineering Pipeline

Production-grade pipeline that uses the unified feature system (src/features_unified/).

Key Features:
- TRD-compliant feature generation (36 indicators)
- Wavelet denoising (Haar, 3-level, soft thresholding)
- 4-stage feature selection (Variance → Pearson → VIF → MI)
- Cross-ticker features (SPY + peers + market breadth)
- MinMax normalization to [-1, 1]
- Strict leakage prevention (all transformations fit on training only)

Author: System Architect
Version: 1.0.0 UNIFIED
"""

import argparse
import gc
import hashlib
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from matplotlib import ticker
import numpy as np
import pandas as pd
import yaml


# Project root setup
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Unified imports
from src.features import (
    FeaturePipeline,
    transform_features,
    save_transformer_state,
)
from src.features.pipeline import FeaturePipeline
from src.features.universe_builder import SymbolUniverseBuilder
from src.data.splitter import DataSplitter
from src.data.alpaca_ingestor import AlpacaIngestor
from src.utils.logger import LogFileMode, setup_logger
from src.utils.config_loader import Config, load_config

logger = logging.getLogger(__name__)

TARGET_COL = "log_return"


NY_TZ = "America/New_York"


def filter_us_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce US equity session: 09:00–16:30 New York time.

    Rules:
    - index must be datetime-like
    - converted to America/New_York
    - filtered to intraday trading session only
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # If naive timestamps → assume UTC (common in parquet pipelines)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Convert to New York time
    df = df.tz_convert(NY_TZ)

    # Filter trading hours
    df = df.between_time("09:00", "16:30")

    # Return as naive timestamps again (recommended for ML pipelines)
    df.index = df.index.tz_localize(None)

    return df


def process_ticker(
    ticker: str,
    dfs_train: dict,
    dfs_val: dict,
    dfs_test: dict,
    universe_builder: SymbolUniverseBuilder,
    output_dir: Path,
    config: Config,
) -> None:

    # ============================================================
    # GUARD
    # ============================================================
    if ticker not in dfs_train:
        logger.info("Skipping %s (not in training data)", ticker)
        return

    logger.info("[%s] START PIPELINE", ticker)

    target_ticker = ticker

    # ============================================================
    # STAGE 0: FEATURE PREPARATION (TARGET ONLY)
    # ============================================================
    def add_log_return(dfs: dict, name: str) -> None:
        logger.info("[%s] Computing log_return for %s", ticker, name)
        dfs[ticker]["log_return"] = np.log(dfs[ticker]["close"]).diff()

    add_log_return(dfs_train, "train")
    add_log_return(dfs_val, "val")
    add_log_return(dfs_test, "test")

    # ============================================================
    # STAGE 1: UNIVERSE + PEERS
    # ============================================================
    universe = universe_builder.get_universe(ticker, dfs_train, fit=True)
    peers = universe_builder.get_fitted_peers(ticker)

    logger.info("[%s] Universe size=%d | peers=%s", ticker, len(universe), peers)

    # ============================================================
    # FILTERING UTILITY
    # ============================================================
    def filter_universe(dfs: dict, split_name: str) -> dict:
        filtered = {t: dfs[t] for t in universe if t in dfs}

        logger.info("\n[%s] ===== %s DATA DEBUG =====", ticker, split_name)
        for t, df in filtered.items():
            logger.info("[%s] %s shape=%s", ticker, t, df.shape)

        logger.info(
            "[%s] TARGET [%s] columns: %s",
            ticker,
            split_name,
            filtered[ticker].columns.tolist(),
        )
        # log out the tickers in this split
        logger.info(
            "[%s] %s split tickers: %s", ticker, split_name, list(filtered.keys())
        )
        return filtered

    dfs_train_f = filter_universe(dfs_train, "TRAIN")

    # log the columns in this dfs
    for t, df in dfs_train_f.items():
        logger.info(f"[{ticker}] TRAIN ticker '{t}' columns: {df.columns.tolist()}")

    # log the min and max timestamps in this dfs
    for t, df in dfs_train_f.items():
        logger.info(
            f"[{ticker}] TRAIN ticker '{t}' date range: {df.index.min()} to {df.index.max()}"
        )
    dfs_val_f = filter_universe(dfs_val, "VAL")
    dfs_test_f = filter_universe(dfs_test, "TEST")

    # ============================================================
    # STAGE 2: FEATURE PIPELINE
    # ============================================================
    # enable_wavelet = config.features.wavelet.enabled
    enable_wavelet = False

    pipeline = FeaturePipeline(
        target_ticker=target_ticker,
        peer_tickers=peers,
        selector_kwargs={
            "variance_threshold": config.features.selector.variance_threshold,
            "correlation_threshold": config.features.selector.correlation_threshold,
            "vif_threshold": config.features.selector.vif_threshold,
            "mi_quantile_threshold": config.features.selector.mi_quantile_threshold,
        },
        enable_wavelet=enable_wavelet,
    )

    logger.info("[%s] Running feature pipeline (FIT)", ticker)

    X_train, y_train, feature_names, train_idx = pipeline.fit_transform(dfs_train_f)

    logger.info(
        "[%s] TRAIN shape=%s features=%d",
        ticker,
        X_train.shape,
        len(feature_names),
    )

    logger.info("[%s] Running feature pipeline (TRANSFORM)", ticker)

    X_val, y_val, val_idx = pipeline.transform(dfs_val_f)
    X_test, y_test, test_idx = pipeline.transform(dfs_test_f)

    logger.info("[%s] VAL shape=%s TEST shape=%s", ticker, X_val.shape, X_test.shape)

    # ============================================================
    # STAGE 3: NORMALIZATION
    # ============================================================
    logger.info("[%s] Stage 3: Normalization", ticker)

    train_df = pd.DataFrame(X_train, columns=feature_names, index=train_idx)
    val_df = pd.DataFrame(X_val, columns=feature_names, index=val_idx)
    test_df = pd.DataFrame(X_test, columns=feature_names, index=test_idx)

    logger.info(
        "[%s] Before normalization: train min=%.4f max=%.4f | val min=%.4f max=%.4f | test min=%.4f max=%.4f",
        ticker,
        train_df.min().min(),
        train_df.max().max(),
        val_df.min().min(),
        val_df.max().max(),
        test_df.min().min(),
        test_df.max().max(),
    )
    # log out the shapes of each DataFrame before normalization
    logger.info(
        "[%s] DataFrame shapes before normalization: train=%s | val=%s | test=%s",
        ticker,
        train_df.shape,
        val_df.shape,
        test_df.shape,
    )

    # log out the columns of each dataframe before normalization
    logger.info(
        "[%s] DataFrame columns before normalization: train=%s | val=%s | test=%s",
        ticker,
        train_df.columns.tolist(),
        val_df.columns.tolist(),
        test_df.columns.tolist(),
    )

    noramlization_wavelet_threshold = (
        pipeline.wavelet_threshold if enable_wavelet else None
    )
    train_scaled, val_scaled, test_scaled, transformer_state = transform_features(
        train_df, val_df, test_df, noramlization_wavelet_threshold
    )

    X_train = train_scaled.values.astype(np.float32)
    X_val = val_scaled.values.astype(np.float32)
    X_test = test_scaled.values.astype(np.float32)

    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # log out the shapes of each DataFrame after normalization
    logger.info(
        "[%s] DataFrame shapes after normalization: train=%s | val=%s | test=%s",
        ticker,
        train_scaled.shape,
        val_scaled.shape,
        test_scaled.shape,
    )
    # log out the shapes of each target_col after normalization
    logger.info(
        "[%s] DataFrame shapes after normalization: train_y=%s | val_y=%s | test_y=%s",
        ticker,
        y_train.shape,
        y_val.shape,
        y_test.shape,
    )

    # log out the columns of each dataframe after normalization
    logger.info(
        "[%s] DataFrame columns after normalization: train=%s | val=%s | test=%s",
        ticker,
        train_scaled.columns.tolist(),
        val_scaled.columns.tolist(),
        test_scaled.columns.tolist(),
    )

    # ============================================================
    # STAGE 4: SAVE ARTIFACTS
    # ============================================================
    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Saving artifacts", ticker)

    artifacts = {
        "X_train.npy": X_train,
        "y_train.npy": y_train,
        "X_val.npy": X_val,
        "y_val.npy": y_val,
        "X_test.npy": X_test,
        "y_test.npy": y_test,
        "train_index.npy": train_idx.values.astype("datetime64[ns]"),
        "val_index.npy": val_idx.values.astype("datetime64[ns]"),
        "test_index.npy": test_idx.values.astype("datetime64[ns]"),
    }

    for name, arr in artifacts.items():
        np.save(ticker_dir / name, arr, allow_pickle=False)

    # Save DataFrames (audit)
    train_scaled.to_parquet(ticker_dir / f"{ticker}_train_features.parquet")
    val_scaled.to_parquet(ticker_dir / f"{ticker}_val_features.parquet")
    test_scaled.to_parquet(ticker_dir / f"{ticker}_test_features.parquet")

    # Save transformer state
    save_transformer_state(
        str(ticker_dir / f"{ticker}_transformer_state.json"),
        transformer_state,
    )

    # ============================================================
    # STAGE 5: METADATA
    # ============================================================
    metadata = {
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline_version": "1.0.0_unified",
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "peers": peers,
        "universe_size": len(universe),
        "shapes": {
            "train": list(X_train.shape),
            "val": list(X_val.shape),
            "test": list(X_test.shape),
        },
        "normalization_range": [-1.0, 1.0],
        "data_quality": {
            "train_nan": bool(np.isnan(X_train).any()),
            "val_nan": bool(np.isnan(X_val).any()),
            "test_nan": bool(np.isnan(X_test).any()),
        },
        "temporal_ranges": {
            "train": [str(train_idx.min()), str(train_idx.max())],
            "val": [str(val_idx.min()), str(val_idx.max())],
            "test": [str(test_idx.min()), str(test_idx.max())],
        },
        "data_fingerprint": hashlib.md5(X_train.tobytes()).hexdigest(),
    }

    with open(ticker_dir / f"{ticker}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(ticker_dir / f"{ticker}_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # ============================================================
    # CLEANUP
    # ============================================================
    del X_train, X_val, X_test, y_train, y_val, y_test
    del train_scaled, val_scaled, test_scaled
    gc.collect()

    logger.info("[%s] PIPELINE COMPLETE | artifacts=%d", ticker, len(artifacts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--output", default="data/features")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--tickers-file", default="config/tickers.txt")
    parser.add_argument("--universe-config", default="config/symbol_universe.yaml")

    def load_tickers(args) -> list[str]:
        # Priority 1: single ticker
        if args.ticker:
            return [args.ticker]

        # Priority 2: CLI list
        if args.tickers:
            return args.tickers

        # Priority 3: file
        with open(args.tickers_file) as f:
            return [l.strip() for l in f if l.strip() and not l.startswith("#")]

    args = parser.parse_args()

    if args.ticker and args.tickers:
        raise ValueError("Use either --ticker or --tickers, not both.")

    setup_logger(
        log_file="logs/02_build_features.log", level="INFO", mode=LogFileMode.OVERWRITE
    )
    config = load_config(args.config)

    logger.info(f"Arguments: \n {args}")

    all_tickers = load_tickers(args)
    logger.info("Loaded %d tickers", len(all_tickers))
    logger.info("Tickers: %s", all_tickers)

    # Universe builder
    universe_builder = SymbolUniverseBuilder.from_config(args.universe_config)

    # Load and split data
    processed_dir = Path("data/processed")
    dfs = {}
    for t in all_tickers:
        p = processed_dir / f"{t}.parquet"
        if p.exists():
            df = pd.read_parquet(p)

            df.index = pd.to_datetime(df.index)

            # ENFORCE MARKET HOURS (CRITICAL FIX)
            # df = filter_us_market_hours(df)

            dfs[t] = df
        else:
            logger.warning("Missing parquet: %s", t)
    logger.info("Loaded %d ticker DataFrames", len(dfs))

    # log out number of rows and columns for dfs
    logger.info("DataFrame shapes:")
    for t, df in dfs.items():
        logger.info(f"{t}: {df.shape}")

    df_aligned = pd.concat(dfs, axis=1, keys=dfs.keys()).sort_index()
    df_train, df_val, df_test = DataSplitter().split(df_aligned)

    OHLCV_COLS = ["open", "high", "low", "close", "volume"]

    def extract(df_split: pd.DataFrame) -> dict:
        out = {}

        for t in all_tickers:
            if t not in df_split.columns.get_level_values(0):
                continue

            # Extract ticker slice
            sub = df_split.xs(t, axis=1, level=0).copy()

            # Ensure datetime index consistency
            sub.index = pd.to_datetime(sub.index).tz_localize(None)

            # KEEP ONLY OHLCV COLUMNS (hard enforcement)
            missing = [c for c in OHLCV_COLS if c not in sub.columns]
            if missing:
                continue  # skip invalid tickers safely

            sub = sub[OHLCV_COLS]

            out[t] = sub

        return out

    dfs_train = extract(df_train)
    dfs_val = extract(df_val)
    dfs_test = extract(df_test)

    # Determine prediction targets
    with open(args.universe_config) as f:
        u_cfg = yaml.safe_load(f)
    targets = u_cfg.get("prediction_targets", all_tickers)

    logger.info(f"Prediction Targets: {targets}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # process_ticker(
    #     "AAPL", dfs_train, dfs_val, dfs_test, universe_builder, output_dir, config
    # )
    for ticker in targets:
        process_ticker(
            ticker, dfs_train, dfs_val, dfs_test, universe_builder, output_dir, config
        )

    peers_path = output_dir / "fitted_peers.json"
    universe_builder.save_peers(peers_path)
    logger.info("Saved peers --> %s", peers_path)
    logger.info("[SELECTED] Feature engineering complete")

    logger.info("=" * 80)
    logger.info("UNIFIED FEATURE PIPELINE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", default="config/default_config.yaml")
#     parser.add_argument("--output", default="data/features")
#     parser.add_argument("--ticker", type=str, default=None)
#     parser.add_argument("--tickers", nargs="+", default=None)
#     parser.add_argument("--tickers-file", default="config/tickers.txt")
#     parser.add_argument("--universe-config", default="config/symbol_universe.yaml")

#     def load_tickers(args) -> list[str]:
#         # Priority 1: single ticker
#         if args.ticker:
#             return [args.ticker]

#         # Priority 2: CLI list
#         if args.tickers:
#             return args.tickers

#         # Priority 3: file
#         with open(args.tickers_file) as f:
#             return [l.strip() for l in f if l.strip() and not l.startswith("#")]

#     args = parser.parse_args()

#     if args.ticker and args.tickers:
#         raise ValueError("Use either --ticker or --tickers, not both.")

#     setup_logger(
#         log_file="logs/02_build_features.log", level="INFO", mode=LogFileMode.OVERWRITE
#     )
#     config = load_config(args.config)

#     logger.info(f"Arguments: \n {args}")

#     all_tickers = load_tickers(args)
#     logger.info("Loaded %d tickers", len(all_tickers))
#     logger.info("Tickers: %s", all_tickers)

#     # Universe builder
#     universe_builder = SymbolUniverseBuilder.from_config(args.universe_config)

#     # Load and split data
#     processed_dir = Path("data/processed")
#     dfs = {}
#     for t in all_tickers:
#         p = processed_dir / f"{t}.parquet"
#         if p.exists():
#             df = pd.read_parquet(p)

#             df.index = pd.to_datetime(df.index)

#             # ENFORCE MARKET HOURS (CRITICAL FIX)
#             # df = filter_us_market_hours(df)

#             dfs[t] = df
#         else:
#             logger.warning("Missing parquet: %s", t)
#     logger.info("Loaded %d ticker DataFrames", len(dfs))

#     # log out number of rows and columns for dfs
#     logger.info("DataFrame shapes:")
#     for t, df in dfs.items():
#         logger.info(f"{t}: {df.shape}")

#     df_aligned = pd.concat(dfs, axis=1, keys=dfs.keys()).sort_index()
#     df_train, df_val, df_test = DataSplitter().split(df_aligned)

#     OHLCV_COLS = ["open", "high", "low", "close", "volume"]

#     def extract(df_split: pd.DataFrame) -> dict:
#         out = {}

#         for t in all_tickers:
#             if t not in df_split.columns.get_level_values(0):
#                 continue

#             # Extract ticker slice
#             sub = df_split.xs(t, axis=1, level=0).copy()

#             # Ensure datetime index consistency
#             sub.index = pd.to_datetime(sub.index).tz_localize(None)

#             # KEEP ONLY OHLCV COLUMNS (hard enforcement)
#             missing = [c for c in OHLCV_COLS if c not in sub.columns]
#             if missing:
#                 continue  # skip invalid tickers safely

#             sub = sub[OHLCV_COLS]

#             out[t] = sub

#         return out

#     dfs_train = extract(df_train)
#     dfs_val = extract(df_val)
#     dfs_test = extract(df_test)

#     # Determine prediction targets
#     with open(args.universe_config) as f:
#         u_cfg = yaml.safe_load(f)
#     targets = u_cfg.get("prediction_targets", all_tickers)

#     logger.info(f"Prediction Targets: {targets}")

#     output_dir = Path(args.output)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # process_ticker(
#     #     "AAPL", dfs_train, dfs_val, dfs_test, universe_builder, output_dir, config
#     # )
#     for ticker in targets:
#         process_ticker(
#             ticker, dfs_train, dfs_val, dfs_test, universe_builder, output_dir, config
#         )

#     peers_path = output_dir / "fitted_peers.json"
#     universe_builder.save_peers(peers_path)
#     logger.info("Saved peers --> %s", peers_path)
#     logger.info("[SELECTED] Feature engineering complete")

#     logger.info("=" * 80)
#     logger.info("UNIFIED FEATURE PIPELINE COMPLETE")
#     logger.info("=" * 80)


# if __name__ == "__main__":
#     main()
