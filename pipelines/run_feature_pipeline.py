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
    UnifiedFeaturePipeline,
    build_windows,
    transform_features,
    save_transformer_state,
)
from src.features.universe_builder import SymbolUniverseBuilder
from src.data.splitter import DataSplitter
from src.data.alpaca_ingestor import AlpacaIngestor
from src.utils.logger import LogFileMode, setup_logger
from src.utils.config_loader import Config, load_config

logger = logging.getLogger(__name__)

TARGET_COL = "log_return"


def load_and_split_data(
    tickers: List[str], processed_dir: Path, config: Config
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Load ticker data and perform temporal split.

    Args:
        tickers: List of ticker symbols
        processed_dir: Directory containing processed parquet files
        config: Configuration dict with split ratios

    Returns:
        Tuple of (dfs_train, dfs_val, dfs_test) dictionaries
    """
    logger.info("Loading and splitting data for %d tickers", len(tickers))

    # Load all ticker DataFrames
    dfs = {}
    for ticker in tickers:
        parquet_path = processed_dir / f"{ticker}.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            df.index = pd.to_datetime(df.index)

            # Ensure required columns present
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(set(df.columns)):
                logger.warning(f"{ticker}: missing required columns, skipping")
                continue

            # Add log_return if not present
            if "log_return" not in df.columns:
                df["log_return"] = np.log(df["close"] / df["close"].shift(1))

            dfs[ticker] = df
        else:
            logger.warning(f"Missing parquet: {ticker}")

    logger.info(f"Loaded {len(dfs)} ticker DataFrames")

    # Concatenate and split temporally
    df_aligned = pd.concat(dfs, axis=1, keys=dfs.keys()).sort_index()

    # Use DataSplitter for temporal split
    splitter = DataSplitter()
    df_train, df_val, df_test = splitter.split(df_aligned)

    # Extract individual ticker DataFrames
    DROP_COLS = ["gap_flag", "gap_length", "post_long_gap", "outlier_flag"]

    def extract_tickers(df_split: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        out = {}
        for ticker in tickers:
            if ticker not in df_split.columns.get_level_values(0):
                continue
            sub = df_split.xs(ticker, axis=1, level=0).copy()
            sub.index = pd.to_datetime(sub.index).tz_localize(None)
            out[ticker] = sub.drop(columns=DROP_COLS, errors="ignore")
        return out

    dfs_train = extract_tickers(df_train)
    dfs_val = extract_tickers(df_val)
    dfs_test = extract_tickers(df_test)

    logger.info(
        f"Split complete: train={len(dfs_train)}, val={len(dfs_val)}, test={len(dfs_test)}"
    )

    return dfs_train, dfs_val, dfs_test


def process_ticker_unified(
    ticker: str,
    dfs_train: Dict[str, pd.DataFrame],
    dfs_val: Dict[str, pd.DataFrame],
    dfs_test: Dict[str, pd.DataFrame],
    universe_builder: SymbolUniverseBuilder,
    output_dir: Path,
    config: Config,
) -> None:
    """
    Process single ticker through unified feature pipeline.

    Pipeline Stages:
    1. Universe selection (peers)
    2. Feature generation (UnifiedFeaturePipeline)
    3. Wavelet denoising (integrated in pipeline)
    4. Feature selection (4-stage)
    5. Normalization (MinMax [-1, 1])
    6. Windowing (20 timesteps for LSTM)

    Args:
        ticker: Target ticker symbol
        dfs_train: Training data dictionary
        dfs_val: Validation data dictionary
        dfs_test: Test data dictionary
        universe_builder: Universe/peer selector
        output_dir: Output directory for artifacts
        config: Configuration dictionary
    """
    if ticker not in dfs_train:
        logger.info(f"[{ticker}] Not in training data, skipping")
        return

    logger.info(f"[{ticker}] START UNIFIED PIPELINE")

    # ========================================================================
    # STAGE 1: UNIVERSE SELECTION (FIT ON TRAINING)
    # ========================================================================
    universe = universe_builder.get_universe(ticker, dfs_train, fit=True)
    # peers = universe_builder.get_fitted_peers(ticker)
    peers = []

    logger.info(f"[{ticker}] Universe: {len(universe)} tickers")
    logger.info(f"[{ticker}] Peers: {peers}")

    def filter_universe(dfs: Dict) -> Dict:
        return {t: dfs[t] for t in universe if t in dfs}

    # ========================================================================
    # STAGE 2-5: UNIFIED FEATURE PIPELINE
    # ========================================================================
    # Initialize unified pipeline
    enable_wavelet = config.features.wavelet.enabled
    lookback = config.lstm.lookback

    selector_kwargs = {
        "variance_threshold": config.features.selector.variance_threshold,
        "correlation_threshold": config.features.selector.correlation_threshold,
        "vif_threshold": config.features.selector.vif_threshold,
        "mi_quantile_threshold": config.features.selector.mi_quantile_threshold,
    }

    pipeline = UnifiedFeaturePipeline(
        target_ticker=ticker,
        peer_tickers=peers,
        selector_kwargs=selector_kwargs,
        enable_wavelet=enable_wavelet,
        lookback=lookback,
    )

    # FIT on training data
    logger.info(f"[{ticker}] Stage 2-5: Running unified pipeline (fit)")
    X_train, y_train, feature_names, train_idx = pipeline.fit_transform(
        filter_universe(dfs_train)
    )

    logger.info(
        f"[{ticker}] Training: {X_train.shape[0]} samples, {X_train.shape[1]} features"
    )
    logger.info(f"[{ticker}] Selected features: {feature_names}")

    # TRANSFORM validation and test
    logger.info(f"[{ticker}] Stage 2-5: Applying to validation/test")
    X_val, y_val, val_idx = pipeline.transform(filter_universe(dfs_val))
    X_test, y_test, test_idx = pipeline.transform(filter_universe(dfs_test))

    logger.info(
        f"[{ticker}] Validation: {X_val.shape[0]} samples, {X_val.shape[1]} features"
    )
    logger.info(
        f"[{ticker}] Test: {X_test.shape[0]} samples, {X_test.shape[1]} features"
    )

    # ========================================================================
    # STAGE 6: NORMALIZATION (MINMAX [-1, 1])
    # ========================================================================
    logger.info(f"[{ticker}] Stage 6: MinMax normalization")

    # Convert to DataFrames for normalization
    train_df = pd.DataFrame(X_train, columns=feature_names, index=train_idx)
    val_df = pd.DataFrame(X_val, columns=feature_names, index=val_idx)
    test_df = pd.DataFrame(X_test, columns=feature_names, index=test_idx)

    # Apply normalization (fit on training only)
    train_scaled, val_scaled, test_scaled, transformer_state = transform_features(
        train_df, val_df, test_df
    )

    # Convert back to arrays
    X_train_scaled = train_scaled.values.astype(np.float32)
    X_val_scaled = val_scaled.values.astype(np.float32)
    X_test_scaled = test_scaled.values.astype(np.float32)

    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)

    logger.info(f"[{ticker}] Normalization complete")

    # ========================================================================
    # STAGE 7: TEMPORAL WINDOWING (LSTM SEQUENCES)
    # ========================================================================
    logger.info(f"[{ticker}] Stage 7: Building temporal windows (lookback={lookback})")

    X_train_seq, y_train_seq = build_windows(X_train_scaled, y_train, lookback)
    X_val_seq, y_val_seq = build_windows(X_val_scaled, y_val, lookback)
    X_test_seq, y_test_seq = build_windows(X_test_scaled, y_test, lookback)

    logger.info(
        f"[{ticker}] Windowed shapes: "
        f"train={X_train_seq.shape}, val={X_val_seq.shape}, test={X_test_seq.shape}"
    )

    # ========================================================================
    # STAGE 8: SAVE ARTIFACTS
    # ========================================================================
    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{ticker}] Saving artifacts to {ticker_dir}")

    # Save arrays (both tabular and windowed)
    artifacts = {
        # Tabular (for XGBoost)
        f"{ticker}_X_train.npy": X_train_scaled,
        f"{ticker}_y_train.npy": y_train,
        f"{ticker}_X_val.npy": X_val_scaled,
        f"{ticker}_y_val.npy": y_val,
        f"{ticker}_X_test.npy": X_test_scaled,
        f"{ticker}_y_test.npy": y_test,
        # Windowed (for LSTM)
        f"{ticker}_X_train_seq.npy": X_train_seq,
        f"{ticker}_y_train_seq.npy": y_train_seq,
        f"{ticker}_X_val_seq.npy": X_val_seq,
        f"{ticker}_y_val_seq.npy": y_val_seq,
        f"{ticker}_X_test_seq.npy": X_test_seq,
        f"{ticker}_y_test_seq.npy": y_test_seq,
        # Indices
        f"{ticker}_train_index.npy": train_idx.values.astype("datetime64[ns]"),
        f"{ticker}_val_index.npy": val_idx.values.astype("datetime64[ns]"),
        f"{ticker}_test_index.npy": test_idx.values.astype("datetime64[ns]"),
    }

    for filename, array in artifacts.items():
        np.save(ticker_dir / filename, array, allow_pickle=False)

    # Save DataFrames for audit
    train_scaled.to_parquet(ticker_dir / f"{ticker}_train_features.parquet")
    val_scaled.to_parquet(ticker_dir / f"{ticker}_val_features.parquet")
    test_scaled.to_parquet(ticker_dir / f"{ticker}_test_features.parquet")

    # Save transformer state (scaler params + wavelet threshold)
    save_transformer_state(
        str(ticker_dir / f"{ticker}_transformer_state.json"), transformer_state
    )

    # Save pipeline metadata
    metadata = {
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline_version": "1.0.0_unified",
        # Feature info
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_raw_features": "~60",
        "n_selected_features": len(feature_names),
        # Peers
        "peer_tickers": peers,
        "universe_size": len(universe),
        # Shapes
        "shapes": {
            "train": {
                "tabular": list(X_train_scaled.shape),
                "windowed": list(X_train_seq.shape),
            },
            "val": {
                "tabular": list(X_val_scaled.shape),
                "windowed": list(X_val_seq.shape),
            },
            "test": {
                "tabular": list(X_test_scaled.shape),
                "windowed": list(X_test_seq.shape),
            },
        },
        # Pipeline configuration
        "wavelet_enabled": enable_wavelet,
        "wavelet_threshold": transformer_state.get("wavelet", {}).get("threshold"),
        "lookback": lookback,
        "normalization_range": [-1.0, 1.0],
        # Data quality
        "data_quality": {
            "train_nan": bool(np.isnan(X_train_scaled).any()),
            "val_nan": bool(np.isnan(X_val_scaled).any()),
            "test_nan": bool(np.isnan(X_test_scaled).any()),
            "train_inf": bool(np.isinf(X_train_scaled).any()),
            "val_inf": bool(np.isinf(X_val_scaled).any()),
            "test_inf": bool(np.isinf(X_test_scaled).any()),
        },
        # Temporal ranges
        "temporal_ranges": {
            "train": [str(train_idx.min()), str(train_idx.max())],
            "val": [str(val_idx.min()), str(val_idx.max())],
            "test": [str(test_idx.min()), str(test_idx.max())],
        },
        # Checksums
        "data_fingerprint": hashlib.md5(X_train_scaled.tobytes()).hexdigest(),
    }

    with open(ticker_dir / f"{ticker}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(ticker_dir / f"{ticker}_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"[{ticker}] UNIFIED PIPELINE COMPLETE")
    logger.info(f"[{ticker}] Artifacts saved: {len(artifacts)} arrays")

    # Cleanup
    del X_train, X_val, X_test, y_train, y_val, y_test
    del X_train_scaled, X_val_scaled, X_test_scaled
    del X_train_seq, X_val_seq, X_test_seq
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Feature Engineering Pipeline (TRD-Compliant)"
    )
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--output", default="data/features_unified")
    parser.add_argument(
        "--ticker", type=str, default=None, help="Single ticker to process"
    )
    parser.add_argument("--tickers", nargs="+", default=None, help="List of tickers")
    parser.add_argument("--tickers-file", default="config/tickers.txt")
    parser.add_argument("--universe-config", default="config/symbol_universe.yaml")

    args = parser.parse_args()

    if args.ticker and args.tickers:
        raise ValueError("Use either --ticker or --tickers, not both")

    # Setup logging
    setup_logger(
        log_file="logs/unified_feature_pipeline.log",
        level="INFO",
        mode=LogFileMode.OVERWRITE,
    )

    # Load configuration
    config = load_config(args.config)

    logger.info("=" * 80)
    logger.info("UNIFIED FEATURE PIPELINE (TRD-COMPLIANT)")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output}")

    # Determine tickers
    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = args.tickers
    else:
        with open(args.tickers_file) as f:
            tickers = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

    logger.info(f"Processing {len(tickers)} tickers: {tickers}")

    # Initialize universe builder
    universe_builder = SymbolUniverseBuilder.from_config(args.universe_config)

    # Load and split data
    processed_dir = Path("data/processed")
    dfs_train, dfs_val, dfs_test = load_and_split_data(tickers, processed_dir, config)

    # Determine prediction targets
    with open(args.universe_config) as f:
        universe_config = yaml.safe_load(f)

    targets = universe_config.get("prediction_targets", tickers)
    logger.info(f"Prediction targets: {targets}")

    # Process each target ticker
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker in targets:
        try:
            process_ticker_unified(
                ticker=ticker,
                dfs_train=dfs_train,
                dfs_val=dfs_val,
                dfs_test=dfs_test,
                universe_builder=universe_builder,
                output_dir=output_dir,
                config=config,
            )
        except Exception as e:
            logger.error(f"[{ticker}] FAILED: {e}", exc_info=True)
            continue

    # Save fitted peers
    peers_path = output_dir / "fitted_peers.json"
    universe_builder.save_peers(peers_path)
    logger.info(f"Saved peers → {peers_path}")

    logger.info("=" * 80)
    logger.info("UNIFIED FEATURE PIPELINE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
