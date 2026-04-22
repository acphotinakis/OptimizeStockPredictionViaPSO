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
from typing import Dict, List, Optional, Tuple

from matplotlib import ticker
import numpy as np
import pandas as pd
import yaml

from src.features.wavelet import _apply_wavelet


# Project root setup
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.base_features import FeatureGenerator

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
import sys

logger = logging.getLogger(__name__)

TARGET_COL = "log_return"


NY_TZ = "America/New_York"


def add_log_return(dfs: Dict[str, pd.DataFrame]) -> None:
    for ticker, df in dfs.items():
        df = df.copy()

        if "close" not in df.columns:
            raise ValueError(f"[{ticker}] Missing 'close' column")

        df["log_return"] = np.log(df["close"]).diff()

        dfs[ticker] = df


def process_ticker(
    ticker: str,
    dfs: Dict[str, pd.DataFrame],
    universe_builder: SymbolUniverseBuilder,
    output_dir: Path,
    config: Config,
) -> None:

    # ============================================================
    # GUARD
    # ============================================================
    if ticker not in dfs:
        logger.info("Skipping %s (not in training data)", ticker)
        return

    logger.info("[%s] START PIPELINE", ticker)

    target_ticker = ticker

    # ============================================================
    # STAGE 1: UNIVERSE + PEERS
    # ============================================================
    universe = universe_builder.get_universe(ticker, dfs, fit=True)
    peers = universe_builder.get_fitted_peers(ticker)

    logger.info("[%s] Universe size=%d | peers=%s", ticker, len(universe), peers)
    logger.info("[%s] Universe: %s", ticker, universe)

    for t in universe:
        if t not in dfs:
            logger.warning("[%s] Missing universe ticker in data: %s", ticker, t)
        else:
            df = dfs[t]
            logger.info(
                f"[{ticker}] Universe ticker '{t}' columns: {df.columns.tolist()}"
            )
            logger.info(
                f"[{ticker}] Universe ticker '{t}' date range: {df.index.min()} to {df.index.max()}"
            )

    # add_log_return(dfs, "all")

    # ============================================================
    # STAGE 2: Generate the base features + cross ticker features
    # ============================================================
    feature_generator = FeatureGenerator(target_ticker=ticker, peer_tickers=peers)

    raw_features, raw_y, feature_names, idx = feature_generator.generate_features(dfs)

    logger.info(
        f"[{ticker}] Generated features shape={raw_features.shape} target shape={raw_y.shape} features={len(feature_names)} samples={len(idx)}"
    )
    logger.info(f"[{ticker}] Sample feature names: {feature_names[:10]}")
    logger.info(f"[{ticker}] Sample index range: {idx.min()} to {idx.max()}")

    if config.features.wavelet.enabled:
        logger.info(f"[{target_ticker}] Stage 2: Wavelet denoising")
        X, names, _wavelet_threshold = _apply_wavelet(
            raw_features, feature_names, fit_mode=True, _wavelet_threshold=None
        )
        logger.info(f"  Threshold fitted: {_wavelet_threshold:.6f}")

    sys.exit(0)

    # ============================================================
    # STAGE 0: FEATURE PREPARATION (TARGET ONLY)
    # ============================================================
    # def add_log_return(dfs: dict, name: str) -> None:
    #     logger.info("[%s] Computing log_return for %s", ticker, name)
    #     dfs[ticker]["log_return"] = np.log(dfs[ticker]["close"]).diff()

    # add_log_return(dfs_train, "train")
    # add_log_return(dfs_val, "val")
    # add_log_return(dfs_test, "test")

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

    # single prediction target (primary mode)
    parser.add_argument("--prediction-target", type=str, default="AAPL")

    parser.add_argument("--universe-config", default="config/symbol_universe.yaml")

    args = parser.parse_args()

    setup_logger(
        log_file="logs/build_features.log",
        level="INFO",
        mode=LogFileMode.OVERWRITE,
    )

    config = load_config(args.config)

    logger.info("Arguments:\n%s", args)

    # ------------------------------------------------------------
    # Load universe config
    # ------------------------------------------------------------
    with open(args.universe_config) as f:
        u_cfg = yaml.safe_load(f)

    valid_targets = set(u_cfg.get("prediction_targets", []))
    all_symbols_in_universe = set(
        u_cfg.get("context_only", [])
        + list(valid_targets)
        + u_cfg.get("market_context", [])
    )

    logger.info("Valid prediction targets: %s", sorted(valid_targets))
    logger.info("All symbols in universe config: %s", sorted(all_symbols_in_universe))

    target = args.prediction_target
    logger.info("Selected prediction target: %s", target)

    # ------------------------------------------------------------
    # Validate prediction target
    # ------------------------------------------------------------
    if target not in valid_targets:
        raise ValueError(
            f"Invalid prediction target: {target}. "
            f"Must be one of: {sorted(valid_targets)}"
        )

    logger.info("Prediction target validated: %s", target)

    # ------------------------------------------------------------
    # Initialize universe builder
    # ------------------------------------------------------------
    universe_builder = SymbolUniverseBuilder.from_config(args.universe_config)

    logger.info("Initialized SymbolUniverseBuilder")

    # ------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------
    processed_dir = Path("data/processed")

    dfs = {}
    for symbol in all_symbols_in_universe:
        p = processed_dir / f"{symbol}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            dfs[symbol] = df
        else:
            logger.warning("Missing parquet: %s", symbol)

    logger.info("Loaded %d symbol DataFrames", len(dfs))
    add_log_return(dfs)

    # log out the columns in all dfs
    for ticker, df in dfs.items():
        logger.info(f"[{ticker}] Columns after log_return: {df.columns.tolist()}")

    logger.info("Added log_return to all DataFrames")
    # ------------------------------------------------------------
    # Align full dataset
    # ------------------------------------------------------------
    df_aligned = pd.concat(dfs, axis=1, keys=dfs.keys()).sort_index()
    dfs_restored = {
        ticker: df_aligned[ticker].copy() for ticker in df_aligned.columns.levels[0]
    }

    process_ticker(
        target,
        dfs_restored,
        universe_builder,
        args.output,
        config,
    )

    import sys

    sys.exit(0)

    df_train, df_val, df_test = DataSplitter().split(df_aligned)

    OHLCV_COLS = ["open", "high", "low", "close", "volume"]

    def extract(df_split: pd.DataFrame) -> dict:
        out = {}

        for sym in dfs.keys():
            if sym not in df_split.columns.get_level_values(0):
                continue

            sub = df_split.xs(sym, axis=1, level=0).copy()
            sub.index = pd.to_datetime(sub.index).tz_localize(None)

            if not all(c in sub.columns for c in OHLCV_COLS):
                continue

            out[sym] = sub[OHLCV_COLS]

        return out

    dfs_train = extract(df_train)
    dfs_val = extract(df_val)
    dfs_test = extract(df_test)

    # ------------------------------------------------------------
    # Build universe + features per target
    # ------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing target: %s", target)

    # STEP 1: build full universe (TRAIN ONLY → fit=True)
    universe = universe_builder.get_universe(
        target_ticker=target,
        dfs=dfs_train,
        fit=True,
    )

    # STEP 2: ensure all required symbols exist
    missing = [s for s in universe if s not in dfs_train]
    if missing:
        logger.warning("Missing symbols in training data: %s", missing)

    # STEP 3: recompute final usable universe
    universe = [s for s in universe if s in dfs_train]

    logger.info("Final universe size for %s: %d", target, len(universe))
    logger.info("Universe: %s", universe)

    # STEP 4: align universe data
    dfs_train_u = {s: dfs_train[s] for s in universe}
    dfs_val_u = {s: dfs_val[s] for s in universe if s in dfs_val}
    dfs_test_u = {s: dfs_test[s] for s in universe if s in dfs_test}

    # STEP 5: compute features (single target pipeline)
    process_ticker(
        target,
        dfs_train_u,
        dfs_val_u,
        dfs_test_u,
        universe_builder,
        output_dir,
        config,
    )

    # ------------------------------------------------------------
    # Save learned peers
    # ------------------------------------------------------------
    peers_path = output_dir / "fitted_peers.json"
    universe_builder.save_peers(peers_path)

    logger.info("Saved peers -> %s", peers_path)

    logger.info("=" * 80)
    logger.info("FEATURE ENGINEERING COMPLETE FOR TARGET=%s", target)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
