#!/usr/bin/env python3
"""Build feature matrices for each ticker.  Leakage-safe: universe selection
is done once per ticker on training data and frozen for val/test."""

import argparse
import gc
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.pipeline import FeaturePipeline
from src.features.universe_builder import SymbolUniverseBuilder
from src.features.scalar import PipelineScaler
from src.data.splitter import DataSplitter
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def process_ticker(
    ticker: str,
    dfs_train: dict,
    dfs_val: dict,
    dfs_test: dict,
    universe_builder: SymbolUniverseBuilder,
    output_dir: Path,
) -> None:
    if ticker not in dfs_train:
        logger.info("Skipping %s (not in training data)", ticker)
        return

    # 1. Universe — fit on train, reuse for val/test
    universe = universe_builder.get_universe(ticker, dfs_train, fit=True)
    peers = universe_builder.get_fitted_peers(ticker)
    filter_dfs = lambda dfs: {t: dfs[t] for t in universe if t in dfs}

    logger.info("[%s] universe=%d %s", ticker, len(universe), universe)
    logger.info("[%s] universe=%d  peers=%s", ticker, len(universe), universe)

    # 2. Feature pipeline
    pipeline = FeaturePipeline(target_ticker=ticker, peer_tickers=peers)
    X_train, y_train, feat_names = pipeline.fit_transform(filter_dfs(dfs_train))
    X_val, y_val = pipeline.transform(filter_dfs(dfs_val))
    X_test, y_test = pipeline.transform(filter_dfs(dfs_test))

    # 3. Scaling — fit on train only
    def to_df(X, y):
        df = pd.DataFrame(X, columns=feat_names)
        df["log_return"] = y
        return df

    scaler = PipelineScaler().fit(to_df(X_train, y_train), feature_cols=feat_names)

    def scale(X, y):
        s = scaler.transform(to_df(X, y))
        return s[feat_names].to_numpy(np.float32), s["log_return"].to_numpy(np.float32)

    X_train_s, y_train_s = scale(X_train, y_train)
    X_val_s, y_val_s = scale(X_val, y_val)
    X_test_s, y_test_s = scale(X_test, y_test)

    # 4. Save
    out = output_dir / ticker
    out.mkdir(parents=True, exist_ok=True)

    for name, arr in {
        "X_train": X_train_s,
        "y_train": y_train_s,
        "X_val": X_val_s,
        "y_val": y_val_s,
        "X_test": X_test_s,
        "y_test": y_test_s,
    }.items():
        np.save(out / f"{name}.npy", arr, allow_pickle=False)

    scaler.save(out / "scaler.pkl")
    with open(out / "metadata.pkl", "wb") as f:
        pickle.dump(
            {
                "feature_names": feat_names,
                "n_features": len(feat_names),
                "train_samples": len(X_train_s),
                "val_samples": len(X_val_s),
                "test_samples": len(X_test_s),
            },
            f,
        )

    logger.info("✓ %s: %d features", ticker, len(feat_names))
    del X_train, X_val, X_test, y_train, y_val, y_test
    del X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--output", default="data/features")
    parser.add_argument("--tickers", default="config/tickers.txt")
    parser.add_argument("--universe-config", default="config/symbol_universe.yaml")
    args = parser.parse_args()

    load_config(args.config)
    setup_logger(log_file="logs/02_build_features.log", level="INFO")

    # Load tickers
    with open(args.tickers) as f:
        all_tickers = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    logger.info("Loaded %d tickers", len(all_tickers))

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
            dfs[t] = df
        else:
            logger.warning("Missing parquet: %s", t)
    logger.info("Loaded %d ticker DataFrames", len(dfs))

    df_aligned = pd.concat(dfs, axis=1, keys=dfs.keys()).sort_index()
    df_train, df_val, df_test = DataSplitter().split(df_aligned)

    DROP_COLS = ["gap_flag", "gap_length", "post_long_gap", "outlier_flag"]

    def extract(df_split: pd.DataFrame) -> dict:
        out = {}
        for t in all_tickers:
            if t not in df_split.columns.get_level_values(0):
                continue
            sub = df_split.xs(t, axis=1, level=0).copy()
            sub.index = pd.to_datetime(sub.index).tz_localize(None)
            out[t] = sub.drop(columns=DROP_COLS, errors="ignore")
        return out

    dfs_train = extract(df_train)
    dfs_val = extract(df_val)
    dfs_test = extract(df_test)

    # Determine prediction targets
    with open(args.universe_config) as f:
        u_cfg = yaml.safe_load(f)
    targets = u_cfg.get("prediction_targets", all_tickers)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker in targets:
        process_ticker(
            ticker, dfs_train, dfs_val, dfs_test, universe_builder, output_dir
        )

    peers_path = output_dir / "fitted_peers.json"
    universe_builder.save_peers(peers_path)
    logger.info("Saved peers --> %s", peers_path)
    logger.info("✓ Feature engineering complete")


if __name__ == "__main__":
    main()
