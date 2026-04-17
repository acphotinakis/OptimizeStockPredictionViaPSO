#!/usr/bin/env python3
"""Build feature matrices for each ticker.  Leakage-safe: universe selection
is done once per ticker on training data and frozen for val/test."""

import argparse
from datetime import datetime
import gc
import hashlib
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# print(sys.path)
import sys
from pathlib import Path

# Resolve project root (adjust depth if needed)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]  # adjust if structure changes

# Ensure only the project root (not file paths) is added
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Debug prints (optional)
print("Current file:", CURRENT_FILE)
print("Project root:", PROJECT_ROOT)
print("sys.path updated:")
print(sys.path)

from src.features.pipeline import FeaturePipeline
from src.features.universe_builder import SymbolUniverseBuilder
from src.features.scalar import PipelineScaler
from src.data.splitter import DataSplitter
from src.utils.logger import LogFileMode, setup_logger
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
    X_train, y_train, feat_names, train_idx = pipeline.fit_transform(
        filter_dfs(dfs_train)
    )
    X_val, y_val, val_idx = pipeline.transform(filter_dfs(dfs_val))
    X_test, y_test, test_idx = pipeline.transform(filter_dfs(dfs_test))

    # 3. Scaling — fit on train only (features only; target is NOT scaled)
    def to_df(X, y):
        df = pd.DataFrame(X, columns=feat_names)
        df["log_return"] = y
        return df

    logger.info(f"Feature Names [SELECTED FOR {ticker}]: {feat_names}")

    scaler = PipelineScaler().fit(to_df(X_train, y_train), feature_cols=feat_names)

    def scale(X, y):
        s = scaler.transform(to_df(X, y))
        return s[feat_names].to_numpy(np.float32), y.astype(np.float32)

    X_train_s, y_train_s = scale(X_train, y_train)
    X_val_s, y_val_s = scale(X_val, y_val)
    X_test_s, y_test_s = scale(X_test, y_test)

    # 4. Save
    out = output_dir / ticker
    out.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "X_train": X_train_s,
        "y_train": y_train_s,
        "X_val": X_val_s,
        "y_val": y_val_s,
        "X_test": X_test_s,
        "y_test": y_test_s,
        "train_index": train_idx.values.astype("datetime64[ns]"),
        "val_index": val_idx.values.astype("datetime64[ns]"),
        "test_index": test_idx.values.astype("datetime64[ns]"),
    }

    for name, arr in artifacts.items():
        final_path = out / f"{name}.npy"

        try:
            np.save(final_path, arr, allow_pickle=False)
            logger.debug("[%s] Saved %s with", ticker, name)
        except Exception as e:
            raise RuntimeError(f"Failed to save {name} for {ticker}: {e}")

    # Save scaler
    scaler.save(out / "scaler.pkl")

    # Save metadata with checksums
    metadata = {
        "nan_checks": {
            "X_train_nan": bool(np.isnan(X_train_s).any()),
            "X_val_nan": bool(np.isnan(X_val_s).any()),
            "X_test_nan": bool(np.isnan(X_test_s).any()),
        },
        "inf_checks": {
            "X_train_inf": bool(np.isinf(X_train_s).any()),
            "X_val_inf": bool(np.isinf(X_val_s).any()),
            "X_test_inf": bool(np.isinf(X_test_s).any()),
        },
        "data_fingerprint": hashlib.md5(str(X_train_s.tobytes()).encode()).hexdigest(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "timestamp": datetime.utcnow().isoformat(),
        "feature_names": feat_names,
        "feature_count": len(feat_names),
        "n_features": len(feat_names),
        "peer_tickers": peers,
        "train_samples": len(X_train_s),
        "val_samples": len(X_val_s),
        "test_samples": len(X_test_s),
        "train_index_range": (
            [str(train_idx[0]), str(train_idx[-1])] if len(train_idx) > 0 else []
        ),
        "val_index_range": (
            [str(val_idx[0]), str(val_idx[-1])] if len(val_idx) > 0 else []
        ),
        "test_index_range": (
            [str(test_idx[0]), str(test_idx[-1])] if len(test_idx) > 0 else []
        ),
        "train_start": str(dfs_train[ticker].index.min()),
        "train_end": str(dfs_train[ticker].index.max()),
        "val_start": str(dfs_val[ticker].index.min()),
        "val_end": str(dfs_val[ticker].index.max()),
        "test_start": str(dfs_test[ticker].index.min()),
        "test_end": str(dfs_test[ticker].index.max()),
        "raw_shapes": {
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        "scaled_shapes": {
            "X_train": list(X_train_s.shape),
            "X_val": list(X_val_s.shape),
            "X_test": list(X_test_s.shape),
        },
        "index_shapes": {
            "train_index": len(train_idx),
            "val_index": len(val_idx),
            "test_index": len(test_idx),
        },
        "index_dtype": "datetime64[ns]",
        "scaler_type": "PipelineScaler",
        "scaler_fitted_on": "train_only",
        "scaled_features": feat_names,
        "feature_columns_order": feat_names,
        "scaler_stats": {
            "center_median": scaler.feature_scaler.center_.tolist(),
            "scale_iqr": scaler.feature_scaler.scale_.tolist(),
        },
    }

    with open(out / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    logger.info(
        "[SELECTED] %s: %d features, checksums verified", ticker, len(feat_names)
    )

    # Cleanup
    del X_train, X_val, X_test, y_train, y_val, y_test
    del X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s
    gc.collect()


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
    load_config(args.config)

    logger.info(f"Arguments: \n {args}")

    all_tickers = load_tickers(args)
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

    logger.info(f"Prediction Targets: {targets}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker in targets:
        process_ticker(
            ticker, dfs_train, dfs_val, dfs_test, universe_builder, output_dir
        )

    peers_path = output_dir / "fitted_peers.json"
    universe_builder.save_peers(peers_path)
    logger.info("Saved peers --> %s", peers_path)
    logger.info("[SELECTED] Feature engineering complete")


if __name__ == "__main__":
    main()
