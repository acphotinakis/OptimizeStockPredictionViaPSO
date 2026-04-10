#!/usr/bin/env python3
"""
Builds feature matrices for each ticker using aligned OHLCV data.
Leakage-safe: universe selection happens outside FeaturePipeline.
"""

import argparse
import logging
from pathlib import Path
import pickle
import sys
import gc
import yaml
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

logger = logging.getLogger(__name__)


# -----------------------------
# Utilities
# -----------------------------
def extract_ticker_dfs(df_multi: pd.DataFrame, tickers: list[str]) -> dict:
    """Return {ticker: DataFrame} from MultiIndex columns."""
    return {
        t: df_multi.xs(t, axis=1, level=0)
        for t in tickers
        if t in df_multi.columns.get_level_values(0)
    }


def drop_columns_if_exist(dfs: dict, cols: list[str]) -> dict:
    """Safely drop columns across all tickers."""
    return {t: df.drop(columns=cols, errors="ignore") for t, df in dfs.items()}


def filter_dfs(dfs: dict, universe: list[str]) -> dict:
    """Keep only tickers in universe."""
    return {t: dfs[t] for t in universe if t in dfs}


def load_ticker_parquets(data_dir: Path, tickers: list[str]) -> dict:
    dfs = {}

    for t in tickers:
        path = data_dir / f"{t}.parquet"
        if not path.exists():
            logger.warning("Missing parquet for %s", t)
            continue

        df = pd.read_parquet(path)

        # enforce datetime index consistency
        df.index = pd.to_datetime(df.index)

        dfs[t] = df

    return dfs


# -----------------------------
# Core processing
# -----------------------------
def process_ticker(
    ticker: str,
    dfs_train,
    dfs_val,
    dfs_test,
    cfg,
    output_dir,
    universe_builder: SymbolUniverseBuilder,
):
    if ticker not in dfs_train:
        logger.info("Skipping %s (not in training data)", ticker)
        return

    # -------------------------------------------------------
    # 1. Build universe ONCE per ticker (no leakage)
    # -------------------------------------------------------
    universe = universe_builder.get_universe(
        ticker,
        dfs_train,
        fit=True,
    )

    logger.info(f"Universe { universe}")

    # import sys
    # sys.exit(0)

    dfs_train_f = filter_dfs(dfs_train, universe)
    dfs_val_f = filter_dfs(dfs_val, universe)
    dfs_test_f = filter_dfs(dfs_test, universe)

    logger.info(
        "[%s] universe size=%d | peers=%s",
        ticker,
        len(universe),
        universe,
    )

    # -------------------------------------------------------
    # 2. Feature pipeline (no internal universe logic)
    # -------------------------------------------------------
    pipeline = FeaturePipeline(
        target_ticker=ticker,
        universe_builder=SymbolUniverseBuilder,  # IMPORTANT: disable internal selection
        universe_tickers=universe_builder._fitted_peers[ticker],
        selector_kwargs={},
    )

    X_train, y_train, feature_names = pipeline.fit_transform(dfs_train_f)
    X_val, y_val = pipeline.transform(dfs_val_f)
    X_test, y_test = pipeline.transform(dfs_test_f)

    # -------------------------------------------------------
    # 3. Scaling (fit only on train)
    # -------------------------------------------------------
    scaler = PipelineScaler()

    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["log_return"] = y_train

    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df["log_return"] = y_val

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["log_return"] = y_test

    scaler.fit(train_df, target_col="log_return", feature_cols=feature_names)

    # Verify no scaler contamination with val/test data
    assert (
        scaler.target_scaler.data_min_[0] <= y_train.min()
    ), "Target scaler contaminated: data_min_ is below training min"
    assert (
        scaler.target_scaler.data_max_[0] >= y_train.max()
    ), "Target scaler contaminated: data_max_ is above training max"

    train_scaled = scaler.transform(train_df, target_col="log_return")
    val_scaled = scaler.transform(val_df, target_col="log_return")
    test_scaled = scaler.transform(test_df, target_col="log_return")

    X_train_scaled = train_scaled[feature_names].to_numpy(np.float32)
    y_train_scaled = train_scaled["log_return"].to_numpy(np.float32)

    X_val_scaled = val_scaled[feature_names].to_numpy(np.float32)
    y_val_scaled = val_scaled["log_return"].to_numpy(np.float32)

    X_test_scaled = test_scaled[feature_names].to_numpy(np.float32)
    y_test_scaled = test_scaled["log_return"].to_numpy(np.float32)

    # -------------------------------------------------------
    # 4. Save outputs
    # -------------------------------------------------------
    ticker_dir = Path(output_dir) / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "X_train": X_train_scaled,
        "y_train": y_train_scaled,
        "X_val": X_val_scaled,
        "y_val": y_val_scaled,
        "X_test": X_test_scaled,
        "y_test": y_test_scaled,
    }

    for name, arr in arrays.items():
        np.save(ticker_dir / f"{name}.npy", arr, allow_pickle=False)

    scaler.save(ticker_dir / "scaler.pkl")

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

    # -------------------------------------------------------
    # 5. Cleanup
    # -------------------------------------------------------
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
        scaler,
    )
    gc.collect()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
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

    # -----------------------------
    # Universe builder
    # -----------------------------
    universe_builder = None
    if Path(args.universe_config).exists():
        logger.info("Loading universe config: %s", args.universe_config)
        universe_builder = SymbolUniverseBuilder.from_config(args.universe_config)
    else:
        logger.warning("No universe config found → legacy mode")

    # -----------------------------
    # Load tickers
    # -----------------------------
    with open(args.tickers) as f:
        tickers = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    logger.info("Loaded %d tickers", len(tickers))

    # -----------------------------
    # Load aligned data
    # -----------------------------
    processed_dir = Path("data/processed")
    dfs = load_ticker_parquets(processed_dir, tickers)
    logger.info("Loaded %d ticker DataFrames", len(dfs))

    df_aligned = pd.concat(dfs, axis=1, keys=dfs.keys())
    df_aligned.index = pd.to_datetime(df_aligned.index)
    df_aligned = df_aligned.sort_index()

    logger.info("Rebuilt aligned dataset: %s", df_aligned.shape)

    splitter = DataSplitter()
    df_train_all, df_val_all, df_test_all = splitter.split(df_aligned)

    dfs_train = extract_ticker_dfs(df_train_all, tickers)
    dfs_val = extract_ticker_dfs(df_val_all, tickers)
    dfs_test = extract_ticker_dfs(df_test_all, tickers)

    # timezone normalize
    for dfs in [dfs_train, dfs_val, dfs_test]:
        for t, df in dfs.items():
            df.index = pd.to_datetime(df.index).tz_localize(None)

    # -----------------------------
    # Drop leakage columns
    # -----------------------------
    DROP_COLS = ["gap_flag", "gap_length", "post_long_gap", "outlier_flag"]

    dfs_train = drop_columns_if_exist(dfs_train, DROP_COLS)
    dfs_val = drop_columns_if_exist(dfs_val, DROP_COLS)
    dfs_test = drop_columns_if_exist(dfs_test, DROP_COLS)

    # inspect
    # splitter.inspect_split("TRAIN", dfs_train)
    # splitter.inspect_split("VAL", dfs_val)
    # splitter.inspect_split("TEST", dfs_test)

    # -----------------------------
    # Output dir
    # -----------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Select tickers
    # -----------------------------
    if universe_builder is not None:
        with open(args.universe_config) as f:
            universe_cfg = yaml.safe_load(f)

        tickers_to_process = universe_cfg.get("prediction_targets", tickers)
    else:
        tickers_to_process = tickers

    logger.info("Processing %d tickers", len(tickers_to_process))

    # -----------------------------
    # Run pipeline
    # -----------------------------
    for ticker in tickers_to_process:
        process_ticker(
            ticker,
            dfs_train,
            dfs_val,
            dfs_test,
            cfg,
            output_dir,
            universe_builder,
        )

    # -----------------------------
    # Save peers
    # -----------------------------
    if universe_builder is not None:
        peers_path = Path(output_dir) / "fitted_peers.json"
        universe_builder.save_peers(peers_path)
        logger.info("Saved peers → %s", peers_path)

    logger.info("✓ Feature engineering complete")


if __name__ == "__main__":
    main()
