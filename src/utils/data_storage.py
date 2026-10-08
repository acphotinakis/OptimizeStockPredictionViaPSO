#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LOAD DATA FUNCTIONS
# ---------------------------------------------------------------------------


def load_training_data(
    ticker_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.info("\n[TRAIN] Loading data...")

    X_train_flat = np.load(ticker_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(ticker_dir / "y_train.npy", mmap_mode="r")
    X_val_flat = np.load(ticker_dir / "X_val.npy", mmap_mode="r")
    y_val = np.load(ticker_dir / "y_val.npy", mmap_mode="r")

    X_train_flat = np.array(X_train_flat)
    y_train = np.array(y_train)
    X_val_flat = np.array(X_val_flat)
    y_val = np.array(y_val)

    logger.info("Raw Shapes:")
    logger.info("  X_train_flat: %s", X_train_flat.shape)
    logger.info("  y_train     : %s", y_train.shape)
    logger.info("  X_val_flat  : %s", X_val_flat.shape)
    logger.info("  y_val       : %s", y_val.shape)

    return X_train_flat, y_train, X_val_flat, y_val


def load_windows(data_dir: Path, ticker: str):
    """Load pre-built numpy arrays."""
    prefix = data_dir / ticker

    file_map = {
        "X_train": prefix / "X_train.npy",
        "X_val": prefix / "X_val.npy",
        "X_test": prefix / "X_test.npy",
        "y_train": prefix / "y_train.npy",
        "y_val": prefix / "y_val.npy",
        "y_test": prefix / "y_test.npy",
    }

    arrays = {}

    for key, path in file_map.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        arrays[key] = np.load(path)

    return (
        arrays["X_train"],
        arrays["y_train"],
        arrays["X_val"],
        arrays["y_val"],
        arrays["X_test"],
        arrays["y_test"],
    )


def load_feature_names(features_dir: Path, ticker: str) -> List[str]:
    """Safely load feature names from JSON metadata for a given ticker.

    Avoids unsafe pickle deserialization by reading standard JSON metadata.
    Searches for feature_names.json, selected_features.json, metadata.json,
    or <ticker>_frozen_state.json in order.
    """
    ticker_dir = features_dir / ticker

    # 1. Direct feature_names.json
    fn_path = ticker_dir / "feature_names.json"
    if fn_path.exists():
        with open(fn_path, "r", encoding="utf-8") as f:
            names = json.load(f)
        if isinstance(names, list):
            return names

    # 2. selected_features.json
    sf_path = ticker_dir / "selected_features.json"
    if sf_path.exists():
        with open(sf_path, "r", encoding="utf-8") as f:
            names = json.load(f)
        if isinstance(names, list):
            return names

    # 3. metadata.json
    meta_path = ticker_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "feature_names" in data:
            names = data["feature_names"]
            if isinstance(names, list):
                return names

    # 4. <ticker>_frozen_state.json
    frozen_path = ticker_dir / f"{ticker}_frozen_state.json"
    if frozen_path.exists():
        with open(frozen_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            names = data.get("feature_names_selected") or data.get("feature_names_raw")
            if isinstance(names, list):
                return names

    raise FileNotFoundError(
        f"No valid feature names JSON file found for ticker '{ticker}' in {ticker_dir}"
    )


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)
    df = df.sort_index()
    logger.info("Loaded %s | Rows: %d", path, len(df))
    logger.info("Columns: %s", df.columns.tolist())
    if not df.index.is_monotonic_increasing:
        logger.warning("Non-monotonic index, sorting")
        df = df.sort_index()

    return df


mapping = {
    "1Min": "1min",
    "5Min": "5min",
    "15Min": "15min",
    "1Hour": "1h",
    "1Day": "1D",
}


def _load_parquet_close_volume(path: Path, tf: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)
    df = df.asfreq(mapping[tf])
    df = df[["close"]]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    logger.info("Loaded %s | Rows: %d", path, len(df))
    logger.info("Columns: %s", df.columns.tolist())

    return df


def load_all_timeframes(
    raw_output_dir: Path, ticker: str, timeframes: List[str]
) -> Dict[str, pd.DataFrame]:
    data = {}

    for tf in timeframes:
        path = raw_output_dir / tf / f"{ticker}.parquet"
        if not path.exists():
            logger.warning("Missing %s", path)
            continue

        df = _load_parquet(path)
        cols = ["open", "high", "low", "close", "volume"]
        df = df[cols]

        # ------------------------------------------------------------
        # 1. Ensure datetime index
        # ------------------------------------------------------------
        df.index = pd.to_datetime(df.index, utc=True)

        # ------------------------------------------------------------
        # 2. Convert UTC --> America/New_York (DST-aware)
        # ------------------------------------------------------------
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")

        data[tf] = df

    return data


# ---------------------------------------------------------------------------
# SAVE DATA FUNCTIONS
# ---------------------------------------------------------------------------
def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=True)
    logger.info("Saved %s (%d rows)", path.name, len(df))
