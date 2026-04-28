#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
import logging
import numpy as np
import psutil
import os
import json
import sys
from pathlib import Path
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LOAD DATA FUNCTIONS
# ---------------------------------------------------------------------------


def load_training_data(ticker_dir: Path):
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


def load_feature_names(features_dir: Path, ticker: str) -> list:
    """Load feature names from metadata.pkl for a given ticker."""
    path = features_dir / ticker / "metadata.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read or parse {path}: {e}")

    if "feature_names" not in data:
        raise KeyError(f"'feature_names' key missing in {path}")

    feature_names = data["feature_names"]

    if not isinstance(feature_names, list):
        raise TypeError(f"'feature_names' in {path} is not a list")

    return feature_names


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)
    df = df.sort_index()
    logger.info(f"Loaded {path} | Rows: {len(df)}")
    logger.info(f"Columns: {df.columns.tolist()}")

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
    logger.info(f"Loaded {path} | Rows: {len(df)}")
    logger.info(f"Columns: {df.columns.tolist()}")

    return df


def load_all_timeframes(raw_output_dir: Path, ticker: str, timeframes: list[str]):
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
        # df.index = df.index.tz_convert("America/New_York")
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
