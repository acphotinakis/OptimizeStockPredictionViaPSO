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
