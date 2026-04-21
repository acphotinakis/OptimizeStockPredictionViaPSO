#!/usr/bin/env python3
"""
Inspect saved NumPy datasets for a given ticker.

Logs:
- Shapes
- Dtypes
- Basic statistics (mean, std, min, max)
- Sample values (first N rows)
- NaN / Inf checks
- Consistency checks between X and y

Usage:
    python inspect_npy.py --ticker AAPL --data-dir data/features/AAPL --samples 5
"""

import argparse
import logging
from pathlib import Path
import numpy as np


# ============================================================
# LOGGER
# ============================================================
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


logger = logging.getLogger(__name__)


# ============================================================
# CORE INSPECTION
# ============================================================
def summarize_array(name: str, arr: np.ndarray, samples: int = 5):
    logger.info(f"\n=== {name} ===")
    logger.info(f"shape: {arr.shape}")
    logger.info(f"dtype: {arr.dtype}")
    logger.info(f"features: {arr.shape[1] if arr.ndim > 1 else 'N/A'}")
    logger.info(
        f"columns: {arr.dtype.names.tolist() if hasattr(arr, 'columns') else 'N/A'}"
    )

    if arr.size == 0:
        logger.warning("Array is empty")
        return

    # Stats
    logger.info(
        f"stats -> mean={np.mean(arr):.6f}, std={np.std(arr):.6f}, "
        f"min={np.min(arr):.6f}, max={np.max(arr):.6f}"
    )

    # NaN / Inf checks
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()

    logger.info(f"NaNs: {nan_count}, Infs: {inf_count}")

    # Sample values
    logger.info(f"sample (first {samples} rows):")
    logger.info(arr[:samples])


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    setup_logger()

    ticker = args.ticker
    base = Path(args.data_dir)

    logger.info(f"Inspecting ticker: {ticker}")
    logger.info(f"Directory: {base.resolve()}")

    # File paths
    paths = {
        "X_train": base / f"{ticker}_X_train.npy",
        "y_train": base / f"{ticker}_y_train.npy",
        "X_val": base / f"{ticker}_X_val.npy",
        "y_val": base / f"{ticker}_y_val.npy",
        "X_test": base / f"{ticker}_X_test.npy",
        "y_test": base / f"{ticker}_y_test.npy",
    }

    # Load arrays
    arrays = {}
    for name, path in paths.items():
        arrays[name] = load_array(path)

    # ============================================================
    # SUMMARIES
    # ============================================================
    for name, arr in arrays.items():
        summarize_array(name, arr, samples=args.samples)

    # ============================================================
    # CONSISTENCY CHECKS
    # ============================================================
    logger.info("\n=== CONSISTENCY CHECKS ===")

    def check_pair(x, y, split):
        if x.shape[0] != y.shape[0]:
            logger.error(f"{split}: X/y row mismatch -> {x.shape[0]} vs {y.shape[0]}")
        else:
            logger.info(f"{split}: X/y aligned ({x.shape[0]} samples)")

    check_pair(arrays["X_train"], arrays["y_train"], "train")
    check_pair(arrays["X_val"], arrays["y_val"], "val")
    check_pair(arrays["X_test"], arrays["y_test"], "test")

    # Feature consistency
    train_features = arrays["X_train"].shape[1]
    val_features = arrays["X_val"].shape[1]
    test_features = arrays["X_test"].shape[1]

    print(
        f"\nFeature dimensions -> train: {train_features}, val: {val_features}, test: {test_features}"
    )

    if train_features == val_features == test_features:
        logger.info(f"Feature dimension consistent: {train_features}")
    else:
        logger.error(
            f"Feature mismatch -> train:{train_features}, val:{val_features}, test:{test_features}"
        )

    logger.info("\nInspection complete.")


if __name__ == "__main__":
    main()
