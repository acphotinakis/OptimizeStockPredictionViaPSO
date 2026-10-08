import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import time
from .feature_creators_funcs import (
    compute_price_features,
    compute_trd_technical_features,
    compute_statistical_features,
    compute_volume_features,
)

logger = logging.getLogger(__name__)


def generate_raw_features(
    target_ticker: str,
    df_target: pd.DataFrame,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
    """
    Generate raw features for a ALL DATA

    Args:
        target_ticker: Ticker to predict
        dfs: Dictionary of ticker -> DataFrame for THIS SPLIT ONLY

    Returns:
        Tuple of (X, y, feature_names, datetime_index)
        - X: (N, F) raw feature matrix
        - feature_names: List of F feature names
        - datetime_index: DatetimeIndex of length N
    """
    start_time = time.perf_counter()

    logger.info(f"[{target_ticker}] {split_name}: Generating raw features")

    n_rows = len(df_target)

    # ========================================================================
    # STEP 1: GENERATE FEATURE BLOCKS
    # ========================================================================
    blocks = [
        compute_price_features(df_target),
        compute_trd_technical_features(df_target),
        compute_statistical_features(df_target),
        compute_volume_features(df_target),
    ]

    feature_names: List[str] = []
    seen = set()

    processed_blocks = []
    total_features = 0

    for block in blocks:
        # drop duplicates inside block
        if block.columns.has_duplicates:
            block = block.loc[:, ~block.columns.duplicated()]

        cols = block.columns.tolist()
        keep_cols = [c for c in cols if c not in seen]

        seen.update(keep_cols)

        block = block[keep_cols]

        arr = block.to_numpy(dtype=np.float32, copy=False)

        processed_blocks.append(arr)
        feature_names.extend(keep_cols)
        total_features += arr.shape[1]

        logger.info(f"[{target_ticker}] {split_name}: Block features={arr.shape[1]}")

    # ============================================================
    # STEP 2: PREALLOCATE FEATURE MATRIX (NO CONCAT COPY)
    # ============================================================
    X = np.empty((n_rows, total_features), dtype=np.float32)

    start = 0
    for arr in processed_blocks:
        end = start + arr.shape[1]
        X[:, start:end] = arr
        start = end

    logger.info(f"[{target_ticker}] {split_name}: Final feature count={total_features}")

    # ============================================================
    # STEP 3: TARGET
    # ============================================================
    y = df_target["target"].to_numpy(dtype=np.float32, copy=False)

    # ============================================================
    # STEP 4: CLEANING (single pass mask)
    # ============================================================
    nan_mask = np.isnan(y) | np.isnan(X).any(axis=1)

    if np.any(nan_mask):
        drop_count = int(nan_mask.sum())

        logger.warning(
            f"[{target_ticker}] {split_name}: Dropping {drop_count}/{len(y)} "
            f"({100 * drop_count / len(y):.2f}%) NaN rows"
        )

        keep = ~nan_mask
        X = X[keep]
        y = y[keep]
        idx = df_target.index.to_numpy()[keep]
    else:
        idx = df_target.index.to_numpy()

    # ============================================================
    # FINAL VALIDATION (debug only)
    # ============================================================
    # if __debug__:
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()

    logger.info(f"[{target_ticker}] {split_name}: Final X={X.shape}, y={y.shape}")
    end_time = time.perf_counter()

    elapsed_seconds = end_time - start_time

    logger.info(f"Execution time: {elapsed_seconds:.6f} seconds")
    return X, y, feature_names, pd.DatetimeIndex(idx)
