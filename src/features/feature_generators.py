"""
Raw Feature Generation Module (TRD-Compliant)

This module generates raw features from OHLCV data WITHOUT any fitting operations.
All features are strictly causal (backward-looking).

CRITICAL: This module does NOT:
- Fit scalers
- Fit selectors
- Compute thresholds
- Mix data across splits

Features Generated:
- Price features (OHLCV)
- Technical indicators (TRD-compliant)
- Statistical features
- Volume features
- Cross-ticker features

Author: System Architect
Version: 2.0.0 - AUDIT REMEDIATION
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .feature_creators_funcs import (
    compute_price_features,
    compute_trd_technical_features,
    compute_statistical_features,
    compute_volume_features,
)
from .cross_ticker_strict import compute_cross_ticker_features_strict
from .target import compute_canonical_target

logger = logging.getLogger(__name__)


def generate_raw_features(
    target_ticker: str,
    dfs: Dict[str, pd.DataFrame],
    peer_tickers: List[str],
    split_name: str = "",
    market_context: List[str] = None,
    sector_etf: str = None,
    market_internals: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
    """
    Generate raw features for a single split (train, val, or test).
    
    CRITICAL INVARIANTS:
    - No fitting operations (all features are deterministic transforms)
    - All features are strictly causal (use only historical data)
    - Target is computed correctly (next-period log return, shifted)
    - No data mixing across splits
    
    Args:
        target_ticker: Ticker to predict
        dfs: Dictionary of ticker -> DataFrame for THIS SPLIT ONLY
        peer_tickers: Pre-selected peer tickers (selected on training data)
        split_name: "TRAIN", "VAL", or "TEST" (for logging)
        market_context: List of market context tickers (e.g., ["SPY", "QQQ"])
        sector_etf: Sector ETF for target (e.g., "XLK")
        market_internals: List of market internal tickers (e.g., ["UVXY", "GLD"])
    
    Returns:
        Tuple of (X, y, feature_names, datetime_index)
        - X: (N, F) raw feature matrix
        - y: (N,) target (next-period log_return)
        - feature_names: List of F feature names
        - datetime_index: DatetimeIndex of length N
    """
    if target_ticker not in dfs:
        raise ValueError(f"{split_name}: Target ticker '{target_ticker}' not in dfs")
    
    df_target = dfs[target_ticker]
    
    logger.info(f"[{target_ticker}] {split_name}: Generating raw features")
    
    # ========================================================================
    # STEP 1: GENERATE FEATURE BLOCKS
    # ========================================================================
    blocks = []
    
    # Price features
    block_price = compute_price_features(df_target)
    blocks.append(block_price)
    logger.info(f"[{target_ticker}] {split_name}: Price features: {block_price.shape[1]}")
    
    # Technical indicators (TRD-compliant)
    block_technical = compute_trd_technical_features(df_target)
    blocks.append(block_technical)
    logger.info(f"[{target_ticker}] {split_name}: Technical features: {block_technical.shape[1]}")
    
    # Statistical features
    block_statistical = compute_statistical_features(df_target)
    blocks.append(block_statistical)
    logger.info(f"[{target_ticker}] {split_name}: Statistical features: {block_statistical.shape[1]}")
    
    # Volume features
    block_volume = compute_volume_features(df_target)
    blocks.append(block_volume)
    logger.info(f"[{target_ticker}] {split_name}: Volume features: {block_volume.shape[1]}")
    
    # Cross-ticker features (strict alignment)
    block_cross = compute_cross_ticker_features_strict(
        target_ticker, 
        dfs, 
        peer_tickers, 
        split_name=split_name,
        market_context=market_context,
        sector_etf=sector_etf,
        market_internals=market_internals,
    )
    blocks.append(block_cross)
    logger.info(f"[{target_ticker}] {split_name}: Cross-ticker features: {block_cross.shape[1]}")
    
    # ========================================================================
    # STEP 2: CONCATENATE BLOCKS WITH DEDUPLICATION
    # ========================================================================
    arrays = []
    feature_names = []
    seen_names = set()
    
    for block in blocks:
        # Remove duplicate columns within block
        block = block.loc[:, ~block.columns.duplicated()]
        
        # Track unique names across blocks
        block_names = []
        keep_indices = []
        
        for i, col in enumerate(block.columns):
            if col not in seen_names:
                seen_names.add(col)
                block_names.append(col)
                keep_indices.append(i)
        
        # Slice to keep only unique columns
        if len(keep_indices) < len(block.columns):
            block = block.iloc[:, keep_indices]
        
        arrays.append(block.values.astype(np.float32))
        feature_names.extend(block_names)
    
    X = np.concatenate(arrays, axis=1)
    
    logger.info(
        f"[{target_ticker}] {split_name}: Concatenated {len(feature_names)} features "
        f"from {len(blocks)} blocks"
    )
    
    # ========================================================================
    # STEP 3: COMPUTE CANONICAL TARGET (NEXT-PERIOD LOG RETURN)
    # ========================================================================
    # CRITICAL: This computes y[t] = log(close[t+1] / close[t])
    # The last sample will be NaN and dropped in Step 4.
    y_series = compute_canonical_target(df_target["close"], horizon=1)
    y = y_series.values.astype(np.float32)
    
    # ========================================================================
    # STEP 4: CLEAN NON-FINITE VALUES
    # ========================================================================
    # Replace Inf with 0.0 in features (from division by zero)
    inf_count = np.isinf(X).sum()
    if inf_count > 0:
        logger.warning(
            f"[{target_ticker}] {split_name}: Replacing {inf_count} Inf values with 0.0"
        )
        X = np.nan_to_num(X, copy=False, nan=np.nan, posinf=0.0, neginf=0.0)
    
    # Check for NaN
    nan_mask_X = np.isnan(X).any(axis=1)
    nan_mask_y = np.isnan(y)
    nan_mask = nan_mask_X | nan_mask_y
    
    if nan_mask.sum() > 0:
        n_dropped = int(nan_mask.sum())
        logger.warning(
            f"[{target_ticker}] {split_name}: Dropping {n_dropped}/{len(y)} rows "
            f"({100*n_dropped/len(y):.2f}%) with NaN"
        )
        
        X = X[~nan_mask]
        y = y[~nan_mask]
        idx = df_target.index[~nan_mask]
    else:
        idx = df_target.index
    
    # Final validation
    if np.isnan(X).any():
        raise ValueError(f"{split_name}: NaN still present in X after cleaning!")
    if np.isnan(y).any():
        raise ValueError(f"{split_name}: NaN still present in y after cleaning!")
    
    logger.info(
        f"[{target_ticker}] {split_name}: Final shape X={X.shape}, y={y.shape}, "
        f"features={len(feature_names)}"
    )
    
    return X, y, feature_names, pd.DatetimeIndex(idx)
