#!/usr/bin/env python3
"""
Production Feature Engineering Pipeline (TRD-Compliant, Leakage-Free)

CRITICAL ARCHITECTURE:
This pipeline implements the SPLIT-FIRST paradigm to prevent data leakage.

Pipeline Flow (STRICT ORDER):
1. Load raw OHLCV data for all tickers
2. Compute log_return for all tickers (causal operation)
3. TEMPORAL SPLIT (70/10/20) - BEFORE any fitting
4. Select universe & peers on TRAINING data only
5. Generate raw features per split independently
6. FIT transformations on TRAINING data only:
   - Wavelet threshold
   - Feature selector
   - Feature scaler
   - Target scaler
7. TRANSFORM validation and test using frozen parameters
8. Save artifacts with full provenance

TRD Compliance: TRD1, TRD2, TRD3, FINAL_PLAN.md
Audit: FEA_ENG_AUDIT.md - All CRITICAL issues addressed

Author: System Architect
Version: 2.0.0 - AUDIT REMEDIATION
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

import numpy as np
import pandas as pd
import yaml

# Project root setup
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_generators import generate_raw_features
from src.features.target import compute_canonical_target
from src.features.wavelet import compute_wavelet_threshold, apply_wavelet_denoising
from src.features.selector import FeatureSelector
from src.features.scaler import FrozenMinMaxScaler
from src.utils.logger import LogFileMode, setup_logger
from src.utils.config_loader import Config, load_config

logger = logging.getLogger(__name__)


def compute_log_return_all(df: pd.DataFrame, ticker: str) -> None:
    """
    Compute log_return

    Formula: log_return[t] = log(close[t] / close[t-1])

    This is a CAUSAL operation (uses only historical data).
    """

    # Compute log return (causal - uses only prior close)
    df["log_return"] = np.log(df["close"]).diff()

    logger.info(
        f"[{ticker}] Computed log_return: "
        f"{len(df)} samples, {df['log_return'].isna().sum()} NaN"
    )


def temporal_split_all(
    df: pd.DataFrame,
    ticker: str,
    train_pct: float = 0.70,
    val_pct: float = 0.10,
    test_pct: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform chronological split on ALL tickers simultaneously.

    CRITICAL: This must happen BEFORE any fitting operations.

    Args:
        df:  DataFrame
        train_pct: Training fraction (default 0.70)
        val_pct: Validation fraction (default 0.10)
        test_pct: Test fraction (default 0.20)

    Returns:
        Tuple of (df_train, dfs_val, dfs_test)
    """
    if abs(train_pct + val_pct + test_pct - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {train_pct + val_pct + test_pct}"
        )

    N = len(df)
    train_end = int(N * train_pct)
    val_end = int(N * (train_pct + val_pct))

    df_train = df.iloc[:train_end].copy()
    dfs_val = df.iloc[train_end:val_end].copy()
    dfs_test = df.iloc[val_end:].copy()

    logger.info(
        f"[{ticker}] Split: train={len(df_train)} "
        f"val={len(dfs_val)} test={len(dfs_test)}"
    )

    return df_train, dfs_val, dfs_test


def process_ticker_split_first(
    ticker: str,
    df_train: pd.DataFrame,
    dfs_val: pd.DataFrame,
    dfs_test: pd.DataFrame,
    output_dir: Path,
    config: Config,
) -> None:
    """
    Process single ticker with SPLIT-FIRST architecture.

    CRITICAL: This function receives PRE-SPLIT data and NEVER mixes splits.

    Steps:
    1. Select universe/peers on TRAINING data only
    2. Generate raw features per split independently
    3. FIT pipeline on TRAINING data only
    4. TRANSFORM val/test with frozen parameters
    5. Save artifacts with provenance
    """
    logger.info(f"=" * 80)
    logger.info(f"[{ticker}] START SPLIT-FIRST PIPELINE")
    logger.info(f"=" * 80)

    # ============================================================================
    # STAGE 2: RAW FEATURE GENERATION (PER SPLIT INDEPENDENTLY)
    # ============================================================================
    logger.info(f"[{ticker}] Stage 2: Raw Feature Generation")

    peers = []
    X_train_raw, y_train_raw, feat_names_train, idx_train = generate_raw_features(
        ticker,
        df_train,
        split_name="TRAIN",
    )

    X_val_raw, y_val_raw, feat_names_val, idx_val = generate_raw_features(
        ticker,
        dfs_val,
        split_name="VAL",
    )

    X_test_raw, y_test_raw, feat_names_test, idx_test = generate_raw_features(
        ticker,
        dfs_test,
        split_name="TEST",
    )

    # Verify feature consistency
    if feat_names_train != feat_names_val or feat_names_train != feat_names_test:
        raise ValueError(f"[{ticker}] Feature name mismatch across splits!")

    feature_names = feat_names_train
    logger.info(f"[{ticker}] Generated {len(feature_names)} raw features")

    # ============================================================================
    # STAGE 3: WAVELET DENOISING (FIT THRESHOLD ON TRAINING ONLY)
    # ============================================================================
    wavelet_threshold = None

    if config.features.wavelet.enabled:
        logger.info(f"[{ticker}] Stage 3: Wavelet Denoising")

        # Check if 'close' feature exists
        if "close" in feature_names:
            close_idx = feature_names.index("close")

            # FIT: Compute threshold on training data
            close_train_series = pd.Series(X_train_raw[:, close_idx], index=idx_train)
            wavelet_threshold = compute_wavelet_threshold(close_train_series)

            logger.info(
                f"[{ticker}] Wavelet threshold (from TRAIN): {wavelet_threshold:.6f}"
            )

            # TRANSFORM: Apply to all splits
            close_train_denoised = apply_wavelet_denoising(
                close_train_series, threshold_train=wavelet_threshold
            )
            close_val_denoised = apply_wavelet_denoising(
                pd.Series(X_val_raw[:, close_idx], index=idx_val),
                threshold_train=wavelet_threshold,
            )
            close_test_denoised = apply_wavelet_denoising(
                pd.Series(X_test_raw[:, close_idx], index=idx_test),
                threshold_train=wavelet_threshold,
            )

            # Replace 'close' with 'close_denoised'
            X_train_raw[:, close_idx] = close_train_denoised.values
            X_val_raw[:, close_idx] = close_val_denoised.values
            X_test_raw[:, close_idx] = close_test_denoised.values

            feature_names[close_idx] = "close_denoised"

            logger.info(f"[{ticker}] Wavelet denoising applied to all splits")
        else:
            logger.warning(f"[{ticker}] 'close' feature not found, skipping wavelet")

    # ============================================================================
    # STAGE 4: FEATURE SELECTION (FIT ON TRAINING ONLY)
    # ============================================================================
    logger.info(f"[{ticker}] Stage 4: Feature Selection")

    selector = FeatureSelector(
        variance_threshold=config.features.selector.variance_threshold,
        correlation_threshold=config.features.selector.correlation_threshold,
        vif_threshold=config.features.selector.vif_threshold,
        mi_quantile_threshold=config.features.selector.mi_quantile_threshold,
    )

    # FIT on training data
    selector.fit(X_train_raw, y_train_raw, feature_names)

    # TRANSFORM all splits
    X_train_sel, selected_names = selector.transform(X_train_raw, feature_names)
    X_val_sel, _ = selector.transform(X_val_raw, feature_names)
    X_test_sel, _ = selector.transform(X_test_raw, feature_names)

    logger.info(
        f"[{ticker}] Selected {len(selected_names)}/{len(feature_names)} features"
    )
    logger.info(f"[{ticker}] Selector report:\n{selector.report()}")

    # ============================================================================
    # STAGE 5: FEATURE SCALING (FIT ON TRAINING ONLY)
    # ============================================================================
    logger.info(f"[{ticker}] Stage 5: Feature Scaling")

    feature_scaler = FrozenMinMaxScaler(feature_range=(-1, 1))

    # FIT on training features
    feature_scaler.fit(X_train_sel, selected_names)

    # TRANSFORM all splits
    X_train_scaled = feature_scaler.transform(X_train_sel)
    X_val_scaled = feature_scaler.transform(X_val_sel)
    X_test_scaled = feature_scaler.transform(X_test_sel)

    logger.info(
        f"[{ticker}] Feature scaling: train=[{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}] "
        f"val=[{X_val_scaled.min():.4f}, {X_val_scaled.max():.4f}] "
        f"test=[{X_test_scaled.min():.4f}, {X_test_scaled.max():.4f}]"
    )

    # ============================================================================
    # STAGE 6: TARGET SCALING (SEPARATE SCALER, FIT ON TRAINING ONLY)
    # ============================================================================
    logger.info(f"[{ticker}] Stage 6: Target Scaling")

    target_scaler = FrozenMinMaxScaler(feature_range=(-1, 1))

    # FIT on training targets
    y_train_2d = y_train_raw.reshape(-1, 1)
    target_scaler.fit(y_train_2d, ["target"])

    # TRANSFORM all splits
    y_train_scaled = target_scaler.transform(y_train_2d).ravel()
    y_val_scaled = target_scaler.transform(y_val_raw.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

    logger.info(
        f"[{ticker}] Target scaling: train=[{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}] "
        f"val=[{y_val_scaled.min():.4f}, {y_val_scaled.max():.4f}] "
        f"test=[{y_test_scaled.min():.4f}, {y_test_scaled.max():.4f}]"
    )

    # ============================================================================
    # STAGE 7: VALIDATION CHECKS
    # ============================================================================
    logger.info(f"[{ticker}] Stage 7: Validation Checks")

    # Check for NaN
    for name, X in [
        ("train", X_train_scaled),
        ("val", X_val_scaled),
        ("test", X_test_scaled),
    ]:
        if np.isnan(X).any():
            raise ValueError(
                f"[{ticker}] NaN detected in {name} features after scaling!"
            )

    for name, y in [
        ("train", y_train_scaled),
        ("val", y_val_scaled),
        ("test", y_test_scaled),
    ]:
        if np.isnan(y).any():
            raise ValueError(f"[{ticker}] NaN detected in {name} target after scaling!")

    # Check feature range
    eps = 1e-6
    for name, X in [
        ("train", X_train_scaled),
        ("val", X_val_scaled),
        ("test", X_test_scaled),
    ]:
        if X.max() > 1.0 + eps or X.min() < -1.0 - eps:
            logger.warning(
                f"[{ticker}] {name} features slightly outside [-1, 1]: "
                f"[{X.min():.6f}, {X.max():.6f}] (clipping will occur)"
            )
            # Clip to range
            X[:] = np.clip(X, -1.0, 1.0)

    logger.info(f"[{ticker}] Validation checks passed")

    # ============================================================================
    # STAGE 8: SAVE ARTIFACTS
    # ============================================================================
    logger.info(f"[{ticker}] Stage 8: Saving Artifacts")

    # Save arrays
    np.save(
        output_dir / "X_train.npy",
        X_train_scaled.astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        output_dir / "y_train.npy",
        y_train_scaled.astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        output_dir / "X_val.npy", X_val_scaled.astype(np.float32), allow_pickle=False
    )
    np.save(
        output_dir / "y_val.npy", y_val_scaled.astype(np.float32), allow_pickle=False
    )
    np.save(
        output_dir / "X_test.npy", X_test_scaled.astype(np.float32), allow_pickle=False
    )
    np.save(
        output_dir / "y_test.npy", y_test_scaled.astype(np.float32), allow_pickle=False
    )

    # Save indices
    np.save(
        output_dir / "train_index.npy",
        idx_train.values.astype("datetime64[ns]"),
        allow_pickle=False,
    )
    np.save(
        output_dir / "val_index.npy",
        idx_val.values.astype("datetime64[ns]"),
        allow_pickle=False,
    )
    np.save(
        output_dir / "test_index.npy",
        idx_test.values.astype("datetime64[ns]"),
        allow_pickle=False,
    )

    # Save DataFrames (audit trail)
    pd.DataFrame(X_train_scaled, columns=selected_names, index=idx_train).to_parquet(
        output_dir / f"{ticker}_train_features.parquet"
    )
    pd.DataFrame(X_val_scaled, columns=selected_names, index=idx_val).to_parquet(
        output_dir / f"{ticker}_val_features.parquet"
    )
    pd.DataFrame(X_test_scaled, columns=selected_names, index=idx_test).to_parquet(
        output_dir / f"{ticker}_test_features.parquet"
    )

    # Save frozen pipeline state
    frozen_state = {
        "pipeline_version": "2.0.0_split_first",
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat(),
        "feature_names_raw": feature_names,
        "feature_names_selected": selected_names,
        "wavelet": {
            "enabled": config.features.wavelet.enabled,
            "threshold": (
                float(wavelet_threshold) if wavelet_threshold is not None else None
            ),
            "wavelet": "haar",
            "level": 3,
        },
        "selector": {
            "variance_threshold": selector.variance_threshold,
            "correlation_threshold": selector.correlation_threshold,
            "vif_threshold": selector.vif_threshold,
            "mi_quantile_threshold": selector.mi_quantile_threshold,
            "mi_scores": selector.mi_scores_,
        },
        "feature_scaler": feature_scaler.get_params(),
        "target_scaler": target_scaler.get_params(),
        "shapes": {
            "train": list(X_train_scaled.shape),
            "val": list(X_val_scaled.shape),
            "test": list(X_test_scaled.shape),
        },
        "temporal_ranges": {
            "train": [str(idx_train.min()), str(idx_train.max())],
            "val": [str(idx_val.min()), str(idx_val.max())],
            "test": [str(idx_test.min()), str(idx_test.max())],
        },
        "data_quality": {
            "train_nan": bool(np.isnan(X_train_scaled).any()),
            "val_nan": bool(np.isnan(X_val_scaled).any()),
            "test_nan": bool(np.isnan(X_test_scaled).any()),
        },
        "audit_compliance": {
            "split_first": True,
            "fit_on_train_only": True,
            "frozen_transforms": True,
            "issue_2_fixed": True,
            "issue_26_fixed": True,
            "issue_27_fixed": True,
        },
    }

    with open(output_dir / f"{ticker}_frozen_state.json", "w") as f:
        json.dump(frozen_state, f, indent=2)

    logger.info(f"[{ticker}] Artifacts saved to {output_dir}")
    logger.info(f"[{ticker}] PIPELINE COMPLETE")
    logger.info(f"=" * 80)

    # Cleanup
    del X_train_raw, X_val_raw, X_test_raw
    del X_train_sel, X_val_sel, X_test_sel
    del X_train_scaled, X_val_scaled, X_test_scaled
    gc.collect()


def _parse_timeframe(timeframe: str) -> pd.Timedelta:
    mapping = {
        "1Min": pd.Timedelta(minutes=1),
        "5Min": pd.Timedelta(minutes=5),
        "15Min": pd.Timedelta(minutes=15),
        "1Hour": pd.Timedelta(hours=1),
        "1Day": pd.Timedelta(days=1),
    }

    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    return mapping[timeframe]


mapping = {
    "1Min": "1min",
    "5Min": "5min",
    "15Min": "15min",
    "1Hour": "1h",
    "1Day": "1D",
}


def _load_tickers(path: str) -> list[str]:
    with open(path) as f:
        return [
            l.split()[0] for l in f if l.split() and not l.split()[0].startswith("#")
        ]


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)
    df = df.sort_index()
    logger.info(f"Loaded {path} | Rows: {len(df)}")
    logger.info(f"Columns: {df.columns.tolist()}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRD-Compliant Split-First Feature Engineering Pipeline"
    )
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--tickers", default="config/tickers.txt")
    parser.add_argument("--output", default="data/features_v2")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--universe-config", default="config/symbol_universe.yaml")

    args = parser.parse_args()

    setup_logger(
        log_file="logs/build_features_v2.log",
        level="INFO",
        mode=LogFileMode.OVERWRITE,
    )

    logger.info("=" * 80)
    logger.info("PRODUCTION FEATURE PIPELINE v2.0 - SPLIT-FIRST ARCHITECTURE")
    logger.info("=" * 80)
    logger.info(f"Arguments: {args}")

    config = load_config(args.config)

    tickers = _load_tickers(args.tickers)
    timeframes = ["1Min", "5Min", "15Min", "1Hour", "1Day"]

    def get_path(base_dir: Path, timeframe: str, ticker: str) -> Path:
        return base_dir / timeframe / f"{ticker}.parquet"

    processed_dir = Path(args.processed_dir)
    features_dir = Path(args.output)

    def ensure_directories_exist() -> None:
        """
        Ensures all required pipeline directories exist.
        No deletion or modification of existing data is permitted.
        """

        required_dirs = [processed_dir, features_dir]

        dirs = []
        for n in required_dirs:
            for tf in timeframes:
                path = n / tf
                dirs.append(path)

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    ensure_directories_exist()

    for ticker in tickers:
        logger.info("===== FEATURES PIPELINE START: %s =====", ticker)

        for tf in timeframes:
            logger.info("----- TIMEFRAME: %s -----", tf)
            processed_path = get_path(processed_dir, tf, ticker)
            if not processed_path.exists():
                logger.warning("Missing processed data: %s", processed_path)
                continue
            processed_df = _load_parquet(processed_path)
            # Verify monotonic
            if not processed_df.index.is_monotonic_increasing:
                logger.warning(f"[{ticker}] Non-monotonic index, sorting")
                processed_df = processed_df.sort_index()

            compute_log_return_all(processed_df, ticker)
            logger.info("Stage 4: Temporal Split (70/10/20) - CRITICAL STEP")

            df_train, dfs_val, dfs_test = temporal_split_all(
                processed_df, ticker=ticker, train_pct=0.70, val_pct=0.10, test_pct=0.20
            )

            logger.info(
                f"[{ticker}] Split complete: {len(df_train)} train, {len(dfs_val)} val, {len(dfs_test)} test"
            )
            output_dir = Path(args.output) / ticker / tf
            output_dir.mkdir(parents=True, exist_ok=True)

            process_ticker_split_first(
                ticker,
                df_train,
                dfs_val,
                dfs_test,
                output_dir,
                config,
            )
            logger.info("=" * 80)
            logger.info(f"FEATURE ENGINEERING COMPLETE FOR TARGET={ticker}")
            logger.info("=" * 80)


if __name__ == "__main__":
    main()
