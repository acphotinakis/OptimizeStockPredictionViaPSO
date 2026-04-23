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

from src.data.splitter import DataSplitter
from src.data.aligner import TickerAligner
from src.features.universe_builder import SymbolUniverseBuilder
from src.features.feature_generators import generate_raw_features
from src.features.target import compute_canonical_target
from src.features.wavelet import compute_wavelet_threshold, apply_wavelet_denoising
from src.features.selector import FeatureSelector
from src.features.scaler import FrozenMinMaxScaler
from src.utils.logger import LogFileMode, setup_logger
from src.utils.config_loader import Config, load_config

logger = logging.getLogger(__name__)


def compute_log_return_all(dfs: Dict[str, pd.DataFrame]) -> None:
    """
    Compute log_return for all tickers in-place.

    Formula: log_return[t] = log(close[t] / close[t-1])

    This is a CAUSAL operation (uses only historical data).
    """
    for ticker, df in dfs.items():
        if "close" not in df.columns:
            raise ValueError(f"[{ticker}] Missing 'close' column")

        # Compute log return (causal - uses only prior close)
        df["log_return"] = np.log(df["close"]).diff()

        logger.info(
            f"[{ticker}] Computed log_return: "
            f"{len(df)} samples, {df['log_return'].isna().sum()} NaN"
        )


def temporal_split_all(
    dfs: Dict[str, pd.DataFrame],
    train_pct: float = 0.60,
    val_pct: float = 0.20,
    test_pct: float = 0.20,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Perform chronological split on ALL tickers simultaneously.

    CRITICAL: This must happen BEFORE any fitting operations.

    Args:
        dfs: Dictionary of ticker -> DataFrame
        train_pct: Training fraction (default 0.70)
        val_pct: Validation fraction (default 0.10)
        test_pct: Test fraction (default 0.20)

    Returns:
        Tuple of (dfs_train, dfs_val, dfs_test)
    """
    if abs(train_pct + val_pct + test_pct - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {train_pct + val_pct + test_pct}"
        )

    dfs_train = {}
    dfs_val = {}
    dfs_test = {}

    for ticker, df in dfs.items():
        N = len(df)
        train_end = int(N * train_pct)
        val_end = int(N * (train_pct + val_pct))

        dfs_train[ticker] = df.iloc[:train_end].copy()
        dfs_val[ticker] = df.iloc[train_end:val_end].copy()
        dfs_test[ticker] = df.iloc[val_end:].copy()

        logger.info(
            f"[{ticker}] Split: train={len(dfs_train[ticker])} "
            f"val={len(dfs_val[ticker])} test={len(dfs_test[ticker])}"
        )

    return dfs_train, dfs_val, dfs_test


def process_ticker_split_first(
    ticker: str,
    dfs_train: Dict[str, pd.DataFrame],
    dfs_val: Dict[str, pd.DataFrame],
    dfs_test: Dict[str, pd.DataFrame],
    universe_builder: SymbolUniverseBuilder,
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
    if ticker not in dfs_train:
        logger.warning(f"Skipping {ticker} (not in training data)")
        return

    logger.info(f"=" * 80)
    logger.info(f"[{ticker}] START SPLIT-FIRST PIPELINE")
    logger.info(f"=" * 80)

    # ============================================================================
    # STAGE 1: UNIVERSE & PEERS (FIT ON TRAINING ONLY)
    # ============================================================================
    logger.info(f"[{ticker}] Stage 1: Universe & Peer Selection (TRAINING ONLY)")

    # Select universe and peers with error handling
    try:
        universe = universe_builder.get_universe(ticker, dfs_train, fit=True)
        peers = universe_builder.get_fitted_peers(ticker)

        logger.info(f"[{ticker}] Universe size: {len(universe)}")
        logger.info(f"[{ticker}] Selected peers: {peers}")

        # Validate peer data availability
        missing_peers = [p for p in peers if p not in dfs_train]
        if missing_peers:
            logger.warning(
                f"[{ticker}] Missing peer data: {missing_peers}. "
                "These will be excluded from cross-ticker features."
            )
            # Remove missing peers
            peers = [p for p in peers if p in dfs_train]

        if not peers:
            logger.warning(
                f"[{ticker}] No valid peers available. "
                "Cross-ticker features will be limited to market context only."
            )
    except Exception as e:
        logger.error(f"[{ticker}] Peer selection failed: {e}", exc_info=True)
        raise

    # Extract universe context for feature generation
    sector_etf = universe_builder.sector_map.get(ticker)
    market_context = universe_builder.market_context
    market_internals = universe_builder.market_internals

    logger.info(f"[{ticker}] Sector ETF: {sector_etf}")
    logger.info(f"[{ticker}] Market context: {market_context}")
    logger.info(f"[{ticker}] Market internals: {market_internals}")

    import sys

    sys.exit(0)

    # Filter to available tickers per split
    def filter_universe(
        dfs: Dict[str, pd.DataFrame], split_name: str
    ) -> Dict[str, pd.DataFrame]:
        filtered = {t: dfs[t] for t in universe if t in dfs}
        missing = set(universe) - set(filtered.keys())
        if missing:
            logger.warning(
                f"[{ticker}] {split_name}: Missing {len(missing)} universe tickers: {missing}"
            )
        return filtered

    dfs_train_u = filter_universe(dfs_train, "TRAIN")
    dfs_val_u = filter_universe(dfs_val, "VAL")
    dfs_test_u = filter_universe(dfs_test, "TEST")

    # ============================================================================
    # STAGE 2: RAW FEATURE GENERATION (PER SPLIT INDEPENDENTLY)
    # ============================================================================
    logger.info(f"[{ticker}] Stage 2: Raw Feature Generation")

    X_train_raw, y_train_raw, feat_names_train, idx_train = generate_raw_features(
        ticker,
        dfs_train_u,
        peers,
        split_name="TRAIN",
        market_context=market_context,
        sector_etf=sector_etf,
        market_internals=market_internals,
    )

    X_val_raw, y_val_raw, feat_names_val, idx_val = generate_raw_features(
        ticker,
        dfs_val_u,
        peers,
        split_name="VAL",
        market_context=market_context,
        sector_etf=sector_etf,
        market_internals=market_internals,
    )

    X_test_raw, y_test_raw, feat_names_test, idx_test = generate_raw_features(
        ticker,
        dfs_test_u,
        peers,
        split_name="TEST",
        market_context=market_context,
        sector_etf=sector_etf,
        market_internals=market_internals,
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

    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.save(
        ticker_dir / "X_train.npy",
        X_train_scaled.astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        ticker_dir / "y_train.npy",
        y_train_scaled.astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        ticker_dir / "X_val.npy", X_val_scaled.astype(np.float32), allow_pickle=False
    )
    np.save(
        ticker_dir / "y_val.npy", y_val_scaled.astype(np.float32), allow_pickle=False
    )
    np.save(
        ticker_dir / "X_test.npy", X_test_scaled.astype(np.float32), allow_pickle=False
    )
    np.save(
        ticker_dir / "y_test.npy", y_test_scaled.astype(np.float32), allow_pickle=False
    )

    # Save indices
    np.save(
        ticker_dir / "train_index.npy",
        idx_train.values.astype("datetime64[ns]"),
        allow_pickle=False,
    )
    np.save(
        ticker_dir / "val_index.npy",
        idx_val.values.astype("datetime64[ns]"),
        allow_pickle=False,
    )
    np.save(
        ticker_dir / "test_index.npy",
        idx_test.values.astype("datetime64[ns]"),
        allow_pickle=False,
    )

    # Save DataFrames (audit trail)
    pd.DataFrame(X_train_scaled, columns=selected_names, index=idx_train).to_parquet(
        ticker_dir / f"{ticker}_train_features.parquet"
    )
    pd.DataFrame(X_val_scaled, columns=selected_names, index=idx_val).to_parquet(
        ticker_dir / f"{ticker}_val_features.parquet"
    )
    pd.DataFrame(X_test_scaled, columns=selected_names, index=idx_test).to_parquet(
        ticker_dir / f"{ticker}_test_features.parquet"
    )

    # Save frozen pipeline state
    frozen_state = {
        "pipeline_version": "2.0.0_split_first",
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat(),
        "peers": peers,
        "universe_size": len(universe),
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

    with open(ticker_dir / f"{ticker}_frozen_state.json", "w") as f:
        json.dump(frozen_state, f, indent=2)

    logger.info(f"[{ticker}] Artifacts saved to {ticker_dir}")
    logger.info(f"[{ticker}] PIPELINE COMPLETE")
    logger.info(f"=" * 80)

    # Cleanup
    del X_train_raw, X_val_raw, X_test_raw
    del X_train_sel, X_val_sel, X_test_sel
    del X_train_scaled, X_val_scaled, X_test_scaled
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRD-Compliant Split-First Feature Engineering Pipeline"
    )

    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--output", default="data/features_v2")
    parser.add_argument("--prediction-target", type=str, default="AAPL")
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

    # Load universe config
    with open(args.universe_config) as f:
        u_cfg = yaml.safe_load(f)

    valid_targets = set(u_cfg.get("prediction_targets", []))
    all_symbols = set(
        u_cfg.get("context_only", [])
        + list(valid_targets)
        + u_cfg.get("market_context", [])
    )

    target = args.prediction_target

    if target not in valid_targets:
        raise ValueError(
            f"Invalid prediction target: {target}. "
            f"Must be one of: {sorted(valid_targets)}"
        )

    logger.info(f"Selected prediction target: {target}")
    logger.info(f"Universe symbols: {sorted(all_symbols)}")

    # ============================================================================
    # STAGE 1: LOAD RAW DATA (ALL TICKERS)
    # ============================================================================
    logger.info("Stage 1: Loading raw OHLCV data")

    processed_dir = Path("data/processed")
    dfs = {}

    for symbol in all_symbols:
        p = processed_dir / f"{symbol}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)

            # Verify monotonic
            if not df.index.is_monotonic_increasing:
                logger.warning(f"[{symbol}] Non-monotonic index, sorting")
                df = df.sort_index()

            dfs[symbol] = df
            logger.info(
                f"[{symbol}] Loaded: {len(df)} samples, range=[{df.index.min()}, {df.index.max()}]"
            )
        else:
            logger.warning(f"[{symbol}] Missing parquet file: {p}")

    logger.info(f"Loaded {len(dfs)} ticker DataFrames")

    # Validate universe data availability
    missing_symbols = [s for s in all_symbols if s not in dfs]
    if missing_symbols:
        logger.warning(
            f"Missing {len(missing_symbols)} universe symbols: {missing_symbols}. "
            "This may affect feature quality. Consider ingesting missing data."
        )
        logger.warning(
            "Impact: Some cross-ticker features may be unavailable or less informative."
        )

    # ============================================================================
    # STAGE 2: SPY-ALIGNED REINDEXING (CRITICAL FOR CROSS-TICKER)
    # ============================================================================
    logger.info("Stage 2: SPY-Aligned Reindexing (BEFORE split)")

    # CRITICAL: Align all tickers to SPY's index BEFORE splitting
    # This ensures all tickers have the SAME date range
    aligner = TickerAligner(benchmark_ticker="SPY")
    aligned_df = aligner.align(dfs, fields=["open", "high", "low", "close", "volume"])

    # Extract back to individual DataFrames
    dfs_aligned = {}
    tickers = aligned_df.columns.get_level_values("ticker").unique()
    for ticker in tickers:
        dfs_aligned[ticker] = aligned_df[ticker].copy()
        # dfs_aligned[ticker].columns = dfs_aligned[ticker].columns.droplevel(0)  # Remove ticker level
        logger.info(
            f"[{ticker}] Aligned: {len(dfs_aligned[ticker])} samples, "
            f"NaN: {dfs_aligned[ticker].isna().sum().sum()}"
        )

    dfs = dfs_aligned  # Replace with aligned data

    logger.info(f"All tickers aligned to SPY index: {len(dfs)} tickers")

    # ============================================================================
    # STAGE 3: COMPUTE LOG_RETURN (CAUSAL OPERATION)
    # ============================================================================
    logger.info("Stage 3: Computing log_return for all tickers")

    compute_log_return_all(dfs)

    # ============================================================================
    # STAGE 4: TEMPORAL SPLIT (70/10/20) - BEFORE ANY FITTING
    # ============================================================================
    logger.info("Stage 4: Temporal Split (70/10/20) - CRITICAL STEP")

    dfs_train, dfs_val, dfs_test = temporal_split_all(
        dfs, train_pct=0.70, val_pct=0.10, test_pct=0.20
    )

    logger.info(
        f"Split complete: {len(dfs_train)} train, {len(dfs_val)} val, {len(dfs_test)} test"
    )

    # ============================================================================
    # STAGE 5: INITIALIZE UNIVERSE BUILDER
    # ============================================================================
    logger.info("Stage 5: Initializing Universe Builder")

    universe_builder = SymbolUniverseBuilder.from_config(args.universe_config)

    # ============================================================================
    # STAGE 6: PROCESS TARGET TICKER
    # ============================================================================
    logger.info(f"Stage 6: Processing target ticker: {target}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    process_ticker_split_first(
        target,
        dfs_train,
        dfs_val,
        dfs_test,
        universe_builder,
        output_dir,
        config,
    )

    # ============================================================================
    # STAGE 7: SAVE UNIVERSE METADATA
    # ============================================================================
    peers_path = output_dir / "fitted_peers.json"
    universe_builder.save_peers(peers_path)

    logger.info(f"Saved peers metadata to {peers_path}")

    logger.info("=" * 80)
    logger.info(f"FEATURE ENGINEERING COMPLETE FOR TARGET={target}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
