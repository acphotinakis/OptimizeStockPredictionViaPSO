#!/usr/bin/env python3
"""Build feature matrices for each ticker.  Leakage-safe: universe selection
is done once per ticker on training data and frozen for val/test."""

import argparse
from datetime import datetime
import gc
import hashlib
import json
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

from src.data.splitter import DataSplitter
from src.feature_eng_revised.generate_features import FeatureGenerator
from src.data.alpaca_ingestor import AlpacaIngestor
from src.utils.logger import LogFileMode, setup_logger
from src.utils.config_loader import load_config

from src.feature_eng_revised.feature_selector import FeatureSelector
from src.feature_eng_revised.normalization import transform_features

logger = logging.getLogger(__name__)


TARGET_COL = "log_return"
DROP_COLS = [TARGET_COL]


# ============================================================
# CORE PIPELINE
# ============================================================
def compute_initial_features(
    ticker: str,
    df: pd.DataFrame,
    output_dir: Path,
) -> None:

    if df is None or df.empty:
        logger.warning(f"{ticker}: empty dataframe — skipping")
        return

    logger.info(f"{ticker}: rows={len(df)}")

    fg = FeatureGenerator(df=df)

    # ─────────────────────────────────────────────
    # Stage 3.1: Price
    # ─────────────────────────────────────────────
    logger.info("Stage 3.1 START: Price Features")
    df_price, target_series = fg.generate_price_features_and_add_target_col(
        df, target_col="log_return"
    )
    logger.info(f"Stage 3.1 END: shape={df_price.shape}")

    # log the target series stats for sanity check
    logger.info(
        f"{ticker}: target_col='{TARGET_COL}' stats -> "
        f"mean={target_series.mean():.6f}, "
        f"std={target_series.std():.6f}, "
        f"min={target_series.min():.6f}, "
        f"max={target_series.max():.6f}, "
        f"num_zeros={(target_series == 0).sum()}, "
        f"num_nonzeros={(target_series != 0).sum()}"
    )

    # CRITICAL: keep target separate immediately
    df_target = target_series.to_frame(TARGET_COL)

    df = df_price.copy()  # update state for subsequent features

    df_trend = fg.compute_trend_following_indicators(df)
    df_vol = fg.compute_volatility_indicators(df)
    df_mom = fg.compute_momentum_oscillator_indicators(df)
    df_volume = fg.compute_volume_features(df)
    # df_stat = fg.compute_statistical_features(df)
    # df_ext = fg.compute_other_features(df)

    # ─────────────────────────────────────────────
    # CONCAT (STRICT ORDER)
    # ─────────────────────────────────────────────
    df_features = pd.concat([df_price, df_trend, df_vol, df_mom, df_volume], axis=1)

    # ─────────────────────────────────────────────
    # ENFORCE TEMPORAL ORDER
    # ─────────────────────────────────────────────
    df_features = df_features.sort_index().dropna()

    # HARD ENFORCEMENT: REMOVE TARGET IF IT SLIPPED IN
    df_features = df_features.drop(columns=[TARGET_COL], errors="ignore")

    logger.info(f"{ticker}: before_drop_final_shape={df_features.shape}")

    # df_features = df_features.iloc[20:].copy()

    # hard safety (optional but recommended)
    # df_features = df_features.dropna(how="any")

    logger.info(f"{ticker}: final_shape={df_features.shape}")

    # --------------------------
    # SAVE FEATURES ONLY
    # --------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    df_features.to_parquet(
        output_dir / "initial_features.parquet",
        engine="pyarrow",
        compression="zstd",
        index=True,
    )

    # SAVE TARGET SEPARATELY (CRITICAL TRD FIX)
    df_target.loc[df_features.index].to_parquet(
        output_dir / "target.parquet", engine="pyarrow", compression="zstd", index=True
    )

    logger.info(f"{ticker}: saved features + target separately")


def build_and_save_transformed_dataset(
    ticker: str,
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    output_dir: Path,
    target_col: str = TARGET_COL,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> None:
    """
    TRD-compliant feature transformation + persistence pipeline.

    Guarantees:
    - No leakage (temporal split only)
    - Transform fitted only on train
    - Selector applied only on train
    - Target preserved independently and re-attached only at final stage
    """

    if feature_df is None or feature_df.empty:
        logger.warning(f"{ticker}: empty feature dataframe")
        return

    logger.info(f"{ticker}: Starting Stage 4 transformation pipeline")

    # ============================================================
    # 1. ALIGN FEATURES + TARGET
    # ============================================================
    feature_df = feature_df.sort_index()
    target_df = target_df.sort_index()

    df = feature_df.join(target_df[[target_col]], how="inner")
    df = df.dropna()

    if target_col not in df.columns:
        raise ValueError(f"{target_col} missing after join")

    # ============================================================
    # 2. TEMPORAL SPLIT (NO SHUFFLE)
    # ============================================================
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        f"{ticker}: split sizes -> "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    # ============================================================
    # 3. SPLIT X / y (DATAFRAME ONLY - NO NUMPY YET)
    # ============================================================
    def split_df(d: pd.DataFrame):
        if target_col not in d.columns:
            raise ValueError(f"{target_col} missing in split")

        y = d[[target_col]].astype("float32")
        X = d.drop(columns=[target_col]).astype("float32")
        return X, y

    X_train_df, y_train_df = split_df(train_df)
    X_val_df, y_val_df = split_df(val_df)
    X_test_df, y_test_df = split_df(test_df)

    # ============================================================
    # 4. TRANSFORM (FIT ONLY ON TRAIN)
    # ============================================================
    train_scaled, val_scaled, test_scaled, transformer_state = transform_features(
        X_train_df, X_val_df, X_test_df
    )

    # ============================================================
    # 5. FEATURE SELECTION (TARGET RE-ATTACH REQUIRED)
    # ============================================================
    def attach_target(X_df: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
        return X_df.join(y_df)

    train_for_sel = attach_target(train_scaled, y_train_df)
    val_for_sel = attach_target(val_scaled, y_val_df)
    test_for_sel = attach_target(test_scaled, y_test_df)

    # HARD CHECK
    assert target_col in train_for_sel.columns
    assert target_col in val_for_sel.columns
    assert target_col in test_for_sel.columns

    selector = FeatureSelector()

    train_sel, val_sel, test_sel, selector_state = selector.fit_transform(
        train_for_sel,
        val_for_sel,
        test_for_sel,
        target_col=target_col,
    )

    # ============================================================
    # 6. FINAL VALIDATION (HARD GUARANTEE)
    # ============================================================
    for name, df_ in [
        ("train", train_sel),
        ("val", val_sel),
        ("test", test_sel),
    ]:
        if target_col not in df_.columns:
            raise ValueError(f"{target_col} missing after selection in {name}")

    # ============================================================
    # 7. FINAL X / y EXPORT
    # ============================================================
    def to_xy(df_: pd.DataFrame):
        y = df_[[target_col]].values.astype("float32")
        X = df_.drop(columns=[target_col]).values.astype("float32")
        return X, y

    X_train, y_train = to_xy(train_sel)
    X_val, y_val = to_xy(val_sel)
    X_test, y_test = to_xy(test_sel)

    # ============================================================
    # 8. OUTPUT DIR
    # ============================================================
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 9. SAVE ARRAYS
    # ============================================================
    np.save(output_dir / f"{ticker}_X_train.npy", X_train)
    np.save(output_dir / f"{ticker}_y_train.npy", y_train)

    np.save(output_dir / f"{ticker}_X_val.npy", X_val)
    np.save(output_dir / f"{ticker}_y_val.npy", y_val)

    np.save(output_dir / f"{ticker}_X_test.npy", X_test)
    np.save(output_dir / f"{ticker}_y_test.npy", y_test)

    # ============================================================
    # 10. SAVE AUDIT TABLES
    # ============================================================
    train_sel.to_parquet(output_dir / f"{ticker}_train.parquet")
    val_sel.to_parquet(output_dir / f"{ticker}_val.parquet")
    test_sel.to_parquet(output_dir / f"{ticker}_test.parquet")

    # ============================================================
    # 11. SAVE ARTIFACTS
    # ============================================================
    with open(output_dir / f"{ticker}_transformer_state.json", "w") as f:
        json.dump(transformer_state, f, indent=2)

    with open(output_dir / f"{ticker}_selector_state.json", "w") as f:
        json.dump(selector_state, f, indent=2)

    with open(output_dir / f"{ticker}_selector_state.pkl", "wb") as f:
        pickle.dump(selector_state, f)

    with open(output_dir / f"{ticker}_transformer_state.pkl", "wb") as f:
        pickle.dump(transformer_state, f)

    logger.info(f"{ticker}: Stage 4 pipeline complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--output", default="data/features")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--tickers-file", default="config/tickers.txt")
    parser.add_argument("--universe-config", default="config/symbol_universe.yaml")

    def load_tickers(args) -> list[str]:
        if args.ticker:
            return [args.ticker]
        if args.tickers:
            return args.tickers
        with open(args.tickers_file) as f:
            return [l.strip() for l in f if l.strip() and not l.startswith("#")]

    args = parser.parse_args()

    if args.ticker and args.tickers:
        raise ValueError("Use either --ticker or --tickers, not both.")

    setup_logger(
        log_file="logs/02_build_features.log", level="INFO", mode=LogFileMode.OVERWRITE
    )

    load_config(args.config)

    logger.info(f"Arguments:\n{args}")

    tickers = load_tickers(args)
    logger.info("Loaded %d tickers", len(tickers))

    # ============================================================
    # PIPELINE DIRECTORIES
    # ============================================================
    raw_dir = Path("data/processed")
    feature_dir = Path(args.output)

    # ============================================================
    # SINGLE PASS PIPELINE (TRD CORRECT)
    # ============================================================
    for t in tickers:

        raw_path = raw_dir / f"{t}.parquet"

        if not raw_path.exists():
            logger.warning("Missing raw data: %s", t)
            continue

        df = AlpacaIngestor._load_bars(raw_path)

        if df is None or df.empty:
            logger.warning("%s: empty dataframe", t)
            continue

        logger.info("%s: START PIPELINE", t)

        # ============================================================
        # STAGE 3: FEATURE ENGINEERING
        # ============================================================
        compute_initial_features(ticker=t, df=df, output_dir=feature_dir / t)

        # ============================================================
        # STAGE 4: TRANSFORMATION (USES SAME RAW FEATURE SOURCE)
        # ============================================================
        feature_path = feature_dir / t / f"initial_features.parquet"
        target_path = feature_dir / t / f"target.parquet"

        if not feature_path.exists():
            logger.warning("%s: missing initial features file", t)
            continue

        feature_df = pd.read_parquet(feature_path)
        target_df = pd.read_parquet(target_path)

        build_and_save_transformed_dataset(
            ticker=t,
            feature_df=feature_df,
            target_df=target_df,
            output_dir=feature_dir / t,
        )

        logger.info("%s: PIPELINE COMPLETE", t)


if __name__ == "__main__":
    main()
