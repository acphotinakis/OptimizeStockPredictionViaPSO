import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from prettytable import PrettyTable
import matplotlib.pyplot as plt

# Project root setup
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.target import compute_canonical_target
from pipelines.data_ingest_align_clean import _load_tickers
from src.evaluation.canonical_split import (
    compute_canonical_split,
    verify_split_integrity,
)
from src.utils.data_storage import _load_parquet
from src.features.feature_generators import generate_raw_features
from src.features.selector import FeatureSelector
from src.utils.logger import LogFileMode, setup_logger
from src.utils.config_loader import Config, load_config

logger = logging.getLogger(__name__)


def _as_dataframe(
    data: Union[np.ndarray, pd.DataFrame],
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if columns is None:
        raise ValueError("columns must be provided when input is numpy.ndarray")

    return pd.DataFrame(data, columns=columns)


def inspect_features(
    ticker: str,
    data: Union[np.ndarray, pd.DataFrame],
    columns: Optional[Sequence[str]] = None,
) -> None:

    df = _as_dataframe(data, columns)

    logger.info("\n" + "=" * 90)
    logger.info(f"[{ticker}] FEATURE INSPECTION")
    logger.info("=" * 90)

    logger.info(f"Shape: {df.shape}\n")

    # ---- FIRST 5 ROWS ----
    logger.info("FIRST 5 ROWS (TRANSPOSED)")

    head = df.head()

    table = PrettyTable()
    table.field_names = ["Column", *[f"Row {i}" for i in range(len(head))]]

    for col in df.columns:
        values = head[col].tolist()
        table.add_row([col, *values])

    logger.info("\n" + table.get_string() + "\n")

    # ---- LAST 5 ROWS ----
    logger.info("LAST 5 ROWS (TRANSPOSED)")

    tail = df.tail()

    table = PrettyTable()
    table.field_names = ["Column", *[f"Row {i}" for i in range(len(tail))]]

    for col in df.columns:
        values = tail[col].tolist()
        table.add_row([col, *values])

    logger.info("\n" + table.get_string() + "\n")

    # ---- summary table ----
    table = PrettyTable()
    table.field_names = ["Feature", "NaNs", "Min", "Max", "Mean", "Mode (top 3)"]

    for col in df.columns:
        s = df[col]

        nan = int(s.isna().sum())

        if pd.api.types.is_numeric_dtype(s):
            min_v = float(s.min())
            max_v = float(s.max())
            mean_v = float(s.mean())
        else:
            min_v = max_v = mean_v = "N/A"

        mode_vals = s.mode(dropna=True).values[:3]
        mode_str = ", ".join(map(str, mode_vals)) if len(mode_vals) else "N/A"

        table.add_row([col, nan, min_v, max_v, mean_v, mode_str])

    logger.info(table)
    logger.info("=" * 90 + "\n")


def inspect_targets(
    ticker: str,
    data: Union[np.ndarray, pd.DataFrame],
    columns: Optional[Sequence[str]] = None,
) -> None:

    df = _as_dataframe(data, columns)

    logger.info("\n" + "=" * 90)
    logger.info(f"[{ticker}] TARGET INSPECTION")
    logger.info("=" * 90)

    logger.info(f"Shape: {df.shape}\n")

    # ---- FIRST 5 ROWS ----
    logger.info("FIRST 5 ROWS (TRANSPOSED)")

    head = df.head()

    table = PrettyTable()
    table.field_names = ["Column", *[f"Row {i}" for i in range(len(head))]]

    for col in df.columns:
        values = head[col].tolist()
        table.add_row([col, *values])

    logger.info("\n" + table.get_string() + "\n")

    # ---- LAST 5 ROWS ----
    logger.info("LAST 5 ROWS (TRANSPOSED)")

    tail = df.tail()

    table = PrettyTable()
    table.field_names = ["Column", *[f"Row {i}" for i in range(len(tail))]]

    for col in df.columns:
        values = tail[col].tolist()
        table.add_row([col, *values])

    logger.info("\n" + table.get_string() + "\n")

    # ---- summary table ----
    table = PrettyTable()
    table.field_names = ["Target", "NaNs", "Min", "Max", "Mean", "Mode (top 3)"]

    for col in df.columns:
        s = df[col]

        nan = int(s.isna().sum())

        if pd.api.types.is_numeric_dtype(s):
            min_v = float(s.min())
            max_v = float(s.max())
            mean_v = float(s.mean())
        else:
            min_v = max_v = mean_v = "N/A"

        mode_vals = s.mode(dropna=True).values[:3]
        mode_str = ", ".join(map(str, mode_vals)) if len(mode_vals) else "N/A"

        table.add_row([col, nan, min_v, max_v, mean_v, mode_str])

    logger.info(table)
    logger.info("=" * 90 + "\n")


def _generate_raw(
    ticker: str, df: pd.DataFrame, target_method: str
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DatetimeIndex]:
    logger.info(f"[{ticker}] Stage 2: Raw Feature Generation")
    # df = compute

    df["target"] = compute_canonical_target(
        df["close"], horizon=1, method=target_method
    )

    X_raw_np, y_raw_np, feature_names, datetime_index = generate_raw_features(
        target_ticker=ticker, df_target=df, split_name="ALL"
    )

    X_raw = pd.DataFrame(X_raw_np, columns=feature_names, index=datetime_index)
    y_raw = pd.DataFrame(y_raw_np, columns=["target"], index=datetime_index)

    logger.info(f"[DEBUG] NaNs in target: {y_raw['target'].isna().sum()}")
    logger.info(f"[DEBUG] First 5 targets:\n{y_raw['target'].head()}")
    logger.info(f"[DEBUG] Last 5 targets:\n{y_raw['target'].tail()}")
    logger.info(f"Generated {len(X_raw)} samples with {X_raw.shape[1]} features")
    logger.info("Feature scales (raw):")
    logger.info(X_raw.describe().loc[["mean", "std", "min", "max"]].round(2))
    logger.info("\n")

    inspect_features(ticker=ticker, data=X_raw, columns=feature_names)
    inspect_targets(ticker=ticker, data=y_raw, columns=["target"])

    return X_raw, y_raw, feature_names, datetime_index


def _split_data(X_raw: pd.DataFrame, y_raw: pd.DataFrame):

    logger.info("Splitting feature data...")
    X_train_raw, X_val_raw, X_test_raw = compute_canonical_split(X_raw)
    split_integrity = verify_split_integrity(X_train_raw, X_val_raw, X_test_raw)
    logger.info(f"FEATURE DATA SPLIT INTEGRITY --> [{split_integrity}]")

    idx_train = X_train_raw.index
    idx_val = X_val_raw.index
    idx_test = X_test_raw.index

    logger.info("Splitting target variable data...")
    y_train_raw, y_val_raw, y_test_raw = compute_canonical_split(y_raw)
    split_integrity = verify_split_integrity(y_train_raw, y_val_raw, y_test_raw)
    logger.info(f"TARGET VARIABLE SPLIT INTEGRITY --> [{split_integrity}]")

    return (
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y_train_raw,
        y_val_raw,
        y_test_raw,
        idx_train,
        idx_val,
        idx_test,
    )


def _scale_data(
    X_train_raw,
    X_val_raw,
    X_test_raw,
    y_train_raw,
    y_val_raw,
    y_test_raw,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    MinMaxScaler,
    MinMaxScaler,
]:
    # Feature Scaler (Stage 7)
    feature_scaler = MinMaxScaler(feature_range=(-1, 1))
    feature_scaler.fit(X_train_raw)

    # Target Scaler (Stage 8) - ISOLATED
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    target_scaler.fit(y_train_raw.values.reshape(-1, 1))

    # Transform all datasets
    X_train_scaled = feature_scaler.transform(X_train_raw)
    X_val_scaled = feature_scaler.transform(X_val_raw)
    X_test_scaled = feature_scaler.transform(X_test_raw)

    y_train_scaled = target_scaler.transform(y_train_raw.values.reshape(-1, 1))
    y_val_scaled = target_scaler.transform(y_val_raw.values.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test_raw.values.reshape(-1, 1))

    logger.info("Scaled ranges:")
    logger.info(f"  Features: [{X_train_scaled.min()}, {X_train_scaled.max()}]")
    logger.info(f"  Target:   [{y_train_scaled.min()}, {y_train_scaled.max()}]\n")

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        feature_scaler,
        target_scaler,
    )


def _feature_selection(
    ticker: str,
    config: Config,
    X_train_scaled,
    X_test_scaled,
    X_val_scaled,
    y_train_scaled,
    feature_names,
):
    logger.info(f"[{ticker}] Stage: Feature Selection")

    selector = FeatureSelector(
        variance_threshold=config.features.selector.variance_threshold,
        correlation_threshold=config.features.selector.correlation_threshold,
        mi_quantile_threshold=config.features.selector.mi_quantile_threshold,
    )

    # FIT on training data
    selector.fit(X_train_scaled, y_train_scaled, feature_names)

    # TRANSFORM all splits
    X_train_sel, selected_names = selector.transform(X_train_scaled, feature_names)
    X_val_sel, _ = selector.transform(X_val_scaled, feature_names)
    X_test_sel, _ = selector.transform(X_test_scaled, feature_names)

    logger.info("Feature Selection:")
    logger.info(f"  Feature Names:    [{feature_names}]")
    logger.info(f"  Selected Names:   [{selected_names}]")
    logger.info(f"  X_train_sel:      [{X_train_sel.min()}, {X_train_sel.max()}]")

    return selector, selected_names, X_train_sel, X_val_sel, X_test_sel


import seaborn as sns


def save_feature_target_correlations(
    ticker: str,
    features: pd.DataFrame,
    target: pd.Series,
    output_dir: Path,
    split_name: str = "none",
) -> None:
    out_dir = Path(output_dir) / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    df = features.copy()
    df["target"] = target.values

    corr = df.corr(numeric_only=True)[["target"]].sort_values(
        by="target", ascending=False
    )

    fig, ax = plt.subplots(figsize=(6, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)

    ax.set_title(f"{ticker} | Feature-Target Correlation")

    fig.tight_layout()

    filename = (
        "feature_target_correlations.png"
        if split_name == "none"
        else f"{split_name}_feature_target_correlations.png"
    )
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_features_vs_target(
    ticker: str,
    features: pd.DataFrame,
    target: pd.Series,
    output_dir: Path,
    chunk_size: int = 10,
    split_name: str = "none",
) -> None:

    out_dir = Path(output_dir) / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    df = features.copy()
    df["target"] = target.values

    feature_cols = [c for c in df.columns if c != "target"]

    # ----------------------------
    # normalize all features (z-score)
    # ----------------------------
    df_norm = df.copy()

    for col in feature_cols:
        mean = df_norm[col].mean()
        std = df_norm[col].std()
        df_norm[col] = (df_norm[col] - mean) / (std + 1e-10)

    # normalize target
    t_mean = df_norm["target"].mean()
    t_std = df_norm["target"].std()
    df_norm["target"] = (df_norm["target"] - t_mean) / (t_std + 1e-10)

    # ----------------------------
    # chunk features
    # ----------------------------
    def chunk_list(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    chunks = list(chunk_list(feature_cols, chunk_size))

    for i, chunk in enumerate(chunks):

        fig, ax = plt.subplots(figsize=(16, 8))

        # plot chunked features
        for col in chunk:
            ax.plot(
                df_norm.index,
                df_norm[col],
                linewidth=0.8,
                alpha=0.4,
                label=col,
            )

        # always plot target
        ax.plot(
            df_norm.index,
            df_norm["target"],
            linewidth=2.0,
            color="black",
            label="target",
            zorder=10,
        )

        ax.set_title(
            f"{ticker} | Features {i*chunk_size}-{i*chunk_size+len(chunk)-1} vs Target"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Z-score normalized value")
        ax.grid(True, alpha=0.3)

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
        )

        fig.tight_layout()

        save_path = out_dir / f"{ticker}_features_chunk_{i}.png"
        if split_name != "none":
            save_path = out_dir / f"{ticker}_{split_name}_features_chunk_{i}.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def process_ticker_split_first(
    ticker: str,
    df: pd.DataFrame,
    output_dir: Path,
    config: Config,
    target_method: str = "log_return",
) -> None:
    logger.info(f"=" * 80)
    logger.info(f"[{ticker}] START SPLIT-FIRST PIPELINE")
    logger.info(f"=" * 80)

    # create splits directory
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True, parents=True)

    # ============================================================================
    # STAGE: RAW FEATURE GENERATION
    # ============================================================================

    logger.info(f"Target Method = {target_method}")
    X_raw, y_raw, feature_names, datetime_index = _generate_raw(
        ticker, df, target_method
    )
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    X_raw.to_parquet(raw_dir / f"{ticker}_X_raw.parquet")
    y_raw.to_parquet(raw_dir / f"{ticker}_y_raw.parquet")

    # plot_features_vs_target(
    #     ticker=ticker,
    #     features=X_raw,
    #     target=y_raw["target"],
    #     output_dir=raw_dir / "diagnostics",
    # )

    # save_feature_target_correlations(
    #     ticker=ticker,
    #     features=X_raw,
    #     target=y_raw["target"],
    #     output_dir=raw_dir / "diagnostics",
    # )

    # sys.exit(0)

    # =============================================================================
    # 2. TRAIN / VALIDATION / TEST SPLIT (Time-Series Aware)
    # =============================================================================

    (
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y_train_raw,
        y_val_raw,
        y_test_raw,
        idx_train,
        idx_val,
        idx_test,
    ) = _split_data(X_raw=X_raw, y_raw=y_raw)
    raw_dir = splits_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    X_train_raw.to_parquet(raw_dir / f"{ticker}_train_raw.parquet")
    X_val_raw.to_parquet(raw_dir / f"{ticker}_val_raw.parquet")
    X_test_raw.to_parquet(raw_dir / f"{ticker}_test_raw.parquet")

    # =============================================================================
    # FIT SCALERS ON TRAIN ONLY, TRANSFORM ALL SETS
    # =============================================================================
    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        feature_scaler,
        target_scaler,
    ) = _scale_data(
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y_train_raw,
        y_val_raw,
        y_test_raw,
    )

    # ============================================================================
    # SAVE SCALED FEATURES (PRE-SELECTION)
    # ============================================================================
    scaled_dir = splits_dir / "scaled"
    scaled_dir.mkdir(exist_ok=True)

    pd.DataFrame(X_train_scaled, columns=feature_names, index=idx_train).to_parquet(
        scaled_dir / f"{ticker}_X_train_scaled.parquet"
    )

    pd.DataFrame(X_val_scaled, columns=feature_names, index=idx_val).to_parquet(
        scaled_dir / f"{ticker}_X_val_scaled.parquet"
    )

    pd.DataFrame(X_test_scaled, columns=feature_names, index=idx_test).to_parquet(
        scaled_dir / f"{ticker}_X_test_scaled.parquet"
    )

    pd.DataFrame(
        y_train_scaled.reshape(-1, 1), columns=[target_method], index=idx_train
    ).to_parquet(scaled_dir / f"{ticker}_y_train_scaled.parquet")

    pd.DataFrame(
        y_val_scaled.reshape(-1, 1), columns=[target_method], index=idx_val
    ).to_parquet(scaled_dir / f"{ticker}_y_val_scaled.parquet")

    pd.DataFrame(
        y_test_scaled.reshape(-1, 1), columns=[target_method], index=idx_test
    ).to_parquet(scaled_dir / f"{ticker}_y_test_scaled.parquet")

    joblib.dump(feature_scaler, output_dir / "feature_scaler.joblib")
    joblib.dump(target_scaler, output_dir / "target_scaler.joblib")

    # ============================================================================
    # STAGE: FEATURE SELECTION (FIT ON TRAINING ONLY)
    # ============================================================================
    logger.info(f"[{ticker}] Stage: Feature Selection")

    selector, selected_names, X_train_sel, X_val_sel, X_test_sel = _feature_selection(
        ticker,
        config,
        X_train_scaled,
        X_test_scaled,
        X_val_scaled,
        y_train_scaled,
        feature_names,
    )
    logger.info(f"[{ticker}] Stage 8: Saving Artifacts")
    # ----------------------------------------------------------------------------
    # directories
    # ----------------------------------------------------------------------------
    selected_dir = splits_dir / "selected"
    diagnostics_dir = selected_dir / "diagnostics"

    selected_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------------
    # save metadata
    # ----------------------------------------------------------------------------
    with open(output_dir / "selected_features.json", "w") as f:
        json.dump(selected_names, f)

    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    joblib.dump(selector, output_dir / "feature_selector.joblib")

    # ----------------------------------------------------------------------------
    # build DataFrames ONCE (avoid repetition + ensure consistency)
    # ----------------------------------------------------------------------------
    X_train_sel_df = pd.DataFrame(X_train_sel, columns=selected_names, index=idx_train)
    X_val_sel_df = pd.DataFrame(X_val_sel, columns=selected_names, index=idx_val)
    X_test_sel_df = pd.DataFrame(X_test_sel, columns=selected_names, index=idx_test)

    y_train_series = pd.Series(y_train_scaled.squeeze(), index=idx_train)
    y_val_series = pd.Series(y_val_scaled.squeeze(), index=idx_val)
    y_test_series = pd.Series(y_test_scaled.squeeze(), index=idx_test)

    # ----------------------------------------------------------------------------
    # save features
    # ----------------------------------------------------------------------------
    X_train_sel_df.to_parquet(selected_dir / f"{ticker}_train_features.parquet")
    X_val_sel_df.to_parquet(selected_dir / f"{ticker}_val_features.parquet")
    X_test_sel_df.to_parquet(selected_dir / f"{ticker}_test_features.parquet")

    # ----------------------------------------------------------------------------
    # diagnostics plots (CORRECTED)
    # ----------------------------------------------------------------------------
    # plot_features_vs_target(
    #     ticker=ticker,
    #     features=X_train_sel_df,
    #     target=y_train_series,
    #     output_dir=diagnostics_dir,
    #     split_name="train",
    # )

    # save_feature_target_correlations(
    #     ticker=ticker,
    #     features=X_train_sel_df,
    #     target=y_train_series,
    #     output_dir=diagnostics_dir,
    #     split_name="train",
    # )
    # plot_features_vs_target(
    #     ticker=ticker,
    #     features=X_train_sel_df,
    #     target=y_train_series,
    #     output_dir=diagnostics_dir,
    #     split_name="train",
    # )

    # save_feature_target_correlations(
    #     ticker=ticker,
    #     features=X_train_sel_df,
    #     target=y_train_series,
    #     output_dir=diagnostics_dir,
    #     split_name="train",
    # )
    # ============================================================================
    # SAVE NUMPY ARRAYS (MODEL INPUT)
    # ============================================================================
    np.save(
        output_dir / "X_train.npy",
        X_train_sel,
        allow_pickle=False,
    )
    np.save(output_dir / "y_train.npy", y_train_scaled.squeeze(), allow_pickle=False)

    np.save(output_dir / "X_val.npy", X_val_sel, allow_pickle=False)
    np.save(output_dir / "y_val.npy", y_val_scaled.squeeze(), allow_pickle=False)

    np.save(output_dir / "X_test.npy", X_test_sel, allow_pickle=False)
    np.save(output_dir / "y_test.npy", y_test_scaled.squeeze(), allow_pickle=False)

    # ============================================================================
    # SAVE INDICES
    # ============================================================================
    np.save(
        output_dir / "train_index.npy",
        X_train_raw.index.values.astype("datetime64[ns]"),
        allow_pickle=False,
    )
    np.save(
        output_dir / "val_index.npy",
        X_val_raw.index.values.astype("datetime64[ns]"),
        allow_pickle=False,
    )
    np.save(
        output_dir / "test_index.npy",
        X_test_raw.index.values.astype("datetime64[ns]"),
        allow_pickle=False,
    )

    # Save frozen pipeline state
    frozen_state = {
        "pipeline_version": "2.0.0_split_first",
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat(),
        "feature_names_raw": feature_names,
        "feature_names_selected": selected_names,
        "selector": {
            "variance_threshold": selector.variance_threshold,
            "correlation_threshold": selector.correlation_threshold,
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


def plot_ohlcv_from_parquet(
    df: pd.DataFrame,
    output_dir: Path,
    title: str = "OHLCV Plot",
    volume: bool = True,
) -> None:

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # IMPORTANT: ensure sorted but DO NOT reindex
    df = df.sort_index()

    # ensure we DO NOT accidentally connect broken sequences
    df = df.copy()

    fig, axes = plt.subplots(
        2 if volume else 1,
        1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1] if volume else [1]},
    )

    if not volume:
        axes = [axes]

    ax = axes[0]

    # ---------------------------------------------------------
    # CRITICAL FIX: matplotlib breaks lines ONLY on NaN
    # so we force gaps by ensuring missing timestamps remain NaN
    # and we DO NOT interpolate or fill anywhere
    # ---------------------------------------------------------

    ax.plot(df.index, df["close"], label="close", linewidth=1.2)
    ax.plot(df.index, df["open"], label="open", linewidth=0.8, alpha=0.7)
    ax.plot(df.index, df["high"], label="high", linewidth=0.6, alpha=0.5)
    ax.plot(df.index, df["low"], label="low", linewidth=0.6, alpha=0.5)

    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # ---------------------------------------------------------
    # VOLUME
    # ---------------------------------------------------------
    if volume:
        axv = axes[1]
        axv.bar(df.index, df["volume"], width=1.0, alpha=0.5)
        axv.set_ylabel("Volume")
        axv.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "ohlcv_plot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRD-Compliant Split-First Feature Engineering Pipeline"
    )
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--tickers", default="config/tickers.txt")
    parser.add_argument("--processed-dir", default="data/processed")

    parser.add_argument("--output", default="data/features_v4")

    # TEMP PARAMETERS TO MANIPULATE
    parser.add_argument(
        "--target-method",
        choices=["next_close", "log_return"],
        default="next_close",
    )

    args = parser.parse_args()

    logger_file = "logs/build_features_v3.log"

    setup_logger(
        log_file=logger_file,
        level="INFO",
        mode=LogFileMode.OVERWRITE,
    )

    logger.info("=" * 80)
    logger.info("PRODUCTION FEATURE PIPELINE v3.0 - SPLIT-FIRST ARCHITECTURE")
    logger.info("=" * 80)
    logger.info(f"Arguments: {args}")

    config = load_config(args.config)

    tickers = _load_tickers(args.tickers)
    timeframes = ["1Min", "5Min", "15Min", "1Hour", "1Day"]
    # timeframes = ["15Min", "1Hour", "1Day"]
    # timeframes = ["1Min", "5Min"]
    # timeframes = ["1Day"]
    # tickers = ["SPY"]

    def get_path(base_dir: Path, timeframe: str, ticker: str) -> Path:
        return base_dir / timeframe / f"{ticker}.parquet"

    processed_dir = Path(args.processed_dir)
    features_dir = Path(args.output)
    features_dir.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        logger.info("===== FEATURES PIPELINE START: %s =====", ticker)

        for tf in timeframes:
            logger.info("----- TIMEFRAME: %s -----", tf)
            processed_path = get_path(processed_dir, tf, ticker)
            if not processed_path.exists():
                logger.warning("Missing processed data: %s", processed_path)
                continue
            processed_df = _load_parquet(processed_path)

            output_dir = Path(args.output) / ticker / tf / args.target_method
            output_dir.mkdir(parents=True, exist_ok=True)

            # ---------------------------------------------------------
            # KEEP ONLY NUMERIC FEATURES (prevents noise + errors)
            # ---------------------------------------------------------
            feature_cols = processed_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()

            if len(feature_cols) == 0:
                logger.warning("[%s][%s] No numeric features found", ticker, tf)
                continue

            # inspect_features(
            #     ticker=ticker,
            #     data=processed_df[feature_cols],
            #     columns=feature_cols,
            # )

            # plot_ohlcv_from_parquet(
            #     df=processed_df, output_dir=output_dir, title=f"{ticker} OHLCV ({tf})"
            # )
            # continue
            # sys.exit(0)

            logger_dir = output_dir / "logs"
            logger_dir.mkdir(parents=True, exist_ok=True)

            log_file = logger_dir / "build_features.log"

            setup_logger(
                log_file=log_file,
                level="INFO",
                mode=LogFileMode.OVERWRITE,
            )

            process_ticker_split_first(
                ticker=ticker,
                df=processed_df,
                output_dir=output_dir,
                config=config,
                target_method=args.target_method,
            )

            setup_logger(
                log_file=logger_file,
                level="INFO",
                mode=LogFileMode.OVERWRITE,
            )

            logger.info("=" * 80)
            logger.info(f"FEATURE ENGINEERING COMPLETE FOR TARGET={ticker}")
            logger.info("=" * 80)


if __name__ == "__main__":
    main()
