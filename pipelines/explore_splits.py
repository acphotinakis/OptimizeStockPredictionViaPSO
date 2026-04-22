#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project root setup
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splitter import DataSplitter
from src.utils.logger import LogFileMode, setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def load_tickers(args) -> list[str]:
    if args.ticker:
        return [args.ticker]
    if args.tickers:
        return args.tickers
    with open(args.tickers_file) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def compute_log_returns(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    df = df.copy()
    df[col_name] = np.log(df["close"] / df["close"].shift(1))
    return df


def plot_series(df: pd.DataFrame, col: str, path: Path, title: str) -> None:
    plt.figure()
    plt.plot(df.index, df[col])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Log Return")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_comparison(df: pd.DataFrame, t: str, path: Path) -> None:
    plt.figure()
    plt.plot(df.index, df["before_split_log_return"], label="Before")
    plt.plot(df.index, df["after_split_log_return"], label="After")
    plt.title(f"{t} Log Returns")
    plt.xlabel("Time")
    plt.ylabel("Log Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def extract(df_split: pd.DataFrame, tickers: list[str]) -> dict:
    out = {}

    for t in tickers:
        if t not in df_split.columns.get_level_values(0):
            continue

        sub = df_split.xs(t, axis=1, level=0).copy()
        sub.index = pd.to_datetime(sub.index).tz_localize(None)

        if any(c not in sub.columns for c in OHLCV_COLS):
            continue

        sub = sub[OHLCV_COLS + ["before_split_log_return"]]
        sub = compute_log_returns(sub, "after_split_log_return")

        out[t] = sub

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--output", default="data/features")
    parser.add_argument("--ticker", type=str)
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--tickers-file", default="config/tickers.txt")

    args = parser.parse_args()

    if args.ticker and args.tickers:
        raise ValueError("Use either --ticker or --tickers, not both.")

    setup_logger(
        log_file="logs/02_build_features.log",
        level="INFO",
        mode=LogFileMode.OVERWRITE,
    )
    load_config(args.config)

    tickers = load_tickers(args)
    logger.info("Tickers: %s", tickers)

    processed_dir = Path("data/processed")
    plot_dir = Path("plots/log_returns")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    dfs = {}
    for t in tickers:
        path = processed_dir / f"{t}.parquet"
        if not path.exists():
            logger.warning("Missing parquet: %s", t)
            continue

        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        dfs[t] = df

    # --- Compute pre-split returns + plot ---
    dfs = {
        t: compute_log_returns(df, "before_split_log_return") for t, df in dfs.items()
    }

    for t, df in dfs.items():
        plot_series(
            df,
            "before_split_log_return",
            plot_dir / f"{t}_before.png",
            f"{t} Log Returns (Before Split)",
        )

    # --- Align + split ---
    df_aligned = pd.concat(dfs, axis=1, keys=dfs.keys()).sort_index()
    df_train, df_val, df_test = DataSplitter().split(df_aligned)

    dfs_train = extract(df_train, tickers)
    dfs_val = extract(df_val, tickers)
    dfs_test = extract(df_test, tickers)

    # --- Merge back ---
    merged = {}
    for t in tickers:
        parts = [dfs_train.get(t), dfs_val.get(t), dfs_test.get(t)]
        parts = [p for p in parts if p is not None]
        if parts:
            merged[t] = pd.concat(parts).sort_index()

    # --- Plot after-split + comparison ---
    for t, df in merged.items():
        plot_series(
            df,
            "after_split_log_return",
            plot_dir / f"{t}_after.png",
            f"{t} Log Returns (After Split)",
        )

        plot_comparison(
            df,
            t,
            plot_dir / f"{t}_comparison.png",
        )

    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
# #!/usr/bin/env python3
# """
# Unified Feature Engineering Pipeline

# Production-grade pipeline that uses the unified feature system (src/features_unified/).

# Key Features:
# - TRD-compliant feature generation (36 indicators)
# - Wavelet denoising (Haar, 3-level, soft thresholding)
# - 4-stage feature selection (Variance → Pearson → VIF → MI)
# - Cross-ticker features (SPY + peers + market breadth)
# - MinMax normalization to [-1, 1]
# - Strict leakage prevention (all transformations fit on training only)

# Author: System Architect
# Version: 1.0.0 UNIFIED
# """

# import argparse
# import logging
# import sys
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# # Project root setup
# CURRENT_FILE = Path(__file__).resolve()
# PROJECT_ROOT = CURRENT_FILE.parents[1]

# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# # Unified imports
# from src.data.splitter import DataSplitter
# from src.utils.logger import LogFileMode, setup_logger
# from src.utils.config_loader import load_config

# logger = logging.getLogger(__name__)


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", default="config/default_config.yaml")
#     parser.add_argument("--output", default="data/features")
#     parser.add_argument("--ticker", type=str, default=None)
#     parser.add_argument("--tickers", nargs="+", default=None)
#     parser.add_argument("--tickers-file", default="config/tickers.txt")
#     parser.add_argument("--universe-config", default="config/symbol_universe.yaml")

#     def load_tickers(args) -> list[str]:
#         if args.ticker:
#             return [args.ticker]
#         if args.tickers:
#             return args.tickers
#         with open(args.tickers_file) as f:
#             return [l.strip() for l in f if l.strip() and not l.startswith("#")]

#     args = parser.parse_args()

#     if args.ticker and args.tickers:
#         raise ValueError("Use either --ticker or --tickers, not both.")

#     setup_logger(
#         log_file="logs/02_build_features.log",
#         level="INFO",
#         mode=LogFileMode.OVERWRITE,
#     )
#     config = load_config(args.config)

#     logger.info(f"Arguments: \n {args}")

#     all_tickers = load_tickers(args)
#     logger.info("Loaded %d tickers", len(all_tickers))
#     logger.info("Tickers: %s", all_tickers)

#     # Load data
#     processed_dir = Path("data/processed")
#     dfs = {}
#     for t in all_tickers:
#         p = processed_dir / f"{t}.parquet"
#         if p.exists():
#             df = pd.read_parquet(p)
#             df.index = pd.to_datetime(df.index)
#             dfs[t] = df
#         else:
#             logger.warning("Missing parquet: %s", t)

#     logger.info("Loaded %d ticker DataFrames", len(dfs))
#     for t, df in dfs.items():
#         logger.info(f"{t}: {df.shape}")

#     # --- Compute continuous log returns (safe copy) ---
#     dfs = {t: df.copy() for t, df in dfs.items()}
#     for t, df in dfs.items():
#         df["before_split_log_return"] = np.log(df["close"] / df["close"].shift(1))

#     # --- Plot log returns BEFORE splitting ---
#     plot_dir = Path("plots/log_returns")
#     plot_dir.mkdir(parents=True, exist_ok=True)

#     for t, df in dfs.items():
#         plt.figure()
#         plt.plot(df.index, df["before_split_log_return"])
#         plt.title(f"{t} Log Returns (Full Series)")
#         plt.xlabel("Time")
#         plt.ylabel("Log Return")
#         plt.tight_layout()
#         plt.savefig(plot_dir / f"before{t}_full.png")
#         plt.close()

#     # Align and split
#     df_aligned = pd.concat(dfs, axis=1, keys=dfs.keys()).sort_index()
#     df_train, df_val, df_test = DataSplitter().split(df_aligned)

#     OHLCV_COLS = ["open", "high", "low", "close", "volume"]

#     def extract(df_split: pd.DataFrame) -> dict:
#         out = {}

#         for t in all_tickers:
#             if t not in df_split.columns.get_level_values(0):
#                 continue

#             sub = df_split.xs(t, axis=1, level=0).copy()
#             sub.index = pd.to_datetime(sub.index).tz_localize(None)

#             missing = [c for c in OHLCV_COLS if c not in sub.columns]
#             if missing:
#                 continue

#             sub = sub[OHLCV_COLS + ["before_split_log_return"]]

#             out[t] = sub
#             out[t]["after_split_log_return"] = np.log(
#                 sub["close"] / sub["close"].shift(1)
#             )
#         return out

#     dfs_train = extract(df_train)
#     dfs_val = extract(df_val)
#     dfs_test = extract(df_test)

#     logger.info("Train/Val/Test sizes:")
#     for name, dset in [("train", dfs_train), ("val", dfs_val), ("test", dfs_test)]:
#         for t, df in dset.items():
#             logger.info(f"{name} - {t}: {df.shape}")

#     # merge the dfs back together for the next stages of the pipeline
#     dfs = {
#         t: pd.concat([dfs_train.get(t), dfs_val.get(t), dfs_test.get(t)], axis=0)
#         for t in all_tickers
#     }

#     logger.info("Merged Train/Val/Test sizes:")
#     for t, df in dfs.items():
#         logger.info(f"{t}: {df.shape}")

#     for t, df in dfs.items():
#         plt.figure()
#         plt.plot(df.index, df["after_split_log_return"])
#         plt.title(f"{t} Log Returns (Full Series)")
#         plt.xlabel("Time")
#         plt.ylabel("Log Return")
#         plt.tight_layout()
#         plt.savefig(plot_dir / f"after{t}_full.png")
#         plt.close()

#     for t, df in dfs.items():
#         plt.figure()
#         plt.plot(
#             df.index,
#             df["after_split_log_return"],
#             label="After Split Log Return",
#             color="orange",
#             linestyle="--",
#         )
#         plt.plot(
#             df.index,
#             df["before_split_log_return"],
#             label="Before Split Log Return",
#             color="blue",
#             linestyle="-",
#         )
#         plt.title(f"{t} Log Returns (Full Series)")
#         plt.xlabel("Time")
#         plt.ylabel("Log Return")
#         plt.tight_layout()
#         plt.savefig(plot_dir / f"merged{t}_full.png")
#         plt.close()

#     logger.info("=" * 80)
#     logger.info("FEATURE PIPELINE READY (LOG RETURNS COMPUTED)")
#     logger.info("=" * 80)


# if __name__ == "__main__":
#     main()
