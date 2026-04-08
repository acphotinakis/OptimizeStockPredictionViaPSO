#!/usr/bin/env python3
"""
Builds feature matrices for each ticker using aligned OHLCV data.
Optimized for memory, parallelization, and GPU-aware XGBoost.
"""

import numpy as np
import argparse
import logging
from pathlib import Path
import pickle
import sys
import gc

import pandas as pd
from joblib import Parallel, delayed

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.pipeline import FeaturePipeline
from src.data.splitter import DataSplitter
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


from prettytable import PrettyTable
import pandas as pd


def pretty_print_df(
    df: pd.DataFrame,
    name: str = "DataFrame",
    n_head: int = 3,
    n_tail: int = 3,
    logger=None,
):
    """
    Pretty print a DataFrame with:
      - shape
      - column names
      - dtypes
      - missing %
      - first N rows
      - last N rows
    """
    if df is None or df.empty:
        msg = f"{name}: EMPTY DataFrame"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return

    # --- Summary table ---
    summary = PrettyTable()
    summary.title = f"{name} Summary"
    summary.field_names = ["Metric", "Value"]

    summary.add_row(["Shape", df.shape])
    summary.add_row(["Rows", len(df)])
    summary.add_row(["Columns", len(df.columns)])
    summary.add_row(["Column Names", ", ".join(map(str, df.columns.tolist()))])

    # Dtypes
    dtype_str = ", ".join([f"{col}:{dtype}" for col, dtype in df.dtypes.items()])
    summary.add_row(["Dtypes", dtype_str])

    # Missing %
    missing_pct = df.isna().mean().mean() * 100
    summary.add_row(["Missing %", f"{missing_pct:.4f}%"])

    # --- Head table ---
    head_table = PrettyTable()
    head_table.title = f"{name} (First {n_head} rows)"
    head_table.field_names = ["Index"] + list(df.columns)

    for idx, row in df.head(n_head).iterrows():
        head_table.add_row([str(idx)] + list(row.values))

    # --- Tail table ---
    tail_table = PrettyTable()
    tail_table.title = f"{name} (Last {n_tail} rows)"
    tail_table.field_names = ["Index"] + list(df.columns)

    for idx, row in df.tail(n_tail).iterrows():
        tail_table.add_row([str(idx)] + list(row.values))

    # --- Output ---
    output = f"\n{summary}\n\n{head_table}\n\n{tail_table}\n"

    if logger:
        logger.info(output)
    else:
        print(output)


def process_ticker(
    ticker, dfs_train, dfs_val, dfs_test, cfg, output_dir, splitter: DataSplitter
):
    if ticker not in dfs_train:
        logger.warning("Skipping %s (not in training data)", ticker)
        return

    # pretty_print_df(dfs_train, name=f"{ticker} (train)", logger=logger)
    # pretty_print_df(dfs_val, name=f"{ticker} (val)", logger=logger)
    # pretty_print_df(dfs_test, name=f"{ticker} (test)", logger=logger)

    pipeline = FeaturePipeline(
        target_ticker=ticker,
        universe_tickers=list(dfs_train.keys()),
        selector_kwargs={
            # "importance_cumulative": cfg.features.selector.importance_threshold
        },
    )

    # --- Fit ---
    X_train, y_train, feature_names = pipeline.fit_transform(dfs_train)

    # --- Transform ---
    X_val, y_val = pipeline.transform(dfs_val)
    X_test, y_test = pipeline.transform(dfs_test)

    # --- Normalize ---
    X_train = splitter.fit_transform(X_train, feature_names)
    X_val = splitter.transform(X_val)
    X_test = splitter.transform(X_test)

    # --- Save ---
    ticker_dir = Path(output_dir) / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    for name, arr in zip(
        ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"],
        [X_train, y_train, X_val, y_val, X_test, y_test],
    ):
        np.save(ticker_dir / f"{name}.npy", arr, allow_pickle=False)

    metadata = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
    }

    with open(ticker_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    logger.info("✓ %s complete (%d features)", ticker, len(feature_names))

    del X_train, X_val, X_test, y_train, y_val, y_test
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Build and select features")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument(
        "--input", type=str, default="data/processed/aligned_universe.parquet"
    )
    parser.add_argument("--output", type=str, default="data/features")
    parser.add_argument("--tickers", type=str, default="config/tickers.txt")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logger(log_file="logs/02_build_features.log", level="INFO")

    # GPU info
    try:
        import torch

        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("GPU detected: %s (%.1f GB)", gpu_name, gpu_memory)
    except ImportError:
        gpu_available = False
        logger.info("PyTorch not available - using CPU only")

    # --- Load tickers ---
    with open(args.tickers) as f:
        tickers = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
    logger.info("Loaded %d tickers", len(tickers))

    # --- Load aligned data ---
    # df_aligned = pd.read_parquet(args.input)
    # logger.info("Loaded aligned data: %s", df_aligned.shape)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.warning(
            "Input Parquet %s not found, merging individual tickers...", input_path
        )
        # Load SPY first
        raw_dir = Path("data/processed")
        spy_path = raw_dir / "SPY.parquet"
        if not spy_path.exists():
            raise FileNotFoundError(f"SPY file not found at {spy_path}")
        df_spy = pd.read_parquet(spy_path)
        logger.info(
            "SPY: shape=%s, columns=%s, rows=%d",
            df_spy.shape,
            df_spy.columns.tolist(),
            len(df_spy),
        )

        merged_dfs = {"SPY": df_spy}

        # Load other tickers
        for ticker in tickers:
            if ticker == "SPY":
                continue
            ticker_path = raw_dir / f"{ticker}.parquet"
            if not ticker_path.exists():
                logger.warning(
                    "Ticker %s not found at %s, skipping", ticker, ticker_path
                )
                continue
            df_ticker = pd.read_parquet(ticker_path)
            logger.info(
                "%s: shape=%s, columns=%s, rows=%d",
                ticker,
                df_ticker.shape,
                df_ticker.columns.tolist(),
                len(df_ticker),
            )
            merged_dfs[ticker] = df_ticker

        # Merge into MultiIndex columns: ticker × fields
        df_aligned = pd.concat(
            merged_dfs, axis=1, names=["Ticker", "Field"]
        ).sort_index(axis=1)
        # Save for future runs
        input_path.parent.mkdir(parents=True, exist_ok=True)
        df_aligned.to_parquet(input_path)
        logger.info("Merged aligned universe saved to %s", input_path)
    else:
        df_aligned = pd.read_parquet(input_path)
        logger.info("Loaded aligned data: %s", df_aligned.shape)

    # --- Split data ---
    splitter = DataSplitter(train_end=cfg.data.train_end, val_end=cfg.data.val_end)
    df_train_all, df_val_all, df_test_all = splitter.split(df_aligned)

    # --- Extract per-ticker DataFrames efficiently ---
    def extract_ticker_dfs(df_multi, tickers):
        """Return a dict of {ticker: DataFrame} for all tickers in df_multi"""
        return {
            ticker: df_multi.xs(ticker, axis=1, level=0)
            for ticker in tickers
            if ticker in df_multi.columns.get_level_values(0)
        }

    # Extract per split
    dfs_train = extract_ticker_dfs(df_train_all, tickers)
    dfs_val = extract_ticker_dfs(df_val_all, tickers)
    dfs_test = extract_ticker_dfs(df_test_all, tickers)

    # --- Log details per ticker using pretty printer ---
    for ticker, df in dfs_train.items():
        pretty_print_df(df, name=f"{ticker} (train)", logger=logger)

    for ticker, df in dfs_val.items():
        pretty_print_df(df, name=f"{ticker} (val)", logger=logger)

    for ticker, df in dfs_test.items():
        pretty_print_df(df, name=f"{ticker} (test)", logger=logger)

    logger.info("Extracted %d tickers for feature engineering", len(dfs_train))
    logger.info(
        "Cleared DataFrames from memory:\n"
        "aligned=%d bytes (%.2f GB), train=%d bytes (%.2f GB), "
        "val=%d bytes (%.2f GB), test=%d bytes (%.2f GB)",
        df_aligned.memory_usage(deep=True).sum(),
        df_aligned.memory_usage(deep=True).sum() / 1e9,
        df_train_all.memory_usage(deep=True).sum(),
        df_train_all.memory_usage(deep=True).sum() / 1e9,
        df_val_all.memory_usage(deep=True).sum(),
        df_val_all.memory_usage(deep=True).sum() / 1e9,
        df_test_all.memory_usage(deep=True).sum(),
        df_test_all.memory_usage(deep=True).sum() / 1e9,
    )
    # --- Clean memory ---
    del df_aligned, df_train_all, df_val_all, df_test_all
    gc.collect()

    # --- Output directory ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Parallel feature construction ---
    logger.info("Starting parallel feature generation with %d jobs", args.n_jobs)
    for ticker in tickers:
        process_ticker(ticker, dfs_train, dfs_val, dfs_test, cfg, output_dir, splitter)
    # Parallel(n_jobs=args.n_jobs, backend="loky")(
    #     delayed(process_ticker)(
    #         ticker,
    #         dfs_train,
    #         dfs_val,
    #         dfs_test,
    #         cfg,
    #         output_dir,
    #         splitter,
    #     )
    #     for ticker in tickers
    # )

    logger.info("=" * 60)
    logger.info("✓ Feature engineering complete!")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# scripts/02_build_features.py

# Builds feature matrices for each ticker using the aligned OHLCV data.
# Loads ALL tickers to compute proper cross-ticker features (SPY correlation, etc.).
# Applies feature selection (XGBoost importance) and saves to disk.

# GPU acceleration: Automatically uses GPU for XGBoost if available.

# Usage:
#     python scripts/02_build_features.py --config config/default_config.yaml
#     python scripts/02_build_features.py --config config/default_config.yaml --n-jobs 4
# """

# import argparse
# import logging
# import pickle
# import sys
# from pathlib import Path

# import numpy as np
# import pandas as pd

# # Add project root to Python path
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# from src.features.pipeline import FeaturePipeline
# from src.data.splitter import DataSplitter
# from src.utils.logger import setup_logger
# from src.utils.config_loader import load_config

# logger = logging.getLogger(__name__)


# def process_ticker(
#     ticker: str,
#     dfs_train,
#     dfs_val,
#     dfs_test,
#     cfg,
#     output_dir,
#     splitter,
# ):
#     import numpy as np
#     import pickle
#     from pathlib import Path

#     logger = logging.getLogger(__name__)

#     if ticker not in dfs_train:
#         logger.warning("Skipping %s (not in training data)", ticker)
#         return

#     pipeline = FeaturePipeline(
#         target_ticker=ticker,
#         universe_tickers=list(dfs_train.keys()),
#         selector_kwargs={
#             "importance_cumulative": cfg.features.selector.importance_threshold,
#         },
#     )

#     # --- Fit ---
#     X_train, y_train, feature_names = pipeline.fit_transform(dfs_train)

#     # --- Transform ---
#     X_val, y_val = pipeline.transform(dfs_val)
#     X_test, y_test = pipeline.transform(dfs_test)

#     # --- Normalize (optimized later in splitter) ---
#     X_train = splitter.fit_transform(X_train, feature_names)
#     X_val = splitter.transform(X_val)
#     X_test = splitter.transform(X_test)

#     # --- Save ---
#     ticker_dir = Path(output_dir) / ticker
#     ticker_dir.mkdir(parents=True, exist_ok=True)

#     np.save(ticker_dir / "X_train.npy", X_train, allow_pickle=False)
#     np.save(ticker_dir / "y_train.npy", y_train, allow_pickle=False)
#     np.save(ticker_dir / "X_val.npy", X_val, allow_pickle=False)
#     np.save(ticker_dir / "y_val.npy", y_val, allow_pickle=False)
#     np.save(ticker_dir / "X_test.npy", X_test, allow_pickle=False)
#     np.save(ticker_dir / "y_test.npy", y_test, allow_pickle=False)

#     metadata = {
#         "feature_names": feature_names,
#         "n_features": len(feature_names),
#         "train_samples": len(X_train),
#         "val_samples": len(X_val),
#         "test_samples": len(X_test),
#     }

#     with open(ticker_dir / "metadata.pkl", "wb") as f:
#         pickle.dump(metadata, f)

#     logger.info("✓ %s complete (%d features)", ticker, len(feature_names))


# def main():
#     parser = argparse.ArgumentParser(description="Build and select features")
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="config/default_config.yaml",
#         help="Path to config file",
#     )
#     parser.add_argument(
#         "--input",
#         type=str,
#         default="data/processed/aligned_universe.parquet",
#         help="Path to aligned universe parquet file",
#     )
#     parser.add_argument(
#         "--output",
#         type=str,
#         default="data/features",
#         help="Output directory for feature matrices",
#     )
#     parser.add_argument(
#         "--tickers",
#         type=str,
#         default="config/tickers.txt",
#         help="Path to ticker list file",
#     )
#     parser.add_argument(
#         "--n-jobs",
#         type=int,
#         default=1,
#         help="Number of parallel jobs (for future use)",
#     )
#     args = parser.parse_args()

#     # Load config
#     cfg = load_config(args.config)
#     setup_logger(log_file="logs/02_build_features.log", level="INFO")

#     # Check GPU availability
#     try:
#         import torch

#         gpu_available = torch.cuda.is_available()
#         if gpu_available:
#             gpu_name = torch.cuda.get_device_name(0)
#             gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
#             logger.info("🚀 GPU detected: %s (%.1f GB)", gpu_name, gpu_memory)
#             logger.info("XGBoost will automatically use GPU acceleration")
#         else:
#             logger.info("No GPU detected - using CPU only")
#     except ImportError:
#         gpu_available = False
#         logger.info("PyTorch not available - using CPU only")

#     # Read ticker list
#     with open(args.tickers) as f:
#         tickers = [
#             line.strip()
#             for line in f
#             if line.strip() and not line.strip().startswith("#")
#         ]
#     logger.info("Loaded %d tickers", len(tickers))

#     # Load aligned data
#     logger.info("Loading aligned data from %s", args.input)
#     df_aligned = pd.read_parquet(args.input)
#     logger.info("Loaded data shape: %s", df_aligned.shape)

#     logger.info(f"Columns in aligned data: {df_aligned.columns.tolist()}")

#     # Split into train/val/test
#     splitter = DataSplitter(
#         train_end=cfg.data.train_end,
#         val_end=cfg.data.val_end,
#     )

#     # Split the aligned data
#     df_train_all, df_val_all, df_test_all = splitter.split(df_aligned)
#     logger.info(
#         "Split complete - Train: %d, Val: %d, Test: %d",
#         len(df_train_all),
#         len(df_val_all),
#         len(df_test_all),
#     )

#     # Convert MultiIndex DataFrames to dict of single-ticker DataFrames
#     def extract_ticker_dfs(df_multi):
#         """Extract individual ticker DataFrames from MultiIndex DataFrame."""
#         # available = set(df_multi.columns.get_level_values(0))
#         # ticker_dfs = {}
#         # for ticker in tickers:
#         #     if ticker in available:
#         #         ticker_dfs[ticker] = df_multi[ticker]
#         # return ticker_dfs
#         return {
#             ticker: df_multi.xs(ticker, axis=1, level=0)
#             for ticker in tickers
#             if ticker in df_multi.columns.get_level_values(0)
#         }

#     dfs_train = extract_ticker_dfs(df_train_all)
#     dfs_val = extract_ticker_dfs(df_val_all)
#     dfs_test = extract_ticker_dfs(df_test_all)

#     logger.info("Extracted %d tickers for feature engineering", len(dfs_train))

#     logger.info(
#         "Cleared DataFrames from memory:\n"
#         "aligned=%d bytes (%.2f GB), train=%d bytes (%.2f GB), "
#         "val=%d bytes (%.2f GB), test=%d bytes (%.2f GB)",
#         df_aligned.memory_usage(deep=True).sum(),
#         df_aligned.memory_usage(deep=True).sum() / 1e9,
#         df_train_all.memory_usage(deep=True).sum(),
#         df_train_all.memory_usage(deep=True).sum() / 1e9,
#         df_val_all.memory_usage(deep=True).sum(),
#         df_val_all.memory_usage(deep=True).sum() / 1e9,
#         df_test_all.memory_usage(deep=True).sum(),
#         df_test_all.memory_usage(deep=True).sum() / 1e9,
#     )
#     del df_aligned, df_train_all, df_val_all, df_test_all
#     import gc

#     gc.collect()

#     # import sys

#     # sys.exit(0)

#     # Build features for each ticker
#     output_dir = Path(args.output)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     from joblib import Parallel, delayed

#     logger.info("Starting parallel feature generation with %d jobs", args.n_jobs)

#     Parallel(n_jobs=args.n_jobs, backend="loky")(
#         delayed(process_ticker)(
#             ticker,
#             dfs_train,
#             dfs_val,
#             dfs_test,
#             cfg,
#             output_dir,
#             DataSplitter(
#                 train_end=cfg.data.train_end,
#                 val_end=cfg.data.val_end,
#             ),
#         )
#         for ticker in tickers
#     )

#     logger.info("=" * 60)
#     logger.info("✓ Feature engineering complete!")


# if __name__ == "__main__":
#     main()

# # #!/usr/bin/env python3
# # """
# # scripts/02_build_features.py

# # Builds feature matrices for each ticker using the aligned OHLCV data.
# # Loads ALL tickers to compute proper cross-ticker features (SPY correlation, etc.).
# # Applies feature selection (XGBoost importance) and saves to disk.

# # GPU acceleration: Automatically uses GPU for XGBoost if available.

# # Usage:
# #     python scripts/02_build_features.py --config config/default_config.yaml
# #     python scripts/02_build_features.py --config config/default_config.yaml --n-jobs 4
# # """

# # import argparse
# # import logging
# # import pickle
# # import sys
# # from pathlib import Path

# # import numpy as np
# # import pandas as pd

# # # Add project root to Python path
# # project_root = Path(__file__).parent.parent
# # sys.path.insert(0, str(project_root))

# # from src.features.pipeline import FeaturePipeline
# # from src.data.splitter import DataSplitter
# # from src.utils.logger import setup_logger
# # from src.utils.config_loader import load_config

# # logger = logging.getLogger(__name__)


# # def main():
# #     parser = argparse.ArgumentParser(description="Build and select features")
# #     parser.add_argument(
# #         "--config",
# #         type=str,
# #         default="config/default_config.yaml",
# #         help="Path to config file",
# #     )
# #     parser.add_argument(
# #         "--input",
# #         type=str,
# #         default="data/processed/aligned_universe.parquet",
# #         help="Path to aligned universe parquet file",
# #     )
# #     parser.add_argument(
# #         "--output",
# #         type=str,
# #         default="data/features",
# #         help="Output directory for feature matrices",
# #     )
# #     parser.add_argument(
# #         "--tickers",
# #         type=str,
# #         default="config/tickers.txt",
# #         help="Path to ticker list file",
# #     )
# #     parser.add_argument(
# #         "--n-jobs",
# #         type=int,
# #         default=1,
# #         help="Number of parallel jobs (for future use)",
# #     )
# #     args = parser.parse_args()

# #     # Load config
# #     cfg = load_config(args.config)
# #     setup_logger(log_file="logs/02_build_features.log", level="INFO")

# #     # Check GPU availability
# #     try:
# #         import torch

# #         gpu_available = torch.cuda.is_available()
# #         if gpu_available:
# #             gpu_name = torch.cuda.get_device_name(0)
# #             gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
# #             logger.info("🚀 GPU detected: %s (%.1f GB)", gpu_name, gpu_memory)
# #             logger.info("XGBoost will automatically use GPU acceleration")
# #         else:
# #             logger.info("No GPU detected - using CPU only")
# #     except ImportError:
# #         gpu_available = False
# #         logger.info("PyTorch not available - using CPU only")

# #     # Read ticker list
# #     with open(args.tickers) as f:
# #         tickers = [
# #             line.strip()
# #             for line in f
# #             if line.strip() and not line.strip().startswith("#")
# #         ]
# #     logger.info("Loaded %d tickers", len(tickers))

# #     # Load aligned data
# #     logger.info("Loading aligned data from %s", args.input)
# #     df_aligned = pd.read_parquet(args.input)
# #     logger.info("Loaded data shape: %s", df_aligned.shape)

# #     logger.info(f"Columns in aligned data: {df_aligned.columns.tolist()}")

# #     # Split into train/val/test
# #     splitter = DataSplitter(
# #         train_end=cfg.data.train_end,
# #         val_end=cfg.data.val_end,
# #     )

# #     # Split the aligned data
# #     df_train_all, df_val_all, df_test_all = splitter.split(df_aligned)
# #     logger.info(
# #         "Split complete - Train: %d, Val: %d, Test: %d",
# #         len(df_train_all),
# #         len(df_val_all),
# #         len(df_test_all),
# #     )

# #     # Convert MultiIndex DataFrames to dict of single-ticker DataFrames
# #     def extract_ticker_dfs(df_multi):
# #         """Extract individual ticker DataFrames from MultiIndex DataFrame."""
# #         ticker_dfs = {}
# #         for ticker in tickers:
# #             if ticker not in df_multi.columns.get_level_values(0):
# #                 continue
# #             ticker_dfs[ticker] = df_multi[ticker].copy()
# #         return ticker_dfs

# #     dfs_train = extract_ticker_dfs(df_train_all)
# #     dfs_val = extract_ticker_dfs(df_val_all)
# #     dfs_test = extract_ticker_dfs(df_test_all)

# #     logger.info(f"Columns of Train DataFrames: {list(dfs_train.keys())}")
# #     logger.info(f"Columns of Val DataFrames: {list(dfs_val.keys())}")
# #     logger.info(f"Columns of Test DataFrames: {list(dfs_test.keys())}")

# #     # print the first 5 rows of each dataframe for the first ticker
# #     for ticker in tickers[:1]:  # Just the first ticker for brevity
# #         if ticker in dfs_train:
# #             logger.info(
# #                 f"First 5 rows of Train DataFrame for {ticker}:\n{dfs_train[ticker].head()}"
# #             )
# #         if ticker in dfs_val:
# #             logger.info(
# #                 f"First 5 rows of Val DataFrame for {ticker}:\n{dfs_val[ticker].head()}"
# #             )
# #         if ticker in dfs_test:
# #             logger.info(
# #                 f"First 5 rows of Test DataFrame for {ticker}:\n{dfs_test[ticker].head()}"
# #             )

# #     import sys

# #     sys.exit(0)

# #     logger.info("Extracted %d tickers for feature engineering", len(dfs_train))

# #     # Build features for each ticker
# #     output_dir = Path(args.output)
# #     output_dir.mkdir(parents=True, exist_ok=True)

# #     for idx, ticker in enumerate(tickers, 1):
# #         if ticker not in dfs_train:
# #             logger.warning("Skipping %s (not in training data)", ticker)
# #             continue

# #         logger.info("=" * 60)
# #         logger.info("[%d/%d] Processing ticker: %s", idx, len(tickers), ticker)

# #         # Initialize feature pipeline with full universe for cross-ticker features
# #         pipeline = FeaturePipeline(
# #             target_ticker=ticker,
# #             universe_tickers=list(dfs_train.keys()),
# #             selector_kwargs={
# #                 "importance_cumulative": cfg.features.selector.importance_threshold,
# #             },
# #         )

# #         # Fit on training data (pass ALL tickers for cross-ticker features)
# #         logger.info("  Computing and selecting features...")
# #         X_train, y_train, feature_names = pipeline.fit_transform(dfs_train)
# #         logger.info("  Train features: %d samples × %d features", *X_train.shape)

# #         # Transform val and test (pass ALL tickers for cross-ticker features)
# #         X_val, y_val = pipeline.transform(dfs_val)
# #         X_test, y_test = pipeline.transform(dfs_test)
# #         logger.info("  Val: %d samples, Test: %d samples", len(X_val), len(X_test))

# #         # Normalize features
# #         logger.info("  Normalizing features...")
# #         X_train_norm = splitter.fit_transform(X_train, feature_names)
# #         X_val_norm = splitter.transform(X_val)
# #         X_test_norm = splitter.transform(X_test)

# #         # Save to disk
# #         ticker_dir = output_dir / ticker
# #         ticker_dir.mkdir(parents=True, exist_ok=True)

# #         np.save(ticker_dir / "X_train.npy", X_train_norm)
# #         np.save(ticker_dir / "y_train.npy", y_train)
# #         np.save(ticker_dir / "X_val.npy", X_val_norm)
# #         np.save(ticker_dir / "y_val.npy", y_val)
# #         np.save(ticker_dir / "X_test.npy", X_test_norm)
# #         np.save(ticker_dir / "y_test.npy", y_test)

# #         # Save metadata
# #         metadata = {
# #             "feature_names": feature_names,
# #             "n_features": len(feature_names),
# #             "train_samples": len(X_train),
# #             "val_samples": len(X_val),
# #             "test_samples": len(X_test),
# #         }
# #         with open(ticker_dir / "metadata.pkl", "wb") as f:
# #             pickle.dump(metadata, f)

# #         logger.info("  ✓ Saved features to %s", ticker_dir)

# #     logger.info("=" * 60)
# #     logger.info("✓ Feature engineering complete!")


# # if __name__ == "__main__":
# #     main()

# #!/usr/bin/env python3
# """
# Builds feature matrices for each ticker using aligned OHLCV data.
# Optimized for memory, parallelization, and GPU-aware XGBoost.
# """

# import numpy as np
# import argparse
# import logging
# from pathlib import Path
# import pickle
# import sys
# import gc

# import pandas as pd
# from joblib import Parallel, delayed

# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# from src.features.pipeline import FeaturePipeline
# from src.data.splitter import DataSplitter
# from src.utils.logger import setup_logger
# from src.utils.config_loader import load_config

# logger = logging.getLogger(__name__)


# from prettytable import PrettyTable
# import pandas as pd


# def pretty_print_df(
#     df: pd.DataFrame,
#     name: str = "DataFrame",
#     n_head: int = 3,
#     n_tail: int = 3,
#     logger=None,
# ):
#     """
#     Pretty print a DataFrame with:
#       - shape
#       - column names
#       - dtypes
#       - missing %
#       - first N rows
#       - last N rows
#     """
#     if df is None or df.empty:
#         msg = f"{name}: EMPTY DataFrame"
#         if logger:
#             logger.info(msg)
#         else:
#             print(msg)
#         return

#     # --- Summary table ---
#     summary = PrettyTable()
#     summary.title = f"{name} Summary"
#     summary.field_names = ["Metric", "Value"]

#     summary.add_row(["Shape", df.shape])
#     summary.add_row(["Rows", len(df)])
#     summary.add_row(["Columns", len(df.columns)])
#     summary.add_row(["Column Names", ", ".join(map(str, df.columns.tolist()))])

#     # Dtypes
#     dtype_str = ", ".join([f"{col}:{dtype}" for col, dtype in df.dtypes.items()])
#     summary.add_row(["Dtypes", dtype_str])

#     # Missing %
#     missing_pct = df.isna().mean().mean() * 100
#     summary.add_row(["Missing %", f"{missing_pct:.4f}%"])

#     # --- Head table ---
#     head_table = PrettyTable()
#     head_table.title = f"{name} (First {n_head} rows)"
#     head_table.field_names = ["Index"] + list(df.columns)

#     for idx, row in df.head(n_head).iterrows():
#         head_table.add_row([str(idx)] + list(row.values))

#     # --- Tail table ---
#     tail_table = PrettyTable()
#     tail_table.title = f"{name} (Last {n_tail} rows)"
#     tail_table.field_names = ["Index"] + list(df.columns)

#     for idx, row in df.tail(n_tail).iterrows():
#         tail_table.add_row([str(idx)] + list(row.values))

#     # --- Output ---
#     output = f"\n{summary}\n\n{head_table}\n\n{tail_table}\n"

#     if logger:
#         logger.info(output)
#     else:
#         print(output)


# def process_ticker(
#     ticker, dfs_train, dfs_val, dfs_test, cfg, output_dir, splitter: DataSplitter
# ):
#     if ticker not in dfs_train:
#         logger.warning("Skipping %s (not in training data)", ticker)
#         return

#     # pretty_print_df(dfs_train, name=f"{ticker} (train)", logger=logger)
#     # pretty_print_df(dfs_val, name=f"{ticker} (val)", logger=logger)
#     # pretty_print_df(dfs_test, name=f"{ticker} (test)", logger=logger)

#     pipeline = FeaturePipeline(
#         target_ticker=ticker,
#         universe_tickers=list(dfs_train.keys()),
#         selector_kwargs={
#             # "importance_cumulative": cfg.features.selector.importance_threshold
#         },
#     )

#     # --- Fit ---
#     X_train, y_train, feature_names = pipeline.fit_transform(dfs_train)

#     # --- Transform ---
#     X_val, y_val = pipeline.transform(dfs_val)
#     X_test, y_test = pipeline.transform(dfs_test)

#     # --- Normalize ---
#     X_train = splitter.fit_transform(X_train, feature_names)
#     X_val = splitter.transform(X_val)
#     X_test = splitter.transform(X_test)

#     # --- Save ---
#     ticker_dir = Path(output_dir) / ticker
#     ticker_dir.mkdir(parents=True, exist_ok=True)

#     for name, arr in zip(
#         ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"],
#         [X_train, y_train, X_val, y_val, X_test, y_test],
#     ):
#         np.save(ticker_dir / f"{name}.npy", arr, allow_pickle=False)

#     metadata = {
#         "feature_names": feature_names,
#         "n_features": len(feature_names),
#         "train_samples": len(X_train),
#         "val_samples": len(X_val),
#         "test_samples": len(X_test),
#     }

#     with open(ticker_dir / "metadata.pkl", "wb") as f:
#         pickle.dump(metadata, f)

#     logger.info("✓ %s complete (%d features)", ticker, len(feature_names))

#     del X_train, X_val, X_test, y_train, y_val, y_test
#     gc.collect()


# def main():
#     parser = argparse.ArgumentParser(description="Build and select features")
#     parser.add_argument("--config", type=str, default="config/default_config.yaml")
#     parser.add_argument(
#         "--input", type=str, default="data/processed/aligned_universe.parquet"
#     )
#     parser.add_argument("--output", type=str, default="data/features")
#     parser.add_argument("--tickers", type=str, default="config/tickers.txt")
#     parser.add_argument("--n-jobs", type=int, default=1)
#     args = parser.parse_args()

#     cfg = load_config(args.config)
#     setup_logger(log_file="logs/02_build_features.log", level="INFO")

#     # GPU info
#     try:
#         import torch

#         gpu_available = torch.cuda.is_available()
#         if gpu_available:
#             gpu_name = torch.cuda.get_device_name(0)
#             gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
#             logger.info("GPU detected: %s (%.1f GB)", gpu_name, gpu_memory)
#     except ImportError:
#         gpu_available = False
#         logger.info("PyTorch not available - using CPU only")

#     # --- Load tickers ---
#     with open(args.tickers) as f:
#         tickers = [
#             line.strip() for line in f if line.strip() and not line.startswith("#")
#         ]
#     logger.info("Loaded %d tickers", len(tickers))

#     # --- Load aligned data ---
#     # df_aligned = pd.read_parquet(args.input)
#     # logger.info("Loaded aligned data: %s", df_aligned.shape)

#     input_path = Path(args.input)
#     if not input_path.exists():
#         logger.warning(
#             "Input Parquet %s not found, merging individual tickers...", input_path
#         )
#         # Load SPY first
#         raw_dir = Path("data/processed")
#         spy_path = raw_dir / "SPY.parquet"
#         if not spy_path.exists():
#             raise FileNotFoundError(f"SPY file not found at {spy_path}")
#         df_spy = pd.read_parquet(spy_path)
#         logger.info(
#             "SPY: shape=%s, columns=%s, rows=%d",
#             df_spy.shape,
#             df_spy.columns.tolist(),
#             len(df_spy),
#         )

#         merged_dfs = {"SPY": df_spy}

#         # Load other tickers
#         for ticker in tickers:
#             if ticker == "SPY":
#                 continue
#             ticker_path = raw_dir / f"{ticker}.parquet"
#             if not ticker_path.exists():
#                 logger.warning(
#                     "Ticker %s not found at %s, skipping", ticker, ticker_path
#                 )
#                 continue
#             df_ticker = pd.read_parquet(ticker_path)
#             logger.info(
#                 "%s: shape=%s, columns=%s, rows=%d",
#                 ticker,
#                 df_ticker.shape,
#                 df_ticker.columns.tolist(),
#                 len(df_ticker),
#             )
#             merged_dfs[ticker] = df_ticker

#         # Merge into MultiIndex columns: ticker × fields
#         df_aligned = pd.concat(
#             merged_dfs, axis=1, names=["Ticker", "Field"]
#         ).sort_index(axis=1)
#         # Save for future runs
#         input_path.parent.mkdir(parents=True, exist_ok=True)
#         df_aligned.to_parquet(input_path)
#         logger.info("Merged aligned universe saved to %s", input_path)
#     else:
#         df_aligned = pd.read_parquet(input_path)
#         logger.info("Loaded aligned data: %s", df_aligned.shape)

#     # --- Split data ---
#     splitter = DataSplitter(train_end=cfg.data.train_end, val_end=cfg.data.val_end)
#     df_train_all, df_val_all, df_test_all = splitter.split(df_aligned)

#     # --- Extract per-ticker DataFrames efficiently ---
#     def extract_ticker_dfs(df_multi, tickers):
#         """Return a dict of {ticker: DataFrame} for all tickers in df_multi"""
#         return {
#             ticker: df_multi.xs(ticker, axis=1, level=0)
#             for ticker in tickers
#             if ticker in df_multi.columns.get_level_values(0)
#         }

#     # Extract per split
#     dfs_train = extract_ticker_dfs(df_train_all, tickers)
#     dfs_val = extract_ticker_dfs(df_val_all, tickers)
#     dfs_test = extract_ticker_dfs(df_test_all, tickers)

#     # --- Log details per ticker using pretty printer ---
#     for ticker, df in dfs_train.items():
#         pretty_print_df(df, name=f"{ticker} (train)", logger=logger)

#     for ticker, df in dfs_val.items():
#         pretty_print_df(df, name=f"{ticker} (val)", logger=logger)

#     for ticker, df in dfs_test.items():
#         pretty_print_df(df, name=f"{ticker} (test)", logger=logger)

#     logger.info("Extracted %d tickers for feature engineering", len(dfs_train))

#     # --- Clean memory ---
#     del df_aligned, df_train_all, df_val_all, df_test_all
#     gc.collect()

#     # --- Output directory ---
#     output_dir = Path(args.output)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # --- Parallel feature construction ---
#     logger.info("Starting parallel feature generation with %d jobs", args.n_jobs)
#     Parallel(n_jobs=args.n_jobs, backend="loky")(
#         delayed(process_ticker)(
#             ticker,
#             dfs_train,
#             dfs_val,
#             dfs_test,
#             cfg,
#             output_dir,
#             splitter,
#         )
#         for ticker in tickers
#     )

#     logger.info("=" * 60)
#     logger.info("✓ Feature engineering complete!")


# if __name__ == "__main__":
#     main()

# # #!/usr/bin/env python3
# # """
# # scripts/02_build_features.py

# # Builds feature matrices for each ticker using the aligned OHLCV data.
# # Loads ALL tickers to compute proper cross-ticker features (SPY correlation, etc.).
# # Applies feature selection (XGBoost importance) and saves to disk.

# # GPU acceleration: Automatically uses GPU for XGBoost if available.

# # Usage:
# #     python scripts/02_build_features.py --config config/default_config.yaml
# #     python scripts/02_build_features.py --config config/default_config.yaml --n-jobs 4
# # """

# # import argparse
# # import logging
# # import pickle
# # import sys
# # from pathlib import Path

# # import numpy as np
# # import pandas as pd

# # # Add project root to Python path
# # project_root = Path(__file__).parent.parent
# # sys.path.insert(0, str(project_root))

# # from src.features.pipeline import FeaturePipeline
# # from src.data.splitter import DataSplitter
# # from src.utils.logger import setup_logger
# # from src.utils.config_loader import load_config

# # logger = logging.getLogger(__name__)


# # def process_ticker(
# #     ticker: str,
# #     dfs_train,
# #     dfs_val,
# #     dfs_test,
# #     cfg,
# #     output_dir,
# #     splitter,
# # ):
# #     import numpy as np
# #     import pickle
# #     from pathlib import Path

# #     logger = logging.getLogger(__name__)

# #     if ticker not in dfs_train:
# #         logger.warning("Skipping %s (not in training data)", ticker)
# #         return

# #     pipeline = FeaturePipeline(
# #         target_ticker=ticker,
# #         universe_tickers=list(dfs_train.keys()),
# #         selector_kwargs={
# #             "importance_cumulative": cfg.features.selector.importance_threshold,
# #         },
# #     )

# #     # --- Fit ---
# #     X_train, y_train, feature_names = pipeline.fit_transform(dfs_train)

# #     # --- Transform ---
# #     X_val, y_val = pipeline.transform(dfs_val)
# #     X_test, y_test = pipeline.transform(dfs_test)

# #     # --- Normalize (optimized later in splitter) ---
# #     X_train = splitter.fit_transform(X_train, feature_names)
# #     X_val = splitter.transform(X_val)
# #     X_test = splitter.transform(X_test)

# #     # --- Save ---
# #     ticker_dir = Path(output_dir) / ticker
# #     ticker_dir.mkdir(parents=True, exist_ok=True)

# #     np.save(ticker_dir / "X_train.npy", X_train, allow_pickle=False)
# #     np.save(ticker_dir / "y_train.npy", y_train, allow_pickle=False)
# #     np.save(ticker_dir / "X_val.npy", X_val, allow_pickle=False)
# #     np.save(ticker_dir / "y_val.npy", y_val, allow_pickle=False)
# #     np.save(ticker_dir / "X_test.npy", X_test, allow_pickle=False)
# #     np.save(ticker_dir / "y_test.npy", y_test, allow_pickle=False)

# #     metadata = {
# #         "feature_names": feature_names,
# #         "n_features": len(feature_names),
# #         "train_samples": len(X_train),
# #         "val_samples": len(X_val),
# #         "test_samples": len(X_test),
# #     }

# #     with open(ticker_dir / "metadata.pkl", "wb") as f:
# #         pickle.dump(metadata, f)

# #     logger.info("✓ %s complete (%d features)", ticker, len(feature_names))


# # def main():
# #     parser = argparse.ArgumentParser(description="Build and select features")
# #     parser.add_argument(
# #         "--config",
# #         type=str,
# #         default="config/default_config.yaml",
# #         help="Path to config file",
# #     )
# #     parser.add_argument(
# #         "--input",
# #         type=str,
# #         default="data/processed/aligned_universe.parquet",
# #         help="Path to aligned universe parquet file",
# #     )
# #     parser.add_argument(
# #         "--output",
# #         type=str,
# #         default="data/features",
# #         help="Output directory for feature matrices",
# #     )
# #     parser.add_argument(
# #         "--tickers",
# #         type=str,
# #         default="config/tickers.txt",
# #         help="Path to ticker list file",
# #     )
# #     parser.add_argument(
# #         "--n-jobs",
# #         type=int,
# #         default=1,
# #         help="Number of parallel jobs (for future use)",
# #     )
# #     args = parser.parse_args()

# #     # Load config
# #     cfg = load_config(args.config)
# #     setup_logger(log_file="logs/02_build_features.log", level="INFO")

# #     # Check GPU availability
# #     try:
# #         import torch

# #         gpu_available = torch.cuda.is_available()
# #         if gpu_available:
# #             gpu_name = torch.cuda.get_device_name(0)
# #             gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
# #             logger.info("🚀 GPU detected: %s (%.1f GB)", gpu_name, gpu_memory)
# #             logger.info("XGBoost will automatically use GPU acceleration")
# #         else:
# #             logger.info("No GPU detected - using CPU only")
# #     except ImportError:
# #         gpu_available = False
# #         logger.info("PyTorch not available - using CPU only")

# #     # Read ticker list
# #     with open(args.tickers) as f:
# #         tickers = [
# #             line.strip()
# #             for line in f
# #             if line.strip() and not line.strip().startswith("#")
# #         ]
# #     logger.info("Loaded %d tickers", len(tickers))

# #     # Load aligned data
# #     logger.info("Loading aligned data from %s", args.input)
# #     df_aligned = pd.read_parquet(args.input)
# #     logger.info("Loaded data shape: %s", df_aligned.shape)

# #     logger.info(f"Columns in aligned data: {df_aligned.columns.tolist()}")

# #     # Split into train/val/test
# #     splitter = DataSplitter(
# #         train_end=cfg.data.train_end,
# #         val_end=cfg.data.val_end,
# #     )

# #     # Split the aligned data
# #     df_train_all, df_val_all, df_test_all = splitter.split(df_aligned)
# #     logger.info(
# #         "Split complete - Train: %d, Val: %d, Test: %d",
# #         len(df_train_all),
# #         len(df_val_all),
# #         len(df_test_all),
# #     )

# #     # Convert MultiIndex DataFrames to dict of single-ticker DataFrames
# #     def extract_ticker_dfs(df_multi):
# #         """Extract individual ticker DataFrames from MultiIndex DataFrame."""
# #         # available = set(df_multi.columns.get_level_values(0))
# #         # ticker_dfs = {}
# #         # for ticker in tickers:
# #         #     if ticker in available:
# #         #         ticker_dfs[ticker] = df_multi[ticker]
# #         # return ticker_dfs
# #         return {
# #             ticker: df_multi.xs(ticker, axis=1, level=0)
# #             for ticker in tickers
# #             if ticker in df_multi.columns.get_level_values(0)
# #         }

# #     dfs_train = extract_ticker_dfs(df_train_all)
# #     dfs_val = extract_ticker_dfs(df_val_all)
# #     dfs_test = extract_ticker_dfs(df_test_all)

# #     logger.info("Extracted %d tickers for feature engineering", len(dfs_train))

# #     logger.info(
# #         "Cleared DataFrames from memory:\n"
# #         "aligned=%d bytes (%.2f GB), train=%d bytes (%.2f GB), "
# #         "val=%d bytes (%.2f GB), test=%d bytes (%.2f GB)",
# #         df_aligned.memory_usage(deep=True).sum(),
# #         df_aligned.memory_usage(deep=True).sum() / 1e9,
# #         df_train_all.memory_usage(deep=True).sum(),
# #         df_train_all.memory_usage(deep=True).sum() / 1e9,
# #         df_val_all.memory_usage(deep=True).sum(),
# #         df_val_all.memory_usage(deep=True).sum() / 1e9,
# #         df_test_all.memory_usage(deep=True).sum(),
# #         df_test_all.memory_usage(deep=True).sum() / 1e9,
# #     )
# #     del df_aligned, df_train_all, df_val_all, df_test_all
# #     import gc

# #     gc.collect()

# #     # import sys

# #     # sys.exit(0)

# #     # Build features for each ticker
# #     output_dir = Path(args.output)
# #     output_dir.mkdir(parents=True, exist_ok=True)

# #     from joblib import Parallel, delayed

# #     logger.info("Starting parallel feature generation with %d jobs", args.n_jobs)

# #     Parallel(n_jobs=args.n_jobs, backend="loky")(
# #         delayed(process_ticker)(
# #             ticker,
# #             dfs_train,
# #             dfs_val,
# #             dfs_test,
# #             cfg,
# #             output_dir,
# #             DataSplitter(
# #                 train_end=cfg.data.train_end,
# #                 val_end=cfg.data.val_end,
# #             ),
# #         )
# #         for ticker in tickers
# #     )

# #     logger.info("=" * 60)
# #     logger.info("✓ Feature engineering complete!")


# # if __name__ == "__main__":
# #     main()

# # # #!/usr/bin/env python3
# # # """
# # # scripts/02_build_features.py

# # # Builds feature matrices for each ticker using the aligned OHLCV data.
# # # Loads ALL tickers to compute proper cross-ticker features (SPY correlation, etc.).
# # # Applies feature selection (XGBoost importance) and saves to disk.

# # # GPU acceleration: Automatically uses GPU for XGBoost if available.

# # # Usage:
# # #     python scripts/02_build_features.py --config config/default_config.yaml
# # #     python scripts/02_build_features.py --config config/default_config.yaml --n-jobs 4
# # # """

# # # import argparse
# # # import logging
# # # import pickle
# # # import sys
# # # from pathlib import Path

# # # import numpy as np
# # # import pandas as pd

# # # # Add project root to Python path
# # # project_root = Path(__file__).parent.parent
# # # sys.path.insert(0, str(project_root))

# # # from src.features.pipeline import FeaturePipeline
# # # from src.data.splitter import DataSplitter
# # # from src.utils.logger import setup_logger
# # # from src.utils.config_loader import load_config

# # # logger = logging.getLogger(__name__)


# # # def main():
# # #     parser = argparse.ArgumentParser(description="Build and select features")
# # #     parser.add_argument(
# # #         "--config",
# # #         type=str,
# # #         default="config/default_config.yaml",
# # #         help="Path to config file",
# # #     )
# # #     parser.add_argument(
# # #         "--input",
# # #         type=str,
# # #         default="data/processed/aligned_universe.parquet",
# # #         help="Path to aligned universe parquet file",
# # #     )
# # #     parser.add_argument(
# # #         "--output",
# # #         type=str,
# # #         default="data/features",
# # #         help="Output directory for feature matrices",
# # #     )
# # #     parser.add_argument(
# # #         "--tickers",
# # #         type=str,
# # #         default="config/tickers.txt",
# # #         help="Path to ticker list file",
# # #     )
# # #     parser.add_argument(
# # #         "--n-jobs",
# # #         type=int,
# # #         default=1,
# # #         help="Number of parallel jobs (for future use)",
# # #     )
# # #     args = parser.parse_args()

# # #     # Load config
# # #     cfg = load_config(args.config)
# # #     setup_logger(log_file="logs/02_build_features.log", level="INFO")

# # #     # Check GPU availability
# # #     try:
# # #         import torch

# # #         gpu_available = torch.cuda.is_available()
# # #         if gpu_available:
# # #             gpu_name = torch.cuda.get_device_name(0)
# # #             gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
# # #             logger.info("🚀 GPU detected: %s (%.1f GB)", gpu_name, gpu_memory)
# # #             logger.info("XGBoost will automatically use GPU acceleration")
# # #         else:
# # #             logger.info("No GPU detected - using CPU only")
# # #     except ImportError:
# # #         gpu_available = False
# # #         logger.info("PyTorch not available - using CPU only")

# # #     # Read ticker list
# # #     with open(args.tickers) as f:
# # #         tickers = [
# # #             line.strip()
# # #             for line in f
# # #             if line.strip() and not line.strip().startswith("#")
# # #         ]
# # #     logger.info("Loaded %d tickers", len(tickers))

# # #     # Load aligned data
# # #     logger.info("Loading aligned data from %s", args.input)
# # #     df_aligned = pd.read_parquet(args.input)
# # #     logger.info("Loaded data shape: %s", df_aligned.shape)

# # #     logger.info(f"Columns in aligned data: {df_aligned.columns.tolist()}")

# # #     # Split into train/val/test
# # #     splitter = DataSplitter(
# # #         train_end=cfg.data.train_end,
# # #         val_end=cfg.data.val_end,
# # #     )

# # #     # Split the aligned data
# # #     df_train_all, df_val_all, df_test_all = splitter.split(df_aligned)
# # #     logger.info(
# # #         "Split complete - Train: %d, Val: %d, Test: %d",
# # #         len(df_train_all),
# # #         len(df_val_all),
# # #         len(df_test_all),
# # #     )

# # #     # Convert MultiIndex DataFrames to dict of single-ticker DataFrames
# # #     def extract_ticker_dfs(df_multi):
# # #         """Extract individual ticker DataFrames from MultiIndex DataFrame."""
# # #         ticker_dfs = {}
# # #         for ticker in tickers:
# # #             if ticker not in df_multi.columns.get_level_values(0):
# # #                 continue
# # #             ticker_dfs[ticker] = df_multi[ticker].copy()
# # #         return ticker_dfs

# # #     dfs_train = extract_ticker_dfs(df_train_all)
# # #     dfs_val = extract_ticker_dfs(df_val_all)
# # #     dfs_test = extract_ticker_dfs(df_test_all)

# # #     logger.info(f"Columns of Train DataFrames: {list(dfs_train.keys())}")
# # #     logger.info(f"Columns of Val DataFrames: {list(dfs_val.keys())}")
# # #     logger.info(f"Columns of Test DataFrames: {list(dfs_test.keys())}")

# # #     # print the first 5 rows of each dataframe for the first ticker
# # #     for ticker in tickers[:1]:  # Just the first ticker for brevity
# # #         if ticker in dfs_train:
# # #             logger.info(
# # #                 f"First 5 rows of Train DataFrame for {ticker}:\n{dfs_train[ticker].head()}"
# # #             )
# # #         if ticker in dfs_val:
# # #             logger.info(
# # #                 f"First 5 rows of Val DataFrame for {ticker}:\n{dfs_val[ticker].head()}"
# # #             )
# # #         if ticker in dfs_test:
# # #             logger.info(
# # #                 f"First 5 rows of Test DataFrame for {ticker}:\n{dfs_test[ticker].head()}"
# # #             )

# # #     import sys

# # #     sys.exit(0)

# # #     logger.info("Extracted %d tickers for feature engineering", len(dfs_train))

# # #     # Build features for each ticker
# # #     output_dir = Path(args.output)
# # #     output_dir.mkdir(parents=True, exist_ok=True)

# # #     for idx, ticker in enumerate(tickers, 1):
# # #         if ticker not in dfs_train:
# # #             logger.warning("Skipping %s (not in training data)", ticker)
# # #             continue

# # #         logger.info("=" * 60)
# # #         logger.info("[%d/%d] Processing ticker: %s", idx, len(tickers), ticker)

# # #         # Initialize feature pipeline with full universe for cross-ticker features
# # #         pipeline = FeaturePipeline(
# # #             target_ticker=ticker,
# # #             universe_tickers=list(dfs_train.keys()),
# # #             selector_kwargs={
# # #                 "importance_cumulative": cfg.features.selector.importance_threshold,
# # #             },
# # #         )

# # #         # Fit on training data (pass ALL tickers for cross-ticker features)
# # #         logger.info("  Computing and selecting features...")
# # #         X_train, y_train, feature_names = pipeline.fit_transform(dfs_train)
# # #         logger.info("  Train features: %d samples × %d features", *X_train.shape)

# # #         # Transform val and test (pass ALL tickers for cross-ticker features)
# # #         X_val, y_val = pipeline.transform(dfs_val)
# # #         X_test, y_test = pipeline.transform(dfs_test)
# # #         logger.info("  Val: %d samples, Test: %d samples", len(X_val), len(X_test))

# # #         # Normalize features
# # #         logger.info("  Normalizing features...")
# # #         X_train_norm = splitter.fit_transform(X_train, feature_names)
# # #         X_val_norm = splitter.transform(X_val)
# # #         X_test_norm = splitter.transform(X_test)

# # #         # Save to disk
# # #         ticker_dir = output_dir / ticker
# # #         ticker_dir.mkdir(parents=True, exist_ok=True)

# # #         np.save(ticker_dir / "X_train.npy", X_train_norm)
# # #         np.save(ticker_dir / "y_train.npy", y_train)
# # #         np.save(ticker_dir / "X_val.npy", X_val_norm)
# # #         np.save(ticker_dir / "y_val.npy", y_val)
# # #         np.save(ticker_dir / "X_test.npy", X_test_norm)
# # #         np.save(ticker_dir / "y_test.npy", y_test)

# # #         # Save metadata
# # #         metadata = {
# # #             "feature_names": feature_names,
# # #             "n_features": len(feature_names),
# # #             "train_samples": len(X_train),
# # #             "val_samples": len(X_val),
# # #             "test_samples": len(X_test),
# # #         }
# # #         with open(ticker_dir / "metadata.pkl", "wb") as f:
# # #             pickle.dump(metadata, f)

# # #         logger.info("  ✓ Saved features to %s", ticker_dir)

# # #     logger.info("=" * 60)
# # #     logger.info("✓ Feature engineering complete!")


# # # if __name__ == "__main__":
# # #     main()
