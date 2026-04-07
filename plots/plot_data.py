#!/usr/bin/env python3
"""
Enhanced Matplotlib line charts for OHLCV data:
- Raw, Cleaned, Processed, Overlay
- High/Low shading, SMA20, grid, and annotations
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def load_parquet(path: Path, resample_to: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if resample_to is not None:
        df = (
            df.resample(resample_to)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
    return df


def downsample(df: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
    n = len(df)
    if n <= max_points:
        return df.copy()
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    return df.iloc[idx]


def plot_ticker_lines_enhanced(
    ticker: str, raw_dir: Path, cleaned_dir: Path, processed_dir: Path
):
    # Load data
    dfs = {}
    for label, path in [
        ("Raw", raw_dir / f"{ticker}.parquet"),
        ("Cleaned", cleaned_dir / f"{ticker}.parquet"),
        ("Processed", processed_dir / f"{ticker}.parquet"),
    ]:
        if path.exists():
            dfs[label] = load_parquet(path)

    if not dfs:
        raise FileNotFoundError(f"No data found for ticker {ticker}")

    # Downsample
    dfs_down = {label: downsample(df) for label, df in dfs.items()}

    # Colors
    colors = {"Raw": "#1f77b4", "Cleaned": "#ff7f0e", "Processed": "#2ca02c"}

    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    subplot_titles = ["Raw", "Cleaned", "Processed", "Overlay"]

    for i, label in enumerate(["Raw", "Cleaned", "Processed"]):
        if label in dfs_down:
            df = dfs_down[label]
            axes[i].plot(
                df.index,
                df["close"],
                label=f"{label} Close",
                color=colors[label],
            )
            # High-Low shading
            axes[i].fill_between(
                df.index, df["low"], df["high"], color=colors[label], alpha=0.1
            )
            # SMA20
            if len(df) >= 20:
                df["SMA20"] = df["close"].rolling(20).mean()
                axes[i].plot(
                    df.index, df["SMA20"], linestyle="--", color="black", label="SMA20"
                )
            # Annotate last close
            last = df["close"].iloc[-1]
            axes[i].annotate(
                f"{last:.2f}",
                xy=(df.index[-1], last),
                xytext=(5, 0),
                textcoords="offset points",
            )
            axes[i].set_title(subplot_titles[i], fontsize=12)
            axes[i].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            axes[i].legend(fontsize=10)
            axes[i].set_facecolor("#f9f9f9")

    # Overlay plot
    for label, df in dfs_down.items():
        axes[3].plot(
            df.index,
            df["close"],
            label=f"{label} Close",
            color=colors[label],
            linewidth=1.5,
        )
    axes[3].set_title("Overlay", fontsize=12)
    axes[3].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    axes[3].legend(fontsize=10)
    axes[3].set_facecolor("#f9f9f9")

    # X-axis formatting
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45)

    # Layout and save
    plt.suptitle(f"{ticker} OHLCV Enhanced Line Charts", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save_path = Path("data/pipelines/ingestion") / f"{ticker}_lines_enhanced.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot enhanced OHLCV line charts for a ticker"
    )
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--cleaned-dir", type=str, default="data/cleaned")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    args = parser.parse_args()

    plot_ticker_lines_enhanced(
        ticker=args.ticker,
        raw_dir=Path(args.raw_dir),
        cleaned_dir=Path(args.cleaned_dir),
        processed_dir=Path(args.processed_dir),
    )


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# scripts/plot_ticker_lines.py

# Plot raw, cleaned, and processed OHLCV data for a single ticker using line charts.
# 4 subplots: Raw, Cleaned, Processed, Overlay.
# """

# import argparse
# from pathlib import Path

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np


# def load_parquet(path: Path, resample_to: str | None = None) -> pd.DataFrame:
#     df = pd.read_parquet(path)

#     # Ensure datetime index
#     if not isinstance(df.index, pd.DatetimeIndex):
#         df.index = pd.to_datetime(df.index)

#     if resample_to is not None:
#         df = (
#             df.resample(resample_to)
#             .agg(
#                 {
#                     "open": "first",
#                     "high": "max",
#                     "low": "min",
#                     "close": "last",
#                     "volume": "sum",
#                 }
#             )
#             .dropna()
#         )

#     return df


# def downsample(df: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
#     n = len(df)
#     if n <= max_points:
#         return df.copy()
#     idx = np.linspace(0, n - 1, num=max_points, dtype=int)
#     return df.iloc[idx]


# def plot_ticker_lines(
#     ticker: str, raw_dir: Path, cleaned_dir: Path, processed_dir: Path
# ):
#     # Load data
#     dfs = {}
#     for label, path in [
#         ("Raw", raw_dir / f"{ticker}.parquet"),
#         ("Cleaned", cleaned_dir / f"{ticker}.parquet"),
#         ("Processed", processed_dir / f"{ticker}.parquet"),
#     ]:
#         if path.exists():
#             dfs[label] = load_parquet(path)

#     if not dfs:
#         raise FileNotFoundError(f"No data found for ticker {ticker}")

#     # Downsample for plotting
#     dfs_down = {label: downsample(df) for label, df in dfs.items()}
#     # dfs_down = dfs

#     # Create 4 subplots (shared x-axis)
#     fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
#     subplot_titles = ["Raw", "Cleaned", "Processed", "Overlay"]

#     # Plot individual datasets
#     for i, label in enumerate(["Raw", "Cleaned", "Processed"]):
#         if label in dfs_down:
#             df = dfs_down[label]
#             axes[i].plot(df.index, df["close"], label=f"{label} Close", color="blue")
#             axes[i].set_title(subplot_titles[i])
#             axes[i].grid(True)
#             axes[i].legend()

#     # Overlay plot
#     for label, df in dfs_down.items():
#         axes[3].plot(df.index, df["close"], label=f"{label} Close")
#     axes[3].set_title("Overlay")
#     axes[3].grid(True)
#     axes[3].legend()

#     # Final layout adjustments
#     plt.suptitle(f"{ticker} OHLCV Line Charts", fontsize=16)
#     plt.tight_layout(rect=(0, 0, 1, 0.96))

#     # Save figure
#     save_path = Path("data/pipelines/ingestion") / f"{ticker}_lines.png"
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(save_path, dpi=150)
#     plt.show()


# def main():
#     parser = argparse.ArgumentParser(description="Plot OHLCV line charts for a ticker")
#     parser.add_argument("--ticker", type=str, required=True)
#     parser.add_argument("--raw-dir", type=str, default="data/raw")
#     parser.add_argument("--cleaned-dir", type=str, default="data/cleaned")
#     parser.add_argument("--processed-dir", type=str, default="data/processed")
#     args = parser.parse_args()

#     plot_ticker_lines(
#         ticker=args.ticker,
#         raw_dir=Path(args.raw_dir),
#         cleaned_dir=Path(args.cleaned_dir),
#         processed_dir=Path(args.processed_dir),
#     )


# if __name__ == "__main__":
#     main()
