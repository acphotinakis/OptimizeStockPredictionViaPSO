#!/usr/bin/env python3
from itertools import combinations
import sys
import argparse
import logging
import sys
from pathlib import Path
import pandas_market_calendars as mcal

import numpy as np
import pandas as pd

from pathlib import Path
from typing import cast

from prettytable import PrettyTable

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

from src.data.debug_logs import (
    flatten_cleaning_report,
    log_all_cleaning_reports,
    log_cleaning_report,
)
from src.data.utils import _load_tickers, _parse_timeframe, _save_parquet
from src.utils.data_storage import _load_parquet
from src.data.cleaner import DataCleaner
from src.data.alpaca_ingestor import AlpacaIngestor
from src.utils.logger import LogFileMode, setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


import pandas as pd
from pathlib import Path
from pathlib import Path
import pandas as pd


import matplotlib

matplotlib.use("Agg")  # important for multiprocessing safety
import matplotlib.pyplot as plt


def _plot_ohlc(ax, df, title):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.3, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_alpha(0.2)

    x = np.arange(len(df))
    width = 0.35

    # vertical line: low to high
    ax.vlines(x, df["low"], df["high"], color="black", linewidth=0.8, alpha=0.7)
    # open tick (left)
    ax.hlines(df["open"], x - width, x, color="green", linewidth=1.2, alpha=0.8)
    # close tick (right)
    ax.hlines(df["close"], x, x + width, color="red", linewidth=1.2, alpha=0.8)


def plot_pipeline_stage(raw_df, aligned_df, cleaned_df, output_dir, ticker, tf):
    plt.style.use("seaborn-v0_8-darkgrid")

    fig, axes = plt.subplots(
        3, 2, figsize=(18, 12), sharex=True, gridspec_kw={"width_ratios": [4, 1]}
    )

    datasets = [
        (raw_df, "RAW"),
        (aligned_df, "ALIGNED"),
        (cleaned_df, "CLEANED"),
    ]

    for i, (df, name) in enumerate(datasets):
        price_ax = axes[i, 0]
        vol_ax = axes[i, 1]

        _plot_ohlc(price_ax, df, name)

        # Volume
        vol_ax.bar(range(len(df)), df["volume"], color="#888888", alpha=0.6)
        vol_ax.set_title("Volume", fontsize=10)
        vol_ax.grid(True, alpha=0.2)
        for spine in vol_ax.spines.values():
            spine.set_alpha(0.2)

        vol_ax.set_yticks([])

    fig.suptitle(f"{ticker} · {tf}", fontsize=16, fontweight="bold", y=0.98)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time Index", fontsize=11)

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = output_dir / f"{ticker}_{tf}_pipeline.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def ingest_all(
    ingestor: AlpacaIngestor,
    tickers: list[str],
    base_dir: Path,
    data_feed: str,
    tf: str,
    start_date: str,
    end_date: str,
    skip_existing: bool = True,
):
    ingestor.download_universe(
        tickers=tickers,
        output_dir=base_dir,
        start=start_date,
        end=end_date,
        skip_existing=skip_existing,
        timeframe=tf,
        data_feed=data_feed,
    )


UTC = "UTC"

FREQ_MAP = {
    "1Min": "1min",
    "5Min": "5min",
    "15Min": "15min",
    "1Hour": "1h",
    "1H": "1h",
    "1Day": "1D",
    "1D": "1D",
}


def normalize_freq(tf: str) -> str:
    try:
        return FREQ_MAP[tf]
    except KeyError:
        raise ValueError(f"Unsupported timeframe: {tf}")


def to_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest, clean, and align OHLCV data")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--tickers", default="config/tickers.txt")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--aligned-dir", default="data/aligned")
    parser.add_argument("--cleaned-dir", default="data/cleaned")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    setup_logger(log_file="logs/testing.log", level="INFO", mode=LogFileMode.APPEND)

    cfg = load_config(args.config)
    tickers = _load_tickers(args.tickers)
    logger.info("Loaded %d tickers", len(tickers))

    start_date = "2020-01-01"
    # start_date = "2026-01-01"
    end_date = "2026-01-28"
    timeframes = ["1Min", "5Min", "15Min", "1Hour", "1Day"]
    # timeframes = ["1Day"]
    # timeframes = ["1Min"]

    ingestor = AlpacaIngestor()

    # ------------------------------------------------------------
    # STAGE 1: INGESTION (ALL TIMEFRAMES)
    # ------------------------------------------------------------

    raw_output = Path(args.raw_dir)
    raw_output.mkdir(parents=True, exist_ok=True)

    for tf in timeframes:
        tf_raw_dir_output = raw_output / tf
        tf_raw_dir_output.mkdir(parents=True, exist_ok=True)
        ingest_all(
            ingestor=ingestor,
            tickers=tickers,
            base_dir=tf_raw_dir_output,
            data_feed="sip",
            tf=tf,
            start_date=start_date,
            end_date=end_date,
        )

    # ------------------------------------------------------------
    # STAGE 2: ALIGN + CLEAN PIPELINE
    # ------------------------------------------------------------
    # data/raw
    raw_dir = Path(args.raw_dir)
    # data/aligned
    aligned_dir = Path("data/aligned")
    # data/processed
    processed_dir = Path(args.processed_dir)
    # data/cleaned
    cleaned_dir = Path(args.cleaned_dir)
    # data/plots
    plots_dir = Path("data/plots")

    def ensure_directories_exist() -> None:
        """
        Ensures all required pipeline directories exist.
        No deletion or modification of existing data is permitted.
        """

        required_dirs = [raw_dir, aligned_dir, cleaned_dir, processed_dir, plots_dir]

        dirs = []
        for n in required_dirs:
            for tf in timeframes:
                path = n / tf
                dirs.append(path)

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    ensure_directories_exist()

    def get_path(base_dir: Path, timeframe: str, ticker: str) -> Path:
        return base_dir / timeframe / f"{ticker}.parquet"

    plots_dir = Path("data/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_cleaning_rows = []

    # tickers = ["AAPL", "SPY"]
    for ticker in tickers:
        logger.info("===== PIPELINE START: %s =====", ticker)

        for tf in timeframes:
            logger.info("----- TIMEFRAME: %s -----", tf)

            # --------------------------------------------------------
            # LOAD RAW
            # --------------------------------------------------------
            plot_dir_ = plots_dir / "plots"
            plot_dir_.mkdir(parents=True, exist_ok=True)
            # raw_path = raw_output / tf / f"{ticker}.parquet"
            raw_path = get_path(raw_dir, tf, ticker)
            # raw_path = raw_output / f"{ticker}.parquet"
            if not raw_path.exists():
                logger.warning("Missing raw data: %s", raw_path)
                continue

            raw_df = _load_parquet(raw_path)

            # ensure OHLCV safety
            raw_df = raw_df[["open", "high", "low", "close", "volume"]]
            raw_df.index = pd.to_datetime(raw_df.index)

            raw_df = raw_df.sort_index()

            # --------------------------------------------------------
            # ALIGN
            # --------------------------------------------------------
            start, end = raw_df.index.min(), raw_df.index.max()
            logger.info(f"Start --> {start} || End --> {end}")

            aligned_df = raw_df

            # sys.exit(0)
            # --------------------------------------------------------
            # CLEAN
            # --------------------------------------------------------
            cleaner = DataCleaner(max_gap_fill=5, ticker=f"{ticker}-{tf}", timeframe=tf)

            cleaned_df, cleaned_report = cleaner.clean(aligned_df)
            log_cleaning_report(cleaned_report)
            # store for cross-timeframe analysis
            all_cleaning_rows.append(
                flatten_cleaning_report(
                    reports=cleaned_report,
                    ticker=ticker,
                    feed="sip",
                    tf=tf,
                )
            )
            # --------------------------------------------------------
            # SAVE OUTPUTS (NO OVERWRITES)
            # --------------------------------------------------------
            _save_parquet(aligned_df, get_path(aligned_dir, tf, ticker))
            _save_parquet(cleaned_df, get_path(cleaned_dir, tf, ticker))
            _save_parquet(cleaned_df, get_path(processed_dir, tf, ticker))

            # if tf != "1Min" and tf != "5Min":
            #     plot_pipeline_stage(
            #         raw_df, aligned_df, cleaned_df, plot_dir_, ticker, tf
            #     )

            # sys.exit(0)
        log_all_cleaning_reports(all_cleaning_rows)
        all_cleaning_rows.clear()


if __name__ == "__main__":
    main()
