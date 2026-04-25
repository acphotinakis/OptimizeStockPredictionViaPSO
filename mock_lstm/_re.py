#!/usr/bin/env python3
"""
scripts/ingest_data.py

Ingest, clean, and align OHLCV data for the ticker universe.

Usage:
    python scripts/ingest_data.py --mode ingest --config config/default_config.yaml
    python scripts/ingest_data.py --mode clean
    python scripts/ingest_data.py --mode align
"""
from collections import defaultdict
import sys
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

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

from mock_lstm.mock_cleaner import MockDataCleaner
from src.data.aligner import TickerAligner
from src.data.alpaca_ingestor import AlpacaIngestor
from src.data.cleaner import DataCleaner
from src.utils.logger import LogFileMode, setup_logger
from src.utils.config_loader import Config, load_config
from constants import MAX_GAP_FILL_BARS
from mock_lstm.log_ import (
    _log_table_stats,
    flatten_cleaning_report,
    log_cleaning_report,
    log_all_cleaning_reports,
    log_all_timeframe_rows,
)

logger = logging.getLogger(__name__)


import pandas as pd
from pathlib import Path


def load_all_timeframes(raw_output_dir: Path, ticker: str, timeframes: list[str]):
    data = {}

    for tf in timeframes:
        path = raw_output_dir / tf / f"{ticker}.parquet"
        if not path.exists():
            logger.warning("Missing %s", path)
            continue

        df = _load_parquet(path)

        # ------------------------------------------------------------
        # 1. Ensure datetime index
        # ------------------------------------------------------------
        df.index = pd.to_datetime(df.index, utc=True)

        # ------------------------------------------------------------
        # 2. Convert UTC → America/New_York (DST-aware)
        # ------------------------------------------------------------
        # df.index = df.index.tz_convert("America/New_York")
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")

        data[tf] = df

    return data


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)

    return df


def ingest_all(
    ingestor: AlpacaIngestor,
    tickers: list[str],
    base_dir: Path,
    data_feed: str,
    timeframes: list[str],
    start_date: str,
    end_date: str,
):
    for tf in timeframes:
        for ticker in tickers:

            out_path = base_dir / tf / f"{ticker}.parquet"

            ingestor.download_universe(
                tickers=[ticker],
                output_dir=out_path,
                start=start_date,
                end=end_date,
                skip_existing=True,
                timeframe=tf,
                data_feed=data_feed,
            )


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:

    Path("logs").mkdir(exist_ok=True)
    setup_logger(log_file="logs/testing.log", level="INFO", mode=LogFileMode.APPEND)

    tickers = ["AAPL"]

    start_date = "2020-01-01"
    end_date = "2026-01-01"
    timeframes = ["1Min", "5Min", "15Min", "1Hour", "1Day"]

    ingestor = AlpacaIngestor()

    # ------------------------------------------------------------
    # STAGE 1: INGESTION
    # ------------------------------------------------------------
    for feed in ["sip", "iex"]:
        base_dir = Path(f"data/_raw_{feed}")

        ingest_all(
            ingestor=ingestor,
            tickers=tickers,
            base_dir=base_dir,
            data_feed=feed,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
        )

    # ------------------------------------------------------------
    # STAGE 2: ANALYSIS PIPELINE (MERGED FEEDS)
    # ------------------------------------------------------------

    def build_full_grid(
        start: pd.Timestamp, end: pd.Timestamp, timeframe: str
    ) -> pd.DatetimeIndex:
        freq = _parse_timeframe(timeframe)

        return pd.date_range(
            start=start,
            end=end,
            freq=freq,
            tz="America/New_York",
        )

    def align_to_grid(df: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
        # Ensure timezone consistency
        if df.index.tz is None:
            df = df.tz_localize("UTC").tz_convert("America/New_York")
        else:
            df = df.tz_convert("America/New_York")

        return df.reindex(grid)

    all_cleaning_rows = []
    all_timeframe_rows = []

    for ticker in tickers:
        logger.info("===== TICKER MERGE PIPELINE: %s =====", ticker)

        for tf in timeframes:
            logger.info("----- TIMEFRAME: %s -----", tf)

            sip_dir = Path("data/_raw_sip")

            sip_data = load_all_timeframes(sip_dir, ticker, [tf])

            if tf not in sip_data:
                logger.warning("No data for %s %s", ticker, tf)
                continue

            sip_df = sip_data.get(tf)

            start = min(sip_df.index.min())
            end = max(sip_df.index.max())

            grid = build_full_grid(start, end, tf)

            sip_aligned = align_to_grid(sip_df, grid) if sip_df is not None else None

            merged_df = sip_aligned

            # ------------------------------------------------------------
            # 5. CLEAN ONLY AFTER MERGE
            # ------------------------------------------------------------
            logger.info(f"merged-df rows: {len(merged_df)}")
            logger.info(f"merged-df first 10:\n{merged_df[:10]}")
            logger.info(f"merged-df cols: {merged_df.columns.tolist()}")
            cleaner = MockDataCleaner(
                max_gap_fill=5, ticker=f"{ticker}-{tf}", timeframe=tf
            )

            cleaned_df, cleaned_report = cleaner.clean(merged_df)
            logger.info(f"merged-df rows: {len(merged_df)}")
            logger.info(f"merged-df first 10:\n{merged_df[:10]}")
            logger.info(f"merged-df cols: {merged_df.columns.tolist()}")

            logger.info(f"merged-df rows: {len(merged_df)}")
            logger.info(f"merged-df cols: {merged_df.columns.tolist()}")

            all_cleaning_rows.append(
                flatten_cleaning_report(cleaned_report, ticker, "merged", tf)
            )

            all_timeframe_rows.append(
                {
                    "ticker": ticker,
                    "feed": "merged",
                    "timeframe": tf,
                    "df": cleaned_df,
                }
            )

    # ------------------------------------------------------------
    # LOG RESULTS
    # ------------------------------------------------------------
    log_all_timeframe_rows(all_timeframe_rows)
    log_all_cleaning_reports(all_cleaning_rows)


if __name__ == "__main__":
    main()
