#!/usr/bin/env python3
"""
scripts/ingest_data.py

Ingest, clean, and align OHLCV data for the ticker universe.

Usage:
    python scripts/ingest_data.py --mode ingest --config config/default_config.yaml
    python scripts/ingest_data.py --mode clean
    python scripts/ingest_data.py --mode align
"""
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
    log_cleaning_report,
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


def flatten_cleaning_report(reports: Dict[str, Any], ticker: str, feed: str, tf: str):
    row = {}

    def add(k, v):
        row[k] = v

    # metadata (IMPORTANT for cross-timeframe table)
    add("ticker", ticker)
    add("feed", feed)
    add("timeframe", tf)

    # ------------------------------------------------------------
    # OHLCV REPORT
    # ------------------------------------------------------------
    r = reports.get("ohlcv_report", {})
    add("initial_rows", r.get("initial_rows"))
    add("final_rows", r.get("final_rows"))
    add("dropped_rows", r.get("dropped_count"))
    add("valid_rows", r.get("valid_count"))
    add("invalid_rows", r.get("invalid_count"))
    add("drop_rate", f"{r.get('drop_rate', 0):.2%}")

    # ------------------------------------------------------------
    # GAP REPORT
    # ------------------------------------------------------------
    g = reports.get("gap_info", {})
    add("expected_interval", g.get("expected_interval"))
    add("total_gaps", g.get("total_gaps"))
    add("max_gap_size", g.get("max_gap"))
    add("min_gap_size", g.get("min_gap"))

    # ------------------------------------------------------------
    # BOUNDED FORWARD FILL REPORT
    # ------------------------------------------------------------
    bff = reports.get("_bounded_forward_fill_report", {})
    add("forward_fill_total_filled", bff.get("total_filled"))
    add("forward_fill_total_skipped", bff.get("total_skipped"))
    add("forward_fill_rows_before", bff.get("rows_before"))
    add("forward_fill_rows_after", bff.get("rows_after"))
    add("forward_fill_nan_before", bff.get("nan_before"))
    add("forward_fill_nan_after", bff.get("nan_after"))

    bff_cols = bff.get("columns", {})

    for col, metrics in bff_cols.items():
        add(f"ff_{col}_filled", metrics.get("filled"))
        add(f"ff_{col}_skipped", metrics.get("skipped"))
        add(f"ff_{col}_fill_ratio", metrics.get("fill_ratio"))

    add("ff_close_filled", bff_cols.get("close", {}).get("filled"))
    add("ff_close_skipped", bff_cols.get("close", {}).get("skipped"))
    add("ff_close_fill_ratio", bff_cols.get("close", {}).get("fill_ratio"))

    # ------------------------------------------------------------
    # LONG GAP REMOVAL
    # ------------------------------------------------------------
    lg = reports.get("_remove_long_gaps_report", {})
    add("long_gap_rows_before", lg.get("rows_before"))
    add("long_gap_rows_after", lg.get("rows_after"))
    add("long_gap_removed", lg.get("rows_removed"))
    add("long_gap_removed_fraction", lg.get("removed_fraction"))
    add("max_gap_threshold", lg.get("max_gap_fill"))

    # ------------------------------------------------------------
    # FINAL VALIDATION
    # ------------------------------------------------------------
    fv = reports.get("_final_validation_report", {})
    add("final_rows", fv.get("rows"))
    add("nan_count", fv.get("nan_count"))
    add("close_invalid", fv.get("close_invalid"))
    add("volume_invalid", fv.get("volume_invalid"))
    add("is_valid", fv.get("is_valid"))

    return row


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


def log_all_timeframe_rows(rows: list[dict]):
    from prettytable import PrettyTable

    df = pd.DataFrame(rows).copy()

    df["rows"] = df["df"].apply(len)
    df["start"] = df["df"].apply(lambda x: x.index.min())
    df["end"] = df["df"].apply(lambda x: x.index.max())
    df["mean_close"] = df["df"].apply(lambda x: x["close"].mean())
    df["volatility"] = df["df"].apply(lambda x: x["close"].pct_change().std())
    df["missing_rate"] = df["df"].apply(lambda x: x.isna().mean().mean())

    df = df.drop(columns=["df"]).sort_values(["ticker", "feed", "timeframe"])

    table = PrettyTable()
    table.field_names = list(df.columns)

    for _, r in df.iterrows():
        table.add_row([r[c] for c in df.columns])

    logger.info("\n========== TIMEFRAME SUMMARY ==========\n%s", table)


# def log_all_cleaning_reports(rows: list[dict]):
#     from prettytable import PrettyTable

#     df = pd.DataFrame(rows)

#     df = df.sort_values(["ticker", "feed", "timeframe"])

#     table = PrettyTable()
#     table.field_names = list(df.columns)

#     for _, r in df.iterrows():
#         table.add_row([r[c] for c in df.columns])

#     logger.info("\n========== CLEANING REPORT (ALL TIMEFRAMES) ==========\n%s", table)


def log_all_cleaning_reports(rows: list[dict]):
    import pandas as pd
    from prettytable import PrettyTable

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # 1. Enforce correct timeframe ordering
    # ------------------------------------------------------------
    timeframe_order = ["1Min", "5Min", "15Min", "1Hour", "1Day"]

    df["timeframe"] = pd.Categorical(
        df["timeframe"], categories=timeframe_order, ordered=True
    )

    # Sort: ticker → timeframe → feed
    df = df.sort_values(["ticker", "timeframe", "feed"])

    # ------------------------------------------------------------
    # 2. Create column key (ticker | timeframe | feed)
    # ------------------------------------------------------------
    df["key"] = (
        df["ticker"].astype(str)
        + " | "
        + df["timeframe"].astype(str)
        + " | "
        + df["feed"].astype(str)
    )

    # ------------------------------------------------------------
    # 3. Select value columns
    # ------------------------------------------------------------
    value_cols = [
        c for c in df.columns if c not in ["ticker", "feed", "timeframe", "key"]
    ]

    # ------------------------------------------------------------
    # 4. Pivot (metrics as rows)
    # ------------------------------------------------------------
    pivot_df = df.set_index("key")[value_cols].T

    # ------------------------------------------------------------
    # 5. Enforce correct column order AFTER pivot
    # ------------------------------------------------------------
    def sort_key(col: str):
        # "AAPL | 1Min | merged"
        ticker, tf, feed = [x.strip() for x in col.split("|")]
        return (
            ticker,
            timeframe_order.index(tf) if tf in timeframe_order else 999,
            feed,
        )

    sorted_cols = sorted(pivot_df.columns, key=sort_key)
    pivot_df = pivot_df[sorted_cols]

    # ------------------------------------------------------------
    # 6. Build PrettyTable
    # ------------------------------------------------------------
    table = PrettyTable()
    table.field_names = ["metric"] + list(pivot_df.columns)

    for metric, row in pivot_df.iterrows():
        table.add_row([metric] + [row[c] for c in pivot_df.columns])

    logger.info("\n========== CLEANING REPORT (PIVOTED) ==========\n%s", table)


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


def _reindex_to_full_grid(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    freq = _parse_timeframe(timeframe)
    logger.info(f"First 10 rows BEFORE reindex")
    logger.info(df[:10])
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq,
        # tz=df.index.tz,
        tz="America/New_York",
    )

    df = df.reindex(full_index)

    logger.info(f"First 10 rows AFTER reindex")
    logger.info(f"\ndf[:10]")
    return df


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

    from collections import defaultdict

    def build_union_index(dfs: list[pd.DataFrame]) -> pd.DatetimeIndex:
        idx = dfs[0].index
        for df in dfs[1:]:
            idx = idx.union(df.index)
        return idx.sort_values()

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

    def align_to_index(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        return df.reindex(index)

    def align_to_grid(df: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
        # Ensure timezone consistency
        if df.index.tz is None:
            df = df.tz_localize("UTC").tz_convert("America/New_York")
        else:
            df = df.tz_convert("America/New_York")

        return df.reindex(grid)

    def merge_feeds(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
        """
        SIP is primary, IEX fallback (adjust if you want opposite priority)
        """
        merged = primary.copy()

        for col in ["open", "high", "low", "close", "volume"]:
            merged[col] = primary[col].combine_first(secondary[col])

        return merged

    all_cleaning_rows = []
    all_timeframe_rows = []

    for ticker in tickers:
        logger.info("===== TICKER MERGE PIPELINE: %s =====", ticker)

        for tf in timeframes:
            logger.info("----- TIMEFRAME: %s -----", tf)

            sip_dir = Path("data/_raw_sip")
            iex_dir = Path("data/_raw_iex")

            sip_data = load_all_timeframes(sip_dir, ticker, [tf])
            iex_data = load_all_timeframes(iex_dir, ticker, [tf])

            if tf not in sip_data and tf not in iex_data:
                logger.warning("No data for %s %s", ticker, tf)
                continue

            sip_df = sip_data.get(tf)
            iex_df = iex_data.get(tf)

            # ------------------------------------------------------------
            # CASE HANDLING
            # ------------------------------------------------------------
            if sip_df is None:
                merged_df = iex_df.copy()
                # merged_df["source"] = "iex"

            elif iex_df is None:
                merged_df = sip_df.copy()
                # merged_df["source"] = "sip"

            else:
                # --------------------------------------------------------
                # 1. Build canonical union index
                # --------------------------------------------------------
                # union_index = build_union_index([sip_df, iex_df])

                # # --------------------------------------------------------
                # # 2. Align both feeds
                # # --------------------------------------------------------
                # sip_aligned = align_to_index(sip_df, union_index)
                # iex_aligned = align_to_index(iex_df, union_index)
                start = min(df.index.min() for df in [sip_df, iex_df] if df is not None)
                end = max(df.index.max() for df in [sip_df, iex_df] if df is not None)

                grid = build_full_grid(start, end, tf)

                sip_aligned = (
                    align_to_grid(sip_df, grid) if sip_df is not None else None
                )
                iex_aligned = (
                    align_to_grid(iex_df, grid) if iex_df is not None else None
                )

                # --------------------------------------------------------
                # 3. Merge with SIP priority + IEX fallback
                # --------------------------------------------------------
                # merged_df = merge_feeds(sip_aligned, iex_aligned)
                merged_df = sip_aligned

                # --------------------------------------------------------
                # 4. Add provenance tracking
                # --------------------------------------------------------
                # merged_df["source"] = build_provenance(sip_aligned, iex_aligned)

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

            # if len(iex_filled) > 0:
            #     logger.info("===== SAMPLE ROWS FILLED FROM IEX (UP TO 10) =====")

            #     for idx, row in iex_filled.iterrows():
            #         logger.info("INDEX=%s | ROW=%s", idx, row.to_dict())
            # else:
            #     logger.info("No rows filled from IEX found.")

            # sys.exit(0)

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
