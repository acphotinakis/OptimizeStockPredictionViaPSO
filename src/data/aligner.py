"""
src/data/aligner.py

Aligns 51 independently cleaned DataFrames to a single master timestamp
index derived from SPY (the most liquid, complete ticker).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

MAX_MISSING_FRACTION = 0.05  # Drop timestamps where >5 % of tickers have NaN
MAX_FFILL_BARS = 5  # Forward-fill at most 5 bars per ticker during alignment


class TickerAligner:
    """Aligns a universe of ticker DataFrames to a common timestamp index.

    Args:
        benchmark_ticker: Ticker to use as the master timestamp index (default "SPY").
        max_missing_fraction: Drop timestamps where more than this fraction of
            tickers have missing close prices.
        max_ffill_bars: Maximum number of bars to forward-fill per ticker.
    """

    def __init__(
        self,
        benchmark_ticker: str = "SPY",
        max_missing_fraction: float = MAX_MISSING_FRACTION,
        max_ffill_bars: int = MAX_FFILL_BARS,
    ) -> None:
        self.benchmark_ticker = benchmark_ticker
        self.max_missing_fraction = max_missing_fraction
        self.max_ffill_bars = max_ffill_bars

    def align(
        self,
        dfs: Dict[str, pd.DataFrame],
        fields: Optional[List] = None,
    ) -> pd.DataFrame:
        """Align all DataFrames to the master index.

        Args:
            dfs: Mapping of ticker --> cleaned DataFrame with DatetimeIndex.
            fields: Which columns to keep per ticker.
                Defaults to [open, high, low, close, volume, log_return, session_start].

        Returns:
            MultiIndex DataFrame: rows = timestamps, columns = (ticker, field).
        """
        if fields is None:
            fields = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "log_return",
                "session_start",
            ]

        if self.benchmark_ticker not in dfs:
            logger.warning(
                "Benchmark ticker %s not found; using first available ticker as master index.",
                self.benchmark_ticker,
            )
            master_ticker = next(iter(dfs))
        else:
            master_ticker = self.benchmark_ticker

        master_index = dfs[master_ticker].index
        logger.info(
            "Aligning %d tickers to master index from %s (%d timestamps)",
            len(dfs),
            master_ticker,
            len(master_index),
        )

        # aligned: Dict[str, pd.DataFrame] = {}
        # for ticker, df in dfs.items():
        #     # Keep only requested fields that exist
        #     available = [f for f in fields if f in df.columns]
        #     df_sub = (
        #         df[available].reindex(master_index).ffill(limit=self.max_ffill_bars)
        #     )
        #     aligned[ticker] = df_sub
        aligned_parts = []
        for ticker, df in dfs.items():
            # Ensure index is DatetimeIndex in UTC
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            elif df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            available = [f for f in fields if f in df.columns]
            df_sub = (
                df[available].reindex(master_index).ffill(limit=self.max_ffill_bars)
            )
            # Optionally convert to float32 to save memory
            df_sub = df_sub.astype(np.float32, errors="ignore")
            # df_sub.columns = pd.MultiIndex.from_product([[ticker], df_sub.columns])
            df_sub.columns = pd.MultiIndex.from_product(
                [[ticker], df_sub.columns], names=["ticker", "field"]
            )

            aligned_parts.append(df_sub)

        result = pd.concat(aligned_parts, axis=1)

        # Stack into MultiIndex column DataFrame
        # result = pd.concat(aligned, axis=1)
        # result.columns = pd.MultiIndex.from_tuples(
        #     [(ticker, col) for ticker in aligned for col in aligned[ticker].columns],
        #     names=["ticker", "field"],
        # )

        # Drop timestamps where too many tickers have NaN close
        if "close" in fields:
            close_cols = result.xs("close", axis=1, level="field")
            missing_frac = close_cols.isna().mean(axis=1)
            n_before = len(result)
            result = result[missing_frac <= self.max_missing_fraction]
            dropped = n_before - len(result)
            if dropped:
                logger.info(
                    "Alignment: dropped %d timestamps with >%.0f%% NaN tickers",
                    dropped,
                    self.max_missing_fraction * 100,
                )

        logger.info("Alignment complete: %d timestamps × %d columns", *result.shape)
        return result

    def save_aligned_tickers(
        self, aligned_df: pd.DataFrame, output_dir: str | Path
    ) -> None:
        """
        Save each ticker's aligned DataFrame to its own CSV file.

        Args:
            aligned_df: MultiIndex DataFrame from TickerAligner.align().
            output_dir: Directory to save per-ticker CSVs.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Loop over tickers in the first level of the MultiIndex columns
        for ticker in aligned_df.columns.get_level_values("ticker").unique():
            ticker_df = aligned_df[ticker].copy()
            file_path = output_dir / f"{ticker}_aligned.csv"
            ticker_df.to_csv(file_path, index=True)
            logger.info(f"Saved {ticker} --> {file_path}")
