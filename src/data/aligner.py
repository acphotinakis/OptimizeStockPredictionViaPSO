"""
src/data/aligner.py

Aligns 51 independently cleaned DataFrames to a single master timestamp
index derived from SPY (the most liquid, complete ticker).
"""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

from constants import MAX_FFILL_BARS, MAX_MISSING_FRACTION


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
            dfs: Mapping of ticker -> cleaned DataFrame with DatetimeIndex.
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
            logger.info(
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

        aligned_parts = []
        for ticker, df in dfs.items():

            df = df.copy()

            # normalize index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            elif df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            available = [f for f in fields if f in df.columns]

            df_sub = df[available].reindex(master_index)

            # Session-aware forward-filling: only fill within the same trading session
            if "session_start" in df_sub.columns:
                session_ids = df_sub["session_start"].fillna(False).astype(bool).cumsum()
                df_sub = df_sub.groupby(session_ids, group_keys=False).apply(
                    lambda g: g.ffill(limit=self.max_ffill_bars)
                )
            else:
                df_sub = df_sub.ffill(limit=self.max_ffill_bars)

            df_sub = df_sub.astype(np.float32, errors="ignore")

            df_sub.columns = pd.MultiIndex.from_product(
                [[ticker], df_sub.columns], names=["ticker", "field"]
            )

            aligned_parts.append(df_sub)

        result = pd.concat(aligned_parts, axis=1)

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
        Save each ticker's aligned DataFrame to compressed Parquet.

        Args:
            aligned_df: MultiIndex DataFrame from TickerAligner.align().
            output_dir: Directory to save per-ticker Parquet files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tickers = aligned_df.columns.get_level_values("ticker").unique()

        for ticker in tickers:
            ticker_df = aligned_df[ticker].copy()

            # Ensure index stays UTC (critical invariant)
            if ticker_df.index.tz is None:
                ticker_df.index = ticker_df.index.tz_localize("UTC")
            else:
                ticker_df.index = ticker_df.index.tz_convert("UTC")

            file_path = output_dir / f"{ticker}.parquet"

            ticker_df.to_parquet(
                file_path,
                engine="pyarrow",  # fastest + most compatible
                compression="zstd",  # best compression ratio + speed tradeoff
                index=True,
            )

            logger.info(
                "Saved %s -> %s (%d rows, %d cols)",
                ticker,
                file_path,
                ticker_df.shape[0],
                ticker_df.shape[1],
            )
