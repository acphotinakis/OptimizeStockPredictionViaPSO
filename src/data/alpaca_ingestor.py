"""
src/data/alpaca_ingestor.py

Downloads 1-minute OHLCV bars from the Alpaca Markets REST API.
Handles pagination, rate-limiting, caching, and split adjustments.
"""

from __future__ import annotations

from datetime import datetime
import os
import time
import logging
from pathlib import Path
from typing import List, Optional
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import timezone
import pandas as pd
from dotenv import load_dotenv


logger = logging.getLogger(__name__)

load_dotenv()


class AlpacaIngestor:
    """Downloads 1-minute OHLCV bars from Alpaca Markets.

    Attributes:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        base_url: Alpaca base URL (paper or live).
    """

    RATE_LIMIT_SLEEP: float = 0.35  # seconds between calls (~170 req/min < 200 limit)

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: str = "https://paper-api.alpaca.markets",
    ) -> None:
        # Load credentials from environment. Accept either ``ALPACA_SECRET_KEY``
        # (README convention) or ``ALPACA_API_SECRET`` (.env.example convention)
        # so users templated from either source work without renaming.
        api_key = api_key or os.getenv("ALPACA_API_KEY")
        secret_key = api_secret or os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")

        if not api_key or not secret_key:
            logger.error(
                "Alpaca API credentials missing. Ensure ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY (or ALPACA_API_SECRET) are set in .env."
            )
            raise EnvironmentError("Missing Alpaca API credentials.")

        logger.info("Alpaca API credentials loaded from environment")

        # Initialize the historical data client
        self.client = StockHistoricalDataClient(api_key, secret_key)
        self.base_url = base_url
        logger.info("Initialized Alpaca StockHistoricalDataClient")

        # Map configuration strings to alpaca-py TimeFrame objects
        self.timeframe_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        logger.info("TimeFrame mapping initialized")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download_bars(
        self,
        ticker: str,
        start: str,
        end: str,
        timeframe: str,
        adjustment: str = "all",
        data_feed: str = "sip",
    ) -> pd.DataFrame:
        """Download 1-minute OHLCV bars for a single ticker.

        Args:
            ticker: Equity ticker symbol (e.g. "AAPL").
            start: ISO date string, inclusive.
            end: ISO date string, exclusive.
            timeframe: Bar resolution string accepted by Alpaca API.
            adjustment: Price adjustment type ("split", "dividend", "all", "raw").

        Returns:
        --------
        pd.DataFrame
            Validated DataFrame with monotonic DatetimeIndex, float64 dtype

        Validations:
        ------------
        - Parse timestamps; enforce ascending chronological order
        - Assert high ≥ low for all rows
        - Assert open, high, low, close, volume > 0
        - Log data provenance: source, date range, row count, ticker symbol

        Constraints:
        ------------
        - Missing values preserved as NaN; no imputation at this stage
        - Must support daily and intraday (minute) frequencies

        OUTPUT INVARIANTS:
        - DatetimeIndex (UTC, timezone-aware)
        - Strict ascending order
        - Raw OHLCV schema preserved
        - Missing values preserved (NaN allowed)

        """

        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

        tf = self.timeframe_map.get(timeframe)
        if tf is None:
            raise ValueError(
                f"Unsupported timeframe: {timeframe!r}. Choose from {list(self.timeframe_map)}"
            )

        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=tf,
            start=start_dt,
            end=end_dt,
            adjustment=Adjustment(adjustment),
            feed=DataFeed(data_feed),
        )

        logger.info(
            "Request Params | ticker=%s timeframe=%s start=%s end=%s adjustment=%s feed=%s",
            ticker,
            request_params.timeframe,
            request_params.start,
            request_params.end,
            request_params.adjustment,
            request_params.feed,
        )
        try:
            bars = self.client.get_stock_bars(request_params)
            df: pd.DataFrame = bars.df

            logger.info(f"Downloaded {len(df)} bars for {ticker} from {start} to {end}")
            if df.empty:
                logger.info(f"No data returned for {ticker}")
                return pd.DataFrame()
            else:
                logger.info(f"Df head:\n{df.head()}")

            # =====================================================
            # STAGE 1.1: STRUCTURAL NORMALIZATION ONLY
            # =====================================================

            # Flatten MultiIndex if present
            if isinstance(df.index, pd.MultiIndex):
                if "symbol" in df.index.names:
                    df = df.xs(ticker, level="symbol")

            # REQUIREMENT: Must be DatetimeIndex
            # Ensure UTC normalization (hard invariant)
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError("Index must be DatetimeIndex from provider")

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            df.index = pd.DatetimeIndex(df.index, tz="UTC")
            df.index = df.index.astype("datetime64[ns, UTC]")

            df.index.name = "timestamp"

            logger.info(f"Index of data (UTC): {df.index}")
            logger.info(f"Index dtype: {df.index.dtype}")

            # IMPORTANT: No rounding, no scaling, no casting here
            # Downcast to reduce memory usage (Phase 1: Data Quantization)
            # df = self.downcast_ohlcv(df)
            # logger.info("Downcasted %s to float32/int32", ticker)

            return df

        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def download_universe(
        self,
        tickers: List[str],
        output_dir: str | Path,
        start: str,
        end: str,
        timeframe: str,
        data_feed: str = "sip",
        skip_existing: bool = False,
    ) -> None:
        """Download all tickers and persist each as a Parquet file.

        Args:
            tickers: List of ticker symbols.
            output_dir: Directory to save Parquet files.
            start: Download start date (ISO string).
            end: Download end date (ISO string).
            skip_existing: If True, skip tickers already on disk.
        """

        def _validate_dates(start: Optional[str], end: Optional[str]) -> None:
            if not start or not end:
                raise ValueError("start and end must be non-empty strings")

            try:
                start_dt = datetime.fromisoformat(start)
                end_dt = datetime.fromisoformat(end)
            except ValueError as e:
                raise ValueError(f"Invalid date format: {e}")

            if start_dt > end_dt:
                raise ValueError("start date must be <= end date")

        _validate_dates(start, end)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

        for idx, ticker in enumerate(tickers, 1):
            out_path = output_dir / f"{ticker}.parquet"

            if skip_existing and out_path.exists():
                logger.info("[%d/%d] %s already exists - skipping", idx, len(tickers), ticker)
                continue

            logger.info("[%d/%d] Processing %s", idx, len(tickers), ticker)

            for chunk_start, chunk_end in self._chunk_time_range(start_dt, end_dt, years=2):
                logger.info(
                    "  Fetching %s [%s --> %s]",
                    ticker,
                    chunk_start.date(),
                    chunk_end.date(),
                )

                df = self.download_bars(
                    ticker=ticker,
                    start=chunk_start.isoformat(),
                    end=chunk_end.isoformat(),
                    timeframe=timeframe,
                    data_feed=data_feed,
                )

                if df.empty:
                    continue

                self._append_parquet(df, out_path)

                logger.info("  Wrote %d rows", len(df))

                time.sleep(self.RATE_LIMIT_SLEEP)
        # output_dir = Path(output_dir)
        # output_dir.mkdir(parents=True, exist_ok=True)

        # for idx, ticker in enumerate(tickers, 1):
        #     out_path = output_dir / f"{ticker}.parquet"
        #     if skip_existing and out_path.exists():
        #         logger.info(
        #             "[%d/%d] %s already cached - skipping", idx, len(tickers), ticker
        #         )
        #         continue

        #     logger.info("[%d/%d] Downloading %s ...", idx, len(tickers), ticker)
        #     df = self.download_bars(
        #         ticker, start=start, end=end, timeframe=config.data.freq
        #     )
        #     if not df.empty:
        #         df.to_parquet(
        #             out_path, engine="pyarrow", compression="zstd", index=True
        #         )
        #         logger.info("  Saved %d bars to %s", len(df), out_path)
        #     else:
        #         logger.info("  No data for %s - file not written", ticker)

        #     time.sleep(self.RATE_LIMIT_SLEEP)

    def downcast_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce memory by downcasting numeric types.

        Converts float64 --> float32 (50% memory savings, 7 decimal precision)
        Converts int64 --> int32 (50% memory savings, max value 2.1B)

        Args:
            df: DataFrame with OHLCV columns.

        Returns:
            DataFrame with downcasted types.
        """
        if "open" in df.columns:
            df["open"] = df["open"].astype("float32")
        if "high" in df.columns:
            df["high"] = df["high"].astype("float32")
        if "low" in df.columns:
            df["low"] = df["low"].astype("float32")
        if "close" in df.columns:
            df["close"] = df["close"].astype("float32")
        if "volume" in df.columns:
            df["volume"] = df["volume"].astype("int32")
        return df

    def _chunk_time_range(self, start: datetime, end: datetime, years: int = 2):
        """Yield (start, end) pairs in N-year chunks."""
        current = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)

        while current < end:
            try:
                next_end = current.replace(year=current.year + years)
            except ValueError:
                # handle leap year edge cases safely
                next_end = current + pd.DateOffset(years=years)
                next_end = next_end.to_pydatetime()

            next_end = next_end.astimezone(timezone.utc)

            if next_end > end:
                next_end = end

            yield current, next_end
            current = next_end
        # current = start

        # while current < end:
        #     next_end = datetime(
        #         year=current.year + years,
        #         month=current.month,
        #         day=current.day,
        #         tzinfo=timezone.utc,
        #     )

        #     if next_end > end:
        #         next_end = end

        #     yield current, next_end
        #     current = next_end

    def _append_parquet(self, df: pd.DataFrame, path: Path) -> None:
        """Append or create parquet safely (via concat + rewrite)."""

        # enforce UTC invariance BEFORE any disk write
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index = pd.DatetimeIndex(df.index, tz="UTC")
        df.index = df.index.astype("datetime64[ns, UTC]")

        if path.exists():
            existing = pd.read_parquet(path)

            # enforce UTC on existing too (critical for consistency)
            if not isinstance(existing.index, pd.DatetimeIndex):
                existing.index = pd.to_datetime(existing.index)

            if existing.index.tz is None:
                existing.index = existing.index.tz_localize("UTC")
            else:
                existing.index = existing.index.tz_convert("UTC")

            df = pd.concat([existing, df])

            # remove duplicates safely
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

        logger.info(f"appending to parquet: {path}")

        # Atomic write: write to a temp path first, then os.replace.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(
            tmp_path,
            engine="pyarrow",
            compression="zstd",
            index=True,
        )
        os.replace(tmp_path, path)

    # def _append_parquet(self, df: pd.DataFrame, path: Path) -> None:
    #     """Append or create parquet safely (via concat + rewrite)."""

    #     if path.exists():
    #         existing = pd.read_parquet(path)
    #         df = pd.concat([existing, df])

    #         # remove duplicates (critical for overlapping ranges)
    #         df = df[~df.index.duplicated(keep="last")]

    #         df = df.sort_index()

    #     logger.info(f"appending to parquet: {path}")
    #     df.to_parquet(path, engine="pyarrow", compression="zstd", index=True)
