"""
src/data/alpaca_ingestor.py

Downloads 1-minute OHLCV bars from the Alpaca Markets REST API.
Handles pagination, rate-limiting, caching, and split adjustments.
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import List, Optional
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.timeframe import TimeFrame
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
        # Load credentials from environment (use consistent naming)
        api_key = api_key or os.getenv("ALPACA_API_KEY")
        secret_key = api_secret or os.getenv("ALPACA_SECRET_KEY")

        logger.debug("Alpaca API credentials loaded from environment")

        if not api_key or not secret_key:
            logger.error(
                "Alpaca API credentials missing. Ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are in .env."
            )
            raise EnvironmentError("Missing Alpaca API credentials.")

        # Initialize the historical data client
        self.client = StockHistoricalDataClient(api_key, secret_key)
        logger.info("Initialized Alpaca StockHistoricalDataClient")

        # Map configuration strings to alpaca-py TimeFrame objects
        self.timeframe_map = {
            "1Min": TimeFrame.Minute,
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        logger.debug("TimeFrame mapping initialized")

        self.base_url = base_url

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download_bars(
        self,
        ticker: str,
        start: str = "2021-04-05",
        end: str = "2026-04-05",
        timeframe: str = "1Min",
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
            DataFrame with DatetimeIndex (UTC) and columns
            [open, high, low, close, volume, ticker].
            Empty DataFrame on failure.
        """

        from datetime import datetime, timezone

        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=self.timeframe_map.get(timeframe, TimeFrame.Minute),
            start=start_dt,
            end=end_dt,
            adjustment=Adjustment(adjustment),
            feed=DataFeed(data_feed),
        )

        try:
            bars = self.client.get_stock_bars(request_params)
            df: pd.DataFrame = bars.df

            logger.info(f"Downloaded {len(df)} bars for {ticker} from {start} to {end}")
            if df.empty:
                logger.info(f"No data returned for {ticker}")
                return pd.DataFrame()
            else:
                logger.info(f"Df head: {df.head}")

            # Flatten MultiIndex
            # if isinstance(df.index, pd.MultiIndex):
            #     df = df.xs(ticker, level=0)

            assert isinstance(df.index, pd.DatetimeIndex), "Expected DatetimeIndex"

            # Ensure timezone correctness for storage in UTC
            if df.index.tz is None:
                df.index = df.index.tz_localize(
                    "UTC"
                )  # Localize naive timestamps to UTC
            else:
                df.index = df.index.tz_convert("UTC")  # Convert any tz-aware to UTC

            df.index = df.index.round("1min")  # Optional rounding
            df.index.name = "timestamp"

            df = df[["open", "high", "low", "close", "volume"]].copy()
            df["ticker"] = ticker

            logger.info(f"Index of data (UTC): {df.index}")
            logger.info(f"Index dtype: {df.index.dtype}")

            # Downcast to reduce memory usage (Phase 1: Data Quantization)
            df = self.downcast_ohlcv(df)
            logger.debug("Downcasted %s to float32/int32", ticker)

            return df

        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def download_universe(
        self,
        tickers: List[str],
        output_dir: str | Path,
        start: str = "2019-01-02",
        end: str = "2024-01-01",
        skip_existing: bool = True,
    ) -> None:
        """Download all tickers and persist each as a Parquet file.

        Args:
            tickers: List of ticker symbols.
            output_dir: Directory to save Parquet files.
            start: Download start date (ISO string).
            end: Download end date (ISO string).
            skip_existing: If True, skip tickers already on disk.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, ticker in enumerate(tickers, 1):
            out_path = output_dir / f"{ticker}.parquet"
            if skip_existing and out_path.exists():
                logger.info(
                    "[%d/%d] %s already cached — skipping", idx, len(tickers), ticker
                )
                continue

            logger.info("[%d/%d] Downloading %s ...", idx, len(tickers), ticker)
            df = self.download_bars(ticker, start=start, end=end)
            if not df.empty:
                df.to_parquet(out_path)
                logger.info("  Saved %d bars to %s", len(df), out_path)
            else:
                logger.warning("  No data for %s — file not written", ticker)

            time.sleep(self.RATE_LIMIT_SLEEP)

    def load_bars(self, path: str | Path) -> pd.DataFrame:
        """Load a previously saved Parquet file.

        Args:
            path: Path to the Parquet file.

        Returns:
            DataFrame with DatetimeIndex (UTC).
        """
        df = pd.read_parquet(path)
        # if not isinstance(df.index, pd.DatetimeIndex):
        #     df.index = pd.to_datetime(df.index, utc=True)
        return df

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
