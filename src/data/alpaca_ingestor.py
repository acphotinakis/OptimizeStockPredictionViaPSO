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

import pandas as pd

logger = logging.getLogger(__name__)


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
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET", "")
        self.base_url = base_url
        self._api = None  # Lazy-loaded

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_api(self):
        """Lazy-load the alpaca_trade_api client."""
        if self._api is None:
            try:
                import alpaca_trade_api as tradeapi
            except ImportError as e:
                raise ImportError(
                    "alpaca-trade-api not installed. Run: pip install alpaca-trade-api"
                ) from e
            self._api = tradeapi.REST(
                self.api_key, self.api_secret, self.base_url, api_version="v2"
            )
        return self._api

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download_bars(
        self,
        ticker: str,
        start: str = "2019-01-02",
        end: str = "2024-01-01",
        timeframe: str = "1Min",
        adjustment: str = "split",
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
        try:
            from alpaca_trade_api.rest import TimeFrame  # noqa: F401
        except ImportError:
            pass

        api = self._get_api()
        try:
            bars = api.get_bars(
                ticker,
                timeframe,
                start=start,
                end=end,
                adjustment=adjustment,
                feed="sip",
            ).df
        except Exception as exc:
            logger.error("Failed to download %s: %s", ticker, exc)
            return pd.DataFrame()

        if bars.empty:
            logger.warning("No data returned for %s", ticker)
            return pd.DataFrame()

        bars = bars[["open", "high", "low", "close", "volume"]].copy()
        bars.index = pd.to_datetime(bars.index, utc=True)
        bars.index.name = "timestamp"
        bars["ticker"] = ticker
        return bars

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

    @staticmethod
    def load_bars(path: str | Path) -> pd.DataFrame:
        """Load a previously saved Parquet file.

        Args:
            path: Path to the Parquet file.

        Returns:
            DataFrame with DatetimeIndex (UTC).
        """
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        return df
