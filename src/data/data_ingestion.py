import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

# Configure module-level logger
logger = logging.getLogger(__name__)


class AlpacaIngestor:
    """
    Data ingestion layer for the PSO-LSTM Stock Tuner.
    Fetches historical OHLCV data using the Alpaca Market Data API and persists to Parquet.
    """

    def __init__(self, cfg):
        """
        Initializes the ingestor with Hydra configuration and environment variables.

        Args:
            cfg: The Hydra configuration object (expected to contain data and path settings).
        """
        self.cfg = cfg
        load_dotenv()

        # Load credentials from .env for security
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            logger.critical(
                "Alpaca API credentials missing. Ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are in .env."
            )
            raise EnvironmentError("Missing Alpaca API credentials.")

        # Initialize the historical data client
        self.client = StockHistoricalDataClient(api_key, secret_key)

        # Map configuration strings to alpaca-py TimeFrame objects
        self.timeframe_map = {
            "1Min": TimeFrame.Minute,
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }

        # Resolve and create the raw data storage path
        self.raw_path = Path(self.cfg.paths.data_storage.raw)
        self.raw_path.mkdir(parents=True, exist_ok=True)

    def fetch_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetches historical bars for a specific ticker based on configured timeframe and date range.
        Handles API communication and initial formatting.

        Args:
            ticker (str): The equity symbol to fetch.

        Returns:
            pd.DataFrame: A DataFrame containing OHLCV data.
        """
        logger.info(f"Fetching historical bars for: {ticker}")

        # Define parameters based on project requirements (5-year historical span)
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=self.timeframe_map.get(self.cfg.timeframe, TimeFrame.Minute),
            start=self.cfg.start_date,
            end=self.cfg.end_date,
            adjustment="all",  # Adjusts for splits and dividends to ensure data continuity
        )

        try:
            # The alpaca-py SDK utilizes urllib3 for internal retry logic and rate limit management
            bars = self.client.get_stock_bars(request_params)
            df = bars.df

            if df is None or df.empty:
                logger.warning(
                    f"No data returned for {ticker} within the range {self.cfg.start_date} to {self.cfg.end_date}."
                )
                return pd.DataFrame()

            # Flatten Alpaca's MultiIndex (Symbol, Timestamp) to a standard DatetimeIndex
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker, level=0)

            return df

        except Exception as e:
            logger.error(
                f"Critical error during Alpaca API request for {ticker}: {str(e)}"
            )
            return pd.DataFrame()

    def validate_response(self, df: pd.DataFrame):
        """
        Validates data integrity and structure before persistence.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
        """
        if df.empty:
            raise ValueError("Validation failed: DataFrame is empty.")

        # Ensure standard OHLCV columns are present
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"Ingested data is missing required columns. Found: {df.columns.tolist()}"
            )

        # Ensure proper time-series indexing
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index validation failed: Data must have a DatetimeIndex.")

        if not df.index.is_monotonic_increasing:
            logger.warning(
                f"Data for ticker was out of order. Sorting index chronologically."
            )
            df.sort_index(inplace=True)

    def save_to_parquet(self, df: pd.DataFrame, ticker: str):
        """
        Saves the validated DataFrame to a Parquet file using pyarrow.

        Args:
            df (pd.DataFrame): The processed data.
            ticker (str): The ticker symbol for naming.
        """
        filename = f"{ticker}_{self.cfg.timeframe}.parquet"
        save_path = self.raw_path / filename

        try:
            # Save using pyarrow engine for production-level speed and metadata support
            df.to_parquet(save_path, engine="pyarrow", index=True)
            logger.info(f"Saved {ticker} raw data to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to write Parquet file for {ticker}: {str(e)}")
            raise

    def run(self, tickers: List[str]):
        """
        Orchestrates the ingestion pipeline for a list of tickers.

        Args:
            tickers (List[str]): List of symbols to ingest (e.g., the 51 tickers in the proposal).
        """
        logger.info(f"Starting Alpaca ingestion pipeline for {len(tickers)} symbols.")

        for ticker in tickers:
            try:
                data = self.fetch_data(ticker)

                if not data.empty:
                    self.validate_response(data)
                    self.save_to_parquet(data, ticker)
                else:
                    logger.warning(
                        f"Ingestion skipped for {ticker} due to lack of source data."
                    )

            except Exception as e:
                logger.error(f"Processing failed for ticker {ticker}: {str(e)}")
                # Continue processing remaining tickers in the queue
                continue

        logger.info("Ingestion pipeline run completed.")
