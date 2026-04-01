import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
from datetime import datetime, timezone
import pytz
import numpy as np
from scipy import stats

import json
from src.report_generation.generate_data_ingestion_report import generate_html_report


class AlpacaIngestor:
    """
    Data ingestion layer for the PSO-LSTM Stock Tuner.
    Fetches historical OHLCV data using the Alpaca Market Data API and persists to Parquet.
    """

    def __init__(self, cfg, logger=None):
        """
        Initializes the ingestor with Hydra configuration and environment variables.

        Args:
            cfg: The Hydra configuration object (expected to contain data and path settings).
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        load_dotenv()
        self.logger.info("Loaded environment variables from .env")

        # Load credentials from .env for security
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.logger.info("Retrieved Alpaca API credentials from environment")

        if not api_key or not secret_key:
            self.logger.critical(
                "Alpaca API credentials missing. Ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are in .env."
            )
            raise EnvironmentError("Missing Alpaca API credentials.")

        # Initialize the historical data client
        self.client = StockHistoricalDataClient(api_key, secret_key)
        self.logger.info("Initialized Alpaca StockHistoricalDataClient")

        # Map configuration strings to alpaca-py TimeFrame objects
        self.timeframe_map = {
            "1Min": TimeFrame.Minute,
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        self.logger.info(f"TimeFrame mapping set: {self.timeframe_map}")

        # Resolve and create the raw data storage path
        self.raw_path = Path(self.cfg.paths.data_storage.raw)
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Raw data path resolved and created: {self.raw_path}")

    def fetch_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetches historical bars for a specific ticker based on configured timeframe and date range.
        Handles API communication and initial formatting.

        Args:
            ticker (str): The equity symbol to fetch.

        Returns:
            pd.DataFrame: A DataFrame containing OHLCV data.
        """
        self.logger.info(f"Fetching historical bars for: {ticker}")

        # Define parameters based on project requirements (5-year historical span)
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=self.timeframe_map.get(self.cfg.data.timeframe, TimeFrame.Minute),
            start=self.cfg.data.start_date,
            end=self.cfg.data.end_date,
            adjustment=self.cfg.data.adjustment,  # Adjusts for splits and dividends to ensure data continuity
            feed=self.cfg.data.data_feed,
        )
        self.logger.info(f"Request parameters: {request_params}")

        try:
            # The alpaca-py SDK utilizes urllib3 for internal retry logic and rate limit management
            bars = self.client.get_stock_bars(request_params)
            df = bars.df
            self.logger.info(
                f"Data fetched for {ticker}. Number of rows received: {len(df) if df is not None else 0}"
            )

            if df is None or df.empty:
                self.logger.warning(
                    f"No data returned for {ticker} within the range {self.cfg.data.start_date} to {self.cfg.data.end_date}."
                )
                return pd.DataFrame()
            else:
                self.logger.info(f"First 5 rows: {df[:5]}")

            # Flatten Alpaca's MultiIndex (Symbol, Timestamp) to a standard DatetimeIndex
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker, level=0)
                self.logger.info(f"Flattened MultiIndex for {ticker}")

            return df

        except Exception as e:
            self.logger.error(f"Critical error during Alpaca API request for {ticker}: {str(e)}")
            return pd.DataFrame()

    def download_symbol(self, symbol: str) -> pd.DataFrame:
        """
        Download historical bars for a single symbol in chunks, handling
        pagination, filtering to market hours (08:00-17:00 ET), and returning
        a consolidated DataFrame for further processing.

        Start and end timestamps are formatted as RFC-3339 (YYYY-MM-DD or ISO 8601)
        to comply with Alpaca API expectations.
        """

        # Map configuration values
        timeframe_obj = self.timeframe_map.get(self.cfg.data.timeframe, TimeFrame.Minute)
        data_adjustment = self.cfg.data.adjustment
        data_feed = self.cfg.data.data_feed

        # Set start and end timestamps
        current_start = pd.Timestamp(self.cfg.data.start_date, tz="UTC")
        global_end = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(minutes=16)

        chunk_days = getattr(self.cfg.data, "chunk_days", 5)  # default chunk size if not set
        eastern = pytz.timezone("US/Eastern")

        self.logger.info(f"Starting download for symbol: {symbol}")

        df_all = pd.DataFrame()

        while current_start < global_end:
            current_chunk_end = min(current_start + pd.Timedelta(days=chunk_days), global_end)
            all_pages = []
            page_token: Optional[str] = None

            self.logger.info(
                f"Fetching data from {current_start} to {current_chunk_end} for {symbol}"
            )

            while True:
                # Format start/end as RFC-3339 strings
                start_str = current_start.isoformat()
                end_str = current_chunk_end.isoformat()

                request_params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=timeframe_obj,
                    start=start_str,
                    end=end_str,
                    adjustment=data_adjustment,
                    feed=data_feed,
                )

                try:
                    response = self.client.get_stock_bars(request_params)
                    df = response.df

                    if df is None or df.empty:
                        self.logger.warning(f"No data returned for {symbol} in this chunk.")
                        break
                    else:
                        self.logger.info(f"Grabbed {len(df)} rows from Alpaca")

                    all_pages.append(df)
                    page_token = getattr(response, "next_page_token", None)
                    if not page_token:
                        break

                except Exception as e:
                    self.logger.error(
                        f"Error fetching data for {symbol}: {e}, page_token={page_token}"
                    )
                    break

            if all_pages:
                df_concat = pd.concat(all_pages)

                # Flatten MultiIndex if present
                if isinstance(df_concat.index, pd.MultiIndex):
                    df_concat = df_concat.xs(symbol, level=0)

                # Convert to Eastern Time and filter to market hours
                df_concat.index = df_concat.index.tz_convert(eastern)
                df_concat = df_concat.between_time("08:00", "17:00")

                # Append to the main DataFrame
                df_all = pd.concat([df_all, df_concat])

            # Move to the next chunk
            current_start = current_chunk_end + pd.Timedelta(minutes=1)

        self.logger.info(f"Completed download for symbol: {symbol}, total rows: {len(df_all)}")
        return df_all

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
            self.logger.warning(f"Data for ticker was out of order. Sorting index chronologically.")
            df.sort_index(inplace=True)

    def save_to_parquet(self, df: pd.DataFrame, ticker: str):
        """
        Saves the validated DataFrame to a Parquet file using pyarrow.

        Args:
            df (pd.DataFrame): The processed data.
            ticker (str): The ticker symbol for naming.
        """
        filename = f"{ticker}_{self.cfg.data.timeframe}.parquet"
        save_path = self.raw_path / filename

        try:
            # Save using pyarrow engine for production-level speed and metadata support
            df.to_parquet(save_path, engine="pyarrow", index=True)
            self.logger.info(f"Saved {ticker} raw data to: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to write Parquet file for {ticker}: {str(e)}")
            raise

    def run(self, tickers: List[str]):
        """
        Orchestrates the ingestion pipeline for a list of tickers, skipping
        already ingested tickers, and generates a JSON ingestion report.
        """
        self.logger.info(f"Starting Alpaca ingestion pipeline for {len(tickers)} symbols.")

        report = []

        for i, ticker in enumerate(tickers, start=1):
            self.logger.info(f"[{i}/{len(tickers)}] Processing ticker: {ticker}")

            filename = f"{ticker}_{self.cfg.data.timeframe}.parquet"
            parquet_path = self.raw_path / filename

            ticker_summary = {
                "ticker": ticker,
                "status": "unknown",
                "rows_fetched": 0,
                "chunks_processed": 0,
                "missing_values": {},
                "summary_statistics": {},
                "start_date": None,
                "end_date": None,
                "parquet_file": str(parquet_path),
                "errors": None,
                "skipped_existing": False,
            }

            # Skip API call if file already exists
            if parquet_path.exists():
                self.logger.info(f"Parquet file already exists for {ticker}, skipping API call.")
                ticker_summary["status"] = "skipped_existing"
                ticker_summary["skipped_existing"] = True

                # Optional: load the existing file to capture statistics
                try:
                    df_existing = pd.read_parquet(parquet_path)
                    ticker_summary["rows_fetched"] = len(df_existing)
                    ticker_summary["start_date"] = str(df_existing.index.min())
                    ticker_summary["end_date"] = str(df_existing.index.max())
                    stats_desc = df_existing.describe().to_dict()
                    ticker_summary["summary_statistics"] = stats_desc
                    missing = df_existing.isnull().sum().to_dict()
                    missing_pct = (df_existing.isnull().mean() * 100).to_dict()
                    ticker_summary["missing_values"] = {
                        col: {"count": missing[col], "percent": missing_pct[col]}
                        for col in df_existing.columns
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to read existing file for {ticker}: {str(e)}")
                    ticker_summary["errors"] = f"Failed to read existing file: {str(e)}"

                report.append(ticker_summary)
                continue

            # If file does not exist, proceed with API fetch
            try:
                df = self.fetch_data(ticker)
                # df = self.download_symbol(ticker)

                if df.empty:
                    ticker_summary["status"] = "skipped"
                    ticker_summary["errors"] = "No data returned from API."
                    self.logger.warning(f"Ingestion skipped for {ticker} due to no source data.")
                    report.append(ticker_summary)
                    continue

                self.validate_response(df)

                # Capture statistics
                ticker_summary["rows_fetched"] = len(df)
                ticker_summary["start_date"] = str(df.index.min())
                ticker_summary["end_date"] = str(df.index.max())

                # Missing values
                missing = df.isnull().sum().to_dict()
                missing_pct = (df.isnull().mean() * 100).to_dict()
                ticker_summary["missing_values"] = {
                    col: {"count": missing[col], "percent": missing_pct[col]} for col in df.columns
                }

                # Summary statistics
                stats_desc = df.describe().to_dict()
                ticker_summary["summary_statistics"] = stats_desc

                # Outliers (z-score > 3) for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                z_scores = np.abs(stats.zscore(df[numeric_cols].dropna()))
                if z_scores.size > 0:
                    outliers_count = dict(zip(numeric_cols, (z_scores > 3).sum(axis=0)))
                    ticker_summary["outliers"] = outliers_count
                else:
                    ticker_summary["outliers"] = {}

                self.save_to_parquet(df, ticker)
                ticker_summary["status"] = "success"

            except Exception as e:
                self.logger.error(f"Processing failed for ticker {ticker}: {str(e)}")
                ticker_summary["status"] = "failed"
                ticker_summary["errors"] = str(e)

            report.append(ticker_summary)

        # Save the JSON report
        report_path = Path(self.cfg.paths.data_storage.report) / "ingestion_report.json"
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
            self.logger.info(f"Ingestion report saved to: {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save ingestion report: {str(e)}")

        self.logger.info("Ingestion pipeline run completed for all tickers.")

        self.logger.info("Generating data ingestion report...")
        html_report_path = Path(self.cfg.paths.data_storage.report) / "ingestion_report.html"
        generate_html_report(report_json_path=report_path, output_path=html_report_path)
