#!/usr/bin/env python3
"""
scripts/01_ingest_data.py

Downloads 1-minute OHLCV bars from Alpaca Markets API for all tickers
in the universe. Cleans and aligns data, then saves to parquet.

Usage:
    python scripts/01_ingest_data.py --config config/default_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.alpaca_ingestor import AlpacaIngestor
from src.data.cleaner import DataCleaner
from src.data.aligner import TickerAligner
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest and clean OHLCV data")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="config/tickers.txt",
        help="Path to ticker list file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tickers that already have cached raw data",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    setup_logger(log_file="logs/01_ingest_data.log", level="INFO")

    # Read ticker list
    with open(args.tickers) as f:
        tickers = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    logger.info("Loaded %d tickers from %s", len(tickers), args.tickers)

    # Initialize ingestor and cleaner
    ingestor = AlpacaIngestor()
    cleaner = DataCleaner(
        session_start=cfg.data.session_start,
        session_end=cfg.data.session_end,
    )

    # Download raw data
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting data download from Alpaca API...")
    ingestor.download_universe(
        tickers=tickers,
        output_dir=raw_dir,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        skip_existing=args.skip_existing,
    )

    # Clean each ticker
    logger.info("Cleaning data...")
    cleaned_dfs = {}
    for ticker in tickers:
        raw_path = raw_dir / f"{ticker}.parquet"
        if not raw_path.exists():
            logger.warning("Skipping %s (no raw data)", ticker)
            continue

        df_raw = ingestor.load_bars(raw_path)
        df_clean = cleaner.clean(df_raw)
        cleaned_dfs[ticker] = df_clean
        logger.info("Cleaned %s: %d bars", ticker, len(df_clean))

    # Align all tickers
    logger.info("Aligning %d tickers to common timestamp index...", len(cleaned_dfs))
    aligner = TickerAligner(benchmark_ticker="SPY")
    df_aligned = aligner.align(cleaned_dfs)

    # Save aligned data
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    aligned_file = output_path / "aligned_universe.parquet"
    df_aligned.to_parquet(aligned_file)
    logger.info("Saved aligned data to %s", aligned_file)
    logger.info("Shape: %s", df_aligned.shape)
    logger.info("Data ingestion complete!")


if __name__ == "__main__":
    main()
