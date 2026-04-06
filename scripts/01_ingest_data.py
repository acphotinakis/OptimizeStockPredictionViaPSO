#!/usr/bin/env python3
"""
scripts/01_ingest_data.py

Memory-efficient ingestion, cleaning, and alignment of OHLCV data
for large universes (~50 tickers × 400k rows). Aligns all tickers
to SPY timestamps, saves individual Parquets, and optionally builds
a combined aligned universe Parquet incrementally.

Usage:
    python scripts/01_ingest_data.py --config config/default_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.alpaca_ingestor import AlpacaIngestor
from src.data.cleaner import DataCleaner
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest, clean, and align OHLCV data")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--tickers", type=str, default="config/tickers.txt")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tickers that already have cached raw data",
    )
    parser.add_argument(
        "--save-combined",
        action="store_true",
        help="Optionally build combined aligned universe parquet",
    )
    args = parser.parse_args()

    # Load config and logger
    cfg = load_config(args.config)
    setup_logger(log_file="logs/01_ingest_data.log", level="INFO")

    logger.info("Arguments: %s", args)

    # Read tickers
    with open(args.tickers) as f:
        tickers = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
    logger.info("Loaded %d tickers from %s", len(tickers), args.tickers)

    # Initialize ingestor and cleaner
    ingestor = AlpacaIngestor()
    cleaner = DataCleaner(
        session_start=cfg.data.session_start,
        session_end=cfg.data.session_end,
    )

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # 1. Load & clean SPY as benchmark
    # -------------------------------
    if "SPY" not in tickers:
        logger.error("Ticker list must include SPY for alignment")
        return

    spy_path = raw_dir / "SPY.parquet"
    if not spy_path.exists():
        logger.error("SPY raw data not found at %s", spy_path)
        return

    df_spy = cleaner.clean(ingestor.load_bars(spy_path))
    df_spy = df_spy[["open", "high", "low", "close", "volume"]].copy()
    df_spy.index = pd.to_datetime(df_spy.index)

    # Save SPY individually
    df_spy.to_parquet(output_path / "SPY.parquet")
    logger.info("Saved SPY aligned data (%d rows)", len(df_spy))

    # Initialize combined Parquet if requested
    combined_file = output_path / "aligned_universe.parquet"
    if args.save_combined and combined_file.exists():
        combined_file.unlink()  # overwrite existing

    # -------------------------------
    # 2. Process other tickers one at a time
    # -------------------------------
    for ticker in tickers:
        if ticker == "SPY":
            continue

        raw_path = raw_dir / f"{ticker}.parquet"
        if not raw_path.exists():
            logger.warning("Skipping %s (no raw data)", ticker)
            continue

        df_ticker = cleaner.clean(ingestor.load_bars(raw_path))
        df_ticker.index = pd.to_datetime(df_ticker.index)

        # Align to SPY timestamps using reindex + fill gaps
        df_ticker_aligned = df_ticker.reindex(df_spy.index)
        # df_ticker_aligned.ffill(inplace=True)
        # df_ticker_aligned.bfill(inplace=True)
        numeric_cols = df_ticker_aligned.select_dtypes(include="number").columns
        df_ticker_aligned[numeric_cols] = (
            df_ticker_aligned[numeric_cols].ffill().bfill()
        )

        # Save individual ticker
        ticker_file = output_path / f"{ticker}.parquet"
        df_ticker_aligned.to_parquet(ticker_file)
        logger.info(
            "Saved aligned %s (%d rows) to %s",
            ticker,
            len(df_ticker_aligned),
            ticker_file,
        )

        # Append to combined Parquet if requested
        if args.save_combined and 1 == 2:
            # Add ticker prefix to columns to avoid collisions
            df_ticker_prefixed = df_ticker_aligned.add_prefix(f"{ticker}_")
            df_ticker_prefixed.reset_index(inplace=True)
            if not combined_file.exists():
                df_ticker_prefixed.to_parquet(combined_file, index=False)
            else:
                # Append in row-wise mode is not possible in Parquet; combine in chunks
                df_existing = pd.read_parquet(combined_file)
                df_combined = pd.concat([df_existing, df_ticker_prefixed], axis=1)
                df_combined.to_parquet(combined_file, index=False)

    logger.info("Data ingestion and alignment complete!")


if __name__ == "__main__":
    main()
