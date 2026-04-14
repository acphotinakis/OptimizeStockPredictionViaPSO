#!/usr/bin/env python3
"""
scripts/ingest_data.py

Ingest, clean, and align OHLCV data for the ticker universe.

Usage:
    python scripts/ingest_data.py --mode ingest --config config/default_config.yaml
    python scripts/ingest_data.py --mode clean
    python scripts/ingest_data.py --mode align
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.aligner import TickerAligner
from src.data.alpaca_ingestor import AlpacaIngestor
from src.data.cleaner import DataCleaner
from src.utils.logger import setup_logger
from src.utils.config_loader import Config, load_config

logger = logging.getLogger(__name__)


def _load_tickers(path: str) -> list[str]:
    with open(path) as f:
        return [
            l.split()[0] for l in f if l.split() and not l.split()[0].startswith("#")
        ]


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=True)
    logger.info("Saved %s (%d rows)", path.name, len(df))


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------


def run_ingest(args, cfg, tickers: list[str]) -> None:
    AlpacaIngestor().download_universe(
        tickers,
        args.raw_dir,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        skip_existing=args.skip_existing,
    )


def run_clean(args, cfg, tickers: list[str]) -> None:
    ingestor = AlpacaIngestor()
    cleaner = DataCleaner(
        session_start=cfg.data.session_start,
        session_end=cfg.data.session_end,
    )
    raw_dir = Path(args.raw_dir)
    cleaned_dir = Path(args.cleaned_dir)

    for ticker in tickers:
        raw_path = raw_dir / f"{ticker}.parquet"
        logger.info(f"Raw Path --> {raw_path} || Raw Directory --> {ticker}")
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw data missing for {ticker}. Run --mode ingest first."
            )
        df = cleaner.clean(ingestor.load_bars(raw_path))
        _save_parquet(df, cleaned_dir / f"{ticker}.parquet")


def run_align(args, cfg: Config, tickers: list[str]) -> None:
    ingestor = AlpacaIngestor()
    cleaner = DataCleaner(
        session_start=cfg.data.session_start,
        session_end=cfg.data.session_end,
    )
    aligner = TickerAligner(
        benchmark_ticker=cfg.data.benchmark_ticker,
        max_missing_fraction=cfg.data.max_missing_fraction,
        max_ffill_bars=cfg.data.max_ffill_bars,
    )
    cleaned_dir = Path(args.cleaned_dir)
    processed_dir = Path(args.processed_dir)

    dfs: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = cleaned_dir / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Cleaned data missing for {ticker}. Run --mode clean first."
            )
        dfs[ticker] = ingestor.load_bars(path)

    # Determine field list from SPY
    spy_raw = ingestor.load_bars(Path(args.raw_dir) / "SPY.parquet")
    fields = cleaner.clean(spy_raw).columns.tolist()

    aligned = aligner.align(dfs=dfs, fields=fields)
    aligner.save_aligned_tickers(aligned_df=aligned, output_dir=str(processed_dir))

    for ticker, df in aligned.items():
        logger.info(
            "%-6s  rows=%-6d  %s --> %s  missing=%.4f%%",
            ticker,
            len(df),
            df.index.min(),
            df.index.max(),
            df.isna().mean().mean() * 100,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest, clean, and align OHLCV data")
    parser.add_argument("--mode", choices=["ingest", "clean", "align"], required=True)
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--tickers", default="config/tickers.txt")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--cleaned-dir", default="data/cleaned")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    setup_logger(log_file="logs/ingest_data.log", level="INFO")
    cfg = load_config(args.config)
    tickers = _load_tickers(args.tickers)
    logger.info("Loaded %d tickers | mode=%s", len(tickers), args.mode)

    if args.mode == "ingest":
        run_ingest(args, cfg, tickers)
    elif args.mode == "clean":
        run_clean(args, cfg, tickers)
    elif args.mode == "align":
        run_align(args, cfg, tickers)


if __name__ == "__main__":
    main()
