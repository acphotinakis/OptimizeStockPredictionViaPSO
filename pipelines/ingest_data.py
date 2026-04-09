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
from typing import Dict
import pandas as pd


# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.aligner import TickerAligner
from src.data.alpaca_ingestor import AlpacaIngestor
from src.data.cleaner import DataCleaner
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def save_parquet(output_path: Path, ticker: str, df: pd.DataFrame):
    # Save individual ticker
    df.to_parquet(
        output_path,
        engine="pyarrow",  # fastest + most compatible
        compression="zstd",  # best compression ratio + speed tradeoff
        index=True,
    )
    logger.info(
        "Saved aligned %s (%d rows) to %s (%d rows, %d cols)",
        ticker,
        len(df),
        output_path,
        df.shape[0],
        df.shape[1],
    )


def table_stats(alpaca_ingestor, raw_dir, tickers):
    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = [
        "Ticker",
        "Rows",
        "Start",
        "End",
        "Mean Close",
        "Std Close",
        "Mean Vol",
        "Std Vol",
        "Missing %",
    ]

    stats_summary = []

    for ticker in tickers:
        path = raw_dir / f"{ticker}.parquet"
        if not path.exists():
            logger.warning("Missing raw data for %s", ticker)
            continue

        df = alpaca_ingestor.load_bars(path)

        # Ensure datetime index
        df.index = pd.to_datetime(df.index)

        rows = len(df)
        start = df.index.min()
        end = df.index.max()

        mean_close = df["close"].mean()
        std_close = df["close"].std()

        mean_vol = df["volume"].mean()
        std_vol = df["volume"].std()

        missing = df.isna().mean().mean()

        stats_summary.append(
            {"ticker": ticker, "mean_close": mean_close, "std_close": std_close}
        )

        table.add_row(
            [
                ticker,
                rows,
                str(start),
                str(end),
                f"{mean_close:.2f}",
                f"{std_close:.2f}",
                f"{mean_vol:.2f}",
                f"{std_vol:.2f}",
                f"{missing:.4%}",
            ]
        )

    logger.info("\n%s", table)

    # ---- Cross-ticker comparison ----
    df_stats = pd.DataFrame(stats_summary)

    logger.info("\n=== Cross-Ticker Dispersion ===")
    logger.info(
        "Mean Close (min/max): %.2f / %.2f",
        df_stats["mean_close"].min(),
        df_stats["mean_close"].max(),
    )

    logger.info(
        "Std Close (min/max): %.2f / %.2f",
        df_stats["std_close"].min(),
        df_stats["std_close"].max(),
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest, clean, and align OHLCV data")
    parser.add_argument(
        "--mode",
        choices=["ingest", "clean", "align", "stats", "plot"],
        required=True,
    )
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--tickers", type=str, default="config/tickers.txt")
    # for raw outputs
    parser.add_argument("--raw-output", type=str, default="data/raw")
    # for cleaned outputs
    parser.add_argument("--cleaned-output", type=str, default="data/cleaned")
    # for aligned outputs
    parser.add_argument("--processed-output", type=str, default="data/processed")
    parser.add_argument(
        "--plot-tickers",
        type=str,
        default=None,
        help="Comma-separated tickers for plotting",
    )
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

    start_date = cfg.data.start_date
    end_date = cfg.data.end_date

    raw_dir = Path(args.raw_output)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = Path(args.cleaned_output)
    cleaned_path.mkdir(parents=True, exist_ok=True)

    processed_path = Path(args.processed_output)
    processed_path.mkdir(parents=True, exist_ok=True)

    pipelines_path = Path("data/pipelines/ingestion")
    pipelines_path.mkdir(parents=True, exist_ok=True)

    alpaca_ingestor = AlpacaIngestor()
    cleaner = DataCleaner(
        session_start=cfg.data.session_start,
        session_end=cfg.data.session_end,
    )
    ticker_aligner = TickerAligner(
        benchmark_ticker=cfg.data.benchmark_ticker,
        max_missing_fraction=cfg.data.max_missing_fraction,
        max_ffill_bars=cfg.data.max_ffill_bars,
    )

    with open(args.tickers) as f:
        tickers = []
        for line in f:
            _strip = line.split()
            if len(_strip) <= 0:
                continue
            if not _strip[0].startswith("#") and len(_strip[0]) > 0:
                tickers.append(_strip[0])
    logger.info("Loaded %d tickers from %s", len(tickers), args.tickers)

    logger.info(f"Tickers: [{tickers}]")

    if args.mode == "ingest":
        """
        The function below:
            -   handles multiindex
            -   ensures timezone correctness
            -   ensures only ["open", "high", "low", "close", "volume"] are columns
            -   Downcast to reduce memory usage (Phase 1: Data Quantization)
            -   Saves the final parquet to file
        """
        # Download the stock data
        alpaca_ingestor.download_universe(
            tickers,
            args.raw_output,
            start=start_date,
            end=end_date,
            skip_existing=args.skip_existing,
        )
    elif args.mode == "clean":
        if "SPY" not in tickers:
            logger.error("Ticker list must include SPY for alignment")
            return

        spy_path = raw_dir / "SPY.parquet"
        if not spy_path.exists():
            logger.error("SPY raw data not found at %s", spy_path)
            return

        df_spy_raw = alpaca_ingestor.load_bars(spy_path)
        df_spy = cleaner.clean(df_spy_raw)
        for ticker in tickers:
            raw_path = raw_dir / f"{ticker}.parquet"
            if not raw_path.exists():
                logger.info("Skipping %s (no raw data)", ticker)
                raise ValueError(
                    f"Raw data for {ticker} doesn't exist. Ensure ingestion for it first."
                )

            df_ticker_raw = alpaca_ingestor.load_bars(raw_path)
            df_ticker = cleaner.clean(df_ticker_raw)
            # df_ticker.index = pd.to_datetime(df_ticker.index)
            logger.info(f"Ticker Index in clean = {df_ticker.index}")

            # Save individual ticker
            ticker_file = cleaned_path / f"{ticker}.parquet"
            save_parquet(Path(ticker_file), ticker=ticker, df=df_ticker)
    elif args.mode == "align":
        spy_path = raw_dir / "SPY.parquet"

        if not spy_path.exists():
            logger.error("SPY raw data not found at %s", spy_path)
            return

        df_spy_raw = alpaca_ingestor.load_bars(spy_path)
        df_spy = cleaner.clean(df_spy_raw)

        fields_list = df_spy.columns.to_list()
        logger.info(f"Fields: {fields_list}")

        dfs: Dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            df_cleaned_path = cleaned_path / f"{ticker}.parquet"
            if not df_cleaned_path.exists():
                logger.info("Skipping %s (no cleaned data)", ticker)
                raise ValueError(
                    f"Cleaned data for {ticker} doesn't exist. Ensure cleaner for it first."
                )

            df_cleaned = alpaca_ingestor.load_bars(df_cleaned_path)
            logger.info(f"Ticker Index in Align = {df_cleaned.index}")

            dfs[ticker] = df_cleaned

        aligned_dfs = ticker_aligner.align(dfs=dfs, fields=fields_list)
        ticker_aligner.save_aligned_tickers(
            aligned_df=aligned_dfs, output_dir=args.processed_output
        )

        # ==========================================================
        # Compact summary across all tickers
        # ==========================================================
        from prettytable import PrettyTable

        # ==========================================================
        # Pretty Summary Table
        # ==========================================================
        table = PrettyTable()
        table.field_names = ["Ticker", "Rows", "Cols", "Start", "End", "Missing %"]

        for ticker, df in aligned_dfs.items():
            if isinstance(df, pd.Series):
                df = df.to_frame(name=df.name or ticker)

            missing_frac = df.isna().mean().mean()
            df.index = df.index.tz_convert("America/New_York")

            table.add_row(
                [
                    ticker,
                    len(df),
                    len(df.columns),
                    str(df.index.min()),
                    str(df.index.max()),
                    f"{missing_frac:.4%}",
                ]
            )

        # Optional formatting tweaks
        table.align = "r"
        table.align["Ticker"] = "l"

        logger.info("\n%s", table)
    elif args.mode == "stats":
        table_stats(alpaca_ingestor=alpaca_ingestor, raw_dir=raw_dir, tickers=tickers)
    elif args.mode == "plot":
        import mplfinance as mpf

        if args.plot_tickers is None:
            raise ValueError("Provide --plot-tickers (comma-separated)")

        plot_list = [t.strip() for t in args.plot_tickers.split(",")]

        for ticker in plot_list:
            path = raw_dir / f"{ticker}.parquet"
            if not path.exists():
                logger.warning("Missing raw data for %s", ticker)
                continue

            df = alpaca_ingestor.load_bars(path)

            # mplfinance requires:
            # index = datetime, columns = Open High Low Close Volume
            df_plot = df.copy()
            df_plot.index = pd.to_datetime(df_plot.index)

            df_plot = df_plot.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )

            df_plot = df_plot[["Open", "High", "Low", "Close", "Volume"]]

            logger.info("Plotting %s (%d rows)", ticker, len(df_plot))

            mpf.plot(
                df_plot.tail(500),  # limit for readability
                type="candle",
                volume=True,
                title=f"{ticker} Candlestick",
                style="charles",
            )
    else:
        raise ValueError(
            "Mode not properly selected. Select either 'ingest', 'clean', 'align'"
        )


if __name__ == "__main__":
    main()
