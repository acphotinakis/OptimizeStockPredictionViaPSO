#!/usr/bin/env python3
import sys
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from pathlib import Path


# Resolve project root (adjust depth if needed)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]  # adjust if structure changes

# Ensure only the project root (not file paths) is added
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Debug prints (optional)
print("Current file:", CURRENT_FILE)
print("Project root:", PROJECT_ROOT)
print("sys.path updated:")
print(sys.path)

from src.utils.data_storage import _load_parquet
from src.data.cleaner import DataCleaner
from src.data.alpaca_ingestor import AlpacaIngestor
from src.utils.logger import LogFileMode, setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


import pandas as pd
from pathlib import Path
from pathlib import Path
import pandas as pd


import matplotlib

matplotlib.use("Agg")  # important for multiprocessing safety


def load_all_timeframes(raw_output_dir: Path, ticker: str, timeframes: list[str]):
    data = {}

    for tf in timeframes:
        path = raw_output_dir / tf / f"{ticker}.parquet"
        if not path.exists():
            logger.warning("Missing %s", path)
            continue

        df = _load_parquet(path)
        cols = ["open", "high", "low", "close", "volume"]
        df = df[cols]

        # ------------------------------------------------------------
        # 1. Ensure datetime index
        # ------------------------------------------------------------
        df.index = pd.to_datetime(df.index, utc=True)

        # ------------------------------------------------------------
        # 2. Convert UTC → America/New_York (DST-aware)
        # ------------------------------------------------------------
        # df.index = df.index.tz_convert("America/New_York")
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")

        data[tf] = df

    return data


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=True)
    logger.info("Saved %s (%d rows)", path.relative_to, len(df))


def ingest_all(
    ingestor: AlpacaIngestor,
    tickers: list[str],
    base_dir: Path,
    data_feed: str,
    tf: str,
    start_date: str,
    end_date: str,
    skip_existing: bool = True,
):
    ingestor.download_universe(
        tickers=tickers,
        output_dir=base_dir,
        start=start_date,
        end=end_date,
        skip_existing=skip_existing,
        timeframe=tf,
        data_feed=data_feed,
    )


def _parse_timeframe(timeframe: str) -> pd.Timedelta:
    mapping = {
        "1Min": pd.Timedelta(minutes=1),
        "5Min": pd.Timedelta(minutes=5),
        "15Min": pd.Timedelta(minutes=15),
        "1Hour": pd.Timedelta(hours=1),
        "1Day": pd.Timedelta(days=1),
    }

    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    return mapping[timeframe]


mapping = {
    "1Min": "1min",
    "5Min": "5min",
    "15Min": "15min",
    "1Hour": "1h",
    "1Day": "1D",
}


def _load_tickers(path: str) -> list[str]:
    with open(path) as f:
        return [
            l.split()[0] for l in f if l.split() and not l.split()[0].startswith("#")
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest, clean, and align OHLCV data")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--tickers", default="config/tickers.txt")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--aligned-dir", default="data/aligned")
    parser.add_argument("--cleaned-dir", default="data/cleaned")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    setup_logger(log_file="logs/testing.log", level="INFO", mode=LogFileMode.APPEND)

    cfg = load_config(args.config)
    tickers = _load_tickers(args.tickers)
    logger.info("Loaded %d tickers", len(tickers))

    start_date = "2020-01-01"
    # start_date = "2026-01-01"
    end_date = "2026-01-28"
    timeframes = ["1Min", "5Min", "15Min", "1Hour", "1Day"]
    # timeframes = ["1Min"]

    ingestor = AlpacaIngestor()

    # ------------------------------------------------------------
    # STAGE 1: INGESTION (ALL TIMEFRAMES)
    # ------------------------------------------------------------

    raw_output = Path(args.raw_dir)
    raw_output.mkdir(parents=True, exist_ok=True)

    for tf in timeframes:
        tf_raw_dir_output = raw_output / tf
        tf_raw_dir_output.mkdir(parents=True, exist_ok=True)
        ingest_all(
            ingestor=ingestor,
            tickers=tickers,
            base_dir=tf_raw_dir_output,
            data_feed="sip",
            tf=tf,
            start_date=start_date,
            end_date=end_date,
        )

    # import sys

    # sys.exit(0)
    # ------------------------------------------------------------
    # STAGE 2: ALIGN + CLEAN PIPELINE
    # ------------------------------------------------------------

    def build_full_grid(
        start: pd.Timestamp, end: pd.Timestamp, timeframe: str
    ) -> pd.DatetimeIndex:
        freq = _parse_timeframe(timeframe)

        # HARD NORMALIZATION CONTRACT
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        # Force both to UTC first (canonical intermediate representation)
        if start.tz is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")

        if end.tz is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")

        # Build grid in UTC first (stable, deterministic)
        grid_utc = pd.date_range(start=start, end=end, freq=freq, tz="UTC")

        # Convert once at the end
        return grid_utc.tz_convert("America/New_York")

    def align_to_grid(df: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
        if df.index.tz is None:
            df = df.tz_localize("UTC").tz_convert("America/New_York")
        else:
            df = df.tz_convert("America/New_York")

        return df.reindex(grid)

    # data/raw
    raw_dir = Path(args.raw_dir)
    # data/aligned
    aligned_dir = Path("data/aligned")
    # data/processed
    processed_dir = Path(args.processed_dir)
    # data/cleaned
    cleaned_dir = Path(args.cleaned_dir)
    # data/plots
    plots_dir = Path("data/plots")

    def ensure_directories_exist() -> None:
        """
        Ensures all required pipeline directories exist.
        No deletion or modification of existing data is permitted.
        """

        required_dirs = [raw_dir, aligned_dir, cleaned_dir, processed_dir, plots_dir]

        dirs = []
        for n in required_dirs:
            for tf in timeframes:
                path = n / tf
                dirs.append(path)

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    ensure_directories_exist()

    def get_path(base_dir: Path, timeframe: str, ticker: str) -> Path:
        return base_dir / timeframe / f"{ticker}.parquet"

    # tickers = ["AAPL"]
    for ticker in tickers:
        logger.info("===== PIPELINE START: %s =====", ticker)

        for tf in timeframes:
            logger.info("----- TIMEFRAME: %s -----", tf)

            # --------------------------------------------------------
            # LOAD RAW
            # --------------------------------------------------------

            # raw_path = raw_output / tf / f"{ticker}.parquet"
            raw_path = get_path(raw_dir, tf, ticker)
            # raw_path = raw_output / f"{ticker}.parquet"
            if not raw_path.exists():
                logger.warning("Missing raw data: %s", raw_path)
                continue

            raw_df = _load_parquet(raw_path)

            # ensure OHLCV safety
            raw_df = raw_df[["open", "high", "low", "close", "volume"]]
            raw_df.index = pd.to_datetime(raw_df.index)
            raw_df = raw_df.sort_index()

            # --------------------------------------------------------
            # ALIGN
            # --------------------------------------------------------
            start, end = raw_df.index.min(), raw_df.index.max()
            grid = build_full_grid(start, end, tf)

            aligned_df = align_to_grid(raw_df, grid)

            # --------------------------------------------------------
            # CLEAN
            # --------------------------------------------------------
            cleaner = DataCleaner(max_gap_fill=5, ticker=f"{ticker}-{tf}", timeframe=tf)

            cleaned_df, cleaned_report = cleaner.clean(aligned_df)

            # --------------------------------------------------------
            # SAVE OUTPUTS (NO OVERWRITES)
            # --------------------------------------------------------
            _save_parquet(aligned_df, get_path(aligned_dir, tf, ticker))
            _save_parquet(cleaned_df, get_path(cleaned_dir, tf, ticker))
            _save_parquet(cleaned_df, get_path(processed_dir, tf, ticker))


if __name__ == "__main__":
    main()
