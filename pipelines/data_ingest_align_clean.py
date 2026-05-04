#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

# Resolve project root (adjust depth if needed)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]  # adjust if structure changes

# Ensure only the project root (not file paths) is added
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.debug_logs import (
    flatten_cleaning_report,
    log_all_cleaning_reports,
    log_cleaning_report,
)
from src.data.utils import _load_tickers, _save_parquet
from src.utils.data_storage import _load_parquet
from src.data.cleaner import DataCleaner
from src.data.alpaca_ingestor import AlpacaIngestor
from src.utils.logger import LogFileMode, setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


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


UTC = "UTC"

FREQ_MAP = {
    "1Min": "1min",
    "5Min": "5min",
    "15Min": "15min",
    "1Hour": "1h",
    "1H": "1h",
    "1Day": "1D",
    "1D": "1D",
}


def normalize_freq(tf: str) -> str:
    try:
        return FREQ_MAP[tf]
    except KeyError:
        raise ValueError(f"Unsupported timeframe: {tf}")


def to_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def align_to_nyse_calendar(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    timeframe: str,
) -> pd.DataFrame:
    """Reindex an OHLCV DataFrame onto the canonical NYSE Regular Trading Hours
    (RTH, 09:30-16:00 ET) calendar for the requested timeframe.

    The resulting index is the union of all NYSE session bars between
    ``start_date`` and ``end_date`` at frequency ``timeframe``. Pre-market and
    after-hours bars are excluded by construction (NYSE schedule + RTH-only
    ``mcal.date_range``). Missing bars in the input are introduced as NaN rows
    so the downstream cleaner (``DataCleaner``) can apply bounded forward-fill
    up to the configured cap.

    SPY-relative alignment is achieved implicitly here: SPY trades on NYSE, so
    the NYSE calendar IS the SPY calendar. Reindexing every ticker onto this
    canonical NYSE RTH index produces a cross-ticker aligned panel without any
    explicit reference to SPY's series.

    Args:
        df: Raw OHLCV DataFrame with a UTC-aware DatetimeIndex.
        start_date: ISO date string, inclusive.
        end_date: ISO date string, inclusive.
        timeframe: Timeframe string accepted by ``FREQ_MAP`` (e.g. "1Min",
            "5Min", "15Min", "1Hour", "1Day").

    Returns:
        DataFrame reindexed onto the canonical NYSE RTH index. Bars present in
        ``df`` are preserved; missing bars are introduced as NaN rows.
    """
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)

    if schedule.empty:
        logger.warning(
            "NYSE schedule empty for %s..%s — returning input unmodified",
            start_date,
            end_date,
        )
        return df

    freq = normalize_freq(timeframe)

    if timeframe in ("1Day", "1D"):
        # For daily bars use one entry per session (session close in UTC).
        canonical_index = pd.DatetimeIndex(schedule["market_close"]).tz_convert("UTC")
    else:
        # ``mcal.date_range`` returns RTH-only timestamps in UTC for intraday
        # frequencies. This naturally excludes pre-market / after-hours.
        canonical_index = mcal.date_range(schedule, frequency=freq)
        canonical_index = pd.DatetimeIndex(canonical_index).tz_convert("UTC")

    canonical_index.name = df.index.name or "timestamp"

    # Ensure the input index is UTC-aware so reindex matches correctly.
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    else:
        df = df.copy()
        df.index = df.index.tz_convert("UTC")

    aligned = df.reindex(canonical_index)
    return aligned


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

    start_date = cfg.data.start_date
    end_date = cfg.data.end_date
    timeframes = ["1Min", "5Min", "15Min", "1Hour", "1Day"]

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
            skip_existing=args.skip_existing,
        )

    # ------------------------------------------------------------
    # STAGE 2: ALIGN + CLEAN PIPELINE
    # ------------------------------------------------------------
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

    plots_dir = Path("data/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_cleaning_rows = []

    for ticker in tickers:
        logger.info("===== PIPELINE START: %s =====", ticker)

        for tf in timeframes:
            logger.info("----- TIMEFRAME: %s -----", tf)

            # --------------------------------------------------------
            # LOAD RAW
            # --------------------------------------------------------
            raw_path = get_path(raw_dir, tf, ticker)
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
            logger.info(f"Start --> {start} || End --> {end}")

            aligned_df = align_to_nyse_calendar(
                raw_df,
                start_date=start_date,
                end_date=end_date,
                timeframe=tf,
            )

            # --------------------------------------------------------
            # CLEAN
            # --------------------------------------------------------
            cleaner = DataCleaner(max_gap_fill=5, ticker=f"{ticker}-{tf}", timeframe=tf)

            cleaned_df, cleaned_report = cleaner.clean(aligned_df)
            log_cleaning_report(cleaned_report)
            # store for cross-timeframe analysis
            all_cleaning_rows.append(
                flatten_cleaning_report(
                    reports=cleaned_report,
                    ticker=ticker,
                    feed="sip",
                    tf=tf,
                )
            )

            # --------------------------------------------------------
            # MAX MISSING FRACTION CHECK (warn, don't drop)
            # --------------------------------------------------------
            if len(cleaned_df) > 0:
                missing_frac = cleaned_df.isna().mean().mean()
                if missing_frac > cfg.data.max_missing_fraction:
                    logger.warning(
                        "Ticker %s [%s] missing fraction %.4f exceeds threshold %.4f",
                        ticker,
                        tf,
                        missing_frac,
                        cfg.data.max_missing_fraction,
                    )

            # --------------------------------------------------------
            # SAVE OUTPUTS (NO OVERWRITES)
            # --------------------------------------------------------
            _save_parquet(aligned_df, get_path(aligned_dir, tf, ticker))
            _save_parquet(cleaned_df, get_path(cleaned_dir, tf, ticker))
            _save_parquet(cleaned_df, get_path(processed_dir, tf, ticker))

    # Cross-ticker analysis runs ONCE after all tickers/timeframes are processed.
    log_all_cleaning_reports(all_cleaning_rows)
    all_cleaning_rows.clear()


if __name__ == "__main__":
    main()
