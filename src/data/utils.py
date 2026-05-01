"""Helpers shared across the data pipeline (parquet I/O, ticker lists,
timeframe parsing).
"""

import logging
from pathlib import Path

import pandas as pd

from src.utils.data_storage import _load_parquet

logger = logging.getLogger(__name__)


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
        # 2. Convert UTC --> America/New_York (DST-aware)
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
    # logger.info("Saved %s (%d rows)", path.relative_to, len(df))
    logger.info("Saved %s (%d rows)", path, len(df))


def _load_tickers(path: str) -> list[str]:
    with open(path) as f:
        return [
            l.split()[0] for l in f if l.split() and not l.split()[0].startswith("#")
        ]


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
