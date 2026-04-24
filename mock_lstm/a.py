import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

from mock_lstm import mock_cleaner
from src.data.aligner import TickerAligner
from src.data.alpaca_ingestor import AlpacaIngestor
from src.data.cleaner import DataCleaner
from src.utils.logger import setup_logger
from src.utils.config_loader import Config, load_config
from constants import MAX_GAP_FILL_BARS
from mock_lstm.log_ import _log_table_stats, _log_ohlcv_validation_report
from mock_lstm.mock_cleaner import MockDataCleaner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load/Save
# ---------------------------------------------------------------------------


def _load_tickers(path: str) -> list[str]:
    with open(path) as f:
        return [
            l.split()[0] for l in f if l.split() and not l.split()[0].startswith("#")
        ]


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=True)
    logger.info("Saved %s (%d rows)", path.name, len(df))


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)

    return df


def _compute_observation_gaps(df: pd.DataFrame) -> dict:
    """
    Classify gaps in time series data (observation-based).

    FIX Issue #3: For daily data, median time delta correctly handles
    weekends/holidays (2-4 day gaps are common and expected).
    Gap sizes are measured in multiples of the expected interval.

    The max_gap_fill=5 parameter means: 5 consecutive MISSING observations,
    not 5 days. For daily data with weekends, this typically means ~2 weeks.
    """
    logger.info("STEP 4 | Gap classification started | rows=%d", len(df))

    time_deltas = df.index.to_series().diff()

    expected = time_deltas.median()

    logger.info("Gap classification | inferred_expected_interval=%s", expected)

    gap_sizes = (time_deltas / expected).fillna(0).astype(int)

    gap_start = gap_sizes > 1

    gap_lengths = gap_sizes.cumsum()

    total_gaps = gap_start.sum()
    max_gap = gap_sizes.max()

    logger.info(
        "Gap classification summary | total_gap_starts=%d | max_gap_size=%d",
        total_gaps,
        max_gap,
    )

    # Optional deeper diagnostics
    if total_gaps > 0:
        logger.info(
            "Gap sizes distribution | min=%d | median=%.2f | max=%d",
            gap_sizes.min(),
            gap_sizes.median(),
            gap_sizes.max(),
        )

        logger.info("Gap start indices sample=%s", list(df.index[gap_start][:10]))

    return {
        "gap_start": gap_start,
        "gap_sizes": gap_sizes,
        "gap_lengths": gap_lengths,
        "expected_interval": expected,
    }


MAX_GAP_FILL_BARS = 5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest, clean, and align OHLCV data")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--tickers", default="config/tickers.txt")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/test")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    setup_logger(log_file="logs/testing.log", level="INFO")

    cfg = load_config(args.config)

    ticker = "AAPL"
    logger.info("Processing ticker: %s", ticker)

    raw_dir = Path(args.raw_dir)

    ticker_path = raw_dir / f"{ticker}.parquet"
    raw_df = _load_parquet(ticker_path)

    _log_table_stats(raw_df, ticker=ticker)

    cleaner = MockDataCleaner(max_gap_fill=5, ticker=ticker)
    cleaned_df = cleaner.clean(raw_df)


if __name__ == "__main__":
    main()
