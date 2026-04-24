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

from src.data.aligner import TickerAligner
from src.data.alpaca_ingestor import AlpacaIngestor
from src.data.cleaner import DataCleaner
from src.utils.logger import setup_logger
from src.utils.config_loader import Config, load_config
from constants import MAX_GAP_FILL_BARS
from .log_ import _log_table_stats, _log_ohlcv_validation_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Inspection
# ---------------------------------------------------------------------------
def _validate_ohlcv(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    logger.info(
        "OHLCV validation started | rows=%d | cols=%d",
        len(df),
        len(df.columns),
    )

    required = ["open", "high", "low", "close", "volume"]

    report: Dict = {
        "initial_rows": len(df),
        "missing_columns": [],
        "invalid_count": 0,
        "valid_count": 0,
        "dropped_count": 0,
    }

    # ------------------------------------------------------------------
    # Column validation
    # ------------------------------------------------------------------
    missing = list(set(required) - set(df.columns))
    report["missing_columns"] = missing

    if missing:
        logger.error("OHLCV validation failed | missing_columns=%s", missing)
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("OHLCV validation | required columns present=%s", required)

    # ------------------------------------------------------------------
    # Row validation mask
    # ------------------------------------------------------------------
    mask = (
        (df["high"] >= df["low"])
        & (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
        & (df["close"] >= df["low"])
        & (df["close"] <= df["high"])
    )

    invalid_mask = ~mask

    invalid_count = int(invalid_mask.sum())
    valid_count = int(mask.sum())

    report["invalid_count"] = invalid_count
    report["valid_count"] = valid_count
    report["drop_rate"] = invalid_count / len(df) if len(df) > 0 else 0.0

    # ------------------------------------------------------------------
    # Breakdown stats
    # ------------------------------------------------------------------
    if invalid_count > 0:
        report["breakdown"] = {
            "high_lt_low": int((df["high"] < df["low"]).sum()),
            "open_le_0": int((df["open"] <= 0).sum()),
            "high_le_0": int((df["high"] <= 0).sum()),
            "low_le_0": int((df["low"] <= 0).sum()),
            "close_le_0": int((df["close"] <= 0).sum()),
            "volume_lt_0": int((df["volume"] < 0).sum()),
            "close_out_of_bounds": int(
                ((df["close"] < df["low"]) | (df["close"] > df["high"])).sum()
            ),
        }

        logger.info(
            "Dropping invalid OHLCV rows | dropped=%d | remaining=%d",
            invalid_count,
            valid_count,
        )

    # ------------------------------------------------------------------
    # Apply filter
    # ------------------------------------------------------------------
    df = df[mask].copy()

    report["dropped_count"] = invalid_count
    report["final_rows"] = len(df)

    logger.info(
        "OHLCV validation complete | rows_before=%d | rows_after=%d | dropped=%d",
        report["initial_rows"],
        report["final_rows"],
        report["dropped_count"],
    )

    return df, report


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


# ---------------------------------------------------------------------------
# Split Data
# ---------------------------------------------------------------------------


def temporal_split_all(
    dfs: Dict[str, pd.DataFrame],
    train_pct: float = 0.70,
    val_pct: float = 0.10,
    test_pct: float = 0.20,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Perform chronological split on ALL tickers simultaneously.

    CRITICAL: This must happen BEFORE any fitting operations.

    Args:
        dfs: Dictionary of ticker -> DataFrame
        train_pct: Training fraction (default 0.70)
        val_pct: Validation fraction (default 0.10)
        test_pct: Test fraction (default 0.20)

    Returns:
        Tuple of (dfs_train, dfs_val, dfs_test)
    """
    if abs(train_pct + val_pct + test_pct - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {train_pct + val_pct + test_pct}"
        )

    dfs_train = {}
    dfs_val = {}
    dfs_test = {}

    for ticker, df in dfs.items():
        N = len(df)
        train_end = int(N * train_pct)
        val_end = int(N * (train_pct + val_pct))

        dfs_train[ticker] = df.iloc[:train_end].copy()
        dfs_val[ticker] = df.iloc[train_end:val_end].copy()
        dfs_test[ticker] = df.iloc[val_end:].copy()

        logger.info(
            f"[{ticker}] Split: train={len(dfs_train[ticker])} "
            f"val={len(dfs_val[ticker])} test={len(dfs_test[ticker])}"
        )

    return dfs_train, dfs_val, dfs_test


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
    ticker_data_raw = _load_parquet(ticker_path)

    _log_table_stats(ticker_data_raw, ticker=ticker)


if __name__ == "__main__":
    main()
