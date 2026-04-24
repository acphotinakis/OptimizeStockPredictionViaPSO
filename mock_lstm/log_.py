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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Log Messages
# ---------------------------------------------------------------------------


def _log_table_stats(df: pd.DataFrame, ticker: str):
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


def _log_ohlcv_validation_report(report: dict, ticker: str):
    from prettytable import PrettyTable

    table = PrettyTable()

    table.field_names = [
        "Ticker",
        "Initial Rows",
        "Final Rows",
        "Dropped",
        "Invalid %",
        "Missing Cols",
    ]

    initial = report.get("initial_rows", 0)
    final = report.get("final_rows", 0)
    dropped = report.get("dropped_count", report.get("invalid_count", 0))

    invalid_pct = (dropped / initial * 100) if initial > 0 else 0.0

    missing_cols = report.get("missing_columns", [])
    missing_str = ",".join(missing_cols) if missing_cols else "None"

    table.add_row(
        [
            ticker,
            initial,
            final,
            dropped,
            f"{invalid_pct:.2f}%",
            missing_str,
        ]
    )

    logger.info("\n%s", table)

    # ------------------------------------------------------------------
    # Breakdown (if available)
    # ------------------------------------------------------------------
    breakdown = report.get("breakdown")

    if breakdown:
        breakdown_table = PrettyTable()
        breakdown_table.field_names = ["Issue", "Count"]

        for k, v in breakdown.items():
            breakdown_table.add_row([k, v])

        logger.info(
            "\n=== OHLCV Validation Breakdown (%s) ===\n%s", ticker, breakdown_table
        )

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    logger.info(
        "OHLCV Summary | %s | initial=%d final=%d dropped=%d (%.2f%%)",
        ticker,
        initial,
        final,
        dropped,
        invalid_pct,
    )


def _log_observation_gap_report(report: dict, ticker: str):
    from prettytable import PrettyTable

    gap_start = report.get("gap_start")
    gap_sizes = report.get("gap_sizes")
    expected = report.get("expected_interval")

    logger.info("=== GAP REPORT (%s) ===", ticker)

    # ------------------------------------------------------------
    # Core summary
    # ------------------------------------------------------------
    total_gaps = int(gap_start.sum()) if gap_start is not None else 0
    max_gap = int(gap_sizes.max()) if gap_sizes is not None else 0
    median_gap = float(gap_sizes.median()) if gap_sizes is not None else 0.0

    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Value"]

    summary_table.add_row(["Expected Interval", str(expected)])
    summary_table.add_row(["Total Gap Starts", total_gaps])
    summary_table.add_row(["Max Gap Size", max_gap])
    summary_table.add_row(["Median Gap Size", f"{median_gap:.2f}"])

    logger.info("\n%s", summary_table)

    # ------------------------------------------------------------
    # Gap size distribution
    # ------------------------------------------------------------
    if gap_sizes is not None and len(gap_sizes) > 0:
        dist_table = PrettyTable()
        dist_table.field_names = ["Statistic", "Value"]

        dist_table.add_row(["Min", int(gap_sizes.min())])
        dist_table.add_row(["Median", f"{gap_sizes.median():.2f}"])
        dist_table.add_row(["Max", int(gap_sizes.max())])
        dist_table.add_row(["Mean", f"{gap_sizes.mean():.2f}"])

        logger.info("\n=== Gap Size Distribution (%s) ===\n%s", ticker, dist_table)

    # ------------------------------------------------------------
    # Sample gap locations
    # ------------------------------------------------------------
    if gap_start is not None and total_gaps > 0:
        sample_idx = list(gap_start[gap_start].index[:10])

        logger.info(
            "Gap start samples (%s) | first_10_indices=%s",
            ticker,
            sample_idx,
        )

    # ------------------------------------------------------------
    # Interpretation summary
    # ------------------------------------------------------------
    if total_gaps == 0:
        logger.info("No gaps detected (%s) | data is continuous", ticker)
    else:
        logger.info(
            "Gaps detected (%s) | %d gap segments found | max gap=%d",
            ticker,
            total_gaps,
            max_gap,
        )
