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
from typing import Dict, List

import numpy as np
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

from src.data.aligner import TickerAligner
from src.data.alpaca_ingestor import AlpacaIngestor
from src.data.cleaner import DataCleaner
from src.utils.logger import setup_logger
from src.utils.config_loader import Config, load_config
from constants import MAX_GAP_FILL_BARS

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


def run_ingest(args, cfg: Config, tickers: List[str]) -> None:
    AlpacaIngestor().download_universe(
        tickers,
        args.raw_dir,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        skip_existing=args.skip_existing,
        config=cfg,
    )


def run_clean(args, cfg: Config, tickers: List[str]) -> None:
    cleaner = DataCleaner()
    raw_dir = Path(args.raw_dir)
    cleaned_dir = Path(args.cleaned_dir)

    for ticker in tickers:
        raw_path = raw_dir / f"{ticker}.parquet"
        logger.info(f"Raw Path --> {raw_path} || Raw Directory --> {ticker}")
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw data missing for {ticker}. Run --mode ingest first."
            )
        df = cleaner.clean(AlpacaIngestor._load_bars(raw_path))
        _save_parquet(df, cleaned_dir / f"{ticker}.parquet")


def run_align(args, cfg: Config, tickers: List[str]) -> None:
    """
    DEPRECATED: Use run_clean_and_align_synchronized() instead.

    This function is kept for backward compatibility but will produce
    desynchronized data. See DATA_PIPELINE_AUDIT.md Issue #2.
    """
    logger.warning(
        "run_align() is deprecated and may produce desynchronized data. "
        "Use --mode clean_align_sync instead."
    )

    aligner = TickerAligner(
        benchmark_ticker=cfg.data.benchmark_ticker,
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
        dfs[ticker] = AlpacaIngestor._load_bars(path)

    # Determine field list from SPY
    spy_clean = AlpacaIngestor._load_bars(Path(args.cleaned_dir) / "SPY.parquet")
    fields = spy_clean.columns.tolist()
    logger.info("Aligning on fields: %s", fields)

    aligned = aligner.align(dfs=dfs, fields=fields)
    aligner.save_aligned(aligned_df=aligned, output_dir=str(processed_dir))

    for ticker, df in aligned.items():
        logger.info(
            "%-6s  rows=%-6d  %s --> %s  missing=%.4f%%",
            ticker,
            len(df),
            df.index.min(),
            df.index.max(),
            df.isna().mean() * 100,
        )


def run_clean_and_align_synchronized(args, cfg: Config, tickers: list[str]) -> None:
    """
    Clean and align with SYNCHRONIZED row removal (FIX Issue #2).

    CRITICAL ARCHITECTURE:
    1. Load all raw data
    2. Align to SPY index FIRST
    3. Apply forward-fill per-ticker (bounded, causal)
    4. Compute GLOBAL invalid mask (if ANY ticker invalid, mark row)
    5. Apply global mask (synchronized removal)
    6. Save cleaned, aligned data

    This ensures all tickers have IDENTICAL valid rows after cleaning.

    TRD Compliance: TRD1 §2 with synchronized cross-ticker enforcement
    """
    logger.info("=" * 80)
    logger.info("SYNCHRONIZED CLEAN-AND-ALIGN PIPELINE (Issue #2 Fix)")
    logger.info("=" * 80)

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    cleaner = DataCleaner(max_gap_fill=MAX_GAP_FILL_BARS)
    aligner = TickerAligner(benchmark_ticker=cfg.data.benchmark_ticker)

    # ========================================================================
    # STEP 1: LOAD ALL RAW DATA
    # ========================================================================
    logger.info("Step 1: Loading raw data for all tickers")

    dfs_raw = {}
    for ticker in tickers:
        path = raw_dir / f"{ticker}.parquet"
        if not path.exists():
            logger.warning(f"[{ticker}] Raw data missing: {path}")
            continue

        df = AlpacaIngestor._load_bars(path)
        dfs_raw[ticker] = df
        logger.info(f"[{ticker}] Loaded: {len(df)} rows")

    if not dfs_raw:
        raise ValueError("No raw data loaded - check raw_dir")

    logger.info(f"Loaded {len(dfs_raw)} tickers")

    # ========================================================================
    # STEP 2: ALIGN TO SPY INDEX (BEFORE CLEANING)
    # ========================================================================
    logger.info("Step 2: Aligning all tickers to SPY index")

    fields = ["open", "high", "low", "close", "volume"]
    aligned_raw = aligner.align(dfs_raw, fields=fields)

    logger.info(f"Alignment complete: {aligned_raw.shape}")

    # ========================================================================
    # STEP 3: APPLY BOUNDED FORWARD-FILL PER-TICKER
    # ========================================================================
    logger.info("Step 3: Applying bounded forward-fill (limit=5) per ticker")

    tickers_in_aligned = aligned_raw.columns.get_level_values("ticker").unique()

    for ticker in tickers_in_aligned:
        df_ticker = aligned_raw[ticker].copy()

        # Apply bounded forward-fill to this ticker
        for col in fields:
            if col not in df_ticker.columns:
                continue

            values = df_ticker[col].values.copy()
            last_valid = None
            gap_count = 0
            filled_count = 0

            for i in range(len(values)):
                if not np.isnan(values[i]):
                    # Reset on valid value
                    last_valid = values[i]
                    gap_count = 0
                else:
                    # Increment gap counter on NaN
                    gap_count += 1

                    # Fill if within limit and we have a valid value
                    if last_valid is not None and gap_count <= MAX_GAP_FILL_BARS:
                        values[i] = last_valid
                        filled_count += 1
                    # else: leave as NaN (gap too long or no prior value)

            df_ticker[col] = values

            remaining_nans = np.isnan(values).sum()
            logger.info(
                f"[{ticker}][{col}] Forward-filled {filled_count} values, "
                f"{remaining_nans} NaN remain"
            )

        # Update aligned_raw with filled values
        aligned_raw[ticker] = df_ticker

    logger.info("Forward-fill complete for all tickers")

    # ========================================================================
    # STEP 4: COMPUTE GLOBAL INVALID MASK (SYNCHRONIZED)
    # ========================================================================
    logger.info("Step 4: Computing global invalid mask (synchronized)")

    global_invalid = pd.Series(False, index=aligned_raw.index)

    for ticker in tickers_in_aligned:
        df_ticker = aligned_raw[ticker].copy()

        # If columns have MultiIndex, flatten to get just field names
        if isinstance(df_ticker.columns, pd.MultiIndex):
            df_ticker.columns = df_ticker.columns.get_level_values(-1)

        # Get invalid mask for this ticker (after forward-fill)
        # This will mark ALL remaining NaN as invalid
        ticker_invalid = cleaner.get_invalid_mask(df_ticker, after_forward_fill=True)

        # Merge into global mask (OR operation)
        global_invalid |= ticker_invalid

        logger.info(
            f"[{ticker}] Invalid rows: {ticker_invalid.sum()} "
            f"({100*ticker_invalid.sum()/len(df_ticker):.2f}%)"
        )

    logger.info(
        f"Global invalid rows: {global_invalid.sum()} "
        f"({100*global_invalid.sum()/len(aligned_raw):.2f}%)"
    )

    # ========================================================================
    # STEP 5: APPLY GLOBAL MASK (SYNCHRONIZED REMOVAL)
    # ========================================================================
    logger.info("Step 5: Applying global mask (synchronized row removal)")

    aligned_clean = aligned_raw[~global_invalid].copy()

    logger.info(
        f"Synchronized cleaning complete: "
        f"{len(aligned_raw)} → {len(aligned_clean)} rows "
        f"({global_invalid.sum()} removed)"
    )

    # ========================================================================
    # STEP 6: FINAL VALIDATION
    # ========================================================================
    logger.info("Step 6: Final validation")

    # Check for NaN in any ticker
    for ticker in tickers_in_aligned:
        df_ticker = aligned_clean[ticker]

        # Handle MultiIndex columns if present
        if isinstance(df_ticker.columns, pd.MultiIndex):
            df_ticker_flat = df_ticker.copy()
            df_ticker_flat.columns = df_ticker_flat.columns.get_level_values(-1)
        else:
            df_ticker_flat = df_ticker

        # Check for NaN in OHLCV fields
        fields_present = [f for f in fields if f in df_ticker_flat.columns]
        nan_count = df_ticker_flat[fields_present].isna().sum().sum()

        if nan_count > 0:
            # Show which columns have NaN
            nan_by_col = df_ticker_flat[fields_present].isna().sum()
            nan_cols = nan_by_col[nan_by_col > 0].to_dict()

            logger.error(
                f"[{ticker}] CRITICAL: {nan_count} NaN remaining after cleaning!"
            )
            logger.error(f"[{ticker}] NaN by column: {nan_cols}")

            # Show sample of rows with NaN
            nan_rows = df_ticker_flat[df_ticker_flat[fields_present].isna().any(axis=1)]
            logger.error(f"[{ticker}] Sample NaN rows (first 5):\n{nan_rows.head()}")

            raise ValueError(
                f"NaN present in {ticker} after synchronized cleaning. "
                f"This indicates a bug in the cleaning logic. "
                f"Columns with NaN: {list(nan_cols.keys())}"
            )

        logger.info(f"[{ticker}] Validation passed: 0 NaN")

    # ========================================================================
    # STEP 7: SAVE ALIGNED, CLEANED DATA
    # ========================================================================
    logger.info("Step 7: Saving synchronized, cleaned data")

    aligner.save_aligned(aligned_clean, output_dir=str(processed_dir))

    logger.info("=" * 80)
    logger.info("SYNCHRONIZED CLEAN-AND-ALIGN COMPLETE")
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest, clean, and align OHLCV data")
    parser.add_argument(
        "--mode",
        choices=[
            "ingest",
            "clean",
            "align",
            "all",
            "clean_and_align",
            "clean_align_sync",
        ],
        required=True,
        help=(
            "ingest: Download raw data | "
            "clean: Clean per-ticker (legacy) | "
            "align: Align cleaned data (legacy) | "
            "clean_align_sync: Synchronized clean+align (RECOMMENDED)"
        ),
    )
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
        logger.warning(
            "Using legacy per-ticker cleaning (may cause desynchronization). "
            "Consider using --mode clean_align_sync instead."
        )
        run_clean(args, cfg, tickers)
    elif args.mode == "align":
        logger.warning(
            "Using legacy alignment (expects per-ticker cleaned data). "
            "Consider using --mode clean_align_sync instead."
        )
        run_align(args, cfg, tickers)
    elif args.mode == "clean_align_sync":
        logger.info("Using SYNCHRONIZED clean+align (RECOMMENDED)")
        run_clean_and_align_synchronized(args, cfg, tickers)
    elif args.mode == "all":
        run_ingest(args, cfg, tickers)
        run_clean_and_align_synchronized(args, cfg, tickers)
    elif args.mode == "clean_and_align":
        logger.warning(
            "Using legacy clean+align sequence (may cause desynchronization). "
            "Consider using --mode clean_align_sync instead."
        )
        run_clean(args, cfg, tickers)
        run_align(args, cfg, tickers)
    else:
        logger.error("Invalid mode: %s", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
