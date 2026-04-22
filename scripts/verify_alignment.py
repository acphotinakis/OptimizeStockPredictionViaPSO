#!/usr/bin/env python3
"""
Verify synchronized cleaning produces identical timestamps across all tickers.

Usage:
    python scripts/verify_alignment.py --processed-dir data/processed
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Resolve project root
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def verify_alignment(processed_dir: Path) -> bool:
    """
    Verify all tickers have identical timestamps.
    
    Args:
        processed_dir: Directory containing aligned parquet files
    
    Returns:
        True if all tickers aligned, False otherwise
    """
    logger.info("=" * 80)
    logger.info("ALIGNMENT VERIFICATION")
    logger.info("=" * 80)
    
    # Find all parquet files
    parquet_files = sorted(processed_dir.glob("*.parquet"))
    
    if not parquet_files:
        logger.error(f"No parquet files found in {processed_dir}")
        return False
    
    logger.info(f"Found {len(parquet_files)} ticker files")
    
    # Load all indices
    indices = {}
    for file in parquet_files:
        ticker = file.stem
        df = pd.read_parquet(file)
        indices[ticker] = df.index
        logger.info(f"[{ticker}] {len(df)} rows | {df.index.min()} → {df.index.max()}")
    
    # Compare all indices
    tickers = list(indices.keys())
    reference_ticker = tickers[0]
    reference_index = indices[reference_ticker]
    
    all_aligned = True
    
    for ticker in tickers[1:]:
        if not indices[ticker].equals(reference_index):
            logger.error(
                f"❌ [{ticker}] NOT ALIGNED with [{reference_ticker}]"
            )
            
            # Show differences
            only_in_ref = reference_index.difference(indices[ticker])
            only_in_ticker = indices[ticker].difference(reference_index)
            
            if len(only_in_ref) > 0:
                logger.error(
                    f"   Dates in [{reference_ticker}] but not [{ticker}]: "
                    f"{len(only_in_ref)} (first 5: {list(only_in_ref[:5])})"
                )
            
            if len(only_in_ticker) > 0:
                logger.error(
                    f"   Dates in [{ticker}] but not [{reference_ticker}]: "
                    f"{len(only_in_ticker)} (first 5: {list(only_in_ticker[:5])})"
                )
            
            all_aligned = False
        else:
            logger.info(f"✅ [{ticker}] aligned with [{reference_ticker}]")
    
    # Summary
    logger.info("=" * 80)
    if all_aligned:
        logger.info("✅ SUCCESS: All tickers have IDENTICAL timestamps")
        logger.info(f"   Total tickers: {len(tickers)}")
        logger.info(f"   Total rows: {len(reference_index)}")
        logger.info(f"   Date range: {reference_index.min()} → {reference_index.max()}")
    else:
        logger.error("❌ FAILURE: Tickers are NOT synchronized")
        logger.error("   This indicates a bug in synchronized cleaning")
    logger.info("=" * 80)
    
    return all_aligned


def main():
    parser = argparse.ArgumentParser(
        description="Verify synchronized cleaning alignment"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing aligned parquet files"
    )
    
    args = parser.parse_args()
    
    if not args.processed_dir.exists():
        logger.error(f"Processed directory does not exist: {args.processed_dir}")
        sys.exit(1)
    
    success = verify_alignment(args.processed_dir)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
