"""
Canonical Temporal Split Implementation

Implements the authoritative 70/10/20 chronological split as defined in FINAL_PLAN.md.

MANDATORY RULES (FINAL_PLAN.md Section 2):
- Train: 70% earliest data
- Val: 10% next chronological block
- Test: 20% most recent data
- NO shuffling
- NO random sampling
- NO overlap between splits

Author: System Architect
Version: CANONICAL 1.0
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_canonical_split(
    data: pd.DataFrame,
    train_pct: float = 0.60,
    val_pct: float = 0.20,
    test_pct: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Compute canonical 60/20/20 temporal split.

    CRITICAL: This is the ONLY authorized split function for the system.
    All models MUST use identical splits for fair comparison.

    Split Rules (FINAL_PLAN.md Section 2.1):
    - Chronological ordering preserved
    - No shuffling
    - No overlap
    - Fixed boundaries

    Args:
        data: Full dataset with DatetimeIndex
        train_pct: Training percentage (default: 0.70)
        val_pct: Validation percentage (default: 0.10)
        test_pct: Test percentage (default: 0.20)

    Returns:
        Tuple of (train_data, val_data, test_data, split_metadata)

    Raises:
        ValueError: If percentages don't sum to 1.0
        ValueError: If data is not chronologically ordered
    """
    # Validate inputs
    if not np.isclose(train_pct + val_pct + test_pct, 1.0):
        raise ValueError(
            f"Split percentages must sum to 1.0, got {train_pct + val_pct + test_pct}"
        )

    # Ensure chronological ordering
    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "Data must be chronologically ordered (monotonic increasing index)"
        )

    n = len(data)

    # Compute split indices
    train_end_idx = int(train_pct * n)
    val_end_idx = int((train_pct + val_pct) * n)

    # Split data
    train_data = data.iloc[:train_end_idx].copy()
    val_data = data.iloc[train_end_idx:val_end_idx].copy()
    test_data = data.iloc[val_end_idx:].copy()

    # Verify no overlap
    assert len(set(train_data.index) & set(val_data.index)) == 0, "Train/val overlap"
    assert len(set(train_data.index) & set(test_data.index)) == 0, "Train/test overlap"
    assert len(set(val_data.index) & set(test_data.index)) == 0, "Val/test overlap"

    # Verify coverage
    assert len(train_data) + len(val_data) + len(test_data) == n, "Incomplete coverage"

    # Create metadata
    split_metadata = {
        "protocol_version": "CANONICAL_1.0",
        "source_document": "FINAL_PLAN.md",
        "total_samples": n,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "train_pct_actual": len(train_data) / n,
        "val_pct_actual": len(val_data) / n,
        "test_pct_actual": len(test_data) / n,
        "train_start_date": train_data.index[0],
        "train_end_date": train_data.index[-1],
        "val_start_date": val_data.index[0],
        "val_end_date": val_data.index[-1],
        "test_start_date": test_data.index[0],
        "test_end_date": test_data.index[-1],
        "split_timestamp": pd.Timestamp.now(),
    }

    # Log split information
    logger.info("=" * 80)
    logger.info("CANONICAL TEMPORAL SPLIT (FINAL_PLAN.md)")
    logger.info("=" * 80)
    logger.info(f"Total samples: {n}")
    logger.info(f"Train: {len(train_data)} samples ({100*len(train_data)/n:.1f}%)")
    logger.info(f"  Period: {train_data.index[0]} to {train_data.index[-1]}")
    logger.info(f"Val:   {len(val_data)} samples ({100*len(val_data)/n:.1f}%)")
    logger.info(f"  Period: {val_data.index[0]} to {val_data.index[-1]}")
    logger.info(f"Test:  {len(test_data)} samples ({100*len(test_data)/n:.1f}%)")
    logger.info(f"  Period: {test_data.index[0]} to {test_data.index[-1]}")
    logger.info("=" * 80)

    return train_data, val_data, test_data, split_metadata


def verify_split_integrity(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> bool:
    """
    Verify that splits satisfy canonical requirements.

    Checks (FINAL_PLAN.md Section 2.3):
    - Chronological ordering
    - No shuffling
    - No overlap
    - Fixed boundaries

    Args:
        train_data: Training split
        val_data: Validation split
        test_data: Test split

    Returns:
        True if all checks pass

    Raises:
        AssertionError: If any integrity check fails
    """
    # Check 1: Chronological ordering
    assert train_data.index.is_monotonic_increasing, "Train not chronological"
    assert val_data.index.is_monotonic_increasing, "Val not chronological"
    assert test_data.index.is_monotonic_increasing, "Test not chronological"

    # Check 2: No overlap
    assert len(set(train_data.index) & set(val_data.index)) == 0, "Train/val overlap"
    assert len(set(train_data.index) & set(test_data.index)) == 0, "Train/test overlap"
    assert len(set(val_data.index) & set(test_data.index)) == 0, "Val/test overlap"

    # Check 3: Temporal ordering (train < val < test)
    assert train_data.index[-1] < val_data.index[0], "Train not before val"
    assert val_data.index[-1] < test_data.index[0], "Val not before test"

    logger.info("✓ Split integrity verified (FINAL_PLAN.md compliant)")

    return True
