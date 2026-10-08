"""Tests verifying strict chronological splitting and prevention of data leakage."""

import pandas as pd
import pytest

from src.evaluation.canonical_split import (
    compute_canonical_split,
    verify_split_integrity,
)


def create_synthetic_market_data(n_bars: int = 1000) -> pd.DataFrame:
    """Create synthetic 1-minute OHLCV series with sequential timestamps."""
    dates = pd.date_range("2026-01-01 09:30", periods=n_bars, freq="1min")
    df = pd.DataFrame(
        {
            "open": [100.0 + i * 0.01 for i in range(n_bars)],
            "high": [100.05 + i * 0.01 for i in range(n_bars)],
            "low": [99.95 + i * 0.01 for i in range(n_bars)],
            "close": [100.02 + i * 0.01 for i in range(n_bars)],
            "volume": [1000 + i for i in range(n_bars)],
        },
        index=dates,
    )
    return df


def test_splits_are_strictly_chronological():
    """Ensure train, validation, and test splits are strictly ordered in time."""
    df = create_synthetic_market_data(1000)
    train_df, val_df, test_df = compute_canonical_split(
        df, train_pct=0.7, val_pct=0.15, test_pct=0.15
    )

    # Chronological ordering checks
    assert train_df.index.max() < val_df.index.min(), (
        f"Leakage: train max {train_df.index.max()} >= val min {val_df.index.min()}"
    )
    assert val_df.index.max() < test_df.index.min(), (
        f"Leakage: val max {val_df.index.max()} >= test min {test_df.index.min()}"
    )

    # Disjoint indices check
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)
    assert train_idx.isdisjoint(val_idx), "Train and validation indices overlap"
    assert val_idx.isdisjoint(test_idx), "Validation and test indices overlap"
    assert train_idx.isdisjoint(test_idx), "Train and test indices overlap"


def test_split_integrity_verification():
    """Test verify_split_integrity helper rejects overlapping or leaked splits."""
    df = create_synthetic_market_data(500)
    train_df, val_df, test_df = compute_canonical_split(df)

    # Canonical split should pass integrity verification
    assert verify_split_integrity(train_df, val_df, test_df) is True

    # Intentionally corrupted / leaked split should fail or raise
    leaked_val = df.iloc[200:400]
    leaked_train = df.iloc[0:300]  # overlaps with val
    with pytest.raises(AssertionError):
        verify_split_integrity(leaked_train, leaked_val, test_df)
