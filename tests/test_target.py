"""Tests verifying the canonical target computation formula and alignment."""

import numpy as np
import pandas as pd
import pytest

from src.features.target import compute_canonical_target


def test_target_formula_and_alignment():
    """Verify y[t] = log(Close[t+1] / Close[t]) and horizon shift."""
    closes = pd.Series([100.0, 105.0, 102.0, 110.0])
    target = compute_canonical_target(closes, horizon=1, method="log_return")

    # y[0] should be log(105 / 100)
    expected_y0 = np.log(105.0 / 100.0)
    assert np.isclose(target.iloc[0], expected_y0)

    # y[1] should be log(102 / 105)
    expected_y1 = np.log(102.0 / 105.0)
    assert np.isclose(target.iloc[1], expected_y1)

    # Last element should be NaN (shifted out of bounds)
    assert pd.isna(target.iloc[-1])


def test_target_horizon_validation():
    """Ensure invalid horizons and unsupported methods raise ValueError."""
    closes = pd.Series([100.0, 105.0])
    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        compute_canonical_target(closes, horizon=0)

    with pytest.raises(ValueError, match="Unsupported method"):
        compute_canonical_target(closes, horizon=1, method="invalid_method")
