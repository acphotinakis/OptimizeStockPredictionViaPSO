"""
src/features/statistical.py

Rolling statistical moment features computed from per-bar log returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

# Define rolling windows per feature type
rolling_specs = {
    "mean": [10, 20, 60, 120],
    "var": [10, 20, 60],
    "skew": [20, 60],
    "kurt": [20, 60],
}


def compute_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistical features from log returns for a single ticker.

    Args:
        df: DataFrame with 'log_return' and 'close'.

    Returns:
        DataFrame of statistical features.
    """
    r = df["log_return"]
    C = df["close"]
    out = pd.DataFrame(index=df.index)

    # ---- Vectorized rolling stats (mean, var, skew, kurt) ------------------
    unique_windows = sorted(set(sum(rolling_specs.values(), [])))
    for w in unique_windows:
        roll = r.rolling(window=w, min_periods=max(1, w // 2))
        # Compute all stats at once
        stats = roll.agg(["mean", "var", "skew", "kurt"])
        # Filter only required stats for this window
        stats = stats[[s for s in stats.columns if w in rolling_specs.get(s, [])]]
        stats.columns = [f"ret_{col}_{w}" for col in stats.columns]
        out = pd.concat([out, stats], axis=1)

    # ---- Lag-1 autocorrelation ---------------------------------------------
    # Precompute unique windows to minimize rolling calls
    for w in (20, 60):
        roll = r.rolling(window=w, min_periods=max(1, w // 2))
        out[f"ret_autocorr_1_{w}"] = roll.apply(_fast_autocorr, raw=True)

    # ---- Price range ratio (compression metric) ----------------------------
    for w in (20, 60):
        roll = C.rolling(window=w, min_periods=1)
        sma_w = roll.mean()
        high = roll.max()
        low = roll.min()
        out[f"range_ratio_{w}"] = (high - low) / (sma_w + 1e-10)

    # ---- Realized volatility (sum of squared returns) ----------------------
    for w in (10, 30):
        roll_sq = r.pow(2).rolling(window=w, min_periods=1)
        out[f"rv_{w}"] = np.sqrt(roll_sq.sum())

    # ---- Hurst exponent (R/S approximation) --------------------------------
    out["hurst_exp_60"] = _rolling_hurst(r, window=60)

    return out.fillna(0.0)


def _fast_autocorr(x):
    return np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0.0


def _rolling_hurst(r: pd.Series, window: int) -> pd.Series:
    """Approximate Hurst exponent via rescaled range (R/S) analysis."""

    # def hurst_single(x: np.ndarray) -> float:
    #     n = len(x)
    #     if n < 10:
    #         return 0.5
    #     mean_x = np.mean(x)
    #     deviations = np.cumsum(x - mean_x)
    #     R = deviations.max() - deviations.min()
    #     S = np.std(x, ddof=1)
    #     if S < 1e-10:
    #         return 0.5
    #     return np.log(R / S) / np.log(n)
    @njit
    def hurst_single(x):
        n = len(x)
        if n < 10:
            return 0.5
        mean_x = np.mean(x)
        dev = np.cumsum(x - mean_x)
        R = dev.max() - dev.min()
        S = np.std(x)
        if S < 1e-10:
            return 0.5
        return np.log(R / S) / np.log(n)

    return (
        r.rolling(window, min_periods=window // 2)
        .apply(hurst_single, raw=True)
        .fillna(0.5)
    )
