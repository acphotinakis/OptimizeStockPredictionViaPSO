"""
src/features/statistical.py

Rolling statistical moment features computed from per-bar log returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 20 rolling statistical features from log returns.

    Args:
        df: Single-ticker DataFrame containing 'log_return' and 'close'.

    Returns:
        DataFrame of statistical feature columns.
    """
    out = pd.DataFrame(index=df.index)
    r = df["log_return"]
    C = df["close"]

    # ---- Rolling mean of returns -------------------------------------------
    for w in (10, 20, 60, 120):
        out[f"ret_mean_{w}"] = r.rolling(w, min_periods=max(1, w // 2)).mean()

    # ---- Rolling variance ---------------------------------------------------
    for w in (10, 20, 60):
        out[f"ret_var_{w}"] = r.rolling(w, min_periods=max(1, w // 2)).var()

    # ---- Rolling skewness ---------------------------------------------------
    for w in (20, 60):
        out[f"ret_skew_{w}"] = r.rolling(w, min_periods=max(1, w // 2)).skew()

    # ---- Rolling excess kurtosis --------------------------------------------
    for w in (20, 60):
        out[f"ret_kurt_{w}"] = r.rolling(w, min_periods=max(1, w // 2)).kurt()

    # ---- Lag-1 autocorrelation ---------------------------------------------
    for w in (20, 60):
        out[f"ret_autocorr_1_{w}"] = r.rolling(w, min_periods=max(1, w // 2)).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 2 else 0.0,
            raw=True,
        )

    # ---- Price range ratio (compression metric) ----------------------------
    for w in (20, 60):
        sma_w = C.rolling(w, min_periods=1).mean()
        out[f"range_ratio_{w}"] = (
            C.rolling(w, min_periods=1).max() - C.rolling(w, min_periods=1).min()
        ) / (sma_w + 1e-10)

    # ---- Realized volatility (sum of squared returns) ----------------------
    for w in (10, 30):
        out[f"rv_{w}"] = np.sqrt(r.pow(2).rolling(w, min_periods=1).sum())

    # ---- Hurst exponent (R/S approximation) --------------------------------
    out["hurst_exp_60"] = _rolling_hurst(r, window=60)

    return out.fillna(0.0)


def _rolling_hurst(r: pd.Series, window: int) -> pd.Series:
    """Approximate Hurst exponent via rescaled range (R/S) analysis."""

    def hurst_single(x: np.ndarray) -> float:
        n = len(x)
        if n < 10:
            return 0.5
        mean_x = np.mean(x)
        deviations = np.cumsum(x - mean_x)
        R = deviations.max() - deviations.min()
        S = np.std(x, ddof=1)
        if S < 1e-10:
            return 0.5
        return np.log(R / S) / np.log(n)

    return (
        r.rolling(window, min_periods=window // 2)
        .apply(hurst_single, raw=True)
        .fillna(0.5)
    )
