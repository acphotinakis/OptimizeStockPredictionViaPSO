"""features/statistical.py — Rolling statistical moment features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    r = df["log_return"]
    C = df["close"]
    out = pd.DataFrame(index=df.index)

    # ── Rolling mean / var / skew / kurt ─────────────────────────────────────
    # Spec: mean@[10,20,60,120]  var@[10,20,60]  skew@[20,60]  kurt@[20,60]
    for w, stats in {
        10: ["mean", "var"],
        20: ["mean", "var", "skew", "kurt"],
        60: ["mean", "var", "skew", "kurt"],
        120: ["mean"],
    }.items():
        roll = r.rolling(w, min_periods=max(1, w // 2))
        for stat in stats:
            out[f"ret_{stat}_{w}"] = getattr(roll, stat)()

    # ── Lag-1 autocorrelation ─────────────────────────────────────────────────
    for w in (20, 60):
        out[f"ret_autocorr_1_{w}"] = r.rolling(w, min_periods=max(1, w // 2)).apply(
            _fast_autocorr, raw=True
        )

    # ── Price range ratio ─────────────────────────────────────────────────────
    for w in (20, 60):
        roll_C = C.rolling(w, min_periods=1)
        out[f"range_ratio_{w}"] = (roll_C.max() - roll_C.min()) / (
            roll_C.mean() + 1e-10
        )

    # ── Realized volatility ───────────────────────────────────────────────────
    r_sq = r**2
    for w in (10, 30):
        out[f"rv_{w}"] = np.sqrt(r_sq.rolling(w, min_periods=1).sum())

    # ── Hurst exponent (R/S) ──────────────────────────────────────────────────
    out["hurst_exp_60"] = (
        r.rolling(60, min_periods=30).apply(_hurst_single, raw=True).fillna(0.5)
    )
    logger.info(
        f"Computed {len(out.columns)} statistical features for {len(out)} samples "
        f"(columns: {list(out.columns.tolist())}...)"
    )

    return out.fillna(0.0)


def _fast_autocorr(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 else 0.0


@njit
def _hurst_single(x: np.ndarray) -> float:
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
