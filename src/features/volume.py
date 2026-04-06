"""
src/features/volume.py

Volume, liquidity, and time-encoding features for 1-minute OHLCV data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_MINUTES = 390  # 09:30 – 16:00
TRADING_DAYS = 5


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 20 volume, liquidity, and temporal features.

    Args:
        df: Single-ticker OHLCV DataFrame with 'session_minute' and 'log_return'
            columns (added by DataCleaner).

    Returns:
        DataFrame of volume feature columns.
    """
    out = pd.DataFrame(index=df.index)
    C, H, L, V = df["close"], df["high"], df["low"], df["volume"]
    r = df["log_return"]
    sm = df.get("session_minute", pd.Series(0, index=df.index))

    # ---- VWAP (session reset) -----------------------------------------------
    tp = (H + L + C) / 3
    session_start = sm == 0
    # Accumulate within session
    cumulative_tp_vol = (tp * V).groupby(session_start.cumsum()).cumsum()
    cumulative_vol = V.groupby(session_start.cumsum()).cumsum()
    vwap = cumulative_tp_vol / (cumulative_vol + 1e-10)
    out["vwap"] = vwap
    out["price_to_vwap"] = C / (vwap + 1e-10)
    out["vwap_dev"] = (C - vwap) / (vwap + 1e-10)

    # ---- Relative Volume ---------------------------------------------------
    sma_vol_20 = V.rolling(20, min_periods=1).mean()
    sma_vol_60 = V.rolling(60, min_periods=1).mean()
    out["rvol_20"] = V / (sma_vol_20 + 1e-10)
    out["rvol_60"] = V / (sma_vol_60 + 1e-10)

    # ---- On-Balance Volume --------------------------------------------------
    obv_delta = np.where(C > C.shift(1), V, np.where(C < C.shift(1), -V, 0))
    obv = pd.Series(np.cumsum(obv_delta), index=df.index)
    out["obv"] = obv
    out["obv_ema_20"] = obv.ewm(span=20, adjust=False).mean()
    out["obv_momentum_20"] = obv.diff(20) / (obv.abs().rolling(20).mean() + 1e-10)

    # ---- Accumulation / Distribution Line ----------------------------------
    clv = ((C - L) - (H - C)) / (H - L + 1e-10)
    adl = (clv * V).cumsum()
    out["adl"] = adl
    out["adl_slope_10"] = adl.diff(10) / (adl.abs().rolling(10).mean() + 1e-10)

    # ---- Chaikin Money Flow -------------------------------------------------
    pos = (clv * V).rolling(20, min_periods=1).sum()
    vol_sum = V.rolling(20, min_periods=1).sum()
    out["cmf_20"] = pos / (vol_sum + 1e-10)

    # ---- Force Index --------------------------------------------------------
    out["force_1"] = r * V
    out["force_ema_13"] = out["force_1"].ewm(span=13, adjust=False).mean()

    # ---- Ease of Movement --------------------------------------------------
    hl_mid = (H + L) / 2
    hl_mid_prev = hl_mid.shift(1)
    box = V / (H - L + 1e-10)
    out["eom_14"] = (
        ((hl_mid - hl_mid_prev) / (box + 1e-10)).rolling(14, min_periods=1).mean()
    )

    # ---- Volume Price Trend ------------------------------------------------
    out["vpt"] = (r * V).cumsum()

    # ---- Intraday Turnover Velocity ----------------------------------------
    out["itv_20"] = V / (V.rolling(20, min_periods=1).sum() + 1e-10)

    # ---- Bar Activity Score ------------------------------------------------
    out["bas"] = np.log1p(V) * (H - L) / (C.shift(1) + 1e-10)

    # ---- Time-of-Day cyclical encoding -------------------------------------
    tod = sm / TRADING_MINUTES
    out["tod_sin"] = np.sin(2 * np.pi * tod)
    out["tod_cos"] = np.cos(2 * np.pi * tod)

    # ---- Day-of-Week cyclical encoding ------------------------------------
    et_index = df.index.tz_convert("America/New_York")
    dow = et_index.dayofweek.values.astype(float)  # Mon=0 … Fri=4
    out["dow_sin"] = np.sin(2 * np.pi * dow / TRADING_DAYS)
    out["dow_cos"] = np.cos(2 * np.pi * dow / TRADING_DAYS)

    return out.fillna(0.0)
