"""features/volume.py — Volume, liquidity, and time-encoding features."""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_MINUTES = 390  # 09:30–16:00
_TRADING_DAYS = 5


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    C, H, L, V = df["close"], df["high"], df["low"], df["volume"]
    r = df["log_return"]
    sm = df.get("session_minute", pd.Series(0, index=df.index))
    out = pd.DataFrame(index=df.index)

    # ── Shared intermediates ─────────────────────────────────────────────────
    tp = (H + L + C) / 3
    clv = ((C - L) - (H - C)) / (H - L + 1e-10)  # close location value

    # ── VWAP (session-reset) ─────────────────────────────────────────────────
    session_id = (sm == 0).cumsum()
    cum_tp_vol = (tp * V).groupby(session_id).cumsum()
    cum_vol = V.groupby(session_id).cumsum()
    vwap = cum_tp_vol / (cum_vol + 1e-10)
    out["vwap"] = vwap
    out["price_to_vwap"] = C / (vwap + 1e-10)
    out["vwap_dev"] = (C - vwap) / (vwap + 1e-10)

    # ── Relative Volume ───────────────────────────────────────────────────────
    out["rvol_20"] = V / (V.rolling(20, min_periods=1).mean() + 1e-10)
    out["rvol_60"] = V / (V.rolling(60, min_periods=1).mean() + 1e-10)

    # ── On-Balance Volume ─────────────────────────────────────────────────────
    obv = pd.Series(
        np.cumsum(np.where(C > C.shift(1), V, np.where(C < C.shift(1), -V, 0))),
        index=df.index,
    )
    out["obv"] = obv
    out["obv_ema_20"] = obv.ewm(span=20, adjust=False).mean()
    out["obv_momentum_20"] = obv.diff(20) / (obv.abs().rolling(20).mean() + 1e-10)

    # ── Accumulation/Distribution Line ───────────────────────────────────────
    adl = (clv * V).cumsum()
    out["adl"] = adl
    out["adl_slope_10"] = adl.diff(10) / (adl.abs().rolling(10).mean() + 1e-10)

    # ── Chaikin Money Flow ────────────────────────────────────────────────────
    out["cmf_20"] = (clv * V).rolling(20, min_periods=1).sum() / (
        V.rolling(20, min_periods=1).sum() + 1e-10
    )

    # ── Force Index ───────────────────────────────────────────────────────────
    force = r * V
    out["force_1"] = force
    out["force_ema_13"] = force.ewm(span=13, adjust=False).mean()

    # ── Ease of Movement ─────────────────────────────────────────────────────
    hl_mid = (H + L) / 2
    out["eom_14"] = (
        ((hl_mid - hl_mid.shift(1)) / (V / (H - L + 1e-10) + 1e-10))
        .rolling(14, min_periods=1)
        .mean()
    )

    # ── Volume Price Trend ────────────────────────────────────────────────────
    out["vpt"] = (r * V).cumsum()

    # ── Intraday Turnover Velocity ────────────────────────────────────────────
    out["itv_20"] = V / (V.rolling(20, min_periods=1).sum() + 1e-10)

    # ── Bar Activity Score ────────────────────────────────────────────────────
    out["bas"] = np.log1p(V) * (H - L) / (C.shift(1) + 1e-10)

    # ── Time-of-Day encoding ──────────────────────────────────────────────────
    tod = sm / _TRADING_MINUTES
    out["tod_sin"] = np.sin(2 * np.pi * tod)
    out["tod_cos"] = np.cos(2 * np.pi * tod)

    # ── Day-of-Week encoding ──────────────────────────────────────────────────
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    dow = idx.tz_convert("America/New_York").dayofweek.values.astype(float)
    out["dow_sin"] = np.sin(2 * np.pi * dow / _TRADING_DAYS)
    out["dow_cos"] = np.cos(2 * np.pi * dow / _TRADING_DAYS)

    return out.fillna(0.0)
