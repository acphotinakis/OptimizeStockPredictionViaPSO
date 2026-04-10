"""features/technical.py — 42 technical indicator features from OHLCV."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    C, H, L, O, V = df["close"], df["high"], df["low"], df["open"], df["volume"]
    out = pd.DataFrame(index=df.index)

    # ── Shared intermediates (compute once, reuse everywhere) ────────────────
    prev_C = C.shift(1)
    tp = (H + L + C) / 3
    tr = pd.concat([(H - L), (H - prev_C).abs(), (L - prev_C).abs()], axis=1).max(
        axis=1
    )
    ema12 = C.ewm(span=12, adjust=False).mean()
    ema26 = C.ewm(span=26, adjust=False).mean()
    sma20 = C.rolling(20, min_periods=1).mean()
    std20 = C.rolling(20, min_periods=1).std()
    low14 = L.rolling(14, min_periods=1).min()
    high14 = H.rolling(14, min_periods=1).max()
    atr14 = tr.rolling(14, min_periods=1).mean()

    # ── Moving averages ──────────────────────────────────────────────────────
    out["sma_5"] = C.rolling(5, min_periods=1).mean()
    out["sma_20"] = sma20
    out["sma_60"] = C.rolling(60, min_periods=1).mean()
    out["ema_12"] = ema12
    out["ema_26"] = ema26
    out["ema_60"] = C.ewm(span=60, adjust=False).mean()

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd - macd_sig

    # ── RSI ──────────────────────────────────────────────────────────────────
    for period in (14, 30):
        delta = C.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        out[f"rsi_{period}"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    out["bb_upper"] = sma20 + 2 * std20
    out["bb_lower"] = sma20 - 2 * std20
    out["bb_width"] = 4 * std20 / (sma20 + 1e-10)
    out["bb_pct_b"] = (C - out["bb_lower"]) / (
        out["bb_upper"] - out["bb_lower"] + 1e-10
    )

    # ── Stochastic ────────────────────────────────────────────────────────────
    stoch_k = 100 * (C - low14) / (high14 - low14 + 1e-10)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_k.rolling(3, min_periods=1).mean()

    # ── ATR ───────────────────────────────────────────────────────────────────
    out["atr_14"] = atr14
    out["atr_30"] = tr.rolling(30, min_periods=1).mean()

    # ── CCI ───────────────────────────────────────────────────────────────────
    sma_tp20 = tp.rolling(20, min_periods=1).mean()
    mad20 = tp.rolling(20, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    out["cci_20"] = (tp - sma_tp20) / (0.015 * mad20 + 1e-10)

    # ── Williams %R ───────────────────────────────────────────────────────────
    out["williams_r"] = -100 * (high14 - C) / (high14 - low14 + 1e-10)

    # ── Rate of Change ────────────────────────────────────────────────────────
    out["roc_5"] = C.pct_change(5, fill_method=None) * 100
    out["roc_10"] = C.pct_change(10, fill_method=None) * 100

    # ── Money Flow Index ──────────────────────────────────────────────────────
    mf = tp * V
    tp_prev = tp.shift(1)
    for period in (14, 30):
        pos_sum = mf.where(tp > tp_prev, 0.0).rolling(period, min_periods=1).sum()
        neg_sum = mf.where(tp < tp_prev, 0.0).rolling(period, min_periods=1).sum()
        out[f"mfi_{period}"] = 100 - 100 / (1 + pos_sum / (neg_sum + 1e-10))

    # ── Ichimoku (chikou omitted — look-ahead bias) ───────────────────────────
    tenkan = (H.rolling(9).max() + L.rolling(9).min()) / 2
    kijun = (H.rolling(26).max() + L.rolling(26).min()) / 2
    out["ichi_tenkan"] = tenkan
    out["ichi_kijun"] = kijun
    out["ichi_senkou_a"] = (tenkan + kijun) / 2

    # ── Parabolic SAR ─────────────────────────────────────────────────────────
    out["psar_value"], out["psar_signal"] = _parabolic_sar(H, L)

    # ── Donchian Channel ──────────────────────────────────────────────────────
    out["dc_upper"] = H.rolling(20, min_periods=1).max()
    out["dc_lower"] = L.rolling(20, min_periods=1).min()

    # ── Linear Regression Slope (normalized) ─────────────────────────────────
    for w in (10, 30):
        out[f"lr_slope_{w}"] = C.rolling(w, min_periods=w).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
        ) / (C + 1e-10)

    # ── Z-Score ───────────────────────────────────────────────────────────────
    for w in (20, 60):
        mu = C.rolling(w, min_periods=1).mean()
        sig = C.rolling(w, min_periods=1).std() + 1e-10
        out[f"zscore_{w}"] = (C - mu) / sig

    # ── ADX / DMI ─────────────────────────────────────────────────────────────
    plus_di = (
        100 * H.diff().clip(lower=0).ewm(span=14, adjust=False).mean() / (atr14 + 1e-10)
    )
    minus_di = (
        100
        * (-L.diff()).clip(lower=0).ewm(span=14, adjust=False).mean()
        / (atr14 + 1e-10)
    )
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    out["adx_14"] = dx.ewm(span=14, adjust=False).mean()
    out["dmi_plus"] = plus_di
    out["dmi_minus"] = minus_di

    return out.ffill().fillna(0.0)


def _parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    af0: float = 0.02,
    af_max: float = 0.20,
) -> tuple[pd.Series, pd.Series]:
    n = len(high)
    sar = np.full(n, np.nan)
    sig = np.zeros(n)
    bull = True
    af = af0
    ep = float(high.iloc[0])
    sar[0] = float(low.iloc[0])

    for i in range(1, n):
        h, l = float(high.iloc[i]), float(low.iloc[i])
        prev = sar[i - 1]
        if bull:
            sar[i] = min(
                prev + af * (ep - prev),
                float(low.iloc[max(0, i - 1)]),
                float(low.iloc[max(0, i - 2)]),
            )
            if h > ep:
                ep, af = h, min(af + af0, af_max)
            if l < sar[i]:
                bull, sar[i], ep, af = False, ep, l, af0
        else:
            sar[i] = max(
                prev + af * (ep - prev),
                float(high.iloc[max(0, i - 1)]),
                float(high.iloc[max(0, i - 2)]),
            )
            if l < ep:
                ep, af = l, min(af + af0, af_max)
            if h > sar[i]:
                bull, sar[i], ep, af = True, ep, h, af0
        sig[i] = 1.0 if bull else -1.0

    return pd.Series(sar, index=high.index), pd.Series(sig, index=high.index)
