"""
src/features/technical.py

Computes all technical indicator features from OHLCV data.
All indicators operate on a single-ticker OHLCV DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 42 technical indicator features.

    Args:
        df: Single-ticker OHLCV DataFrame with columns
            [open, high, low, close, volume].

    Returns:
        DataFrame of the same index with technical feature columns added.
    """
    out = pd.DataFrame(index=df.index)
    C, H, L, O, V = df["close"], df["high"], df["low"], df["open"], df["volume"]

    # ---- Moving Averages ---------------------------------------------------
    out["sma_5"] = _sma(C, 5)
    out["sma_20"] = _sma(C, 20)
    out["sma_60"] = _sma(C, 60)
    out["ema_12"] = _ema(C, 12)
    out["ema_26"] = _ema(C, 26)
    out["ema_60"] = _ema(C, 60)

    # ---- MACD --------------------------------------------------------------
    macd = out["ema_12"] - out["ema_26"]
    macd_signal = _ema(macd, 9)
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd - macd_signal

    # ---- RSI ---------------------------------------------------------------
    for period in (14, 30):
        delta = C.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        out[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    # ---- Bollinger Bands ---------------------------------------------------
    sma20 = _sma(C, 20)
    std20 = C.rolling(20, min_periods=1).std()
    out["bb_upper"] = sma20 + 2 * std20
    out["bb_lower"] = sma20 - 2 * std20
    out["bb_width"] = (2 * 2 * std20) / (sma20 + 1e-10)
    out["bb_pct_b"] = (C - out["bb_lower"]) / (
        out["bb_upper"] - out["bb_lower"] + 1e-10
    )

    # ---- Stochastic Oscillator ---------------------------------------------
    low14 = L.rolling(14, min_periods=1).min()
    high14 = H.rolling(14, min_periods=1).max()
    stoch_k = 100 * (C - low14) / (high14 - low14 + 1e-10)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_k.rolling(3, min_periods=1).mean()

    # ---- True Range & ATR --------------------------------------------------
    prev_close = C.shift(1)
    tr = pd.concat([H - L, (H - prev_close).abs(), (L - prev_close).abs()], axis=1).max(
        axis=1
    )
    out["atr_14"] = tr.rolling(14, min_periods=1).mean()
    out["atr_30"] = tr.rolling(30, min_periods=1).mean()

    # ---- CCI ---------------------------------------------------------------
    tp = (H + L + C) / 3
    sma20_tp = _sma(tp, 20)
    mad20 = tp.rolling(20, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    out["cci_20"] = (tp - sma20_tp) / (0.015 * mad20 + 1e-10)

    # ---- Williams %R -------------------------------------------------------
    low14 = L.rolling(14, min_periods=1).min()
    high14 = H.rolling(14, min_periods=1).max()
    out["williams_r"] = -100 * (high14 - C) / (high14 - low14 + 1e-10)

    # ---- Rate of Change ----------------------------------------------------
    # out["roc_5"] = C.pct_change(5) * 100
    # out["roc_10"] = C.pct_change(10) * 100
    out["roc_5"] = C.pct_change(5, fill_method=None) * 100
    out["roc_10"] = C.pct_change(10, fill_method=None) * 100

    # ---- Money Flow Index --------------------------------------------------
    for period in (14, 30):
        mf = tp * V
        pos_mf = mf.where(tp > tp.shift(1), 0.0)
        neg_mf = mf.where(tp < tp.shift(1), 0.0)
        pos_sum = pos_mf.rolling(period, min_periods=1).sum()
        neg_sum = neg_mf.rolling(period, min_periods=1).sum()
        out[f"mfi_{period}"] = 100 - 100 / (1 + pos_sum / (neg_sum + 1e-10))

    # ---- Ichimoku ----------------------------------------------------------
    out["ichi_tenkan"] = (H.rolling(9).max() + L.rolling(9).min()) / 2
    out["ichi_kijun"] = (H.rolling(26).max() + L.rolling(26).min()) / 2
    out["ichi_senkou_a"] = (out["ichi_tenkan"] + out["ichi_kijun"]) / 2
    # REMOVED ichi_chikou: was using shift(-26) which creates look-ahead bias
    # Chikou span shows close shifted 26 bars back for chart display, not a predictive feature

    # ---- Parabolic SAR (vectorised approximation) --------------------------
    psar, psar_sig = _parabolic_sar(H, L, C)
    out["psar_value"] = psar
    out["psar_signal"] = psar_sig

    # ---- Donchian Channel --------------------------------------------------
    out["dc_upper"] = H.rolling(20, min_periods=1).max()
    out["dc_lower"] = L.rolling(20, min_periods=1).min()

    # ---- Linear Regression Slope -------------------------------------------
    for w in (10, 30):
        out[f"lr_slope_{w}"] = C.rolling(w, min_periods=w).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
        ) / (C + 1e-10)

    # ---- Z-Score of Close --------------------------------------------------
    for w in (20, 60):
        roll_mean = C.rolling(w, min_periods=1).mean()
        roll_std = C.rolling(w, min_periods=1).std() + 1e-10
        out[f"zscore_{w}"] = (C - roll_mean) / roll_std

    # ---- ADX / DMI ---------------------------------------------------------
    plus_dm = (H.diff()).clip(lower=0)
    minus_dm = (-L.diff()).clip(lower=0)
    atr14 = out["atr_14"]
    plus_di = 100 * _ema(plus_dm, 14) / (atr14 + 1e-10)
    minus_di = 100 * _ema(minus_dm, 14) / (atr14 + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    out["adx_14"] = _ema(dx, 14)
    out["dmi_plus"] = plus_di
    out["dmi_minus"] = minus_di

    # return out.fillna(method="ffill").fillna(0.0)
    return out.ffill().fillna(0.0)


def _parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    af0: float = 0.02,
    af_max: float = 0.20,
) -> tuple[pd.Series, pd.Series]:
    """Compute Parabolic SAR and trend signal (+1 / -1)."""
    n = len(high)
    sar = np.full(n, np.nan)
    sig = np.zeros(n)
    bull = True
    af = af0
    ep = float(high.iloc[0])
    sar[0] = float(low.iloc[0])

    for i in range(1, n):
        h, l = float(high.iloc[i]), float(low.iloc[i])
        prev_sar = sar[i - 1]

        if bull:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(
                sar[i], float(low.iloc[max(0, i - 1)]), float(low.iloc[max(0, i - 2)])
            )
            if h > ep:
                ep = h
                af = min(af + af0, af_max)
            if l < sar[i]:
                bull = False
                sar[i] = ep
                ep = l
                af = af0
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(
                sar[i], float(high.iloc[max(0, i - 1)]), float(high.iloc[max(0, i - 2)])
            )
            if l < ep:
                ep = l
                af = min(af + af0, af_max)
            if h > sar[i]:
                bull = True
                sar[i] = ep
                ep = h
                af = af0

        sig[i] = 1.0 if bull else -1.0

    return pd.Series(sar, index=high.index), pd.Series(sig, index=high.index)
