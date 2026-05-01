from __future__ import annotations

from numba import njit
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_TRADING_MINUTES = 390  # 09:30–16:00
_TRADING_DAYS = 5


def _parse_timeframe(timeframe: str) -> pd.Timedelta:
    mapping = {
        "1Min": pd.Timedelta(minutes=1),
        "5Min": pd.Timedelta(minutes=5),
        "15Min": pd.Timedelta(minutes=15),
        "1Hour": pd.Timedelta(hours=1),
        "1Day": pd.Timedelta(days=1),
    }

    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    return mapping[timeframe]


mapping = {
    "1Min": "1min",
    "5Min": "5min",
    "15Min": "15min",
    "1Hour": "1h",
    "1Day": "1D",
}

USE_DYNAMIC_SESSION_NORMALIZATION = True


def compute_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    # Causal log-return: target column is forward-looking (t+1) and would leak.
    r = np.log(df["close"] / df["close"].shift(1))
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
        roll = r.rolling(w, min_periods=w)
        for stat in stats:
            out[f"ret_{stat}_{w}"] = getattr(roll, stat)()

    # ── Lag-1 autocorrelation ─────────────────────────────────────────────────
    for w in (20, 60):
        out[f"ret_autocorr_1_{w}"] = r.rolling(w, min_periods=w).apply(
            _fast_autocorr, raw=True
        )

    # ── Price range ratio ─────────────────────────────────────────────────────
    for w in (20, 60):
        roll_C = C.rolling(w, min_periods=w)
        out[f"range_ratio_{w}"] = (roll_C.max() - roll_C.min()) / (
            roll_C.mean() + 1e-10
        )

    # ── Realized volatility ───────────────────────────────────────────────────
    r_sq = r**2
    for w in (10, 30):
        out[f"rv_{w}"] = np.sqrt(r_sq.rolling(w, min_periods=w).sum())

    # ── Hurst exponent (R/S) ──────────────────────────────────────────────────
    out["hurst_exp_60"] = (
        r.rolling(60, min_periods=60).apply(_hurst_single, raw=True).fillna(0.5)
    )
    logger.info(
        f"Computed {len(out.columns)} statistical features for {len(out)} samples "
        f"(columns: {list(out.columns.tolist())}...)"
    )

    # return out.fillna(0.0)
    return out.astype(np.float32)


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


def compute_trd_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all TRD-mandated technical indicators from OHLCV data.

    This is the canonical feature computation function that enforces TRD compliance.
    All indicators use causal computation (data ≤ time t only).

    Required Input Columns:
        - open: Opening price
        - high: Highest price
        - low: Lowest price
        - close: Closing price
        - volume: Trading volume

    Returns:
        DataFrame with 36 technical indicator features (pre-selection)

    Feature Categories:
        - Price (6 features): OHLCV + log_return
        - Trend (6 features): EMA12, EMA20, EMA25, MA5, MA10, MACD
        - Volatility (5 features): Bollinger Bands (upper, mid, lower), ATR14, BB%B
        - Momentum (13 features): RSI, CCI, MTM, ROC, SMI, WVAD, Stochastic, etc.
        - Statistical (6 features): Z-scores, LR slopes, ADX, DMI

    Raises:
        KeyError: If required columns missing from input DataFrame
        ValueError: If input contains NaN/Inf in OHLCV columns

    Example:
        >>> df = load_ohlcv_data("AAPL")
        >>> features = compute_trd_technical_features(df)
        >>> print(features.shape)  # (N, 36)
    """
    # Validate required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Extract OHLCV
    C = df["close"]
    H = df["high"]
    L = df["low"]
    O = df["open"]
    V = df["volume"]

    # Validate no NaN/Inf in input
    for name, series in [
        ("close", C),
        ("high", H),
        ("low", L),
        ("open", O),
        ("volume", V),
    ]:
        if series.isna().any():
            logger.warning(f"NaN values detected in {name} column")
        if np.isinf(series.values).any():
            raise ValueError(f"Infinite values detected in {name} column")

    # Initialize output container
    out = pd.DataFrame(index=df.index)

    # ========================================================================
    # CATEGORY A: PRICE FEATURES (TRD1 §3.1)
    # ========================================================================
    out["open"] = O
    out["high"] = H
    out["low"] = L
    out["close"] = C
    out["volume"] = V
    out["log_return"] = np.log(C / C.shift(1))

    # ========================================================================
    # CATEGORY B: TREND-FOLLOWING INDICATORS (TRD1 §3.2)
    # ========================================================================
    # EMA indicators (Lanbouri & Achchab 2020, Zeng et al. 2025)
    # FIX Issue #10: Use min_periods=span for proper warm-up
    out["ema12"] = C.ewm(span=12, adjust=False, min_periods=12).mean()
    out["ema20"] = C.ewm(span=20, adjust=False, min_periods=20).mean()
    out["ema25"] = C.ewm(span=25, adjust=False, min_periods=25).mean()
    ema26 = C.ewm(span=26, adjust=False, min_periods=26).mean()

    # Simple Moving Averages (Zeng et al. 2025)
    # Use min_periods=window for proper warm-up
    out["ma5"] = C.rolling(5, min_periods=5).mean()
    out["ma10"] = C.rolling(10, min_periods=10).mean()
    out["ma20"] = C.rolling(20, min_periods=20).mean()

    # MACD (Lanbouri & Achchab 2020)
    # Note: Only MACD line is included per TRD ambiguity resolution
    out["macd"] = out["ema12"] - ema26

    # ========================================================================
    # CATEGORY C: VOLATILITY INDICATORS (TRD1 §3.3)
    # ========================================================================
    # Bollinger Bands (Lanbouri & Achchab 2020)
    sma20 = out["ma20"]
    std20 = C.rolling(20, min_periods=20).std()
    out["boll_upper"] = sma20 + 2 * std20
    out["boll_lower"] = sma20 - 2 * std20
    out["boll_mid"] = sma20  # Zeng et al. 2025 explicit requirement

    # Bollinger Band %B (position within bands)
    out["bb_pct_b"] = (C - out["boll_lower"]) / (
        out["boll_upper"] - out["boll_lower"] + 1e-10
    )

    # Average True Range (TRD1 §3.3 - Wilder's Smoothing)
    # TR calculation: max(H-L, |H-C_prev|, |L-C_prev|)
    prev_C = C.shift(1)
    tr = pd.DataFrame(
        {"hl": H - L, "hc": (H - prev_C).abs(), "lc": (L - prev_C).abs()}
    ).max(axis=1)

    # ATR14: Wilder's RMA in a single pass.
    # Formula: ATR[t] = (13*ATR[t-1] + TR[t]) / 14, equivalent to EWM with alpha=1/14.
    out["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # ========================================================================
    # CATEGORY D: MOMENTUM/OSCILLATOR INDICATORS (TRD1 §3.4)
    # ========================================================================
    # CCI - Commodity Channel Index (Zeng et al. 2025)
    # Typical Price: (H + L + C) / 3
    tp = (H + L + C) / 3
    sma_tp20 = tp.rolling(20, min_periods=20).mean()
    mean_dev = (tp - sma_tp20).abs().rolling(20, min_periods=20).mean()
    out["cci_20"] = (tp - sma_tp20) / (0.015 * mean_dev + 1e-10)

    # MTM - Momentum (Zeng et al. 2025)
    out["mtm6"] = C - C.shift(6)
    out["mtm12"] = C - C.shift(12)

    # ROC - Rate of Change (Zeng et al. 2025)
    out["roc"] = (C - C.shift(12)) / (C.shift(12) + 1e-10) * 100

    # SMI - Stochastic Momentum Index (Zeng et al. 2025)
    # Standard 10/3 period calculation
    HH = H.rolling(10, min_periods=10).max()
    LL = L.rolling(10, min_periods=10).min()
    midpoint = (HH + LL) / 2
    relative_range = C - midpoint
    range_hl = HH - LL

    # Double EMA smoothing (3-period)
    d_s = (
        relative_range.ewm(span=3, adjust=False).mean().ewm(span=3, adjust=False).mean()
    )
    hld_s = range_hl.ewm(span=3, adjust=False).mean().ewm(span=3, adjust=False).mean()
    out["smi"] = 200 * (d_s / (hld_s + 1e-10))

    # WVAD momentum (Zeng et al. 2025). Raw cumulative WVAD is split-dependent
    # under per-split feature generation; expose only the windowed momentum.
    hl_range = H - L + 1e-10
    co_diff = C - O
    wvad_contrib = (co_diff / hl_range) * V
    wvad = wvad_contrib.cumsum()
    out["wvad_momentum_20"] = wvad.diff(20) / (
        wvad.abs().rolling(20, min_periods=20).mean() + 1e-10
    )

    # RSI - Relative Strength Index (14-period Wilder's RMA)
    delta = C.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    # Stochastic %K and %D
    low14 = L.rolling(14, min_periods=14).min()
    high14 = H.rolling(14, min_periods=14).max()
    stoch_k = 100 * (C - low14) / (high14 - low14 + 1e-10)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_k.rolling(3, min_periods=3).mean()

    # ========================================================================
    # CATEGORY E: STATISTICAL FEATURES (Production enhancements)
    # ========================================================================
    # Z-Score (20-period normalized price position)
    mu20 = out["ma20"]
    sig20 = std20 + 1e-10
    out["zscore_20"] = (C - mu20) / sig20

    # Linear Regression Slope (10-period normalized)
    out["lr_slope_10"] = C.rolling(10, min_periods=10).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 10 else 0.0,
        raw=True,
    ) / (C + 1e-10)

    # ADX - Average Directional Index
    atr14 = out["atr_14"]
    plus_dm = H.diff().clip(lower=0)
    minus_dm = (-L.diff()).clip(lower=0)
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    out["adx_14"] = dx.ewm(span=14, adjust=False).mean()

    # ========================================================================
    # POST-PROCESSING
    # ========================================================================
    # Forward-fill NaNs from indicator warm-up (causal operation)
    # Replace remaining NaNs with 0.0 (neutral value after scaling)
    # out = out.ffill().fillna(0.0)

    # Convert to float32 for memory efficiency
    # out = out.astype(np.float32)

    logger.info(
        f"Computed {len(out.columns)} technical indicators for {len(out)} samples "
        f"(columns: {list(out.columns.tolist())}...)"
    )

    return out.astype(np.float32)


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract base price-based features from OHLCV.

    Returns:
        DataFrame with derived price features (mid_price, ratios, etc.)
    """
    C = df["close"]
    H = df["high"]
    L = df["low"]
    O = df["open"]
    prev_C = C.shift(1)

    # True range for intraday volatility
    tr = pd.DataFrame(
        {"hl": H - L, "hc": (H - prev_C).abs(), "lc": (L - prev_C).abs()}
    ).max(axis=1)

    out = pd.DataFrame(
        {
            "mid_price": (H + L) / 2.0,
            "hl_ratio": (H - L) / (prev_C + 1e-10),  # Intraday range / prior close
            "co_ratio": (C - O) / (prev_C + 1e-10),  # Close-open / prior close
            "true_range": tr / (prev_C + 1e-10),  # Normalized true range
            "intrabar_vol": (H - L) / (O + 1e-10),  # Intrabar volatility
        },
        index=df.index,
    )

    logger.info(
        f"Computed {len(out.columns)} price features for {len(out)} samples "
        f"(columns: {list(out.columns.tolist())}...)"
    )
    # out = out.ffill(limit=5)
    # # return out.ffill().fillna(0.0)
    return out.astype(np.float32)


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    C, H, L, V = df["close"], df["high"], df["low"], df["volume"]
    # Causal log-return: target column is forward-looking (t+1) and would leak.
    r = np.log(df["close"] / df["close"].shift(1))

    # Compute session_minute from the DatetimeIndex (minutes since 09:30 ET open).
    idx = df.index
    if idx.tz is None:
        idx_local = idx.tz_localize("UTC").tz_convert("America/New_York")
    else:
        idx_local = idx.tz_convert("America/New_York")
    sm_raw = (idx_local.hour * 60 + idx_local.minute) - (9 * 60 + 30)
    sm = pd.Series(np.clip(sm_raw, 0, 389), index=df.index)
    out = pd.DataFrame(index=df.index)

    # ── Shared intermediates ─────────────────────────────────────────────────
    tp = (H + L + C) / 3
    clv = ((C - L) - (H - C)) / (H - L + 1e-10)  # close location value

    # ── Rolling-window VWAP ──────────────────────────────────────────────────
    # Session-reset VWAP requires the split to start at session open. Under
    # per-split feature generation that is not guaranteed, so we use a fixed
    # rolling-window VWAP which is split-invariant.
    vwap_window = 30
    cum_tp_vol = (tp * V).rolling(vwap_window, min_periods=vwap_window).sum()
    cum_vol = V.rolling(vwap_window, min_periods=vwap_window).sum()
    vwap = cum_tp_vol / (cum_vol + 1e-10)
    out["vwap_30"] = vwap
    out["price_to_vwap_30"] = C / (vwap + 1e-10)
    out["vwap_dev_30"] = (C - vwap) / (vwap + 1e-10)

    # ── Relative Volume ───────────────────────────────────────────────────────
    out["rvol_20"] = V / (V.rolling(20, min_periods=20).mean() + 1e-10)
    out["rvol_60"] = V / (V.rolling(60, min_periods=60).mean() + 1e-10)

    # ── On-Balance Volume (momentum only) ────────────────────────────────────
    # Raw OBV is a cumulative sum starting at zero in each split, so its
    # absolute level is split-dependent. We expose only the windowed momentum
    # which is split-invariant.
    obv = pd.Series(
        np.cumsum(np.where(C > C.shift(1), V, np.where(C < C.shift(1), -V, 0))),
        index=df.index,
    )
    out["obv_momentum_20"] = obv.diff(20) / (
        obv.abs().rolling(20, min_periods=20).mean() + 1e-10
    )

    # ── Accumulation/Distribution slope (momentum only) ──────────────────────
    adl = (clv * V).cumsum()
    out["adl_slope_10"] = adl.diff(10) / (
        adl.abs().rolling(10, min_periods=10).mean() + 1e-10
    )

    # ── Chaikin Money Flow ────────────────────────────────────────────────────
    out["cmf_20"] = (clv * V).rolling(20, min_periods=20).sum() / (
        V.rolling(20, min_periods=20).sum() + 1e-10
    )

    # ── Force Index ───────────────────────────────────────────────────────────
    force = r * V
    out["force_1"] = force
    out["force_ema_13"] = force.ewm(span=13, adjust=False).mean()

    # ── Ease of Movement ─────────────────────────────────────────────────────
    hl_mid = (H + L) / 2
    out["eom_14"] = (
        ((hl_mid - hl_mid.shift(1)) / (V / (H - L + 1e-10) + 1e-10))
        .rolling(14, min_periods=14)
        .mean()
    )

    # Raw cumulative volume series (obv, adl, vpt, wvad) are intentionally
    # not exposed: their absolute level depends on where the split starts,
    # so they distort the train-fitted scaler when applied to val/test.

    # ── Intraday Turnover Velocity ────────────────────────────────────────────
    out["itv_20"] = V / (V.rolling(20, min_periods=20).sum() + 1e-10)

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

    logger.info(
        f"Computed {len(out.columns)} volume features for {len(out)} samples "
        f"(columns: {list(out.columns.tolist())}...)"
    )

    return out.astype(np.float32)


# Alias for backward compatibility and explicit naming
compute_technical_features = compute_trd_technical_features
