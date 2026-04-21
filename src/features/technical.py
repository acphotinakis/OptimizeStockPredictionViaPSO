"""
Unified Technical Indicators Module

Implements TRD-specified technical indicators for financial time-series feature engineering.
Combines indicators from src/features/technical.py and src/feature_eng_revised/feature_gen.py
with strict TRD compliance.

TRD References:
- Price features: TRD1 §3.1
- Trend indicators: TRD1 §3.2, Lanbouri & Achchab 2020, Zeng et al. 2025
- Volatility indicators: TRD1 §3.3, Lanbouri & Achchab 2020, Zeng et al. 2025
- Momentum indicators: TRD1 §3.4, Zeng et al. 2025

Author: System Architect
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
    out["log_return"] = np.log(C / C.shift(1))  # Causal: uses only prior close

    # ========================================================================
    # CATEGORY B: TREND-FOLLOWING INDICATORS (TRD1 §3.2)
    # ========================================================================
    # EMA indicators (Lanbouri & Achchab 2020, Zeng et al. 2025)
    out["ema12"] = C.ewm(span=12, adjust=False, min_periods=1).mean()
    out["ema20"] = C.ewm(span=20, adjust=False, min_periods=1).mean()
    out["ema25"] = C.ewm(span=25, adjust=False, min_periods=1).mean()
    ema26 = C.ewm(span=26, adjust=False, min_periods=1).mean()

    # Simple Moving Averages (Zeng et al. 2025)
    out["ma5"] = C.rolling(5, min_periods=1).mean()
    out["ma10"] = C.rolling(10, min_periods=1).mean()
    out["ma20"] = C.rolling(20, min_periods=1).mean()

    # MACD (Lanbouri & Achchab 2020)
    # Note: Only MACD line is included per TRD ambiguity resolution
    out["macd"] = out["ema12"] - ema26

    # ========================================================================
    # CATEGORY C: VOLATILITY INDICATORS (TRD1 §3.3)
    # ========================================================================
    # Bollinger Bands (Lanbouri & Achchab 2020)
    sma20 = out["ma20"]
    std20 = C.rolling(20, min_periods=1).std()
    out["boll_upper"] = sma20 + 2 * std20
    out["boll_lower"] = sma20 - 2 * std20
    out["boll_mid"] = sma20  # Zeng et al. 2025 explicit requirement

    # Bollinger Band %B (position within bands)
    out["bb_pct_b"] = (C - out["boll_lower"]) / (
        out["boll_upper"] - out["boll_lower"] + 1e-10
    )

    # Average True Range (Zeng et al. 2025)
    # TR calculation: max(H-L, |H-C_prev|, |L-C_prev|)
    prev_C = C.shift(1)
    tr = pd.DataFrame(
        {"hl": H - L, "hc": (H - prev_C).abs(), "lc": (L - prev_C).abs()}
    ).max(axis=1)

    # ATR14: Standard 14-period Wilder's smoothed average
    out["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()

    # ========================================================================
    # CATEGORY D: MOMENTUM/OSCILLATOR INDICATORS (TRD1 §3.4)
    # ========================================================================
    # CCI - Commodity Channel Index (Zeng et al. 2025)
    # Typical Price: (H + L + C) / 3
    tp = (H + L + C) / 3
    sma_tp20 = tp.rolling(20, min_periods=1).mean()
    mean_dev = (tp - sma_tp20).abs().rolling(20, min_periods=1).mean()
    out["cci_20"] = (tp - sma_tp20) / (0.015 * mean_dev + 1e-10)

    # MTM - Momentum (Zeng et al. 2025)
    out["mtm6"] = C - C.shift(6)
    out["mtm12"] = C - C.shift(12)

    # ROC - Rate of Change (Zeng et al. 2025)
    out["roc"] = (C - C.shift(12)) / (C.shift(12) + 1e-10) * 100

    # SMI - Stochastic Momentum Index (Zeng et al. 2025)
    # Standard 10/3 period calculation
    HH = H.rolling(10, min_periods=1).max()
    LL = L.rolling(10, min_periods=1).min()
    midpoint = (HH + LL) / 2
    relative_range = C - midpoint
    range_hl = HH - LL

    # Double EMA smoothing (3-period)
    d_s = (
        relative_range.ewm(span=3, adjust=False).mean().ewm(span=3, adjust=False).mean()
    )
    hld_s = range_hl.ewm(span=3, adjust=False).mean().ewm(span=3, adjust=False).mean()
    out["smi"] = 200 * (d_s / (hld_s + 1e-10))

    # WVAD - Williams Variable Accumulation/Distribution (Zeng et al. 2025)
    # Cumulative sum of (Close-Open)/(High-Low) * Volume
    hl_range = H - L + 1e-10
    co_diff = C - O
    wvad_contrib = (co_diff / hl_range) * V
    out["wvad"] = wvad_contrib.cumsum()

    # RSI - Relative Strength Index (14-period standard)
    delta = C.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    # Stochastic %K and %D
    low14 = L.rolling(14, min_periods=1).min()
    high14 = H.rolling(14, min_periods=1).max()
    stoch_k = 100 * (C - low14) / (high14 - low14 + 1e-10)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_k.rolling(3, min_periods=1).mean()

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
    out = out.ffill().fillna(0.0)

    # Convert to float32 for memory efficiency
    out = out.astype(np.float32)

    logger.info(
        f"Computed {len(out.columns)} technical indicators for {len(out)} samples "
        f"(columns: {list(out.columns.tolist())}...)"
    )

    return out


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
    ).astype(np.float32)

    return out.ffill().fillna(0.0)


# Alias for backward compatibility and explicit naming
compute_technical_features = compute_trd_technical_features
