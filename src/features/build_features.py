import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def calculate_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Orchestrates the generation of technical and statistical features.

    Args:
        df (pd.DataFrame): Cleaned financial data with OHLCV columns.
        cfg: Hydra configuration object (cfg.features).

    Returns:
        pd.DataFrame: Dataframe augmented with engineered features.
    """
    logger.info("Starting vectorized feature engineering...")
    df_feat = df.copy()

    # 1. Momentum: Relative Strength Index (RSI)
    df_feat = add_rsi(df_feat, window=cfg.indicators.rsi.window)

    # 2. Trend: MACD
    df_feat = add_macd(
        df_feat,
        fast=cfg.indicators.macd.fast,
        slow=cfg.indicators.macd.slow,
        signal=cfg.indicators.macd.signal,
    )

    # 3. Volatility: Bollinger Bands
    df_feat = add_bollinger_bands(
        df_feat,
        window=cfg.indicators.bollinger_bands.window,
        std_dev=cfg.indicators.bollinger_bands.std_dev,
    )

    # 4. Moving Averages (EMA & SMA)
    for window in cfg.indicators.ema.windows:
        df_feat[f"ema_{window}"] = (
            df_feat["close"].ewm(span=window, adjust=False).mean()
        )

    for window in cfg.indicators.sma.windows:
        df_feat[f"sma_{window}"] = df_feat["close"].rolling(window=window).mean()

    # 5. Statistical Rolling Features
    df_feat["rolling_mean"] = (
        df_feat["close"].rolling(window=cfg.statistics.rolling_mean).mean()
    )
    df_feat["rolling_std"] = (
        df_feat["close"].rolling(window=cfg.statistics.rolling_std).std()
    )

    # Drop intermediate NaNs created by rolling windows to prevent LSTM gradient issues
    df_feat = df_feat.dropna()

    logger.info(f"Feature engineering complete. Total features: {len(df_feat.columns)}")
    return df_feat


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Vectorized RSI calculation."""
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """Vectorized MACD calculation."""
    exp1 = df["close"].ewm(span=fast, adjust=False).mean()
    exp2 = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int, std_dev: int) -> pd.DataFrame:
    """Vectorized Bollinger Bands."""
    ma = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()
    df["bb_mid"] = ma
    df["bb_high"] = ma + (std * std_dev)
    df["bb_low"] = ma - (std * std_dev)
    return df
