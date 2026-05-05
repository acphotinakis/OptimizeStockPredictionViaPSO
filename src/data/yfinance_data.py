from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn

PAPER_INDEX_SYMBOLS: Dict[str, str] = {
    "DJIA": "^DJI",
    "SP500": "^GSPC",
    "NIKKEI225": "^N225",
    "HANGSENG": "^HSI",
    "CSI300": "000300.SS",
    "NIFTY50": "^NSEI",
}
PAPER_START = "2008-07-02"
PAPER_END = "2016-10-01"
DEFAULT_START = PAPER_START

def latest_yfinance_end_date() -> str:
    """Return tomorrow's date because yfinance treats --end as exclusive."""
    return (date.today() + timedelta(days=1)).isoformat()


def download_close_prices(symbol: str, start: str, end: Optional[str] = None) -> pd.Series:
    """Download daily close prices from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit("yfinance is required. Install with: pip install yfinance") from exc

    end = end or latest_yfinance_end_date()
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, threads=True)
    if df.empty:
        raise ValueError(f"No data returned for symbol={symbol!r} between {start} and {end}.")

    close_col = "Close"
    if isinstance(df.columns, pd.MultiIndex):
        close_df = df[close_col] if close_col in df.columns.get_level_values(0) else None
        if close_df is None:
            raise ValueError(f"Column 'Close' not found in downloaded data for {symbol!r}.")
        series = close_df.iloc[:, 0].dropna().astype(float)
    else:
        if close_col not in df.columns:
            raise ValueError(f"Column 'Close' not found in downloaded data for {symbol!r}.")
        series = df[close_col].dropna().astype(float)

    if len(series) < 100:
        raise ValueError(f"Too few data points ({len(series)}) for {symbol!r}.")
    return series


def train_test_split_series(series: pd.Series, train_ratio: float = 0.7) -> Tuple[pd.Series, pd.Series]:
    n_train = int(len(series) * train_ratio)
    return series.iloc[:n_train].copy(), series.iloc[n_train:].copy()


def fit_scaler_on_train(train: pd.Series) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(train.to_numpy().reshape(-1, 1))
    return scaler


def transform_series(scaler: MinMaxScaler, series: pd.Series) -> np.ndarray:
    return scaler.transform(series.to_numpy().reshape(-1, 1)).astype(np.float32)


def make_supervised(values: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a univariate normalized series (N,1) into LSTM windows."""
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError(f"Expected shape (N,1), got {values.shape}")
    if len(values) <= lookback:
        raise ValueError(f"Series length must exceed lookback ({lookback}).")

    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i, 0])
        y.append(values[i, 0])
    X_arr = np.array(X, dtype=np.float32).reshape(-1, lookback, 1)
    y_arr = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X_arr, y_arr
