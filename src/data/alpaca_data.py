from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from datetime import date

from ..data.alpaca_config import AlpacaConfig, INDEX_PROXY_SYMBOLS, RAW_BAR_COLUMNS, DEFAULT_FEATURES, TARGET_COL

def latest_end_date() -> str:
    """Return today's date. Alpaca's `end` accepts an inclusive RFC-3339/date value."""
    return date.today().isoformat()


def parse_features(features_arg: str) -> List[str]:
    if not features_arg:
        return DEFAULT_FEATURES.copy()
    features = [x.strip().lower() for x in features_arg.split(",") if x.strip()]
    invalid = [f for f in features if f not in RAW_BAR_COLUMNS and f not in {"return", "log_return", "range", "body"}]
    if invalid:
        raise ValueError(f"Unsupported feature(s): {invalid}. Valid raw columns: {RAW_BAR_COLUMNS}, plus return, log_return, range, body")
    if TARGET_COL not in features:
        features.append(TARGET_COL)
    return features


def parse_symbol_map(symbols_arg: str) -> Dict[str, str]:
    """Parse comma-separated symbols or NAME:SYMBOL pairs."""
    out: Dict[str, str] = {}
    for item in (symbols_arg or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, symbol = item.split(":", 1)
            out[name.strip().upper()] = symbol.strip().upper()
        else:
            out[item.upper()] = item.upper()
    return out


def get_symbol_map(indices_arg: str, custom_symbols: str = "") -> Dict[str, str]:
    if custom_symbols:
        return parse_symbol_map(custom_symbols)

    symbols = INDEX_PROXY_SYMBOLS.copy()
    if indices_arg:
        wanted = [x.strip().upper() for x in indices_arg.split(",") if x.strip()]
        symbols = {k: v for k, v in INDEX_PROXY_SYMBOLS.items() if k.upper() in wanted or v.upper() in wanted}
        if not symbols:
            raise ValueError(f"No matching indices found for --indices={indices_arg!r}. Valid keys: {list(INDEX_PROXY_SYMBOLS)}")
    return symbols


def alpaca_bars_request(
    cfg: AlpacaConfig,
    symbols: Sequence[str],
    start: str,
    end: str,
    timeframe: str = "1Day",
    limit: int = 10000,
    max_retries: int = 4,
) -> Dict[str, List[Dict[str, object]]]:
    """Fetch historical bars from Alpaca /v2/stocks/bars, handling pagination."""
    url = f"{cfg.data_url}/stocks/bars"
    all_bars: Dict[str, List[Dict[str, object]]] = {s: [] for s in symbols}
    page_token: Optional[str] = None

    while True:
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "limit": limit,
            "adjustment": cfg.adjustment,
            "feed": cfg.feed,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token

        for attempt in range(max_retries):
            response = requests.get(url, headers=cfg.headers, params=params, timeout=60)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                sleep_s = 2 ** attempt
                print(f"Alpaca returned {response.status_code}; retrying in {sleep_s}s...")
                time.sleep(sleep_s)
                continue
            break

        if response.status_code >= 400:
            detail = response.text[:500]
            raise RuntimeError(f"Alpaca bars request failed with HTTP {response.status_code}: {detail}")

        payload = response.json()
        bars = payload.get("bars", {})
        if isinstance(bars, dict):
            for symbol, rows in bars.items():
                all_bars.setdefault(symbol, []).extend(rows or [])
        else:
            raise RuntimeError(f"Unexpected Alpaca response shape: {payload.keys()}")

        page_token = payload.get("next_page_token")
        if not page_token:
            break

    return all_bars


def bars_to_frame(symbol: str, bars: List[Dict[str, object]]) -> pd.DataFrame:
    if not bars:
        raise ValueError(f"No bars returned for {symbol}.")
    df = pd.DataFrame(bars)
    missing = [c for c in ["t", "o", "h", "l", "c", "v"] if c not in df.columns]
    if missing:
        raise ValueError(f"Alpaca response for {symbol} is missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    for col in RAW_BAR_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep_cols = [c for c in RAW_BAR_COLUMNS if c in df.columns]
    df = df[keep_cols].dropna(subset=["o", "h", "l", "c", "v"])
    df["symbol"] = symbol
    if len(df) < 100:
        raise ValueError(f"Too few usable bars ({len(df)}) for {symbol}.")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return"] = out["c"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["log_return"] = np.log(out["c"] / out["c"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["range"] = ((out["h"] - out["l"]) / out["c"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["body"] = ((out["c"] - out["o"]) / out["o"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def preprocess_alpaca_frame(df: pd.DataFrame, feature_cols: Sequence[str], target_col: str = TARGET_COL) -> pd.DataFrame:
    """Clean Alpaca OHLCV bars and add optional derived features before windowing."""
    df = add_engineered_features(df)
    needed = list(dict.fromkeys(list(feature_cols) + [target_col]))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing requested feature columns: {missing}")

    out = df[needed].copy()
    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    if len(out) < 100:
        raise ValueError(f"Too few rows after preprocessing ({len(out)}).")
    return out.astype(float)


def download_alpaca_symbol_frame(
    cfg: AlpacaConfig,
    symbol: str,
    start: str,
    end: str,
    timeframe: str,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    bars_by_symbol = alpaca_bars_request(cfg, [symbol], start=start, end=end, timeframe=timeframe)
    raw = bars_to_frame(symbol, bars_by_symbol.get(symbol, []))
    return preprocess_alpaca_frame(raw, feature_cols=feature_cols, target_col=TARGET_COL)


def train_test_split_frame(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_train = int(len(df) * train_ratio)
    if n_train <= 0 or n_train >= len(df):
        raise ValueError("Invalid train/test split. Use more data or adjust train_ratio.")
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def make_supervised_matrix(features_scaled: np.ndarray, target_scaled: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    if features_scaled.ndim != 2:
        raise ValueError(f"Expected feature matrix shape (N,F), got {features_scaled.shape}")
    if target_scaled.ndim == 1:
        target_scaled = target_scaled.reshape(-1, 1)
    if len(features_scaled) <= lookback:
        raise ValueError(f"Series length must exceed lookback ({lookback}).")
    X, y = [], []
    for i in range(lookback, len(features_scaled)):
        X.append(features_scaled[i - lookback:i, :])
        y.append(target_scaled[i, 0])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32).reshape(-1, 1)


@dataclass
class WindowData:
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test_scaled: np.ndarray
    y_true: np.ndarray
    n_features: int


def prepare_windows(df: pd.DataFrame, feature_cols: Sequence[str], lookback: int, train_ratio: float = 0.7) -> WindowData:
    train_df, test_df = train_test_split_frame(df, train_ratio=train_ratio)

    feature_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    target_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    feature_scaler.fit(train_df[list(feature_cols)].to_numpy())
    target_scaler.fit(train_df[[TARGET_COL]].to_numpy())

    train_X_scaled = feature_scaler.transform(train_df[list(feature_cols)].to_numpy()).astype(np.float32)
    train_y_scaled = target_scaler.transform(train_df[[TARGET_COL]].to_numpy()).astype(np.float32)

    combined_for_test = pd.concat([train_df.iloc[-lookback:], test_df], axis=0)
    test_X_scaled_full = feature_scaler.transform(combined_for_test[list(feature_cols)].to_numpy()).astype(np.float32)
    test_y_scaled_full = target_scaler.transform(combined_for_test[[TARGET_COL]].to_numpy()).astype(np.float32)

    X_train, y_train = make_supervised_matrix(train_X_scaled, train_y_scaled, lookback)
    X_test, y_test_scaled = make_supervised_matrix(test_X_scaled_full, test_y_scaled_full, lookback)
    y_true = target_scaler.inverse_transform(y_test_scaled).reshape(-1)

    return WindowData(
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test_scaled=y_test_scaled,
        y_true=y_true,
        n_features=len(feature_cols),
    )
