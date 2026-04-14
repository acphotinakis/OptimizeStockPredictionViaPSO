from __future__ import annotations

"""features/pipeline.py — Feature engineering pipeline for LSTM stock prediction."""


import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .technical import compute_technical_features
from .statistical import compute_statistical_features
from .volume import compute_volume_features
from .cross_ticker import compute_cross_ticker_features
from .selector import FeatureSelector

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume", "log_return"}

# Features to lag and their depths
LAG_SPEC: Dict[str, List[int]] = {
    "log_return": [1, 2, 3],
    "rsi_14": [1, 2],
    "vwap_dev": [1],
    "rvol_20": [1],
    "corr_spy_20": [1],
    "macd": [1],
    "atr_14": [1],
    "bb_pct_b": [1],
    "stoch_k": [1],
    "zscore_20": [1],
}


class FeaturePipeline:
    """Build and select features for a single target ticker.

    Lifecycle:
        X_train, y_train, names = pipeline.fit_transform(dfs_train)
        X_val,   y_val          = pipeline.transform(dfs_val)
        X_test,  y_test         = pipeline.transform(dfs_test)

    Args:
        target_ticker:  Stock to predict.
        peer_tickers:   Pre-selected correlated peers (from SymbolUniverseBuilder).
        selector_kwargs: Forwarded to FeatureSelector.
    """

    def __init__(
        self,
        target_ticker: str,
        peer_tickers: List[str],
        selector_kwargs: Optional[Dict] = None,
    ) -> None:
        self.target_ticker = target_ticker
        self.peer_tickers = peer_tickers
        self.selector = FeatureSelector(**(selector_kwargs or {}))
        self._feature_names: List[str] = []
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Fit on training data; return (X, y, feature_names)."""
        self._validate(dfs)
        X, y, names = self._build_features(dfs)
        X_sel, sel_names = self.selector.fit_transform(X, y, names)
        self._feature_names = names  # full names — needed by transform()
        self._fitted = True
        logger.info(
            "[%s] fit_transform: %d --> %d features, %d samples",
            self.target_ticker,
            len(names),
            len(sel_names),
            len(X_sel),
        )
        return X_sel, y, sel_names

    def transform(self, dfs: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply fitted pipeline to val/test data; return (X, y)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        self._validate(dfs)
        X, y, _ = self._build_features(dfs)
        X_sel, _ = self.selector.transform(X, self._feature_names)
        logger.info(
            "[%s] transform: %d samples, %d features",
            self.target_ticker,
            len(X_sel),
            X_sel.shape[1],
        )
        return X_sel, y

    @property
    def feature_names(self) -> List[str]:
        return list(self.selector.selected_features_)

    @property
    def n_features(self) -> int:
        return len(self.selector.selected_features_)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _validate(self, dfs: Dict[str, pd.DataFrame]) -> None:
        if self.target_ticker not in dfs:
            raise KeyError(f"'{self.target_ticker}' not in dfs. Available: {list(dfs)}")
        df = dfs[self.target_ticker]
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"[{self.target_ticker}] Missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"[{self.target_ticker}] Index must be DatetimeIndex.")
        if not df.index.is_monotonic_increasing:
            raise ValueError(f"[{self.target_ticker}] Index not sorted ascending.")

    def _build_features(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        df = dfs[self.target_ticker]
        C, H, L, O = df["close"], df["high"], df["low"], df["open"]
        r = df["log_return"]
        prev_C = C.shift(1)

        # ── Block 1: Base price features ─────────────────────────────────────
        tr = pd.concat([(H - L), (H - prev_C).abs(), (L - prev_C).abs()], axis=1).max(
            axis=1
        )
        base = pd.DataFrame(
            {
                "log_return": r,
                "mid_price": (H + L) / 2.0,
                "hl_ratio": (H - L) / (prev_C + 1e-10),
                "co_ratio": (C - O) / (prev_C + 1e-10),
                "ho_ratio": (H - O) / (prev_C + 1e-10),
                "lc_ratio": (C - L) / (prev_C + 1e-10),
                "gap": (O - prev_C) / (prev_C + 1e-10),
                "true_range": tr / (prev_C + 1e-10),
                "intrabar_vol": (H - L) / (O + 1e-10),
                "cum_return_session": _session_cum_return(df, C),
            },
            index=df.index,
        ).astype(np.float32)

        # ── Blocks 2-5: computed feature modules ─────────────────────────────
        blocks = [
            base,
            compute_technical_features(df),
            compute_statistical_features(df),
            compute_volume_features(df),
            compute_cross_ticker_features(self.target_ticker, dfs, self.peer_tickers),
        ]

        # ── Concatenate into one matrix ───────────────────────────────────────
        arrays, names = [], []
        for blk in blocks:
            blk = blk.loc[:, ~blk.columns.duplicated()]  # drop within-block dupes
            arrays.append(blk.values.astype(np.float32, copy=False))
            names.extend(blk.columns.tolist())

        X = np.concatenate(arrays, axis=1)

        # ── Block 6: Lag features ─────────────────────────────────────────────
        lag_X, lag_names = _build_lags(X, names)
        if lag_X.shape[1]:
            X = np.concatenate([X, lag_X], axis=1)
            names = names + lag_names

        # ── Deduplicate across blocks ─────────────────────────────────────────
        seen, keep_idx, keep_names = set(), [], []
        for i, n in enumerate(names):
            if n not in seen:
                seen.add(n)
                keep_idx.append(i)
                keep_names.append(n)
        X = X[:, keep_idx]

        # ── Clean non-finite values ───────────────────────────────────────────
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Drop rows where target is non-finite ──────────────────────────────
        y = r.values.astype(np.float32)
        valid = np.isfinite(y)
        if (~valid).any():
            logger.info(
                "[%s] Dropping %d non-finite target rows.",
                self.target_ticker,
                (~valid).sum(),
            )
            X, y = X[valid], y[valid]

        logger.info(
            "[%s] Feature matrix: %d × %d", self.target_ticker, X.shape[0], X.shape[1]
        )
        return X, y, keep_names


# ── Module-level helpers ──────────────────────────────────────────────────────


def _session_cum_return(df: pd.DataFrame, close: pd.Series) -> pd.Series:
    """Within-session cumulative return (resets each day)."""
    if "session_start" in df.columns:
        session_ids = df["session_start"].astype(bool).cumsum()
    else:
        session_ids = pd.Series(
            pd.factorize(pd.to_datetime(close.index).date)[0],
            index=close.index,
        )
    first_close = close.groupby(session_ids).transform("first")
    return (close / first_close - 1.0).fillna(0.0).astype(np.float32)


def _build_lags(X: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Build lag features via slicing (no np.roll — avoids wrap-around leakage)."""
    name_idx = {n: i for i, n in enumerate(names)}
    arrays, lag_names = [], []
    for col, lags in LAG_SPEC.items():
        if col not in name_idx:
            continue
        col_data = X[:, name_idx[col]]
        for lag in lags:
            lagged = np.zeros(len(col_data), dtype=np.float32)
            if lag < len(col_data):
                lagged[lag:] = col_data[:-lag]
            arrays.append(lagged)
            lag_names.append(f"{col}_lag{lag}")
    if not arrays:
        return np.empty((X.shape[0], 0), dtype=np.float32), []
    return np.stack(arrays, axis=1).astype(np.float32), lag_names
