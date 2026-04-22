"""features/cross_ticker.py — Canonical cross-ticker feature engine (TRD-compliant)."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MARKET_CONTEXT_TICKERS = ["SPY", "QQQ", "IWM", "DIA"]
MARKET_INTERNAL_TICKERS = ["UVXY", "GLD", "TLT"]

ROLLING_SHORT = 20
ROLLING_LONG = 60
ROLLING_VLONG = 252

EPS = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR MAPPING (CANONICAL - EXTENDABLE)
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_MAP = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "GOOGL": "XLK",
    "META": "XLK",
    "CRM": "XLK",
    "ADBE": "XLK",
    "NVDA": "SOXX",
    "AMD": "SOXX",
    "INTC": "SOXX",
}


def get_sector_etf(target: str) -> Optional[str]:
    return SECTOR_MAP.get(target, None)


def _ensure_log_return(df: pd.DataFrame) -> pd.Series:
    """
    Computes log_return if missing.
    Must be causal and index-safe.
    """
    if "log_return" in df.columns:
        return df["log_return"]

    if "close" not in df.columns:
        raise ValueError("Missing 'close' column required to compute log_return")

    return np.log(df["close"]).diff()


# ─────────────────────────────────────────────────────────────────────────────
# CORE FEATURE ENGINE
# ─────────────────────────────────────────────────────────────────────────────


def compute_cross_ticker_features(
    target: str,
    dfs: Dict[str, pd.DataFrame],
    peers: List[str],
    rolling_window: int = 20,
) -> pd.DataFrame:
    """
    Canonical cross-ticker feature generator.

    STRICT RULES:
    - all features are causal (shifted implicitly via rolling windows)
    - no forward-looking operations
    - all tickers aligned to target index
    """

    if target not in dfs:
        raise ValueError(f"Missing target ticker: {target}")

    idx = dfs[target].index
    r_t = dfs[target]["log_return"].reindex(idx).fillna(0.0)
    c_t = dfs[target]["close"].reindex(idx).ffill()

    out = pd.DataFrame(index=idx)

    # ======================================================================
    # 1. MARKET CONTEXT LAYER
    # ======================================================================
    for t in MARKET_CONTEXT_TICKERS:
        if t not in dfs:
            out[f"{t}_log_return"] = 0.0
            out[f"{t}_rolling_mean_20"] = 0.0
            out[f"{t}_rolling_std_20"] = 0.0
            continue

        df = dfs[t]

        # ── Ensure log_return exists (NO ASSUMPTION) ─────────────────────────────
        if "log_return" in df.columns:
            r = df["log_return"]
        else:
            if "close" not in df.columns:
                logger.warning(f"[{t}] missing both log_return and close → zero-fill")
                r = pd.Series(0.0, index=df.index)
            else:
                # deterministic log return computation
                r = np.log(df["close"]).diff()

        # align to target index
        r = r.reindex(idx).fillna(0.0)

        out[f"{t}_log_return"] = r
        out[f"{t}_rolling_mean_20"] = r.rolling(20, min_periods=5).mean()
        out[f"{t}_rolling_std_20"] = r.rolling(20, min_periods=5).std()

    # ======================================================================
    # 2. MARKET CROSS FEATURES (SPY-centric)
    # ======================================================================
    if "SPY" in dfs:
        spy_df = dfs["SPY"]

        r_spy = _ensure_log_return(spy_df).reindex(idx).fillna(0.0)
        c_spy = dfs["SPY"]["close"].reindex(idx).ffill()

        cov = r_t.rolling(ROLLING_LONG, min_periods=10).cov(r_spy)
        var_spy = r_spy.rolling(ROLLING_LONG, min_periods=10).var() + EPS
        beta = cov / var_spy

        out["rolling_corr_target_SPY_20"] = r_t.rolling(20).corr(r_spy)
        out["target_beta_SPY_static"] = beta
        out["target_relative_SPY"] = (c_t / c_t.shift(20) - 1) - (
            c_spy / c_spy.shift(20) - 1
        )
        out["vol_ratio_target_SPY"] = r_t.rolling(20).std() / (
            r_spy.rolling(20).std() + EPS
        )
    else:
        out["rolling_corr_target_SPY_20"] = 0.0
        out["target_beta_SPY_static"] = 0.0
        out["target_relative_SPY"] = 0.0
        out["vol_ratio_target_SPY"] = 0.0

    # ======================================================================
    # 3. SECTOR LAYER
    # ======================================================================
    sector = get_sector_etf(target)

    if sector and sector in dfs:
        df = dfs[sector]
        r_s = _ensure_log_return(df).reindex(idx).fillna(0.0)
        c_s = df["close"].reindex(idx).ffill()

        cov = r_t.rolling(ROLLING_LONG).cov(r_s)
        var_s = r_s.rolling(ROLLING_LONG).var() + EPS
        beta_s = cov / var_s

        out[f"{sector}_log_return"] = r_s
        out[f"{sector}_rolling_std_20"] = r_s.rolling(20).std()

        out["rolling_corr_target_sector_20"] = r_t.rolling(20).corr(r_s)
        out["sector_beta_static"] = beta_s
        out["target_relative_sector"] = (c_t / c_t.shift(20) - 1) - (
            c_s / c_s.shift(20) - 1
        )
    else:
        out["rolling_corr_target_sector_20"] = 0.0
        out["sector_beta_static"] = 0.0
        out["target_relative_sector"] = 0.0

    # ======================================================================
    # 4. PEER EQUITY LAYER (TOP-K FIXED OR PRESELECTED)
    # ======================================================================
    peers = peers[:3]

    peer_returns = []

    for i in range(3):
        if i < len(peers) and peers[i] in dfs:
            p = peers[i]
            r_p = _ensure_log_return(dfs[p]).reindex(idx).fillna(0.0)

            out[f"peer_{i+1}_log_return"] = r_p
            out[f"peer_{i+1}_rolling_std_20"] = r_p.rolling(20).std()
            out[f"peer_{i+1}_rolling_corr_target_20"] = r_t.rolling(60).corr(r_p)
            out[f"target_relative_peer_{i+1}"] = r_t - r_p

            peer_returns.append(r_p)
        else:
            out[f"peer_{i+1}_log_return"] = 0.0
            out[f"peer_{i+1}_rolling_std_20"] = 0.0
            out[f"peer_{i+1}_rolling_corr_target_20"] = 0.0
            out[f"target_relative_peer_{i+1}"] = 0.0

    # Peer aggregates
    if peer_returns:
        peer_df = pd.concat(peer_returns, axis=1)

        out["peer_mean_return"] = peer_df.mean(axis=1)
        out["peer_dispersion"] = peer_df.std(axis=1)
        out["target_relative_peer_mean"] = r_t - peer_df.mean(axis=1)
    else:
        out["peer_mean_return"] = 0.0
        out["peer_dispersion"] = 0.0
        out["target_relative_peer_mean"] = 0.0

    # ======================================================================
    # 5. MARKET RISK INTERNALS
    # ======================================================================
    for t in MARKET_INTERNAL_TICKERS:
        if t not in dfs:
            out[f"{t}_log_return"] = 0.0
            out[f"{t}_rolling_std_20"] = 0.0
            out[f"target_beta_{t}_static"] = 0.0
            continue

        r_i = _ensure_log_return(dfs[t]).reindex(idx).fillna(0.0)
        # r_i = dfs[t]["log_return"].reindex(idx).fillna(0.0)

        cov = r_t.rolling(ROLLING_LONG).cov(r_i)
        var_i = r_i.rolling(ROLLING_LONG).var() + EPS
        beta_i = cov / var_i

        out[f"{t}_log_return"] = r_i
        out[f"{t}_rolling_std_20"] = r_i.rolling(20).std()
        out[f"target_beta_{t}_static"] = beta_i

    # ======================================================================
    # FINAL CLEANING
    # ======================================================================
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    logger.info(
        f"Computed {len(out.columns)} cross-ticker features for {len(out)} samples "
        f"(columns: {list(out.columns.tolist())}...)"
    )
    return out
