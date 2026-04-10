"""features/cross_ticker.py — Cross-ticker and market-level features."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# SPY feature columns emitted when SPY is absent or target == SPY
_SPY_COLS = [
    "beta_spy_60",
    "corr_spy_20",
    "corr_spy_60",
    "alpha_spy",
    "rel_strength_spy_20",
    "spy_lag_return_1",
    "sector_rotation_20",
    "spy_atr_14",
    "spy_volume_ratio",
    "vix_proxy",
]


def compute_cross_ticker_features(
    target_ticker: str,
    dfs: Dict[str, pd.DataFrame],
    peer_tickers: List[str],
    rolling_window: int = 60,
) -> pd.DataFrame:
    """15 cross-ticker / market features for *target_ticker*.

    Args:
        target_ticker:  Ticker being predicted.
        dfs:            Full universe dict with 'log_return', 'close', 'volume'.
        peer_tickers:   Pre-selected peers (fit on training data, never reselected here).
        rolling_window: Window for beta / correlation calculations.
    """
    df_target = dfs[target_ticker]
    idx = df_target.index
    r_target = df_target["log_return"]
    C_target = df_target["close"]
    out = pd.DataFrame(index=idx)

    # ── SPY features ─────────────────────────────────────────────────────────
    if "SPY" in dfs and target_ticker != "SPY":
        spy = dfs["SPY"]
        r_spy = spy["log_return"].reindex(idx).fillna(0.0)
        C_spy = spy["close"].reindex(idx).ffill()

        cov = r_target.rolling(rolling_window, min_periods=10).cov(r_spy)
        var_spy = r_spy.rolling(rolling_window, min_periods=10).var() + 1e-10
        beta = cov / var_spy

        out[f"beta_spy_{rolling_window}"] = beta
        out["corr_spy_20"] = r_target.rolling(20, min_periods=5).corr(r_spy)
        out[f"corr_spy_{rolling_window}"] = r_target.rolling(
            rolling_window, min_periods=10
        ).corr(r_spy)
        out["alpha_spy"] = r_target - beta * r_spy
        out["rel_strength_spy_20"] = (C_target / C_target.shift(20) - 1) / (
            (C_spy / C_spy.shift(20) - 1).abs() + 1e-10
        )
        out["spy_lag_return_1"] = r_spy.shift(1)

        vol_t = r_target.rolling(20, min_periods=5).std() + 1e-10
        vol_s = r_spy.rolling(20, min_periods=5).std() + 1e-10
        out["sector_rotation_20"] = (
            r_target.rolling(20, min_periods=5).sum() / vol_t
            - r_spy.rolling(20, min_periods=5).sum() / vol_s
        )

        spy_df = spy.reindex(idx).ffill()
        if {"high", "low"}.issubset(spy_df.columns):
            out["spy_atr_14"] = (
                (spy_df["high"] - spy_df["low"]).rolling(14, min_periods=1).mean()
            )
        else:
            out["spy_atr_14"] = 0.0

        spy_vol = spy_df.get("volume", pd.Series(0, index=idx))
        out["spy_volume_ratio"] = spy_vol / (
            spy_vol.rolling(20, min_periods=1).mean() + 1e-10
        )
        out["vix_proxy"] = np.sqrt(252 * 390) * np.sqrt(
            r_spy.pow(2).rolling(30, min_periods=5).sum()
        )
    else:
        for col in _SPY_COLS:
            out[col] = 0.0

    # ── Universe breadth ──────────────────────────────────────────────────────
    all_rets = pd.DataFrame(
        {t: dfs[t]["log_return"].reindex(idx).fillna(0.0) for t in dfs}
    )
    out["mkt_breadth"] = (all_rets > 0).mean(axis=1)
    out["universe_mean_ret"] = all_rets.mean(axis=1)

    # ── Peer correlations (up to 3) ───────────────────────────────────────────
    for rank, peer in enumerate(peer_tickers[:3], 1):
        col = f"peer_corr_{peer}"
        if peer in dfs:
            r_peer = dfs[peer]["log_return"].reindex(idx).fillna(0.0)
            out[col] = r_target.rolling(60, min_periods=10).corr(r_peer)
        else:
            out[col] = 0.0

    # Pad to exactly 3 peer columns so feature count stays constant
    for rank in range(len(peer_tickers[:3]) + 1, 4):
        out[f"peer_corr_{rank}"] = 0.0

    return out.fillna(0.0)
