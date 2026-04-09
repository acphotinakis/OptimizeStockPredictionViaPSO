"""
src/features/cross_ticker.py

Cross-ticker features: beta to SPY, rolling correlation, relative strength,
market breadth, VIX proxy, and peer correlations.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_cross_ticker_features(
    target_ticker: str,
    dfs: Dict[str, pd.DataFrame],
    peer_tickers: List[str] | None = None,
    rolling_window: int = 60,
    return_peer_tickers: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, List[str]]:
    """Compute 15 cross-ticker and market-level features.

    Args:
        target_ticker: The ticker whose features we are building.
        dfs: Dict mapping ticker --> cleaned DataFrame (with 'log_return', 'close', 'volume').
        peer_tickers: Top correlated peers (up to 3); computed from training data if None.
        rolling_window: Rolling window in minutes for correlation/beta.
        return_peer_tickers: If True, return (features, peer_tickers) tuple.

    Returns:
        DataFrame of cross-ticker feature columns on the target ticker's index.
        If return_peer_tickers=True, returns (DataFrame, List[str]) tuple.
    """
    if target_ticker not in dfs:
        raise KeyError(f"Target ticker {target_ticker} not in dfs.")

    df_target = dfs[target_ticker]
    out = pd.DataFrame(index=df_target.index)

    r_target = df_target["log_return"]
    C_target = df_target["close"]

    # ---- SPY features -------------------------------------------------------
    if "SPY" in dfs and target_ticker != "SPY":
        r_spy = dfs["SPY"]["log_return"].reindex(df_target.index).fillna(0.0)
        # Only forward fill (no bfill which would use future data)
        C_spy = dfs["SPY"]["close"].reindex(df_target.index).ffill()
        # Remaining NaN (leading gaps) will be handled by fillna(0.0) at end

        # Rolling beta to SPY (use rolling_window parameter in name)
        cov = r_target.rolling(rolling_window, min_periods=10).cov(r_spy)
        var_spy = r_spy.rolling(rolling_window, min_periods=10).var() + 1e-10
        out[f"beta_spy_{rolling_window}"] = cov / var_spy

        # Rolling Pearson correlations
        out["corr_spy_20"] = r_target.rolling(20, min_periods=5).corr(r_spy)
        out[f"corr_spy_{rolling_window}"] = r_target.rolling(
            rolling_window, min_periods=10
        ).corr(r_spy)

        # Residual (alpha) return
        out["alpha_spy"] = r_target - out[f"beta_spy_{rolling_window}"] * r_spy

        # Relative strength vs SPY (20-bar window)
        ret_20_target = C_target / C_target.shift(20) - 1
        ret_20_spy = C_spy / C_spy.shift(20) - 1
        out["rel_strength_spy_20"] = ret_20_target / (ret_20_spy.abs() + 1e-10)

        # SPY lagged return
        out["spy_lag_return_1"] = r_spy.shift(1)

        # Sector rotation score (normalised momentum difference)
        vol_target = r_target.rolling(20, min_periods=5).std() + 1e-10
        vol_spy = r_spy.rolling(20, min_periods=5).std() + 1e-10
        mom_target = r_target.rolling(20, min_periods=5).sum() / vol_target
        mom_spy = r_spy.rolling(20, min_periods=5).sum() / vol_spy
        out["sector_rotation_20"] = mom_target - mom_spy

        # SPY ATR
        spy_df = dfs["SPY"].reindex(df_target.index).ffill()
        if "high" in spy_df and "low" in spy_df:
            spy_tr = spy_df["high"] - spy_df["low"]
            out["spy_atr_14"] = spy_tr.rolling(14, min_periods=1).mean()
        else:
            out["spy_atr_14"] = 0.0

        # SPY relative volume
        spy_vol = (
            spy_df["volume"]
            if "volume" in spy_df.columns
            else pd.Series(0, index=df_target.index)
        )
        out["spy_volume_ratio"] = spy_vol / (
            spy_vol.rolling(20, min_periods=1).mean() + 1e-10
        )

        # VIX proxy: annualised RV30 of SPY
        out["vix_proxy"] = np.sqrt(252 * 390) * np.sqrt(
            r_spy.pow(2).rolling(30, min_periods=5).sum()
        )
    else:
        if target_ticker == "SPY":
            logger.info(
                "Target is SPY; skipping self-referential SPY features. "
                "Universe features (mkt_breadth, universe_mean_ret, peer_corr) "
                "will provide context from other tickers."
            )
        else:
            logger.info("SPY not in dfs; SPY-based features will be zero.")

        for col in [
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
        ]:
            out[col] = 0.0

    # ---- Universe market breadth -------------------------------------------
    all_returns = pd.DataFrame(
        {t: dfs[t]["log_return"].reindex(df_target.index).fillna(0.0) for t in dfs}
    )
    out["mkt_breadth"] = (all_returns > 0).mean(axis=1)
    out["universe_mean_ret"] = all_returns.mean(axis=1)

    # ---- Peer correlations (top-3 most correlated) -------------------------
    # WARNING: peer_tickers should be pre-selected on TRAINING data only to avoid look-ahead bias
    # If None is passed here, we compute on the full series which includes validation/test data
    computed_peer_tickers = None
    if peer_tickers is None:
        logger.info(
            f"peer_tickers is None for {target_ticker}. Computing correlations on FULL series "
            "which may include validation/test data. This creates LOOK-AHEAD BIAS. "
            "Peers should be selected on training data only and passed explicitly."
        )
        others = [t for t in dfs if t != target_ticker]
        if len(others) > 0:
            corrs = {}
            for t in others:
                r_other = dfs[t]["log_return"].reindex(df_target.index).fillna(0.0)
                corrs[t] = float(r_target.corr(r_other))
            peer_tickers = sorted(corrs, key=lambda x: abs(corrs[x]), reverse=True)[:3]
            computed_peer_tickers = peer_tickers  # Store for return
            logger.info(
                f"Auto-selected peers for {target_ticker} on FULL data: {peer_tickers}"
            )
        else:
            peer_tickers = []
            computed_peer_tickers = []

    for rank, peer in enumerate(peer_tickers[:3], 1):
        if peer in dfs:
            r_peer = dfs[peer]["log_return"].reindex(df_target.index).fillna(0.0)
            out[f"peer_corr_{peer}"] = r_target.rolling(60, min_periods=10).corr(r_peer)
        else:
            out[f"peer_corr_{rank}"] = 0.0

    # Fill missing peer columns with zeros
    for rank in range(len(peer_tickers[:3]) + 1, 4):
        out[f"peer_corr_{rank}"] = 0.0

    result = out.fillna(0.0)

    if return_peer_tickers:
        return result, (
            computed_peer_tickers if computed_peer_tickers is not None else peer_tickers
        )
    return result
