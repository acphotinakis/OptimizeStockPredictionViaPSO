"""
Symbol universe construction for target-specific multivariate LSTM.

Assembles the correct set of symbols for each prediction target:
  - Tier 1: Market structure (SPY, QQQ, IWM, DIA)
  - Tier 2: Sector ETF specific to target
  - Tier 3: Top-N correlated peers (selected on training data)
  - Tier 4: Market internals (UVXY, GLD, TLT)

Leakage prevention:
  - Peer selection runs on training data only
  - Peers frozen for validation/test splits
  - Correlation computed on returns, not prices
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class SymbolUniverseBuilder:
    """
    Constructs target-specific symbol universes for multivariate LSTM.

    Usage:
        builder = SymbolUniverseBuilder.from_config("config/symbol_universe.yaml")

        # Get universe for AAPL (includes sector ETF XLK)
        symbols = builder.get_universe("AAPL", dfs_train, fit=True)
        # Returns: ['AAPL', 'SPY', 'QQQ', 'IWM', 'DIA', 'XLK',
        #           'MSFT', 'GOOGL', 'META', 'UVXY', 'GLD', 'TLT']

        # Later, for validation (uses stored peers)
        symbols = builder.get_universe("AAPL", dfs_val, fit=False)
    """

    def __init__(
        self,
        market_context: List[str],
        sector_map: Dict[str, str],  # ticker -> sector_etf
        market_internals: List[str],
        peer_config: Dict[str, Any],
    ):
        self.market_context = market_context
        self.sector_map = sector_map
        self.market_internals = market_internals
        self.peer_config = peer_config

        # Stores fitted peers: {target_ticker: [peer1, peer2, peer3]}
        self._fitted_peers: Dict[str, List[str]] = {}

    @classmethod
    def from_config(cls, config_path: str | Path) -> "SymbolUniverseBuilder":
        """Load configuration from YAML file."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Build ticker -> sector_etf mapping
        sector_map = {}
        for sector_name, sector_data in cfg["sector_etfs"].items():
            etf = sector_data["etf"]
            for stock in sector_data["stocks"]:
                sector_map[stock] = etf

        return cls(
            market_context=cfg["market_context"],
            sector_map=sector_map,
            market_internals=cfg["market_internals"],
            peer_config=cfg["peer_selection"],
        )

    def get_universe(
        self,
        target_ticker: str,
        dfs: Dict[str, pd.DataFrame],
        fit: bool = False,
    ) -> List[str]:
        """
        Construct symbol universe for target ticker.

        Args:
            target_ticker: Stock to predict (e.g., 'AAPL')
            dfs: Dict of {ticker: DataFrame} with 'log_return' column
            fit: If True, select peers from data; if False, use stored peers

        Returns:
            List of ticker symbols in order: [target, market_ctx, sector, peers, internals]
        """
        universe = [target_ticker]

        # Tier 1: Market context (always included)
        universe.extend(self.market_context)

        # Tier 2: Sector ETF
        sector_etf = self.sector_map.get(target_ticker)
        if sector_etf:
            universe.append(sector_etf)
            logger.info(f"[{target_ticker}] Sector ETF: {sector_etf}")
        else:
            logger.warning(
                f"[{target_ticker}] No sector ETF mapping found. "
                f"Add to symbol_universe.yaml or it will use SPY as fallback."
            )

        # Tier 3: Peers
        if fit:
            peers = self._select_peers(target_ticker, dfs)
            self._fitted_peers[target_ticker] = peers
        else:
            if target_ticker not in self._fitted_peers:
                raise RuntimeError(
                    f"Peers for {target_ticker} not fitted. "
                    f"Call get_universe(..., fit=True) on training data first."
                )
            peers = self._fitted_peers[target_ticker]

        universe.extend(peers)
        logger.info(f"[{target_ticker}] Peers: {peers}")

        # Tier 4: Market internals
        universe.extend(self.market_internals)

        # Remove duplicates while preserving order
        seen = set()
        unique_universe = []
        for symbol in universe:
            if symbol not in seen:
                seen.add(symbol)
                unique_universe.append(symbol)

        logger.info(
            f"[{target_ticker}] Universe: {len(unique_universe)} symbols: {unique_universe}"
        )
        return unique_universe

    def _select_peers(
        self,
        target_ticker: str,
        dfs: Dict[str, pd.DataFrame],
    ) -> List[str]:
        """
        Select top-N correlated peers from same sector.

        Uses Pearson correlation on log returns over the full training period.
        Excludes:
          - The target itself
          - Market context symbols (SPY, QQQ, etc.)
          - Sector ETFs
          - Market internals

        Returns:
            List of up to max_peers ticker symbols
        """
        if target_ticker not in dfs:
            logger.warning(f"Target {target_ticker} not in dfs, returning empty peers")
            return []

        max_peers = self.peer_config["max_peers"]
        min_corr = self.peer_config["min_correlation"]

        # Get target returns
        target_returns = dfs[target_ticker]["log_return"]

        # Build candidate universe (exclude context symbols)
        exclude_set = set(self.market_context + self.market_internals + [target_ticker])
        exclude_set.update(self.sector_map.values())  # exclude all sector ETFs

        candidates = [t for t in dfs.keys() if t not in exclude_set]

        if not candidates:
            logger.warning(f"No peer candidates for {target_ticker}")
            return []

        # Compute correlations
        correlations = {}
        for candidate in candidates:
            if "log_return" not in dfs[candidate].columns:
                continue

            candidate_returns = dfs[candidate]["log_return"]

            # Align indices (inner join)
            aligned = pd.DataFrame(
                {
                    "target": target_returns,
                    "candidate": candidate_returns,
                }
            ).dropna()

            if len(aligned) < 100:  # require at least 100 bars
                continue

            corr = aligned["target"].corr(aligned["candidate"])
            if pd.notna(corr) and abs(corr) >= min_corr:
                correlations[candidate] = abs(corr)

        # Select top N by absolute correlation
        if not correlations:
            logger.warning(
                f"No peers found for {target_ticker} with min_corr={min_corr}"
            )
            return []

        top_peers = sorted(correlations, key=correlations.get, reverse=True)[:max_peers]

        logger.info(
            f"[{target_ticker}] Selected {len(top_peers)} peers from {len(candidates)} candidates: "
            f"{[(p, f'{correlations[p]:.3f}') for p in top_peers]}"
        )

        return top_peers

    def get_fitted_peers(self, target_ticker: str) -> List[str]:
        """Return stored peers for a target (after fitting)."""
        return self._fitted_peers.get(target_ticker, [])

    def save_peers(self, output_path: str | Path) -> None:
        """Save fitted peers to disk for reproducibility."""
        import json

        with open(output_path, "w") as f:
            json.dump(self._fitted_peers, f, indent=2)
        logger.info(f"Saved fitted peers to {output_path}")

    def load_peers(self, input_path: str | Path) -> None:
        """Load previously fitted peers from disk."""
        import json

        with open(input_path) as f:
            self._fitted_peers = json.load(f)
        logger.info(f"Loaded fitted peers from {input_path}")


def align_symbol_universe(
    symbol_list: List[str],
    dfs: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Inner-join all symbols in universe on timestamp index.

    Ensures every symbol has data at every timestamp. Drops timestamps
    where any symbol is missing (conservative approach to avoid forward-fill
    leakage across symbols).

    Args:
        symbol_list: Ordered list of symbols to include
        dfs: Dict of {ticker: DataFrame} with DatetimeIndex

    Returns:
        Dict of {ticker: DataFrame} with aligned indices
    """
    # Collect all DataFrames
    dfs_to_align = {}
    for symbol in symbol_list:
        if symbol not in dfs:
            logger.warning(f"Symbol {symbol} not in dfs, skipping from universe")
            continue
        dfs_to_align[symbol] = dfs[symbol]

    if not dfs_to_align:
        raise ValueError("No symbols available for alignment")

    # Find common index (inner join)
    common_index = None
    for symbol, df in dfs_to_align.items():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

    logger.info(
        f"Aligned {len(dfs_to_align)} symbols: "
        f"{len(common_index)} common timestamps"
    )

    # Reindex all DataFrames to common index
    aligned = {}
    for symbol, df in dfs_to_align.items():
        aligned[symbol] = df.loc[common_index].copy()

    return aligned
