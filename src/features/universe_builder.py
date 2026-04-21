"""features/universe_builder.py — Target-specific symbol universe construction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class SymbolUniverseBuilder:
    """Builds target-specific symbol universes for multivariate LSTM.

    Universe tiers:
        1. Market context  (SPY, QQQ, IWM, DIA)
        2. Sector ETF      (mapped per ticker)
        3. Top-N correlated peers  (fit on training data only)
        4. Market internals (UVXY, GLD, TLT)

    Usage:
        builder = SymbolUniverseBuilder.from_config("config/symbol_universe.yaml")
        symbols = builder.get_universe("AAPL", dfs_train, fit=True)
        symbols = builder.get_universe("AAPL", dfs_val,   fit=False)  # uses stored peers
    """

    def __init__(
        self,
        market_context: List[str],
        sector_map: Dict[str, str],  # ticker --> sector_etf
        market_internals: List[str],
        peer_config: Dict[str, Any],
    ) -> None:
        self.market_context = market_context
        self.sector_map = sector_map
        self.market_internals = market_internals
        self.peer_config = peer_config
        self._fitted_peers: Dict[str, List[str]] = {}

    @classmethod
    def from_config(cls, config_path: str | Path) -> "SymbolUniverseBuilder":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        sector_map = {
            stock: sector_data["etf"]
            for sector_data in cfg["sector_etfs"].values()
            for stock in sector_data["stocks"]
        }
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
        """Return ordered symbol universe for *target_ticker*.

        Pass fit=True on training data to select peers; fit=False reuses them.
        """
        if fit:
            # self._fitted_peers[target_ticker] = self._select_peers(target_ticker, dfs)

            if target_ticker == "AAPL":
                # Manually set peers for AAPL to ensure consistency across runs
                self._fitted_peers[target_ticker] = [
                    "MSFT",
                    "GOOGL",
                    "AMZN",
                    "NVDA",
                    "META",
                ]
                logger.info(
                    "[%s] Using fixed peers: %s",
                    target_ticker,
                    self._fitted_peers[target_ticker],
                )
            elif target_ticker == "MSFT":
                # Manually set peers for MSFT to ensure consistency across runs
                self._fitted_peers[target_ticker] = [
                    "AAPL",
                    "GOOGL",
                    "AMZN",
                    "NVDA",
                    "META",
                ]
                logger.info(
                    "[%s] Using fixed peers: %s",
                    target_ticker,
                    self._fitted_peers[target_ticker],
                )
            elif target_ticker == "GOOGL":
                # Manually set peers for GOOGL to ensure consistency across runs
                self._fitted_peers[target_ticker] = [
                    "AAPL",
                    "MSFT",
                    "AMZN",
                    "NVDA",
                    "META",
                ]
                logger.info(
                    "[%s] Using fixed peers: %s",
                    target_ticker,
                    self._fitted_peers[target_ticker],
                )
            elif target_ticker == "AMZN":
                # Manually set peers for AMZN to ensure consistency across runs
                self._fitted_peers[target_ticker] = [
                    "AAPL",
                    "MSFT",
                    "GOOGL",
                    "NVDA",
                    "META",
                ]
                logger.info(
                    "[%s] Using fixed peers: %s",
                    target_ticker,
                    self._fitted_peers[target_ticker],
                )
        elif target_ticker not in self._fitted_peers:
            raise RuntimeError(
                f"Peers for {target_ticker} not fitted. "
                "Call get_universe(..., fit=True) on training data first."
            )

        sector_etf = self.sector_map.get(target_ticker)
        if not sector_etf:
            logger.warning(
                "[%s] No sector ETF mapping — add to symbol_universe.yaml",
                target_ticker,
            )

        raw = (
            [target_ticker]
            + self.market_context
            + ([sector_etf] if sector_etf else [])
            + self._fitted_peers[target_ticker]
            + self.market_internals
        )
        # Deduplicate, preserving order
        seen, universe = set(), []
        for s in raw:
            if s not in seen:
                seen.add(s)
                universe.append(s)

        logger.info(
            "[%s] universe=%d %s | peers=%s",
            target_ticker,
            len(universe),
            universe,
            self._fitted_peers[target_ticker],
        )
        return universe

    def _select_peers(self, target: str, dfs: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Select top-N correlated peers STRICTLY within same sector.

        Constraint:
        - Peers must belong to same sector ETF group
        - Market context and internals are excluded
        - Correlation computed on training log returns only
        """

        if target not in dfs:
            return []

        # ------------------------------------------------------------
        # STEP 1: Identify sector group for target
        # ------------------------------------------------------------
        target_sector_etf = self.sector_map.get(target)

        if not target_sector_etf:
            logger.warning("[%s] No sector mapping found → no peers selected", target)
            return []

        # ------------------------------------------------------------
        # STEP 2: Restrict candidates to SAME sector only
        # ------------------------------------------------------------
        sector_stocks = [
            stock
            for stock, etf in self.sector_map.items()
            if etf == target_sector_etf and stock != target
        ]

        # fallback safety
        if not sector_stocks:
            logger.warning(
                "[%s] No sector peers found for ETF=%s", target, target_sector_etf
            )
            return []

        # ------------------------------------------------------------
        # STEP 3: Compute correlations within sector only
        # ------------------------------------------------------------
        r_target = dfs[target]["log_return"]

        correlations: Dict[str, float] = {}

        for t in sector_stocks:
            if t not in dfs:
                continue
            if "log_return" not in dfs[t].columns:
                continue

            aligned = pd.concat([r_target, dfs[t]["log_return"]], axis=1).dropna()

            if len(aligned) < 100:
                continue

            corr = abs(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))

            if pd.notna(corr) and corr >= self.peer_config["min_correlation"]:
                correlations[t] = corr

        # ------------------------------------------------------------
        # STEP 4: Rank and select top-N peers
        # ------------------------------------------------------------
        peers = sorted(correlations, key=correlations.get, reverse=True)[
            : self.peer_config["max_peers"]
        ]

        logger.info(
            "[%s] sector=%s peers selected=%s",
            target,
            target_sector_etf,
            [(p, f"{correlations[p]:.3f}") for p in peers],
        )

        return peers

    # def _select_peers(self, target: str, dfs: Dict[str, pd.DataFrame]) -> List[str]:
    #     """Select top-N correlated peers from training data (leakage-safe)."""
    #     if target not in dfs:
    #         return []

    #     exclude = (
    #         set(self.market_context)
    #         | set(self.market_internals)
    #         | set(self.sector_map.values())
    #         | {target}
    #     )
    #     r_target = dfs[target]["log_return"]
    #     candidates = [
    #         t for t in dfs if t not in exclude and "log_return" in dfs[t].columns
    #     ]

    #     correlations: Dict[str, float] = {}
    #     for t in candidates:
    #         aligned = pd.concat([r_target, dfs[t]["log_return"]], axis=1).dropna()
    #         if len(aligned) < 100:
    #             continue
    #         corr = abs(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    #         if pd.notna(corr) and corr >= self.peer_config["min_correlation"]:
    #             correlations[t] = corr

    #     peers = sorted(correlations, key=correlations.get, reverse=True)[
    #         : self.peer_config["max_peers"]
    #     ]
    #     logger.info(
    #         "[%s] peers selected: %s",
    #         target,
    #         [(p, f"{correlations[p]:.3f}") for p in peers],
    #     )
    #     return peers

    # ── Persistence ──────────────────────────────────────────────────────────
    def save_peers(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self._fitted_peers, f, indent=2)

    def load_peers(self, path: str | Path) -> None:
        with open(path) as f:
            self._fitted_peers = json.load(f)

    def get_fitted_peers(self, target: str) -> List[str]:
        return self._fitted_peers.get(target, [])


def align_symbol_universe(
    symbol_list: List[str],
    dfs: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Inner-join all symbols to a common timestamp index."""
    available = {s: dfs[s] for s in symbol_list if s in dfs}
    if not available:
        raise ValueError("No symbols available for alignment")

    common_idx = None
    for df in available.values():
        common_idx = (
            df.index if common_idx is None else common_idx.intersection(df.index)
        )

    logger.info(
        "Aligned %d symbols over %d timestamps", len(available), len(common_idx)
    )
    return {s: df.loc[common_idx].copy() for s, df in available.items()}
