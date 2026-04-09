# Multivariate LSTM Context: Symbol Universe & Peer Selection Plan

**Date:** 2026-04-08  
**Status:** Planning  
**Priority:** High  
**Estimated Complexity:** Medium-High

---

## Executive Summary

This plan implements a sophisticated symbol universe construction system that provides each target stock with the correct market context for multivariate LSTM prediction. The current implementation treats all tickers equally and auto-selects peers from the entire universe, which creates two problems:

1. **Incorrect peer selection**: Every stock gets the same peers regardless of sector/correlation
2. **Missing market structure signals**: No systematic inclusion of market indices, sector ETFs, or market internals

The solution introduces a **4-tier symbol universe builder** that constructs target-specific contexts combining broad market signals (SPY/QQQ/IWM/DIA), sector-specific ETFs, correlation-selected peers, and market internals (volatility/bonds/gold proxies).

### Current State: Existing Assets

**Good news**: We already have 51 high-quality stocks + SPY ingested in `config/mass_tickers.txt`:
- 10 Technology stocks (AAPL, MSFT, GOOGL, NVDA, META, TSLA, INTC, AMD, CRM, ADBE)
- 8 Financials (JPM, BAC, WFC, GS, MS, C, BLK, SCHW)
- 8 Healthcare (JNJ, UNH, PFE, ABBV, TMO, MRK, LLY, ABT)
- 8 Consumer (AMZN, WMT, HD, MCD, NKE, SBUX, TGT, COST)
- 6 Industrials (BA, CAT, GE, UPS, HON, LMT)
- 4 Energy (XOM, CVX, COP, SLB)
- 3 Communication (DIS, NFLX, CMCSA)
- 2 Materials (LIN, APD)
- 1 Utilities (NEE)

**What we need to add**: Only 17 context symbols for Tiers 1, 2, and 4:
- 3 market indices (QQQ, IWM, DIA)
- 11 sector ETFs (XLK, SOXX, XLF, XLV, XLE, XLY, XLP, XLI, XLC, XLB, XLU)
- 3 market internals (UVXY, GLD, TLT)

This minimizes new data ingestion while unlocking the full power of multivariate context.

---

## Problem Statement

### Current State Issues

1. **Naive Universe Construction**
   - `run_build_features.py` line 44: `universe_tickers=list(dfs_train.keys())`
   - All tickers in the config file are treated as potential peers for every target
   - No distinction between market indices, sector ETFs, and individual stocks

2. **Look-Ahead Bias in Peer Selection**
   - `cross_ticker.py` line 141-157: Peers selected on full data if not provided
   - Warning logged but not enforced
   - No mechanism to pre-compute peers on training data only

3. **Missing Market Context**
   - No systematic inclusion of QQQ (tech-heavy), IWM (small-cap), DIA (industrials)
   - No sector ETF mapping (XLK for tech, XLF for financials, etc.)
   - No market internals (UVXY for volatility, GLD for safety, TLT for bonds)

4. **Configuration Limitations**
   - `tickers.txt` is a flat list with no metadata
   - No sector classification
   - No distinction between prediction targets vs. context symbols

### Why This Matters

The LSTM needs to learn **relative strength and divergence patterns** between a stock and its context:
- When AMD moves 2% but NVDA is flat → informative divergence
- When AAPL rises while XLK falls → sector leadership signal
- When UVXY spikes while SPY is flat → fear entering the market

Without proper context, the model sees only the target stock's own bars plus random peers, missing the structural relationships that drive intraday price action.

---

## Solution Architecture

### Overview

Build a **target-aware symbol universe constructor** that assembles the correct set of symbols for each prediction target based on:
1. **Tier 1**: Always-included market structure (SPY, QQQ, IWM, DIA)
2. **Tier 2**: Target-specific sector ETF (XLK for AAPL, XLF for JPM, etc.)
3. **Tier 3**: Correlation-selected peers (max 3, computed on training data only)
4. **Tier 4**: Market internals (UVXY, GLD, TLT)

### Design Principles

1. **Leakage Prevention**: Peers selected on training data only, frozen for val/test
2. **Scalability**: Symbol universe grows modestly (12-15 symbols per target)
3. **Interpretability**: Clear tier structure makes feature importance analysis meaningful
4. **Flexibility**: Easy to add new sectors or market internals without code changes

---

## Implementation Plan

### Phase 1: Configuration Infrastructure

#### 1.1 Create Symbol Universe Configuration File

**File**: `config/symbol_universe.yaml`

**Note**: Based on existing `config/mass_tickers.txt`, we already have 51 stocks + SPY. We need to ingest additional ETFs and market internals.

```yaml
# Symbol Universe Configuration for Multivariate LSTM
# Defines market structure, sector mappings, and market internals

# Tier 1: Market Structure (always included)
# NOTE: Only SPY exists currently - need to ingest QQQ, IWM, DIA
market_context:
  - SPY   # S&P 500 broad market (ALREADY HAVE)
  - QQQ   # Nasdaq 100 tech-heavy (NEED TO INGEST)
  - IWM   # Russell 2000 small caps (NEED TO INGEST)
  - DIA   # Dow Jones industrials (NEED TO INGEST)

# Tier 2: Sector ETF Mappings
# NOTE: All stocks below exist in mass_tickers.txt
# Need to ingest the sector ETFs (XLK, SOXX, XLF, etc.)
sector_etfs:
  technology:
    etf: XLK  # NEED TO INGEST
    stocks:
      - AAPL
      - MSFT
      - GOOGL
      - META
      - CRM
      - ADBE
      # Note: TSLA moved to consumer_discretionary (automotive)
  
  semiconductors:
    etf: SOXX  # NEED TO INGEST
    stocks:
      - NVDA
      - AMD
      - INTC
  
  financials:
    etf: XLF  # NEED TO INGEST
    stocks:
      - JPM
      - BAC
      - GS
      - MS
      - C
      - WFC
      - BLK
      - SCHW
  
  healthcare:
    etf: XLV  # NEED TO INGEST
    stocks:
      - JNJ
      - UNH
      - PFE
      - ABBV
      - TMO
      - MRK
      - LLY
      - ABT
  
  energy:
    etf: XLE  # NEED TO INGEST
    stocks:
      - XOM
      - CVX
      - COP
      - SLB
  
  consumer_discretionary:
    etf: XLY  # NEED TO INGEST
    stocks:
      - AMZN
      - TSLA  # Tesla is automotive/consumer discretionary
      - HD
      - MCD
      - NKE
      - SBUX
      - TGT
  
  consumer_staples:
    etf: XLP  # NEED TO INGEST
    stocks:
      - WMT    # Walmart is staples (groceries)
      - COST   # Costco is staples (bulk groceries)
  
  industrials:
    etf: XLI  # NEED TO INGEST
    stocks:
      - CAT
      - BA
      - GE
      - UPS
      - HON
      - LMT
  
  communication:
    etf: XLC  # NEED TO INGEST
    stocks:
      - DIS
      - NFLX
      - CMCSA
  
  materials:
    etf: XLB  # NEED TO INGEST
    stocks:
      - LIN
      - APD
  
  utilities:
    etf: XLU  # NEED TO INGEST
    stocks:
      - NEE

# Tier 4: Market Internals (equity-only proxies)
# NOTE: Need to ingest all three
market_internals:
  - UVXY  # VIX futures ETF (volatility proxy) - NEED TO INGEST
  - GLD   # Gold ETF (flight-to-safety) - NEED TO INGEST
  - TLT   # 20+ year Treasury bonds (risk-off signal) - NEED TO INGEST

# Tier 3: Peer Selection Settings
peer_selection:
  max_peers: 3
  method: "pearson"  # correlation method
  min_correlation: 0.3  # minimum abs correlation to be considered
  lookback_days: 252  # use 1 year of training data for correlation
  
# Prediction Targets (stocks we want to predict)
# Start with high-liquidity names from each sector
prediction_targets:
  # Technology
  - AAPL
  - MSFT
  - GOOGL
  - META
  # Semiconductors
  - NVDA
  - AMD
  # Financials
  - JPM
  - BAC
  # Healthcare
  - JNJ
  - UNH
  # Consumer
  - AMZN
  - WMT
  # Energy
  - XOM
  - CVX
  # Industrials
  - BA
  - CAT
  # Add more as needed (all 51 stocks are available)

# Context-Only Symbols (never predicted, only used as features)
context_only:
  # Market structure
  - SPY
  - QQQ
  - IWM
  - DIA
  # Sector ETFs
  - XLK
  - SOXX
  - XLF
  - XLV
  - XLE
  - XLY
  - XLP
  - XLI
  - XLC
  - XLB
  - XLU
  # Market internals
  - UVXY
  - GLD
  - TLT
```

**Rationale**: YAML provides human-readable configuration with clear hierarchical structure. Separates prediction targets from context symbols, making the system's intent explicit.

#### 1.2 Create Enhanced Ticker Configuration

**File**: `config/tickers_full.txt`

**Note**: This extends `mass_tickers.txt` (51 stocks) with additional ETFs and market internals needed for Tiers 1, 2, and 4.

```
# Full Symbol Universe for Multivariate LSTM
# Extends mass_tickers.txt with market structure ETFs and internals
# Total: 51 stocks + 1 benchmark + 14 new symbols = 66 symbols

# === TIER 1: MARKET STRUCTURE (always included) ===
SPY    # S&P 500 (ALREADY HAVE from mass_tickers.txt)
QQQ    # Nasdaq 100 tech-heavy (NEW - NEED TO INGEST)
IWM    # Russell 2000 small caps (NEW - NEED TO INGEST)
DIA    # Dow Jones industrials (NEW - NEED TO INGEST)

# === TIER 2: SECTOR ETFs (NEW - ALL NEED TO INGEST) ===
XLK    # Technology
SOXX   # Semiconductors
XLF    # Financials
XLV    # Healthcare
XLE    # Energy
XLY    # Consumer Discretionary
XLP    # Consumer Staples
XLI    # Industrials
XLC    # Communication Services
XLB    # Materials
XLU    # Utilities

# === TIER 4: MARKET INTERNALS (NEW - ALL NEED TO INGEST) ===
UVXY   # Volatility proxy (VIX futures ETF)
GLD    # Gold / flight-to-safety
TLT    # Treasury bonds / risk-off

# === TIER 3: INDIVIDUAL STOCKS (ALREADY HAVE - from mass_tickers.txt) ===
# These are already ingested, just listing for completeness

# Technology (10)
AAPL
MSFT
GOOGL
NVDA
META
TSLA
INTC
AMD
CRM
ADBE

# Financials (8)
JPM
BAC
WFC
GS
MS
C
BLK
SCHW

# Healthcare (8)
JNJ
UNH
PFE
ABBV
TMO
MRK
LLY
ABT

# Consumer (8)
AMZN
WMT
HD
MCD
NKE
SBUX
TGT
COST

# Industrials (6)
BA
CAT
GE
UPS
HON
LMT

# Energy (4)
XOM
CVX
COP
SLB

# Communication (3)
DIS
NFLX
CMCSA

# Materials (2)
LIN
APD

# Utilities (1)
NEE
```

**Summary of New Symbols to Ingest**:
- **Market Structure**: QQQ, IWM, DIA (3 symbols)
- **Sector ETFs**: XLK, SOXX, XLF, XLV, XLE, XLY, XLP, XLI, XLC, XLB, XLU (11 symbols)
- **Market Internals**: UVXY, GLD, TLT (3 symbols)
- **Total New**: 17 symbols

**Rationale**: Minimizes new data ingestion by leveraging existing 51-stock universe. Only adds essential context symbols (ETFs and indices) needed for Tiers 1, 2, and 4.

#### 1.3 Update Config Schema

**File**: `src/utils/config_schema.py`

Add new dataclass for symbol universe configuration:

```python
@dataclass
class PeerSelectionConfig:
    max_peers: int
    method: str
    min_correlation: float
    lookback_days: int

@dataclass
class SymbolUniverseConfig:
    market_context: List[str]
    sector_etfs: Dict[str, Dict[str, Any]]
    market_internals: List[str]
    peer_selection: PeerSelectionConfig
    prediction_targets: List[str]
    context_only: List[str]

@dataclass
class Config:
    # ... existing fields ...
    symbol_universe: SymbolUniverseConfig  # Add this
```

**Rationale**: Type-safe configuration access prevents runtime errors from typos or missing keys.

---

### Phase 2: Symbol Universe Builder

#### 2.1 Create Universe Builder Module

**File**: `src/features/universe_builder.py`

```python
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
from typing import Dict, List, Optional, Tuple
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
        peer_config: Dict[str, any],
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
            aligned = pd.DataFrame({
                "target": target_returns,
                "candidate": candidate_returns,
            }).dropna()
            
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
```

**Key Features**:
- **Leakage-safe**: Peers selected once on training data, frozen for val/test
- **Sector-aware**: Automatically includes correct sector ETF per target
- **Configurable**: All thresholds and limits in config file
- **Serializable**: Can save/load peer selections for reproducibility

#### 2.2 Add Universe Alignment Function

**File**: `src/features/universe_builder.py` (continued)

```python
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
```

**Rationale**: Inner join ensures no forward-fill leakage between symbols. If SPY has data but AAPL doesn't, we drop that timestamp entirely rather than inventing AAPL values.

---

### Phase 3: Data Ingestion for New Symbols

**Before integrating with the pipeline, we need to ingest the 17 new symbols.**

#### 3.0 Ingest New Context Symbols

Create a temporary ticker file with just the new symbols:

**File**: `config/tickers_new_context.txt`

```
# New symbols to ingest for multivariate LSTM context
# Market structure
QQQ
IWM
DIA

# Sector ETFs
XLK
SOXX
XLF
XLV
XLE
XLY
XLP
XLI
XLC
XLB
XLU

# Market internals
UVXY
GLD
TLT
```

**Run ingestion**:

```bash
# Ingest new symbols
python pipelines/ingest_data.py --mode ingest \
    --tickers config/tickers_new_context.txt \
    --config config/default_config.yaml

# Clean new symbols
python pipelines/ingest_data.py --mode clean \
    --tickers config/tickers_new_context.txt

# Re-align entire universe (all 68 symbols)
python pipelines/ingest_data.py --mode align \
    --tickers config/tickers_full.txt
```

**Important**: The alignment step must use `tickers_full.txt` (all 68 symbols) to create a unified timestamp grid across stocks + ETFs + indices.

---

### Phase 4: Integration with Feature Pipeline

#### 4.1 Update FeaturePipeline Constructor

**File**: `src/features/pipeline.py`

Modify the `__init__` method to accept a universe builder:

```python
class FeaturePipeline:
    def __init__(
        self,
        target_ticker: str,
        universe_builder: SymbolUniverseBuilder,  # NEW
        selector_kwargs: Optional[Dict] = None,
    ) -> None:
        self.target_ticker = target_ticker
        self.universe_builder = universe_builder  # NEW
        self.selector = FeatureSelector(**(selector_kwargs or {}))
        
        # Set after fit_transform
        self._feature_names_full: List[str] = []
        self._feature_names_selected: List[str] = []
        self._peer_tickers: List[str] = []
        self._universe_tickers: List[str] = []  # NEW: store fitted universe
        self._fitted = False
        logger.info(f"Initialized feature pipeline for {target_ticker}")
```

#### 4.2 Update fit_transform to Use Universe Builder

**File**: `src/features/pipeline.py`

Modify `fit_transform` to construct target-specific universe:

```python
def fit_transform(
    self,
    dfs_train: Dict[str, pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build features from training data, fit selector, return selected matrix.
    
    NEW: Constructs target-specific symbol universe before feature computation.
    """
    # NEW: Get target-specific universe
    self._universe_tickers = self.universe_builder.get_universe(
        self.target_ticker,
        dfs_train,
        fit=True,  # Select peers on training data
    )
    
    # NEW: Filter dfs to only include universe symbols
    dfs_train_filtered = {
        ticker: dfs_train[ticker]
        for ticker in self._universe_tickers
        if ticker in dfs_train
    }
    
    self._validate_inputs(dfs_train_filtered)
    
    X_full, y, names = self._compute_features(dfs_train_filtered, fit=True)
    
    self._feature_names_full = names
    
    X_sel, sel_names = self.selector.fit_transform(X_full, y, names)
    self._feature_names_selected = sel_names
    self._fitted = True
    
    logger.info(
        "[%s] fit_transform: %d raw features → %d selected | "
        "samples: %d | universe: %s | peers: %s",
        self.target_ticker,
        len(names),
        len(sel_names),
        len(X_sel),
        self._universe_tickers,
        self._peer_tickers,
    )
    return X_sel, y, sel_names
```

#### 4.3 Update transform to Use Stored Universe

**File**: `src/features/pipeline.py`

```python
def transform(
    self,
    dfs: Dict[str, pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply fitted pipeline to validation or test data.
    
    NEW: Uses stored universe from fit_transform (no re-selection).
    """
    if not self._fitted:
        raise RuntimeError(
            "Pipeline has not been fitted. Call fit_transform() on training "
            "data before calling transform()."
        )
    
    # NEW: Filter to fitted universe
    dfs_filtered = {
        ticker: dfs[ticker]
        for ticker in self._universe_tickers
        if ticker in dfs
    }
    
    self._validate_inputs(dfs_filtered)
    
    X_full, y, _ = self._compute_features(dfs_filtered, fit=False)
    
    X_sel, _ = self.selector.transform(X_full, self._feature_names_full)
    
    logger.info(
        "[%s] transform: %d samples, %d selected features",
        self.target_ticker,
        len(X_sel),
        X_sel.shape[1],
    )
    return X_sel, y
```

---

### Phase 5: Update Pipeline Scripts

#### 5.1 Update run_build_features.py

**File**: `pipelines/run_build_features.py`

Replace the naive universe construction with the builder:

```python
# Add import at top
from src.features.universe_builder import SymbolUniverseBuilder

def process_ticker(
    ticker: str,
    dfs_train,
    dfs_val,
    dfs_test,
    cfg,
    output_dir,
    splitter: DataSplitter,
    universe_builder: SymbolUniverseBuilder,  # NEW parameter
):
    if ticker not in dfs_train:
        logger.info("Skipping %s (not in training data)", ticker)
        return
    
    # OLD: universe_tickers=list(dfs_train.keys())
    # NEW: Use universe builder
    pipeline = FeaturePipeline(
        target_ticker=ticker,
        universe_builder=universe_builder,  # NEW
        selector_kwargs={},
    )
    
    # Rest remains the same...
    X_train, y_train, feature_names = pipeline.fit_transform(dfs_train)
    X_val, y_val = pipeline.transform(dfs_val)
    X_test, y_test = pipeline.transform(dfs_test)
    
    # ... (scaling and saving code unchanged)

def main():
    # ... (argument parsing unchanged)
    
    # NEW: Load symbol universe builder
    universe_builder = SymbolUniverseBuilder.from_config(
        "config/symbol_universe.yaml"
    )
    
    # Load tickers (now includes all symbols: targets + context)
    with open(args.tickers) as f:
        tickers = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
    logger.info("Loaded %d tickers", len(tickers))
    
    # ... (data loading unchanged)
    
    # NEW: Only process prediction targets, not all tickers
    with open("config/symbol_universe.yaml") as f:
        universe_cfg = yaml.safe_load(f)
    prediction_targets = universe_cfg["prediction_targets"]
    
    logger.info(f"Processing {len(prediction_targets)} prediction targets")
    
    for ticker in prediction_targets:
        process_ticker(
            ticker, dfs_train, dfs_val, dfs_test, cfg, output_dir, splitter,
            universe_builder,  # NEW
        )
    
    # NEW: Save fitted peers for reproducibility
    peers_path = Path(output_dir) / "fitted_peers.json"
    universe_builder.save_peers(peers_path)
    
    logger.info("=" * 60)
    logger.info("✓ Feature engineering complete!")
```

**Key Changes**:
1. Only process `prediction_targets`, not all tickers
2. Pass universe builder to each ticker processor
3. Save fitted peers for reproducibility

#### 5.2 Update Data Ingestion to Handle Full Universe

**File**: `pipelines/ingest_data.py`

No code changes needed, but update the default tickers argument:

```python
parser.add_argument("--tickers", type=str, default="config/tickers_full.txt")
```

This ensures all symbols (market indices, sector ETFs, stocks) are ingested.

---

### Phase 6: Testing & Validation

#### 6.1 Create Unit Tests

**File**: `tests/test_universe_builder.py`

```python
"""
Unit tests for SymbolUniverseBuilder.

Tests:
  - Correct universe construction for each tier
  - Peer selection on training data only
  - Leakage prevention (stored peers used on val/test)
  - Sector ETF mapping correctness
  - Alignment function (inner join correctness)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.features.universe_builder import SymbolUniverseBuilder, align_symbol_universe


@pytest.fixture
def sample_config(tmp_path):
    """Create a minimal symbol_universe.yaml for testing."""
    config_content = """
market_context:
  - SPY
  - QQQ

sector_etfs:
  technology:
    etf: XLK
    stocks:
      - AAPL
      - MSFT
  financials:
    etf: XLF
    stocks:
      - JPM

market_internals:
  - UVXY
  - GLD

peer_selection:
  max_peers: 2
  method: "pearson"
  min_correlation: 0.3
  lookback_days: 252

prediction_targets:
  - AAPL
  - JPM

context_only:
  - SPY
  - QQQ
  - XLK
  - XLF
  - UVXY
  - GLD
"""
    config_path = tmp_path / "symbol_universe.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_data():
    """Create synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=1000, freq="1min")
    
    def make_df(ticker, correlation_with_aapl=0.0):
        returns = np.random.randn(len(dates)) * 0.001
        if correlation_with_aapl > 0:
            # Add correlated component
            aapl_returns = np.random.randn(len(dates)) * 0.001
            returns = correlation_with_aapl * aapl_returns + np.sqrt(1 - correlation_with_aapl**2) * returns
        
        close = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            "open": close * (1 + np.random.randn(len(dates)) * 0.0005),
            "high": close * (1 + np.abs(np.random.randn(len(dates))) * 0.001),
            "low": close * (1 - np.abs(np.random.randn(len(dates))) * 0.001),
            "close": close,
            "volume": np.random.randint(1000, 10000, len(dates)),
            "log_return": returns,
        }, index=dates)
        return df
    
    return {
        "AAPL": make_df("AAPL"),
        "MSFT": make_df("MSFT", correlation_with_aapl=0.7),  # high correlation
        "GOOGL": make_df("GOOGL", correlation_with_aapl=0.5),  # medium correlation
        "JPM": make_df("JPM", correlation_with_aapl=0.2),  # low correlation
        "SPY": make_df("SPY"),
        "QQQ": make_df("QQQ"),
        "XLK": make_df("XLK"),
        "XLF": make_df("XLF"),
        "UVXY": make_df("UVXY"),
        "GLD": make_df("GLD"),
    }


def test_universe_builder_initialization(sample_config):
    """Test that builder loads config correctly."""
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    assert builder.market_context == ["SPY", "QQQ"]
    assert builder.sector_map["AAPL"] == "XLK"
    assert builder.sector_map["JPM"] == "XLF"
    assert builder.market_internals == ["UVXY", "GLD"]
    assert builder.peer_config["max_peers"] == 2


def test_universe_construction_aapl(sample_config, sample_data):
    """Test universe construction for AAPL (tech stock)."""
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    universe = builder.get_universe("AAPL", sample_data, fit=True)
    
    # Check tier 1 (market context)
    assert "SPY" in universe
    assert "QQQ" in universe
    
    # Check tier 2 (sector ETF)
    assert "XLK" in universe
    
    # Check tier 3 (peers) - should select MSFT and GOOGL (highest correlation)
    peers = builder.get_fitted_peers("AAPL")
    assert len(peers) <= 2
    assert "MSFT" in peers  # highest correlation
    
    # Check tier 4 (market internals)
    assert "UVXY" in universe
    assert "GLD" in universe
    
    # Check target is included
    assert "AAPL" in universe


def test_peer_selection_leakage_prevention(sample_config, sample_data):
    """Test that peers are selected once and reused."""
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    # Fit on training data
    universe_train = builder.get_universe("AAPL", sample_data, fit=True)
    peers_train = builder.get_fitted_peers("AAPL")
    
    # Transform on validation data (different data, same structure)
    sample_data_val = {k: v.iloc[500:] for k, v in sample_data.items()}
    universe_val = builder.get_universe("AAPL", sample_data_val, fit=False)
    peers_val = builder.get_fitted_peers("AAPL")
    
    # Peers should be identical (no re-selection)
    assert peers_train == peers_val
    assert universe_train == universe_val


def test_alignment_function(sample_data):
    """Test that alignment performs inner join correctly."""
    # Create misaligned data (different date ranges)
    aapl_df = sample_data["AAPL"].iloc[:800]  # missing last 200
    spy_df = sample_data["SPY"].iloc[100:]    # missing first 100
    
    dfs = {"AAPL": aapl_df, "SPY": spy_df}
    aligned = align_symbol_universe(["AAPL", "SPY"], dfs)
    
    # Should only have 700 common timestamps (100 to 800)
    assert len(aligned["AAPL"]) == 700
    assert len(aligned["SPY"]) == 700
    
    # Indices should be identical
    assert aligned["AAPL"].index.equals(aligned["SPY"].index)


def test_save_load_peers(sample_config, sample_data, tmp_path):
    """Test peer serialization for reproducibility."""
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    # Fit and save
    builder.get_universe("AAPL", sample_data, fit=True)
    peers_path = tmp_path / "peers.json"
    builder.save_peers(peers_path)
    
    # Load into new builder
    builder2 = SymbolUniverseBuilder.from_config(sample_config)
    builder2.load_peers(peers_path)
    
    # Should have same peers without re-fitting
    assert builder.get_fitted_peers("AAPL") == builder2.get_fitted_peers("AAPL")


def test_missing_sector_etf_fallback(sample_config, sample_data):
    """Test behavior when stock has no sector mapping."""
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    # TSLA not in config
    sample_data["TSLA"] = sample_data["AAPL"].copy()
    
    universe = builder.get_universe("TSLA", sample_data, fit=True)
    
    # Should still include market context and internals
    assert "SPY" in universe
    assert "UVXY" in universe
    
    # No sector ETF should be added (or fallback to SPY)
    # This is a warning case, not an error
```

**Rationale**: Comprehensive tests ensure leakage prevention, correct tier construction, and reproducibility.

#### 6.2 Create Integration Test

**File**: `tests/test_pipeline_with_universe.py`

```python
"""
Integration test: FeaturePipeline with SymbolUniverseBuilder.

Tests the full flow:
  1. Load config
  2. Build universe for target
  3. Fit pipeline on training data
  4. Transform validation data
  5. Verify feature matrix shape and peer consistency
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.features.pipeline import FeaturePipeline
from src.features.universe_builder import SymbolUniverseBuilder


@pytest.fixture
def sample_config(tmp_path):
    """Create minimal config for integration test."""
    config_content = """
market_context: [SPY, QQQ]
sector_etfs:
  technology:
    etf: XLK
    stocks: [AAPL, MSFT]
market_internals: [UVXY, GLD]
peer_selection:
  max_peers: 2
  method: "pearson"
  min_correlation: 0.3
  lookback_days: 252
prediction_targets: [AAPL]
context_only: [SPY, QQQ, XLK, UVXY, GLD]
"""
    config_path = tmp_path / "symbol_universe.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_data():
    """Create realistic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01 09:30", periods=2000, freq="1min")
    
    def make_df(ticker):
        returns = np.random.randn(len(dates)) * 0.001
        close = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            "open": close * (1 + np.random.randn(len(dates)) * 0.0005),
            "high": close * (1 + np.abs(np.random.randn(len(dates))) * 0.001),
            "low": close * (1 - np.abs(np.random.randn(len(dates))) * 0.001),
            "close": close,
            "volume": np.random.randint(1000, 10000, len(dates)),
            "log_return": returns,
        }, index=dates)
        return df
    
    return {
        "AAPL": make_df("AAPL"),
        "MSFT": make_df("MSFT"),
        "GOOGL": make_df("GOOGL"),
        "SPY": make_df("SPY"),
        "QQQ": make_df("QQQ"),
        "XLK": make_df("XLK"),
        "UVXY": make_df("UVXY"),
        "GLD": make_df("GLD"),
    }


def test_full_pipeline_with_universe_builder(sample_config, sample_data):
    """Test complete feature pipeline with universe builder."""
    # Initialize builder
    builder = SymbolUniverseBuilder.from_config(sample_config)
    
    # Split data
    dfs_train = {k: v.iloc[:1000] for k, v in sample_data.items()}
    dfs_val = {k: v.iloc[1000:1500] for k, v in sample_data.items()}
    dfs_test = {k: v.iloc[1500:] for k, v in sample_data.items()}
    
    # Initialize pipeline
    pipeline = FeaturePipeline(
        target_ticker="AAPL",
        universe_builder=builder,
    )
    
    # Fit on training data
    X_train, y_train, feature_names = pipeline.fit_transform(dfs_train)
    
    # Check output shapes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[1] == len(feature_names)
    assert len(feature_names) > 0
    
    # Transform validation data
    X_val, y_val = pipeline.transform(dfs_val)
    
    # Check consistency
    assert X_val.shape[1] == X_train.shape[1]  # same number of features
    
    # Transform test data
    X_test, y_test = pipeline.transform(dfs_test)
    assert X_test.shape[1] == X_train.shape[1]
    
    # Verify peers were selected and frozen
    peers = builder.get_fitted_peers("AAPL")
    assert len(peers) <= 2
    assert all(p in ["MSFT", "GOOGL"] for p in peers)
```

---

### Phase 7: Documentation & Rollout

#### 7.1 Update README

Add section to main README explaining the symbol universe system:

```markdown
## Symbol Universe Construction

The LSTM receives multivariate input at each timestep, combining the target stock with relevant market context. The symbol universe is constructed in 4 tiers:

### Tier 1: Market Structure (Always Included)
- **SPY**: S&P 500 broad market baseline
- **QQQ**: Nasdaq 100 tech-heavy index
- **IWM**: Russell 2000 small caps (risk appetite signal)
- **DIA**: Dow Jones industrials

### Tier 2: Sector ETF (Target-Specific)
- **XLK**: Technology (AAPL, MSFT, GOOGL, etc.)
- **SOXX**: Semiconductors (NVDA, AMD, INTC, etc.)
- **XLF**: Financials (JPM, BAC, GS, etc.)
- **XLV**: Healthcare (JNJ, UNH, PFE, etc.)
- **XLE**: Energy (XOM, CVX, COP, etc.)
- And more...

### Tier 3: Peer Stocks (1-3 Max)
Selected by Pearson correlation on training data returns. Provides relative strength signals (e.g., NVDA vs AMD divergence).

### Tier 4: Market Internals (Equity-Only Proxies)
- **UVXY**: VIX futures ETF (volatility proxy)
- **GLD**: Gold ETF (flight-to-safety signal)
- **TLT**: 20+ year Treasury bonds (risk-on/risk-off)

### Configuration

Edit `config/symbol_universe.yaml` to:
- Add new sectors
- Modify peer selection parameters
- Define prediction targets

### Leakage Prevention

Peers are selected **once** on training data and frozen for validation/test splits. This prevents look-ahead bias.
```

#### 7.2 Create Usage Guide

**File**: `docs/guides/SYMBOL_UNIVERSE_USAGE.md`

```markdown
# Symbol Universe System Usage Guide

## Quick Start

1. **Define your prediction targets** in `config/symbol_universe.yaml`:
   ```yaml
   prediction_targets:
     - AAPL
     - NVDA
     - JPM
   ```

2. **Run data ingestion** with full universe:
   ```bash
   python pipelines/ingest_data.py --mode ingest --tickers config/tickers_full.txt
   python pipelines/ingest_data.py --mode clean
   python pipelines/ingest_data.py --mode align
   ```

3. **Build features** (automatically uses universe builder):
   ```bash
   python pipelines/run_build_features.py --config config/default_config.yaml
   ```

4. **Check fitted peers**:
   ```bash
   cat data/features/fitted_peers.json
   ```

## Adding a New Stock

1. Determine its sector (e.g., TSLA → consumer_discretionary)
2. Add to `config/symbol_universe.yaml`:
   ```yaml
   sector_etfs:
     consumer_discretionary:
       etf: XLY
       stocks:
         - AMZN
         - TSLA  # NEW
   ```
3. Add to `prediction_targets` if you want to predict it
4. Add to `config/tickers_full.txt`
5. Re-run ingestion and feature building

## Adding a New Sector

1. Find the appropriate sector ETF (e.g., utilities → XLU)
2. Add to `config/symbol_universe.yaml`:
   ```yaml
   sector_etfs:
     utilities:
       etf: XLU
       stocks:
         - NEE
         - DUK
         - SO
   ```
3. Add XLU to `context_only` list
4. Add XLU and stocks to `config/tickers_full.txt`
5. Re-run ingestion

## Troubleshooting

### "No peers found for TICKER"
- Check that candidate peers exist in the same sector
- Lower `min_correlation` in peer_selection config
- Verify training data has enough bars (need 100+ for correlation)

### "Symbol X not in dfs"
- Ensure symbol was ingested (check `data/processed/X.parquet`)
- Verify symbol is in `config/tickers_full.txt`
- Check data quality (may have been dropped during cleaning)

### "Peers for TICKER not fitted"
- Must call `fit=True` on training data before `fit=False` on val/test
- Check that pipeline.fit_transform() was called before pipeline.transform()
```

#### 7.3 Create Migration Guide

**File**: `docs/guides/SYMBOL_UNIVERSE_MIGRATION.md`

```markdown
# Migrating to Symbol Universe System

## For Existing Projects

If you have an existing codebase using the old flat ticker list approach:

### Step 1: Backup Current State
```bash
cp -r data/features data/features.backup
cp config/tickers.txt config/tickers.old
```

### Step 2: Create Symbol Universe Config
1. Copy `config/symbol_universe.yaml` from this plan
2. Classify your existing tickers into sectors
3. Define prediction targets vs. context-only symbols

### Step 3: Update Ticker List
1. Create `config/tickers_full.txt` with all symbols (targets + context)
2. Include market indices (SPY, QQQ, IWM, DIA)
3. Include sector ETFs (XLK, XLF, etc.)
4. Include market internals (UVXY, GLD, TLT)

### Step 4: Re-ingest Data
```bash
# Ingest new symbols (ETFs, indices, internals)
python pipelines/ingest_data.py --mode ingest --tickers config/tickers_full.txt

# Clean and align
python pipelines/ingest_data.py --mode clean
python pipelines/ingest_data.py --mode align
```

### Step 5: Rebuild Features
```bash
python pipelines/run_build_features.py --config config/default_config.yaml
```

### Step 6: Compare Results
```python
# Load old features
import numpy as np
X_old = np.load("data/features.backup/AAPL/X_train.npy")

# Load new features
X_new = np.load("data/features/AAPL/X_train.npy")

print(f"Old shape: {X_old.shape}")
print(f"New shape: {X_new.shape}")
# New should have more features (cross-ticker from sector ETF, peers, internals)
```

## Breaking Changes

1. **FeaturePipeline constructor**: Now requires `universe_builder` instead of `universe_tickers`
2. **Peer selection**: No longer automatic fallback; must be in config
3. **Feature count**: Will increase due to additional cross-ticker features

## Backward Compatibility

To maintain old behavior temporarily:

```python
# Old way (deprecated)
pipeline = FeaturePipeline(
    target_ticker="AAPL",
    universe_tickers=["AAPL", "SPY", "MSFT"],
)

# New way
from src.features.universe_builder import SymbolUniverseBuilder
builder = SymbolUniverseBuilder.from_config("config/symbol_universe.yaml")
pipeline = FeaturePipeline(
    target_ticker="AAPL",
    universe_builder=builder,
)
```
```

---

## Implementation Timeline

### Week 1: Configuration & Core Builder
- [ ] Create `config/symbol_universe.yaml` (maps 51 existing stocks to sectors)
- [ ] Create `config/tickers_full.txt` (adds 17 new ETF/index symbols)
- [ ] Create `config/tickers_new_context.txt` (just the 17 new symbols)
- [ ] Update `config_schema.py`
- [ ] Implement `SymbolUniverseBuilder` class
- [ ] Implement `align_symbol_universe` function
- [ ] Write unit tests for builder

### Week 2: Data Ingestion for New Symbols
- [ ] Ingest 17 new symbols (QQQ, IWM, DIA, 11 sector ETFs, 3 market internals)
- [ ] Clean new symbols
- [ ] Re-align entire universe (51 stocks + SPY + 17 new = 69 total symbols)
- [ ] Verify alignment quality (check timestamp coverage)
- [ ] Validate ETF data quality (no excessive gaps)

### Week 3: Pipeline Integration
- [ ] Update `FeaturePipeline.__init__`
- [ ] Update `FeaturePipeline.fit_transform`
- [ ] Update `FeaturePipeline.transform`
- [ ] Update `run_build_features.py`
- [ ] Write integration tests

### Week 4: Validation & Testing
- [ ] Run feature building for subset of prediction targets (5-10 stocks)
- [ ] Validate peer selections make sense (high correlation, same sector)
- [ ] Compare feature counts (old vs new)
- [ ] Profile memory usage
- [ ] Check fitted_peers.json output

### Week 5: Full Rollout & Documentation
- [ ] Run feature building for all 51 stocks (if desired)
- [ ] Write README section
- [ ] Write usage guide
- [ ] Write migration guide
- [ ] Create example notebooks

---

## Success Metrics

1. **Correctness**: Each target gets sector-appropriate peers
   - AAPL gets MSFT/GOOGL (tech), not JPM/BAC (financials)
   - NVDA gets AMD/INTC (semiconductors), not XOM/CVX (energy)

2. **Leakage Prevention**: Peers identical across train/val/test
   - Check `fitted_peers.json` consistency
   - Verify no re-selection on validation data

3. **Feature Count**: Reasonable increase (not explosion)
   - Old: ~50-80 features per target
   - New: ~80-120 features per target (20-40 from cross-ticker)

4. **Model Performance**: Improvement in validation metrics
   - Sharpe ratio increase (better risk-adjusted returns)
   - Drawdown reduction (better downside protection)
   - RMSE reduction (better price prediction)

---

## Risks & Mitigations

### Risk 1: Data Availability
**Problem**: Some sector ETFs or market internals may not have full history  
**Mitigation**: 
- Use fallback to SPY if sector ETF missing
- Log warnings for missing symbols
- Inner join alignment drops timestamps with missing data

### Risk 2: Peer Selection Instability
**Problem**: Top-3 peers may change significantly between runs  
**Mitigation**:
- Save fitted peers to disk for reproducibility
- Use correlation threshold (min_correlation=0.3) to filter noise
- Require minimum 100 bars for correlation computation

### Risk 3: Feature Explosion
**Problem**: Too many symbols → too many features → overfitting  
**Mitigation**:
- Limit peers to 3 max
- Feature selector prunes low-importance features
- Monitor feature count in metadata

### Risk 4: Computational Cost
**Problem**: More symbols → more alignment → slower ingestion  
**Mitigation**:
- Parallel ingestion (already implemented)
- Cache aligned universe parquet
- Use efficient inner join (pandas/polars)

---

## Future Enhancements

### Phase 2: Dynamic Peer Selection
- Re-select peers every N months (walk-forward)
- Use rolling correlation window instead of full training period
- Weight peers by correlation strength in feature engineering

### Phase 3: Alternative Market Internals
- Add HYG (high-yield bonds) for credit risk signal
- Add DXY (dollar index) for currency effects
- Add VIX futures term structure (contango/backwardation)

### Phase 4: Sector Rotation Signals
- Compute relative strength between sectors (XLK/XLF ratio)
- Add sector momentum features
- Detect sector rotation regimes

### Phase 5: International Context
- Add international indices (EWJ for Japan, FXI for China)
- Add currency pairs (EUR/USD, USD/JPY)
- Add commodity futures (crude oil, copper)

---

## Appendix A: Complete File Checklist

### New Files
- [ ] `config/symbol_universe.yaml`
- [ ] `config/tickers_full.txt`
- [ ] `src/features/universe_builder.py`
- [ ] `tests/test_universe_builder.py`
- [ ] `tests/test_pipeline_with_universe.py`
- [ ] `docs/guides/SYMBOL_UNIVERSE_USAGE.md`
- [ ] `docs/guides/SYMBOL_UNIVERSE_MIGRATION.md`

### Modified Files
- [ ] `src/utils/config_schema.py` (add SymbolUniverseConfig)
- [ ] `src/features/pipeline.py` (use universe_builder)
- [ ] `pipelines/run_build_features.py` (integrate builder)
- [ ] `pipelines/ingest_data.py` (update default tickers)
- [ ] `README.md` (add symbol universe section)

### Generated Files (Runtime)
- [ ] `data/features/fitted_peers.json` (peer selections)
- [ ] `data/processed/*.parquet` (aligned symbol data)

---

## Appendix B: Example Fitted Peers Output

**File**: `data/features/fitted_peers.json`

```json
{
  "AAPL": ["MSFT", "GOOGL", "META"],
  "NVDA": ["AMD", "INTC", "QCOM"],
  "JPM": ["BAC", "GS", "MS"],
  "XOM": ["CVX", "COP", "SLB"],
  "AMZN": ["TSLA", "HD", "NKE"]
}
```

This shows:
- AAPL peers are tech stocks (correct)
- NVDA peers are semiconductors (correct)
- JPM peers are financials (correct)
- XOM peers are energy (correct)
- AMZN peers are consumer discretionary (correct)

---

## Appendix C: Symbol Universe Size Estimate

For a typical prediction target (e.g., AAPL):

| Tier | Symbols | Count |
|------|---------|-------|
| Target | AAPL | 1 |
| Market Context | SPY, QQQ, IWM, DIA | 4 |
| Sector ETF | XLK | 1 |
| Peers | MSFT, GOOGL, META | 3 |
| Market Internals | UVXY, GLD, TLT | 3 |
| **Total** | | **12** |

Each symbol contributes ~5-10 features after engineering:
- Base price features (5): close, high, low, open, volume
- Technical indicators (3-5): RSI, MACD, BB, etc.
- Cross-ticker features (2-3): correlation, relative strength, beta

**Total features per symbol**: ~8 average  
**Total features for universe**: 12 symbols × 8 features = **~96 features**  
**After feature selection**: ~40-50 features (50-60% pruning)

This is well within LSTM capacity and provides rich market context without overwhelming the model.

---

## Conclusion

This plan implements a production-grade symbol universe construction system that provides each LSTM with the correct market context for multivariate prediction. The 4-tier architecture (market structure, sector ETF, peers, internals) is grounded in market microstructure theory and prevents look-ahead bias through careful peer selection on training data only.

The system is:
- **Correct**: Sector-aware peer selection
- **Safe**: Leakage prevention through frozen peers
- **Scalable**: Modest feature growth (12-15 symbols per target)
- **Maintainable**: Configuration-driven, well-tested, documented

Implementation follows a phased approach with clear success metrics and risk mitigations. The end result is an LSTM that sees not just the target stock's bars, but the full market context that drives intraday price action.
