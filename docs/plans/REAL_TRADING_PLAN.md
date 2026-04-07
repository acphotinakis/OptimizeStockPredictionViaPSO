# Real Trading Implementation Plan
## Transition from Academic Project to Production Trading System

**Document Version:** 1.0  
**Date:** April 7, 2026  
**Status:** Planning Phase  
**Priority:** High

---

## Executive Summary

This document outlines the architectural and implementation changes required to transform the current academic PSO-LSTM system into a production-ready trading system. The current system has strong foundations (feature engineering, model architecture, backtesting framework) but suffers from critical flaws that would cause failure in live trading:

1. **Universe too narrow**: 5 tech stocks (highly correlated)
2. **Timeframe inappropriate**: 1-minute bars without HFT infrastructure
3. **Circular dependencies**: Using SPY as both benchmark and target
4. **Missing macro context**: No real VIX, rates, currency, sector rotation
5. **Overfitting risk**: Trained only on 2021-2026 tech bull market

**Estimated Implementation Time**: 4-6 weeks  
**Risk Level**: Medium (architecture is sound, data pipeline needs expansion)

---

## Table of Contents

1. [Current System Assessment](#1-current-system-assessment)
2. [Critical Changes Required](#2-critical-changes-required)
3. [Implementation Roadmap](#3-implementation-roadmap)
4. [Code Changes by Module](#4-code-changes-by-module)
5. [Data Pipeline Modifications](#5-data-pipeline-modifications)
6. [Configuration Updates](#6-configuration-updates)
7. [Testing & Validation Strategy](#7-testing--validation-strategy)
8. [Risk Management Additions](#8-risk-management-additions)
9. [Deployment Checklist](#9-deployment-checklist)
10. [Success Metrics](#10-success-metrics)

---

## 1. Current System Assessment

### 1.1 What Works Well ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Feature engineering pipeline | **Excellent** | 117 features, well-designed cross-ticker logic |
| PSO-LSTM architecture | **Good** | Novel approach, solid mathematical foundation |
| Backtesting framework | **Good** | Transaction costs, slippage, walk-forward validation |
| XGBoost feature selection | **Good** | Reduces dimensionality effectively |
| Code modularity | **Good** | Clean separation of concerns |
| GPU acceleration | **Excellent** | Proper CUDA integration |

### 1.2 Critical Flaws ❌

| Issue | Severity | Impact on Live Trading |
|-------|----------|------------------------|
| 5-stock universe (all tech) | **CRITICAL** | Concentration risk, no diversification |
| 1-minute bars | **CRITICAL** | Transaction costs eat profits, execution lag |
| SPY as both benchmark + target | **HIGH** | Circular dependency, conceptual confusion |
| No macro features | **HIGH** | Blind to regime changes (rates, VIX) |
| 2021-2026 training only | **MEDIUM** | Overfitted to tech bull market |
| No position sizing logic | **MEDIUM** | Risk management incomplete |
| No regime detection | **MEDIUM** | Same strategy in all market conditions |

### 1.3 Current Configuration

```yaml
# Current settings (config/default_config.yaml)
data:
  freq: "1Min"              # ❌ Too high frequency
  start_date: "2021-04-05"  # ❌ Tech bull market only
  train_end: "2024-04-05"
  
# Current universe (config/tickers.txt)
AAPL, MSFT, GOOGL, NVDA, TSLA, SPY  # ❌ All tech + SPY
```

---

## 2. Critical Changes Required

### 2.1 Universe Expansion (Priority: CRITICAL)

**Current**: 5 tech stocks + SPY  
**Target**: 25-30 stocks across 6+ sectors + SPY (benchmark only)

#### Proposed Universe (30 stocks)

```python
# Technology (5) - Keep existing
AAPL, MSFT, GOOGL, NVDA, TSLA

# Financials (5)
JPM, BAC, GS, MS, WFC

# Healthcare (5)
UNH, JNJ, PFE, ABBV, TMO

# Energy (4)
XOM, CVX, COP, SLB

# Consumer Discretionary (4)
AMZN, HD, MCD, NKE

# Consumer Staples (3)
PG, KO, WMT

# Industrials (4)
CAT, BA, UNP, HON

# Benchmark (not traded)
SPY
```

**Rationale**:
- **Sector diversification**: Reduces correlation from 0.7+ to 0.3-0.5
- **Market cap coverage**: ~40% of S&P 500 market cap
- **Liquidity**: All stocks have >$5M daily volume
- **Cross-sectional signals**: Sector rotation becomes meaningful

#### Code Changes Required

**File**: `config/tickers.txt`
```diff
- # 6-ticker universe for PSO-LSTM stock prediction
- # 5 equities + SPY benchmark
+ # 30-ticker universe for production trading
+ # 25 equities across 7 sectors + SPY benchmark (not traded)
+ # SPY used only for beta/alpha decomposition

# Technology (5)
AAPL
MSFT
GOOGL
NVDA
TSLA

+ # Financials (5)
+ JPM
+ BAC
+ GS
+ MS
+ WFC
+ 
+ # Healthcare (5)
+ UNH
+ JNJ
+ PFE
+ ABBV
+ TMO
+ 
+ # Energy (4)
+ XOM
+ CVX
+ COP
+ SLB
+ 
+ # Consumer Discretionary (4)
+ AMZN
+ HD
+ MCD
+ NKE
+ 
+ # Consumer Staples (3)
+ PG
+ KO
+ WMT
+ 
+ # Industrials (4)
+ CAT
+ BA
+ UNP
+ HON

- # Benchmark ETF
+ # Benchmark (not traded - used only for features)
SPY
```

**File**: `scripts/build_features.py`
```diff
def main():
    # ... existing code ...
    
    with open(args.tickers) as f:
        tickers = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
+   
+   # Remove SPY from prediction targets (use only as benchmark)
+   tickers_to_predict = [t for t in tickers if t != "SPY"]
+   logger.info("Loaded %d tickers (%d for prediction, SPY as benchmark only)", 
+               len(tickers), len(tickers_to_predict))
    
-   logger.info("Loaded %d tickers", len(tickers))
    
    # ... rest of code ...
    
    Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(process_ticker)(
            ticker,
            dfs_train,
            dfs_val,
            dfs_test,
            cfg,
            output_dir,
            splitter,
        )
-       for ticker in tickers
+       for ticker in tickers_to_predict  # Don't predict SPY
    )
```

---

### 2.2 Timeframe Change: 1-Minute → Daily (Priority: CRITICAL)

**Current**: 1-minute bars  
**Target**: Daily bars (EOD data)

#### Why Daily Bars?

| Factor | 1-Minute Bars | Daily Bars |
|--------|---------------|------------|
| **Transaction costs** | 0.1% per trade × 10 trades/day = 1% daily | 0.1% per trade × 1 trade/day = 0.1% daily |
| **Execution lag** | 100ms lag = 2-3 bars of slippage | 1-hour lag = negligible |
| **Signal stability** | High noise (microstructure) | Cleaner signal |
| **Data requirements** | 390 bars/day × 252 days = 98,280 bars/year | 252 bars/year |
| **Backtesting speed** | Slow (millions of bars) | Fast (thousands of bars) |
| **Overfitting risk** | Very high (too many parameters) | Lower |

**Estimated Impact**:
- **Sharpe ratio improvement**: +0.3 to +0.5 (lower noise, lower costs)
- **Drawdown reduction**: -20% to -30% (fewer whipsaws)
- **Computational cost**: -95% (fewer bars to process)

#### Code Changes Required

**File**: `config/default_config.yaml`
```diff
data:
  start_date: "2021-04-05"
  end_date: "2026-04-05"
  train_end: "2024-04-05"
  val_end: "2025-04-05" 
  test_end: "2026-04-05"
- freq: "1Min"
- session_start: "09:30"
- session_end: "16:00"
+ freq: "1D"              # Daily bars (EOD)
+ session_start: null     # Not applicable for daily
+ session_end: null
  timezone: "America/New_York"
```

**File**: `src/data/alpaca_ingestor.py`
```diff
def fetch_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        bars = self.api.get_bars(
            ticker,
-           TimeFrame.Minute,  # 1-minute bars
+           TimeFrame.Day,     # Daily bars
            start=start,
            end=end,
            adjustment="all",  # Split/dividend adjusted
        ).df
```

**File**: `src/features/cross_ticker.py`
```diff
# Update rolling windows for daily data
def compute_cross_ticker_features(
    target_ticker: str,
    dfs: Dict[str, pd.DataFrame],
    peer_tickers: List[str] | None = None,
-   rolling_window: int = 60,  # 60 minutes
+   rolling_window: int = 60,  # 60 days (3 months)
    return_peer_tickers: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, List[str]]:
```

**File**: `src/features/technical.py`
```python
# Update all technical indicator periods for daily bars
# Example: RSI
def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... existing code ...
    
-   out["rsi_14"] = ta.rsi(C, length=14)      # 14 minutes
+   out["rsi_14"] = ta.rsi(C, length=14)      # 14 days (standard)
    
-   out["bb_upper"], out["bb_middle"], out["bb_lower"] = ta.bbands(C, length=20)  # 20 min
+   out["bb_upper"], out["bb_middle"], out["bb_lower"] = ta.bbands(C, length=20)  # 20 days
    
    # ... rest of code ...
```

---

### 2.3 Remove SPY as Prediction Target (Priority: HIGH)

**Current**: SPY is both benchmark and prediction target  
**Target**: SPY used ONLY for beta/alpha decomposition

#### Rationale

1. **Conceptual clarity**: Benchmark ≠ tradable asset
2. **Avoid redundancy**: SPY prediction = weighted average of constituent predictions
3. **Reduce computation**: 25 models instead of 26

#### Code Changes Required

Already covered in Section 2.1 (Universe Expansion).

**Additional change** in `src/features/cross_ticker.py`:
```python
# Already implemented in previous fix
if "SPY" in dfs and target_ticker != "SPY":
    # Compute SPY features
    ...
else:
    if target_ticker == "SPY":
        logger.info("Target is SPY; skipping self-referential SPY features.")
```

---

### 2.4 Add Macro Features (Priority: HIGH)

**Current**: Only SPY proxy for VIX, no rates, no currency  
**Target**: Real VIX, 10Y yield, DXY, sector ETFs

#### Proposed Macro Features (12 new features)

| Feature | Ticker | Description | Predictive Power |
|---------|--------|-------------|------------------|
| **VIX** | `^VIX` | CBOE Volatility Index | Regime detection (risk-on/off) |
| **10Y Yield** | `^TNX` | 10-Year Treasury Yield | Discount rate for equities |
| **2Y Yield** | `^IRX` | 2-Year Treasury Yield | Short-term rates |
| **Yield Curve** | Computed | 10Y - 2Y spread | Recession predictor |
| **DXY** | `DX-Y.NYB` | US Dollar Index | Currency headwinds for multinationals |
| **XLK** | `XLK` | Tech sector ETF | Sector rotation (tech) |
| **XLF** | `XLF` | Financial sector ETF | Sector rotation (finance) |
| **XLE** | `XLE` | Energy sector ETF | Sector rotation (energy) |
| **XLV** | `XLV` | Healthcare sector ETF | Sector rotation (healthcare) |
| **XLY** | `XLY` | Consumer Disc. ETF | Sector rotation (consumer) |
| **XLP** | `XLP` | Consumer Staples ETF | Defensive rotation |
| **XLI** | `XLI` | Industrials ETF | Cyclical rotation |

#### Implementation Plan

**Step 1**: Create new macro feature module

**File**: `src/features/macro.py` (NEW FILE)
```python
"""
src/features/macro.py

Macro-economic and market regime features.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_macro_features(
    target_ticker: str,
    dfs: Dict[str, pd.DataFrame],
    macro_tickers: Dict[str, str] = None,
) -> pd.DataFrame:
    """Compute macro-economic features.
    
    Args:
        target_ticker: The ticker whose features we are building.
        dfs: Dict mapping ticker → cleaned DataFrame.
        macro_tickers: Dict mapping feature name → ticker symbol.
        
    Returns:
        DataFrame of macro feature columns on the target ticker's index.
    """
    if macro_tickers is None:
        macro_tickers = {
            "vix": "^VIX",
            "tnx_10y": "^TNX",
            "irx_2y": "^IRX",
            "dxy": "DX-Y.NYB",
            "xlk": "XLK",
            "xlf": "XLF",
            "xle": "XLE",
            "xlv": "XLV",
            "xly": "XLY",
            "xlp": "XLP",
            "xli": "XLI",
        }
    
    df_target = dfs[target_ticker]
    out = pd.DataFrame(index=df_target.index)
    
    # --- VIX (volatility regime) ---
    if macro_tickers["vix"] in dfs:
        vix = dfs[macro_tickers["vix"]]["close"].reindex(df_target.index).ffill().bfill()
        out["vix"] = vix
        out["vix_ma_20"] = vix.rolling(20, min_periods=1).mean()
        out["vix_spike"] = (vix > vix.rolling(60, min_periods=1).quantile(0.8)).astype(float)
    else:
        logger.warning("VIX not in dfs; VIX features will be zero.")
        out["vix"] = 0.0
        out["vix_ma_20"] = 0.0
        out["vix_spike"] = 0.0
    
    # --- Interest rates ---
    if macro_tickers["tnx_10y"] in dfs and macro_tickers["irx_2y"] in dfs:
        tnx = dfs[macro_tickers["tnx_10y"]]["close"].reindex(df_target.index).ffill().bfill()
        irx = dfs[macro_tickers["irx_2y"]]["close"].reindex(df_target.index).ffill().bfill()
        
        out["tnx_10y"] = tnx
        out["irx_2y"] = irx
        out["yield_curve"] = tnx - irx  # 10Y-2Y spread (recession indicator)
        out["tnx_change"] = tnx.pct_change(5)  # 5-day rate change
    else:
        logger.warning("Treasury yields not in dfs; rate features will be zero.")
        out["tnx_10y"] = 0.0
        out["irx_2y"] = 0.0
        out["yield_curve"] = 0.0
        out["tnx_change"] = 0.0
    
    # --- Dollar strength ---
    if macro_tickers["dxy"] in dfs:
        dxy = dfs[macro_tickers["dxy"]]["close"].reindex(df_target.index).ffill().bfill()
        out["dxy"] = dxy
        out["dxy_change"] = dxy.pct_change(5)  # 5-day dollar move
    else:
        logger.warning("DXY not in dfs; dollar features will be zero.")
        out["dxy"] = 0.0
        out["dxy_change"] = 0.0
    
    # --- Sector rotation (relative strength vs SPY) ---
    sector_etfs = ["xlk", "xlf", "xle", "xlv", "xly", "xlp", "xli"]
    
    if "SPY" in dfs:
        spy_ret = dfs["SPY"]["log_return"].reindex(df_target.index).fillna(0.0)
        
        for sector_key in sector_etfs:
            sector_ticker = macro_tickers.get(sector_key)
            if sector_ticker and sector_ticker in dfs:
                sector_ret = dfs[sector_ticker]["log_return"].reindex(df_target.index).fillna(0.0)
                # Relative strength: sector return - SPY return (20-day rolling)
                out[f"{sector_key}_rel_strength"] = (
                    sector_ret.rolling(20, min_periods=5).sum() 
                    - spy_ret.rolling(20, min_periods=5).sum()
                )
            else:
                out[f"{sector_key}_rel_strength"] = 0.0
    else:
        logger.warning("SPY not in dfs; sector rotation features will be zero.")
        for sector_key in sector_etfs:
            out[f"{sector_key}_rel_strength"] = 0.0
    
    return out.fillna(0.0)
```

**Step 2**: Integrate into feature pipeline

**File**: `src/features/pipeline.py`
```diff
from .technical import compute_technical_features
from .statistical import compute_statistical_features
from .volume import compute_volume_features
from .cross_ticker import compute_cross_ticker_features
+ from .macro import compute_macro_features
from .selector import FeatureSelector

# ... existing code ...

def _compute_features(
    self,
    dfs: Dict[str, pd.DataFrame],
    fit: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    
    df_target = dfs[self.target_ticker]
    
    # --- Feature blocks ---
    tech = compute_technical_features(df_target)
    stat = compute_statistical_features(df_target)
    vol = compute_volume_features(df_target)
+   macro = compute_macro_features(self.target_ticker, dfs)
    
    if fit:
        cross, peer_tickers = compute_cross_ticker_features(
            self.target_ticker,
            dfs,
            peer_tickers=None,
            return_peer_tickers=True,
        )
        self._peer_tickers = peer_tickers
    else:
        cross = compute_cross_ticker_features(
            self.target_ticker,
            dfs,
            peer_tickers=self._peer_tickers,
            return_peer_tickers=False,
        )
    
    # ... base features ...
    
    _append_block(base)
    _append_block(tech)
    _append_block(stat)
    _append_block(vol)
    _append_block(cross)
+   _append_block(macro)
    
    # ... rest of code ...
```

**Step 3**: Update data ingestion to fetch macro tickers

**File**: `scripts/ingest_data.py`
```diff
def main():
    # ... existing code ...
    
    with open(args.tickers) as f:
        tickers = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
+   
+   # Add macro tickers (not in tickers.txt, but needed for features)
+   macro_tickers = ["^VIX", "^TNX", "^IRX", "DX-Y.NYB", 
+                    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI"]
+   all_tickers = tickers + macro_tickers
+   logger.info("Loaded %d equity tickers + %d macro tickers = %d total",
+               len(tickers), len(macro_tickers), len(all_tickers))
    
-   logger.info("Loaded %d tickers", len(tickers))
    
    # ... fetch data for all_tickers instead of tickers ...
```

---

### 2.5 Extend Training Period (Priority: MEDIUM)

**Current**: 2021-2026 (tech bull market)  
**Target**: 2015-2026 (includes 2015-2016 correction, 2018 vol spike, 2020 COVID crash)

#### Rationale

Training only on 2021-2026 means the model has never seen:
- **Rate hike cycles** (2022-2023 was unique)
- **Energy sector dominance** (2021-2022 oil rally)
- **Tech drawdowns** (2022: -30% Nasdaq)
- **Pandemic volatility** (2020: VIX >80)
- **Sector rotation** (2016: financials outperformed tech)

**Proposed timeline**:
```
2015-01-01 to 2026-04-05 (11 years)
├── Train: 2015-01-01 to 2023-12-31 (9 years)
├── Val:   2024-01-01 to 2024-12-31 (1 year)
└── Test:  2025-01-01 to 2026-04-05 (1.25 years)
```

#### Code Changes Required

**File**: `config/default_config.yaml`
```diff
data:
- start_date: "2021-04-05"
+ start_date: "2015-01-01"  # Include multiple market regimes
  end_date: "2026-04-05"
- train_end: "2024-04-05"
- val_end: "2025-04-05" 
+ train_end: "2023-12-31"   # 9 years of training data
+ val_end: "2024-12-31"     # 1 year validation
  test_end: "2026-04-05"
```

---

## 3. Implementation Roadmap

### Phase 1: Data Pipeline (Week 1-2)

| Task | Priority | Estimated Time | Dependencies |
|------|----------|----------------|--------------|
| Expand ticker universe to 30 stocks | CRITICAL | 2 days | None |
| Add macro ticker ingestion | HIGH | 1 day | None |
| Change frequency to daily bars | CRITICAL | 2 days | None |
| Extend date range to 2015-2026 | MEDIUM | 1 day | None |
| Re-ingest all data | - | 3 days | All above |
| Validate data quality | HIGH | 1 day | Ingestion complete |

**Deliverables**:
- `data/raw/` contains 30 stocks + 11 macro tickers × 11 years of daily data
- `data/processed/aligned_universe.parquet` has 30 stocks aligned

### Phase 2: Feature Engineering (Week 2-3)

| Task | Priority | Estimated Time | Dependencies |
|------|----------|----------------|--------------|
| Create `src/features/macro.py` | HIGH | 2 days | Macro data ingested |
| Update rolling windows for daily data | HIGH | 1 day | None |
| Remove SPY from prediction targets | HIGH | 0.5 days | None |
| Add sector rotation features | MEDIUM | 1 day | Macro module |
| Re-run feature engineering | - | 2 days | All above |
| Validate feature distributions | HIGH | 1 day | Features built |

**Deliverables**:
- `data/features/` contains 25 ticker folders (no SPY)
- Each ticker has ~130 features (117 original + 12 macro)

### Phase 3: Model Training (Week 3-4)

| Task | Priority | Estimated Time | Dependencies |
|------|----------|----------------|--------------|
| Update PSO search space for daily data | MEDIUM | 1 day | Features ready |
| Run XGBoost baseline on new data | HIGH | 2 days | Features ready |
| Run LSTM baseline on new data | HIGH | 2 days | Features ready |
| Run PSO-LSTM on 5 test tickers | HIGH | 3 days | Baselines complete |
| Compare results vs. old system | HIGH | 1 day | All models trained |

**Deliverables**:
- Baseline metrics (RMSE, Sharpe, Drawdown) for 5 tickers
- PSO-LSTM results for 5 tickers
- Performance comparison report

### Phase 4: Risk Management (Week 4-5)

| Task | Priority | Estimated Time | Dependencies |
|------|----------|----------------|--------------|
| Implement position sizing logic | HIGH | 2 days | None |
| Add regime detection module | MEDIUM | 2 days | VIX features |
| Implement portfolio-level risk limits | HIGH | 2 days | Position sizing |
| Add correlation-based diversification | MEDIUM | 1 day | Portfolio logic |
| Backtest with risk management | HIGH | 2 days | All above |

**Deliverables**:
- `src/risk/position_sizer.py`
- `src/risk/regime_detector.py`
- Portfolio-level backtest results

### Phase 5: Production Deployment (Week 5-6)

| Task | Priority | Estimated Time | Dependencies |
|------|----------|----------------|--------------|
| Create live data pipeline | CRITICAL | 3 days | None |
| Implement order execution module | CRITICAL | 2 days | Live data |
| Add monitoring & alerting | HIGH | 2 days | Execution |
| Paper trading for 1 week | CRITICAL | 5 days | All above |
| Go live with small capital | CRITICAL | Ongoing | Paper trading success |

**Deliverables**:
- Live trading system
- Monitoring dashboard
- Alerting system

---

## 4. Code Changes by Module

### 4.1 Data Ingestion (`src/data/`)

**File**: `src/data/alpaca_ingestor.py`

```python
# Changes:
# 1. Switch from 1-minute to daily bars
# 2. Add macro ticker support
# 3. Handle special symbols (^VIX, ^TNX, etc.)

class AlpacaIngestor:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version="v2")
        
        # Map special symbols to Alpaca-compatible tickers
        self.symbol_map = {
            "^VIX": "VIX",      # Alpaca uses VIX instead of ^VIX
            "^TNX": "TNX",      # Treasury 10Y
            "^IRX": "IRX",      # Treasury 2Y
            "DX-Y.NYB": "DXY",  # Dollar index
        }
    
    def fetch_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        # Map special symbols
        alpaca_ticker = self.symbol_map.get(ticker, ticker)
        
        try:
            bars = self.api.get_bars(
                alpaca_ticker,
                TimeFrame.Day,  # Changed from TimeFrame.Minute
                start=start,
                end=end,
                adjustment="all",
            ).df
            
            # ... rest of code ...
```

### 4.2 Feature Engineering (`src/features/`)

**New File**: `src/features/macro.py`  
(See Section 2.4 for full implementation)

**Modified File**: `src/features/pipeline.py`  
(See Section 2.4 for integration)

**Modified File**: `src/features/cross_ticker.py`  
(Already fixed in previous conversation)

### 4.3 Model Training (`src/models/`)

**File**: `src/models/trainer.py`

```python
# No major changes needed - architecture is timeframe-agnostic
# But update default lookback for daily data

class LSTMTrainer:
    def __init__(self, model, optimizer, criterion, device, lookback=30):
        # lookback=30 now means 30 days (1.5 months) instead of 30 minutes
        self.lookback = lookback
        # ... rest of code ...
```

### 4.4 Backtesting (`src/evaluation/`)

**File**: `src/evaluation/backtester.py`

```python
# Update transaction cost assumptions for daily trading
class Backtester:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,  # 10 bps (reasonable for daily)
        slippage: float = 0.0005,         # 5 bps (lower for daily)
        position_fraction: float = 0.04,  # 4% per position (25 positions max)
        stop_loss: float = 0.05,          # 5% stop loss (wider for daily)
        daily_loss_limit: float = 0.10,  # 10% daily loss limit
    ):
        # ... rest of code ...
```

### 4.5 Risk Management (NEW MODULE)

**New File**: `src/risk/__init__.py`

**New File**: `src/risk/position_sizer.py`

```python
"""
src/risk/position_sizer.py

Position sizing logic based on Kelly Criterion and volatility scaling.
"""

import numpy as np
import pandas as pd
from typing import Dict


class PositionSizer:
    """Dynamic position sizing based on signal strength and volatility."""
    
    def __init__(
        self,
        base_position_size: float = 0.04,  # 4% base position
        max_position_size: float = 0.10,   # 10% max position
        min_position_size: float = 0.01,   # 1% min position
        kelly_fraction: float = 0.25,      # Use 25% of full Kelly
        vol_target: float = 0.15,          # 15% annualized vol target
    ):
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.kelly_fraction = kelly_fraction
        self.vol_target = vol_target
    
    def compute_position_size(
        self,
        signal_strength: float,  # Model prediction (log return)
        realized_vol: float,     # 20-day realized volatility
        win_rate: float = 0.55,  # Historical win rate
        avg_win: float = 0.02,   # Historical avg win
        avg_loss: float = 0.015, # Historical avg loss
    ) -> float:
        """Compute position size using Kelly Criterion + vol scaling.
        
        Returns:
            Position size as fraction of portfolio (0.01 to 0.10).
        """
        # Kelly Criterion: f = (p*b - q) / b
        # where p = win rate, q = 1-p, b = avg_win / avg_loss
        if avg_loss == 0:
            kelly_size = self.base_position_size
        else:
            b = avg_win / avg_loss
            kelly_size = (win_rate * b - (1 - win_rate)) / b
            kelly_size = max(0, kelly_size) * self.kelly_fraction
        
        # Volatility scaling: reduce size if stock is too volatile
        if realized_vol > 0:
            vol_scalar = self.vol_target / realized_vol
            vol_scalar = np.clip(vol_scalar, 0.5, 2.0)  # Don't scale too aggressively
        else:
            vol_scalar = 1.0
        
        # Signal strength scaling: increase size for strong signals
        signal_scalar = np.clip(abs(signal_strength) / 0.01, 0.5, 2.0)
        
        # Final position size
        position_size = self.base_position_size * kelly_size * vol_scalar * signal_scalar
        position_size = np.clip(position_size, self.min_position_size, self.max_position_size)
        
        return position_size
```

**New File**: `src/risk/regime_detector.py`

```python
"""
src/risk/regime_detector.py

Market regime detection using VIX and trend indicators.
"""

import numpy as np
import pandas as pd
from enum import Enum


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_LOW_VOL = "bull_low_vol"      # VIX < 15, SPY uptrend
    BULL_HIGH_VOL = "bull_high_vol"    # VIX 15-25, SPY uptrend
    BEAR_HIGH_VOL = "bear_high_vol"    # VIX > 25, SPY downtrend
    SIDEWAYS = "sideways"              # No clear trend


class RegimeDetector:
    """Detect market regime for strategy adaptation."""
    
    def __init__(
        self,
        vix_low_threshold: float = 15.0,
        vix_high_threshold: float = 25.0,
        trend_lookback: int = 60,  # 60 days for trend
    ):
        self.vix_low = vix_low_threshold
        self.vix_high = vix_high_threshold
        self.trend_lookback = trend_lookback
    
    def detect_regime(
        self,
        spy_prices: pd.Series,
        vix_values: pd.Series,
    ) -> MarketRegime:
        """Detect current market regime.
        
        Args:
            spy_prices: SPY close prices (last 60+ days).
            vix_values: VIX values (last 60+ days).
            
        Returns:
            MarketRegime enum.
        """
        current_vix = vix_values.iloc[-1]
        
        # Detect trend using 20-day vs 60-day moving average
        ma_20 = spy_prices.rolling(20).mean().iloc[-1]
        ma_60 = spy_prices.rolling(60).mean().iloc[-1]
        
        is_uptrend = ma_20 > ma_60
        is_downtrend = ma_20 < ma_60
        
        # Classify regime
        if current_vix < self.vix_low and is_uptrend:
            return MarketRegime.BULL_LOW_VOL
        elif current_vix < self.vix_high and is_uptrend:
            return MarketRegime.BULL_HIGH_VOL
        elif current_vix >= self.vix_high and is_downtrend:
            return MarketRegime.BEAR_HIGH_VOL
        else:
            return MarketRegime.SIDEWAYS
    
    def get_strategy_params(self, regime: MarketRegime) -> dict:
        """Get strategy parameters based on regime.
        
        Returns:
            Dict with position_scalar, stop_loss, take_profit.
        """
        params = {
            MarketRegime.BULL_LOW_VOL: {
                "position_scalar": 1.2,   # Increase positions
                "stop_loss": 0.05,        # Wider stops
                "take_profit": 0.10,      # Higher targets
            },
            MarketRegime.BULL_HIGH_VOL: {
                "position_scalar": 1.0,   # Normal positions
                "stop_loss": 0.04,        # Normal stops
                "take_profit": 0.06,      # Normal targets
            },
            MarketRegime.BEAR_HIGH_VOL: {
                "position_scalar": 0.5,   # Reduce positions
                "stop_loss": 0.03,        # Tight stops
                "take_profit": 0.04,      # Quick profits
            },
            MarketRegime.SIDEWAYS: {
                "position_scalar": 0.8,   # Slightly reduce
                "stop_loss": 0.04,        # Normal stops
                "take_profit": 0.05,      # Normal targets
            },
        }
        return params[regime]
```

---

## 5. Data Pipeline Modifications

### 5.1 New Directory Structure

```
data/
├── raw/                              # Raw Alpaca downloads
│   ├── equities/                     # 25 stocks (daily bars, 2015-2026)
│   │   ├── AAPL.parquet
│   │   ├── MSFT.parquet
│   │   └── ... (25 files)
│   ├── macro/                        # 11 macro tickers
│   │   ├── ^VIX.parquet
│   │   ├── ^TNX.parquet
│   │   ├── DX-Y.NYB.parquet
│   │   └── ... (11 files)
│   └── benchmark/
│       └── SPY.parquet               # SPY (not predicted)
│
├── processed/
│   └── aligned_universe.parquet      # 36 tickers aligned (25 + 11 macro + SPY)
│
└── features/                         # Feature matrices
    ├── AAPL/
    │   ├── X_train.npy               # [N_train × ~130 features]
    │   ├── y_train.npy
    │   ├── X_val.npy
    │   ├── y_val.npy
    │   ├── X_test.npy
    │   ├── y_test.npy
    │   └── metadata.pkl
    └── ... (25 ticker folders, no SPY)
```

### 5.2 Data Ingestion Script Updates

**File**: `scripts/ingest_data.py`

```python
#!/usr/bin/env python3
"""
scripts/ingest_data.py

Ingest daily OHLCV data for 25 equities + 11 macro tickers + SPY.
"""

import argparse
import logging
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.alpaca_ingestor import AlpacaIngestor
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest stock data from Alpaca")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--output", type=str, default="data/raw")
    parser.add_argument("--tickers", type=str, default="config/tickers.txt")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    setup_logger(log_file="logs/ingest_data.log", level="INFO")
    
    # Load equity tickers
    with open(args.tickers) as f:
        equity_tickers = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
    
    # Add macro tickers
    macro_tickers = [
        "^VIX", "^TNX", "^IRX", "DX-Y.NYB",
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI"
    ]
    
    all_tickers = equity_tickers + macro_tickers
    
    logger.info("=" * 60)
    logger.info("Data Ingestion - Real Trading Configuration")
    logger.info("=" * 60)
    logger.info("Equity tickers: %d", len(equity_tickers))
    logger.info("Macro tickers: %d", len(macro_tickers))
    logger.info("Total tickers: %d", len(all_tickers))
    logger.info("Date range: %s to %s", cfg.data.start_date, cfg.data.end_date)
    logger.info("Frequency: %s", cfg.data.freq)
    logger.info("=" * 60)
    
    # Initialize ingestor
    ingestor = AlpacaIngestor(
        api_key=cfg.alpaca.api_key,
        api_secret=cfg.alpaca.api_secret,
        base_url=cfg.alpaca.base_url,
    )
    
    # Fetch data
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, ticker in enumerate(all_tickers, 1):
        logger.info("[%d/%d] Fetching %s...", idx, len(all_tickers), ticker)
        
        try:
            df = ingestor.fetch_bars(
                ticker,
                start=cfg.data.start_date,
                end=cfg.data.end_date,
            )
            
            # Save to parquet
            output_path = output_dir / f"{ticker}.parquet"
            df.to_parquet(output_path)
            
            logger.info("  ✓ Saved %d bars to %s", len(df), output_path)
        
        except Exception as e:
            logger.error("  ✗ Failed to fetch %s: %s", ticker, e)
            continue
    
    logger.info("=" * 60)
    logger.info("✓ Data ingestion complete!")


if __name__ == "__main__":
    main()
```

---

## 6. Configuration Updates

### 6.1 Updated `config/default_config.yaml`

```yaml
# PSO-LSTM Stock Price Prediction Configuration
# Real Trading Version - April 2026

pso:
  n_particles: 30
  n_iterations: 50
  w_min: 0.4
  w_max: 0.9
  c1: 1.5
  c2: 1.5
  v_clamp_fraction: 0.20
  seed: 42
  checkpoint_dir: "results/checkpoints"
  n_workers: 4  # Increased for 25 tickers

lstm:
  # Search space bounds (adjusted for daily data)
  num_layers:
    min: 1
    max: 3  # Reduced (daily data has less temporal complexity)
  hidden_units:
    min: 64
    max: 256  # Reduced (fewer features needed for daily)
    step: 32
  dropout:
    min: 0.1
    max: 0.4
  learning_rate:
    min: 1.0e-4
    max: 1.0e-2
    scale: "log"
  lookback:
    choices: [20, 30, 60, 90]  # Days instead of minutes
  
  # Training settings
  max_epochs: 200  # Increased (daily data trains faster per epoch)
  batch_size: 64   # Reduced (fewer samples per day)
  early_stopping_patience: 20
  grad_clip: 1.0

lstm_baseline:
  num_layers: 2
  hidden_units: 128
  dropout: 0.2
  learning_rate: 0.001
  lookback: 30  # 30 days
  max_epochs: 200
  patience: 20
  batch_size: 64
  grad_clip: 1.0
  use_amp: true
  accumulation_steps: 1
  
  # Walk-forward validation (daily data)
  wfv_fold_size: 252  # 1 year of trading days
  wfv_folds: 5        # 5 years of walk-forward
  
  # Backtesting (daily trading)
  initial_capital: 100000.0
  position_fraction: 0.04  # 4% per position (25 positions max)
  transaction_cost: 0.001  # 10 bps
  slippage: 0.0005         # 5 bps
  stop_loss: 0.05          # 5% stop loss
  daily_loss_limit: 0.10   # 10% daily loss limit

xgboost:
  objective: "reg:squarederror"
  n_estimators: 300  # Increased (daily data is cleaner)
  max_depth: 6       # Deeper trees (more stable signals)
  learning_rate: 0.01
  subsample: 0.8
  colsample_bytree: 0.7
  min_child_weight: 3
  gamma: 0.1
  reg_alpha: 0.0
  reg_lambda: 1.0
  early_stopping_rounds: 50
  use_optuna: true
  optuna_trials: 30
  tree_method: "gpu_hist"
  max_bin: 256
  lookback: 30  # 30 days

fitness:
  rmse_weight: 0.3      # Reduced (care more about trading metrics)
  sharpe_weight: 0.5    # Increased (primary goal)
  drawdown_weight: 0.2
  signal_threshold: 5.0e-4  # 5 bps minimum prediction
  transaction_cost: 0.001

data:
  # Extended date range for regime diversity
  start_date: "2015-01-01"
  end_date: "2026-04-05"
  train_end: "2023-12-31"  # 9 years training
  val_end: "2024-12-31"    # 1 year validation
  test_end: "2026-04-05"   # 1.25 years test
  
  # Daily bars
  freq: "1D"
  session_start: null  # Not applicable for daily
  session_end: null
  timezone: "America/New_York"

features:
  selector:
    method: "xgboost"
    importance_threshold: 0.75  # Slightly higher (more features available)
    xgb_params:
      n_estimators: 150
      max_depth: 5
      learning_rate: 0.1
      subsample: 0.8
      colsample_bytree: 0.8

backtesting:
  transaction_cost: 0.001  # 10 bps
  slippage: 0.0005         # 5 bps
  position_size: 0.04      # 4% per position
  stop_loss: 0.05          # 5% stop loss
  take_profit: 0.10        # 10% take profit

risk:
  # Position sizing
  base_position_size: 0.04
  max_position_size: 0.10
  min_position_size: 0.01
  kelly_fraction: 0.25
  vol_target: 0.15
  
  # Regime detection
  vix_low_threshold: 15.0
  vix_high_threshold: 25.0
  trend_lookback: 60
  
  # Portfolio limits
  max_portfolio_leverage: 1.0  # No leverage
  max_sector_exposure: 0.40    # Max 40% in any sector
  max_correlation: 0.70        # Don't hold highly correlated positions

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/pso_lstm.log"

alpaca:
  api_key: "${ALPACA_API_KEY}"      # Environment variable
  api_secret: "${ALPACA_API_SECRET}"
  base_url: "https://paper-api.alpaca.markets"  # Paper trading first
```

### 6.2 Updated `config/tickers.txt`

```txt
# 30-ticker universe for production trading
# 25 equities across 7 sectors + SPY benchmark (not traded)
# SPY used only for beta/alpha decomposition

# Technology (5)
AAPL
MSFT
GOOGL
NVDA
TSLA

# Financials (5)
JPM
BAC
GS
MS
WFC

# Healthcare (5)
UNH
JNJ
PFE
ABBV
TMO

# Energy (4)
XOM
CVX
COP
SLB

# Consumer Discretionary (4)
AMZN
HD
MCD
NKE

# Consumer Staples (3)
PG
KO
WMT

# Industrials (4)
CAT
BA
UNP
HON

# Benchmark (not traded - used only for features)
SPY
```

---

## 7. Testing & Validation Strategy

### 7.1 Data Quality Checks

**Script**: `scripts/validate_data.py` (NEW)

```python
#!/usr/bin/env python3
"""
scripts/validate_data.py

Validate data quality after ingestion.
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def validate_ticker_data(ticker: str, df: pd.DataFrame) -> dict:
    """Validate a single ticker's data."""
    issues = []
    
    # Check for missing data
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_pct > 0.05:
        issues.append(f"Missing data: {missing_pct:.2%}")
    
    # Check for zero volume days
    if "volume" in df.columns:
        zero_vol_pct = (df["volume"] == 0).sum() / len(df)
        if zero_vol_pct > 0.10:
            issues.append(f"Zero volume days: {zero_vol_pct:.2%}")
    
    # Check for price anomalies (>20% daily moves)
    if "close" in df.columns:
        returns = df["close"].pct_change()
        extreme_moves = (returns.abs() > 0.20).sum()
        if extreme_moves > 5:
            issues.append(f"Extreme moves (>20%): {extreme_moves}")
    
    # Check date range
    expected_days = 252 * 11  # ~11 years of trading days
    actual_days = len(df)
    if actual_days < expected_days * 0.90:
        issues.append(f"Insufficient data: {actual_days} days (expected ~{expected_days})")
    
    return {
        "ticker": ticker,
        "rows": len(df),
        "start_date": df.index.min(),
        "end_date": df.index.max(),
        "issues": issues,
        "status": "PASS" if len(issues) == 0 else "FAIL",
    }


def main():
    data_dir = Path("data/raw")
    results = []
    
    for parquet_file in data_dir.glob("*.parquet"):
        ticker = parquet_file.stem
        df = pd.read_parquet(parquet_file)
        result = validate_ticker_data(ticker, df)
        results.append(result)
        
        status_icon = "✓" if result["status"] == "PASS" else "✗"
        logger.info(f"{status_icon} {ticker}: {result['rows']} rows, {result['start_date']} to {result['end_date']}")
        
        if result["issues"]:
            for issue in result["issues"]:
                logger.warning(f"  ⚠ {issue}")
    
    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = len(results) - passed
    
    logger.info("=" * 60)
    logger.info(f"Validation complete: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = main()
    exit(0 if success else 1)
```

### 7.2 Feature Quality Checks

**Script**: `scripts/validate_features.py` (NEW)

```python
#!/usr/bin/env python3
"""
scripts/validate_features.py

Validate feature engineering output.
"""

import numpy as np
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def validate_ticker_features(ticker: str, ticker_dir: Path) -> dict:
    """Validate a single ticker's features."""
    issues = []
    
    # Load data
    X_train = np.load(ticker_dir / "X_train.npy")
    y_train = np.load(ticker_dir / "y_train.npy")
    X_val = np.load(ticker_dir / "X_val.npy")
    y_val = np.load(ticker_dir / "y_val.npy")
    X_test = np.load(ticker_dir / "X_test.npy")
    y_test = np.load(ticker_dir / "y_test.npy")
    
    with open(ticker_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    # Check for NaN/inf
    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        if np.isnan(arr).any():
            issues.append(f"{name} contains NaN")
        if np.isinf(arr).any():
            issues.append(f"{name} contains inf")
    
    # Check feature count
    expected_features = 130  # ~117 original + 12 macro
    actual_features = metadata["n_features"]
    if actual_features < 50:
        issues.append(f"Too few features: {actual_features} (expected ~{expected_features})")
    
    # Check sample counts
    expected_train_samples = 252 * 9  # 9 years
    if metadata["train_samples"] < expected_train_samples * 0.80:
        issues.append(f"Too few train samples: {metadata['train_samples']}")
    
    # Check target distribution
    y_std = np.std(y_train)
    if y_std < 0.005 or y_std > 0.10:
        issues.append(f"Unusual target std: {y_std:.4f} (expected 0.01-0.03)")
    
    return {
        "ticker": ticker,
        "n_features": actual_features,
        "train_samples": metadata["train_samples"],
        "val_samples": metadata["val_samples"],
        "test_samples": metadata["test_samples"],
        "target_std": y_std,
        "issues": issues,
        "status": "PASS" if len(issues) == 0 else "FAIL",
    }


def main():
    features_dir = Path("data/features")
    results = []
    
    for ticker_dir in features_dir.iterdir():
        if not ticker_dir.is_dir():
            continue
        
        ticker = ticker_dir.name
        result = validate_ticker_features(ticker, ticker_dir)
        results.append(result)
        
        status_icon = "✓" if result["status"] == "PASS" else "✗"
        logger.info(
            f"{status_icon} {ticker}: {result['n_features']} features, "
            f"{result['train_samples']} train samples"
        )
        
        if result["issues"]:
            for issue in result["issues"]:
                logger.warning(f"  ⚠ {issue}")
    
    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = len(results) - passed
    
    logger.info("=" * 60)
    logger.info(f"Feature validation complete: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = main()
    exit(0 if success else 1)
```

### 7.3 Backtest Validation

**Checklist**:
- [ ] Sharpe ratio > 1.0 on test set (2025-2026)
- [ ] Max drawdown < 20%
- [ ] Win rate > 50%
- [ ] Profit factor > 1.5
- [ ] No look-ahead bias (features computed only on past data)
- [ ] Transaction costs included (10 bps)
- [ ] Slippage included (5 bps)
- [ ] Results stable across 5 random seeds

---

## 8. Risk Management Additions

### 8.1 Portfolio-Level Risk Limits

**File**: `src/risk/portfolio_manager.py` (NEW)

```python
"""
src/risk/portfolio_manager.py

Portfolio-level risk management and position allocation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class PortfolioManager:
    """Manage portfolio-level risk and position allocation."""
    
    def __init__(
        self,
        max_portfolio_leverage: float = 1.0,
        max_sector_exposure: float = 0.40,
        max_correlation: float = 0.70,
        max_positions: int = 25,
    ):
        self.max_leverage = max_portfolio_leverage
        self.max_sector_exposure = max_sector_exposure
        self.max_correlation = max_correlation
        self.max_positions = max_positions
        
        # Sector mapping
        self.sector_map = {
            "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "NVDA": "Tech", "TSLA": "Tech",
            "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "MS": "Finance", "WFC": "Finance",
            "UNH": "Healthcare", "JNJ": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare", "TMO": "Healthcare",
            "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
            "AMZN": "Consumer", "HD": "Consumer", "MCD": "Consumer", "NKE": "Consumer",
            "PG": "Consumer", "KO": "Consumer", "WMT": "Consumer",
            "CAT": "Industrial", "BA": "Industrial", "UNP": "Industrial", "HON": "Industrial",
        }
    
    def allocate_positions(
        self,
        signals: Dict[str, float],  # ticker → predicted return
        position_sizes: Dict[str, float],  # ticker → position size (0-1)
        current_positions: Dict[str, float],  # ticker → current weight
        correlation_matrix: pd.DataFrame,  # ticker × ticker correlation
    ) -> Dict[str, float]:
        """Allocate positions with risk constraints.
        
        Returns:
            Dict mapping ticker → target weight (0-1).
        """
        # Sort signals by strength
        ranked_tickers = sorted(signals.keys(), key=lambda t: abs(signals[t]), reverse=True)
        
        # Initialize target weights
        target_weights = {}
        sector_exposure = {}
        
        for ticker in ranked_tickers:
            # Skip if signal is too weak
            if abs(signals[ticker]) < 0.001:
                target_weights[ticker] = 0.0
                continue
            
            # Get proposed position size
            proposed_size = position_sizes[ticker]
            
            # Check sector exposure
            sector = self.sector_map.get(ticker, "Other")
            current_sector_exposure = sector_exposure.get(sector, 0.0)
            
            if current_sector_exposure + proposed_size > self.max_sector_exposure:
                # Reduce size to fit sector limit
                proposed_size = max(0, self.max_sector_exposure - current_sector_exposure)
            
            # Check correlation with existing positions
            if len(target_weights) > 0:
                avg_corr = self._compute_avg_correlation(ticker, target_weights, correlation_matrix)
                if avg_corr > self.max_correlation:
                    # Reduce size for highly correlated positions
                    proposed_size *= 0.5
            
            # Check max positions limit
            if len([w for w in target_weights.values() if w > 0]) >= self.max_positions:
                target_weights[ticker] = 0.0
                continue
            
            # Assign weight
            target_weights[ticker] = proposed_size
            sector_exposure[sector] = current_sector_exposure + proposed_size
        
        # Normalize to max leverage
        total_weight = sum(target_weights.values())
        if total_weight > self.max_leverage:
            scale_factor = self.max_leverage / total_weight
            target_weights = {t: w * scale_factor for t, w in target_weights.items()}
        
        return target_weights
    
    def _compute_avg_correlation(
        self,
        ticker: str,
        existing_positions: Dict[str, float],
        correlation_matrix: pd.DataFrame,
    ) -> float:
        """Compute average correlation with existing positions."""
        if ticker not in correlation_matrix.index:
            return 0.0
        
        correlations = []
        for other_ticker, weight in existing_positions.items():
            if weight > 0 and other_ticker in correlation_matrix.columns:
                corr = correlation_matrix.loc[ticker, other_ticker]
                correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
```

---

## 9. Deployment Checklist

### 9.1 Pre-Deployment

- [ ] All data quality checks pass (`scripts/validate_data.py`)
- [ ] All feature quality checks pass (`scripts/validate_features.py`)
- [ ] Backtest Sharpe > 1.0 on test set (2025-2026)
- [ ] Backtest max drawdown < 20%
- [ ] Walk-forward validation shows consistent performance
- [ ] No data leakage detected (manual audit)
- [ ] Transaction costs and slippage included in backtest
- [ ] Risk management modules tested
- [ ] Code reviewed and documented

### 9.2 Paper Trading

- [ ] Paper trading account set up (Alpaca)
- [ ] Live data pipeline tested
- [ ] Order execution tested (market orders, limit orders)
- [ ] Position sizing logic tested
- [ ] Stop loss / take profit logic tested
- [ ] Monitoring dashboard deployed
- [ ] Alerting system configured (email, Slack)
- [ ] Paper trading for 1 week with no issues

### 9.3 Live Trading

- [ ] Paper trading Sharpe > 0.8 for 1 week
- [ ] No execution errors in paper trading
- [ ] Risk limits enforced correctly
- [ ] Start with $10,000 capital (small)
- [ ] Max 5 positions initially
- [ ] Daily monitoring for first month
- [ ] Weekly performance review

---

## 10. Success Metrics

### 10.1 Performance Targets

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| **Sharpe Ratio** (annual) | 1.0 | 1.5 | 2.0 |
| **Max Drawdown** | < 25% | < 20% | < 15% |
| **Win Rate** | > 50% | > 55% | > 60% |
| **Profit Factor** | > 1.3 | > 1.5 | > 2.0 |
| **CAGR** | > 10% | > 15% | > 20% |
| **Calmar Ratio** | > 0.5 | > 0.75 | > 1.0 |

### 10.2 Risk Metrics

| Metric | Limit |
|--------|-------|
| **Max position size** | 10% |
| **Max sector exposure** | 40% |
| **Max portfolio leverage** | 1.0 (no leverage) |
| **Daily loss limit** | 10% |
| **Max correlation** | 0.70 |

### 10.3 Operational Metrics

| Metric | Target |
|--------|--------|
| **Data ingestion uptime** | > 99% |
| **Model inference latency** | < 1 second |
| **Order execution latency** | < 5 seconds |
| **False positive rate** (bad signals) | < 30% |
| **System downtime** | < 1 hour/month |

---

## Appendix A: Estimated Costs

### A.1 Data Costs

| Item | Cost | Frequency |
|------|------|-----------|
| Alpaca API (free tier) | $0 | Monthly |
| Alpaca API (unlimited tier) | $99 | Monthly |
| AWS S3 storage (100 GB) | $2.30 | Monthly |

**Total**: $0-$100/month

### A.2 Compute Costs

| Item | Cost | Frequency |
|------|------|-----------|
| AWS EC2 g4dn.xlarge (GPU) | $0.526/hour | On-demand |
| Daily training (2 hours/day) | $31.56 | Monthly |
| Data storage (S3) | $2.30 | Monthly |

**Total**: ~$35/month

### A.3 Transaction Costs

| Item | Cost | Frequency |
|------|------|-----------|
| Alpaca commission | $0 | Per trade |
| SEC fees | $0.0000278 × trade value | Per trade |
| Estimated slippage | 0.05% × trade value | Per trade |

**Total**: ~0.05-0.10% per trade

---

## Appendix B: Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1**: Data Pipeline | 2 weeks | 30 stocks + 11 macro tickers ingested |
| **Phase 2**: Feature Engineering | 1 week | 25 ticker feature matrices built |
| **Phase 3**: Model Training | 1 week | Baselines + PSO-LSTM trained |
| **Phase 4**: Risk Management | 1 week | Position sizing + regime detection |
| **Phase 5**: Deployment | 1 week | Paper trading live |
| **Total** | **6 weeks** | Production trading system |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-04-07 | AI Assistant | Initial draft |

---

**Next Steps**:
1. Review this plan with stakeholders
2. Prioritize changes (Critical → High → Medium)
3. Begin Phase 1 (Data Pipeline) immediately
4. Schedule weekly progress reviews

**Questions? Contact**: [Your contact info]
