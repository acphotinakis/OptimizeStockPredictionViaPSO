# Symbol Universe Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SYMBOL UNIVERSE BUILDER                         │
│                                                                     │
│  Input: Target Stock (e.g., AAPL)                                  │
│  Output: Target-Specific Symbol Universe (12-15 symbols)           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │         4-TIER UNIVERSE CONSTRUCTION            │
        └─────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │                                                   │
        ▼                                                   ▼
┌──────────────┐                                   ┌──────────────┐
│  TIER 1      │                                   │  TIER 2      │
│  Market      │                                   │  Sector ETF  │
│  Structure   │                                   │  (Target-    │
│              │                                   │  Specific)   │
│  • SPY       │                                   │              │
│  • QQQ       │                                   │  AAPL → XLK  │
│  • IWM       │                                   │  NVDA → SOXX │
│  • DIA       │                                   │  JPM  → XLF  │
│              │                                   │  XOM  → XLE  │
│  (Always)    │                                   │  JNJ  → XLV  │
└──────────────┘                                   └──────────────┘
        │                                                   │
        └─────────────────────────┬─────────────────────────┘
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │                                                 │
        ▼                                                 ▼
┌──────────────┐                                 ┌──────────────┐
│  TIER 3      │                                 │  TIER 4      │
│  Peers       │                                 │  Market      │
│  (Corr-      │                                 │  Internals   │
│  Selected)   │                                 │              │
│              │                                 │  • UVXY      │
│  Max 3       │                                 │  • GLD       │
│  Training    │                                 │  • TLT       │
│  Data Only   │                                 │              │
│              │                                 │  (Always)    │
└──────────────┘                                 └──────────────┘
```

---

## Example: AAPL Universe Construction

```
Target: AAPL (Technology Stock)

Step 1: Add Market Structure (Tier 1)
┌─────────────────────────────────┐
│ Universe = [AAPL]               │
│          + [SPY, QQQ, IWM, DIA] │
│          = [AAPL, SPY, QQQ,     │
│             IWM, DIA]           │
└─────────────────────────────────┘

Step 2: Add Sector ETF (Tier 2)
┌─────────────────────────────────┐
│ Lookup: AAPL → Technology       │
│ Sector ETF: XLK                 │
│ Universe += [XLK]               │
│          = [AAPL, SPY, QQQ,     │
│             IWM, DIA, XLK]      │
└─────────────────────────────────┘

Step 3: Select Peers (Tier 3)
┌─────────────────────────────────┐
│ Candidates: All tech stocks     │
│ (MSFT, GOOGL, META, CRM, ADBE)  │
│                                 │
│ Compute correlations on         │
│ training data returns:          │
│   MSFT:  0.72                   │
│   GOOGL: 0.68                   │
│   META:  0.65                   │
│   CRM:   0.58                   │
│   ADBE:  0.55                   │
│                                 │
│ Select top 3:                   │
│ Universe += [MSFT, GOOGL, META] │
│          = [AAPL, SPY, QQQ,     │
│             IWM, DIA, XLK,      │
│             MSFT, GOOGL, META]  │
└─────────────────────────────────┘

Step 4: Add Market Internals (Tier 4)
┌─────────────────────────────────┐
│ Universe += [UVXY, GLD, TLT]    │
│          = [AAPL, SPY, QQQ,     │
│             IWM, DIA, XLK,      │
│             MSFT, GOOGL, META,  │
│             UVXY, GLD, TLT]     │
│                                 │
│ Total: 12 symbols               │
└─────────────────────────────────┘
```

---

## Comparison: AAPL vs. JPM vs. XOM

### AAPL (Technology)
```
Universe (12 symbols):
  Target:    AAPL
  Market:    SPY, QQQ, IWM, DIA
  Sector:    XLK (Technology)
  Peers:     MSFT, GOOGL, META
  Internals: UVXY, GLD, TLT
```

### JPM (Financials)
```
Universe (12 symbols):
  Target:    JPM
  Market:    SPY, QQQ, IWM, DIA
  Sector:    XLF (Financials)
  Peers:     BAC, GS, MS
  Internals: UVXY, GLD, TLT
```

### XOM (Energy)
```
Universe (12 symbols):
  Target:    XOM
  Market:    SPY, QQQ, IWM, DIA
  Sector:    XLE (Energy)
  Peers:     CVX, COP, SLB
  Internals: UVXY, GLD, TLT
```

**Key Insight**: Same structure (Tiers 1 & 4), different sector ETF and peers (Tiers 2 & 3).

---

## Data Flow: Training vs. Validation

```
┌────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                          │
└────────────────────────────────────────────────────────────────┘

Input: dfs_train (all 69 symbols, training split)

  ┌──────────────────────────────────┐
  │  SymbolUniverseBuilder           │
  │  .get_universe(AAPL, fit=True)   │
  └──────────────────────────────────┘
                  │
                  ▼
  ┌──────────────────────────────────┐
  │  1. Add Tier 1 (market)          │
  │  2. Add Tier 2 (sector ETF)      │
  │  3. SELECT PEERS (correlation)   │  ← COMPUTED HERE
  │  4. Add Tier 4 (internals)       │
  └──────────────────────────────────┘
                  │
                  ▼
  ┌──────────────────────────────────┐
  │  Store peers: AAPL → [MSFT,      │
  │                       GOOGL,      │
  │                       META]       │
  └──────────────────────────────────┘
                  │
                  ▼
  ┌──────────────────────────────────┐
  │  Return universe:                │
  │  [AAPL, SPY, QQQ, IWM, DIA,      │
  │   XLK, MSFT, GOOGL, META,        │
  │   UVXY, GLD, TLT]                │
  └──────────────────────────────────┘
                  │
                  ▼
  ┌──────────────────────────────────┐
  │  FeaturePipeline                 │
  │  .fit_transform(dfs_train)       │
  │                                  │
  │  → Computes features for all     │
  │    12 symbols                    │
  │  → Fits feature selector         │
  │  → Returns X_train, y_train      │
  └──────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                      VALIDATION PHASE                          │
└────────────────────────────────────────────────────────────────┘

Input: dfs_val (all 69 symbols, validation split)

  ┌──────────────────────────────────┐
  │  SymbolUniverseBuilder           │
  │  .get_universe(AAPL, fit=False)  │
  └──────────────────────────────────┘
                  │
                  ▼
  ┌──────────────────────────────────┐
  │  1. Add Tier 1 (market)          │
  │  2. Add Tier 2 (sector ETF)      │
  │  3. USE STORED PEERS             │  ← NO RE-SELECTION
  │  4. Add Tier 4 (internals)       │
  └──────────────────────────────────┘
                  │
                  ▼
  ┌──────────────────────────────────┐
  │  Retrieve stored peers:          │
  │  AAPL → [MSFT, GOOGL, META]      │
  │  (same as training)              │
  └──────────────────────────────────┘
                  │
                  ▼
  ┌──────────────────────────────────┐
  │  Return universe:                │
  │  [AAPL, SPY, QQQ, IWM, DIA,      │
  │   XLK, MSFT, GOOGL, META,        │
  │   UVXY, GLD, TLT]                │
  │  (identical to training)         │
  └──────────────────────────────────┘
                  │
                  ▼
  ┌──────────────────────────────────┐
  │  FeaturePipeline                 │
  │  .transform(dfs_val)             │
  │                                  │
  │  → Computes features for same    │
  │    12 symbols                    │
  │  → Uses fitted feature selector  │
  │  → Returns X_val, y_val          │
  └──────────────────────────────────┘
```

**Critical**: Peers are selected ONCE on training data and FROZEN for val/test. This prevents look-ahead bias.

---

## Feature Engineering Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    FEATURE COMPUTATION                         │
└────────────────────────────────────────────────────────────────┘

Input: Universe = [AAPL, SPY, QQQ, IWM, DIA, XLK, 
                   MSFT, GOOGL, META, UVXY, GLD, TLT]

For each symbol, compute:
  ┌──────────────────────────────────┐
  │  Base Features (10)              │
  │  - log_return                    │
  │  - hl_ratio                      │
  │  - co_ratio                      │
  │  - true_range                    │
  │  - gap                           │
  │  - ...                           │
  └──────────────────────────────────┘
  ┌──────────────────────────────────┐
  │  Technical Features (15)         │
  │  - RSI                           │
  │  - MACD                          │
  │  - Bollinger Bands               │
  │  - ATR                           │
  │  - ...                           │
  └──────────────────────────────────┘
  ┌──────────────────────────────────┐
  │  Volume Features (8)             │
  │  - VWAP                          │
  │  - Volume ratio                  │
  │  - OBV                           │
  │  - ...                           │
  └──────────────────────────────────┘

Cross-Ticker Features:
  ┌──────────────────────────────────┐
  │  Target vs. SPY (10)             │
  │  - beta_spy                      │
  │  - corr_spy                      │
  │  - alpha_spy                     │
  │  - rel_strength_spy              │
  │  - ...                           │
  └──────────────────────────────────┘
  ┌──────────────────────────────────┐
  │  Target vs. Peers (3 × 3 = 9)    │
  │  - peer_corr_MSFT                │
  │  - peer_corr_GOOGL               │
  │  - peer_corr_META                │
  │  - ...                           │
  └──────────────────────────────────┘
  ┌──────────────────────────────────┐
  │  Market Breadth (2)              │
  │  - mkt_breadth                   │
  │  - universe_mean_ret             │
  └──────────────────────────────────┘

Total: ~80-120 features before selection
       ~40-50 features after selection
```

---

## Leakage Prevention Mechanisms

```
┌────────────────────────────────────────────────────────────────┐
│                    LEAKAGE SAFEGUARDS                          │
└────────────────────────────────────────────────────────────────┘

1. Peer Selection
   ┌──────────────────────────────────┐
   │  ✓ Computed on training data     │
   │  ✓ Frozen for val/test           │
   │  ✓ Saved to fitted_peers.json    │
   │  ✗ Never re-computed on val/test │
   └──────────────────────────────────┘

2. Feature Engineering
   ┌──────────────────────────────────┐
   │  ✓ Rolling windows (right-align) │
   │  ✓ shift(1) for lagged features  │
   │  ✓ No future data in any feature │
   │  ✗ No np.roll (wraps end→start)  │
   └──────────────────────────────────┘

3. Feature Selection
   ┌──────────────────────────────────┐
   │  ✓ Fit on training data only     │
   │  ✓ Same features for val/test    │
   │  ✗ No re-selection on val/test   │
   └──────────────────────────────────┘

4. Scaling
   ┌──────────────────────────────────┐
   │  ✓ Fit on training data only     │
   │  ✓ Transform val/test with       │
   │    training statistics           │
   │  ✗ No re-fitting on val/test     │
   └──────────────────────────────────┘
```

---

## Memory & Performance

### Symbol Count per Target
```
Tier 1: 4 symbols  (market structure)
Tier 2: 1 symbol   (sector ETF)
Tier 3: 3 symbols  (peers)
Tier 4: 3 symbols  (market internals)
Target: 1 symbol
─────────────────
Total:  12 symbols per target

Features per symbol: ~8 average
Total features: 12 × 8 = 96
After selection: ~40-50 (50% pruning)
```

### Memory Estimate
```
Per target:
  - Raw features: 96 × N_samples × 4 bytes (float32)
  - Selected features: 50 × N_samples × 4 bytes
  
For N_samples = 500,000 (1 year of minute data):
  - Raw: 96 × 500k × 4 = 192 MB
  - Selected: 50 × 500k × 4 = 100 MB

For 51 targets:
  - Total raw: 51 × 192 MB = 9.8 GB
  - Total selected: 51 × 100 MB = 5.1 GB
  
Fits comfortably in 16-32 GB RAM.
```

---

## Configuration Example

```yaml
# config/symbol_universe.yaml

market_context:
  - SPY
  - QQQ
  - IWM
  - DIA

sector_etfs:
  technology:
    etf: XLK
    stocks: [AAPL, MSFT, GOOGL, META, CRM, ADBE]
  
  semiconductors:
    etf: SOXX
    stocks: [NVDA, AMD, INTC]
  
  financials:
    etf: XLF
    stocks: [JPM, BAC, GS, MS, C, WFC, BLK, SCHW]

market_internals:
  - UVXY
  - GLD
  - TLT

peer_selection:
  max_peers: 3
  method: "pearson"
  min_correlation: 0.3
  lookback_days: 252

prediction_targets:
  - AAPL
  - NVDA
  - JPM
  # ... (up to all 51 stocks)
```

---

## Usage Example

```python
from src.features.universe_builder import SymbolUniverseBuilder
from src.features.pipeline import FeaturePipeline

# Load universe builder
builder = SymbolUniverseBuilder.from_config(
    "config/symbol_universe.yaml"
)

# Initialize pipeline for AAPL
pipeline = FeaturePipeline(
    target_ticker="AAPL",
    universe_builder=builder,
)

# Fit on training data
X_train, y_train, features = pipeline.fit_transform(dfs_train)
# → Selects peers: [MSFT, GOOGL, META]
# → Universe: [AAPL, SPY, QQQ, IWM, DIA, XLK, 
#              MSFT, GOOGL, META, UVXY, GLD, TLT]

# Transform validation data
X_val, y_val = pipeline.transform(dfs_val)
# → Uses same peers: [MSFT, GOOGL, META]
# → Same universe as training

# Check fitted peers
peers = builder.get_fitted_peers("AAPL")
print(f"AAPL peers: {peers}")
# Output: AAPL peers: ['MSFT', 'GOOGL', 'META']

# Save for reproducibility
builder.save_peers("data/features/fitted_peers.json")
```

---

## Benefits Summary

### Signal Quality
- ✅ Sector-appropriate peers (AAPL gets tech stocks, not banks)
- ✅ Market structure context (SPY/QQQ/IWM/DIA relationships)
- ✅ Sector-specific signals (XLK for tech, XLF for financials)
- ✅ Risk regime signals (UVXY/GLD/TLT)

### Leakage Prevention
- ✅ Peers selected on training data only
- ✅ No re-selection on validation/test
- ✅ Reproducible (saved to JSON)

### Scalability
- ✅ Modest universe size (12-15 symbols per target)
- ✅ Reasonable feature count (~40-50 after selection)
- ✅ Fits in memory (5-10 GB for 51 targets)

### Maintainability
- ✅ Configuration-driven (YAML)
- ✅ Clear tier structure
- ✅ Easy to add new sectors/stocks
- ✅ Well-tested and documented
