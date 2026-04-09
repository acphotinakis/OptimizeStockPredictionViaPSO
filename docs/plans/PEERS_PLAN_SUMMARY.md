# Peer Selection Implementation - Quick Summary

**Date:** 2026-04-08  
**Status:** Ready to implement  

---

## The Problem

Currently, every stock gets the same peers regardless of sector. AAPL might get JPM as a peer (tech vs finance), which provides no useful signal. We're also missing critical market context like sector ETFs and volatility signals.

---

## The Solution: 4-Tier Symbol Universe

Each target stock gets a custom universe:

### Tier 1: Market Structure (Always Included)
- SPY, QQQ, IWM, DIA
- Provides broad market context

### Tier 2: Sector ETF (Target-Specific)
- AAPL → XLK (technology)
- NVDA → SOXX (semiconductors)  
- JPM → XLF (financials)
- Provides sector-specific signals

### Tier 3: Peers (Correlation-Selected, Max 3)
- Selected on training data only (no leakage)
- AAPL gets MSFT, GOOGL, META (not JPM, BAC)
- Provides relative strength signals

### Tier 4: Market Internals
- UVXY (volatility), GLD (safety), TLT (bonds)
- Provides risk-on/risk-off context

---

## What We Have vs. What We Need

### ✅ Already Have (52 symbols in `mass_tickers.txt`)
- SPY (benchmark)
- 51 stocks across 9 sectors:
  - Technology: AAPL, MSFT, GOOGL, NVDA, META, TSLA, INTC, AMD, CRM, ADBE
  - Financials: JPM, BAC, WFC, GS, MS, C, BLK, SCHW
  - Healthcare: JNJ, UNH, PFE, ABBV, TMO, MRK, LLY, ABT
  - Consumer: AMZN, WMT, HD, MCD, NKE, SBUX, TGT, COST
  - Industrials: BA, CAT, GE, UPS, HON, LMT
  - Energy: XOM, CVX, COP, SLB
  - Communication: DIS, NFLX, CMCSA
  - Materials: LIN, APD
  - Utilities: NEE

### 🔴 Need to Ingest (17 symbols in `tickers_new_context.txt`)

**Market Indices (3)**:
- QQQ, IWM, DIA

**Sector ETFs (11)**:
- XLK (Technology)
- SOXX (Semiconductors)
- XLF (Financials)
- XLV (Healthcare)
- XLE (Energy)
- XLY (Consumer Discretionary)
- XLP (Consumer Staples)
- XLI (Industrials)
- XLC (Communication)
- XLB (Materials)
- XLU (Utilities)

**Market Internals (3)**:
- UVXY (volatility)
- GLD (gold/safety)
- TLT (bonds)

---

## Quick Start: Ingest New Symbols

```bash
# 1. Ingest the 17 new context symbols
python pipelines/ingest_data.py --mode ingest \
    --tickers config/tickers_new_context.txt \
    --config config/default_config.yaml

# 2. Clean them
python pipelines/ingest_data.py --mode clean \
    --tickers config/tickers_new_context.txt

# 3. Re-align entire universe (52 existing + 17 new = 69 total)
python pipelines/ingest_data.py --mode align \
    --tickers config/tickers_full.txt
```

---

## Implementation Phases

1. **Configuration** (Week 1)
   - Create `symbol_universe.yaml` (sector mappings)
   - Create `SymbolUniverseBuilder` class
   - Write unit tests

2. **Data Ingestion** (Week 2)
   - Ingest 17 new symbols
   - Re-align all 69 symbols

3. **Pipeline Integration** (Week 3)
   - Update `FeaturePipeline` to use universe builder
   - Update `run_build_features.py`

4. **Validation** (Week 4)
   - Test on subset of stocks
   - Verify peer selections make sense
   - Check feature counts

5. **Rollout** (Week 5)
   - Full feature building
   - Documentation

---

## Expected Results

### Before (Current State)
- Universe: All 52 stocks treated equally
- Peers: Random selection, often wrong sector
- Features: ~50-80 per target
- Context: Minimal (just SPY)

### After (With Universe Builder)
- Universe: Target-specific (12-15 symbols)
- Peers: Sector-appropriate, correlation-selected
- Features: ~80-120 per target (20-40 new cross-ticker features)
- Context: Rich (market structure + sector + peers + internals)

### Example for AAPL
**Before**: AAPL + SPY + random peers (maybe JPM, CVX)  
**After**: AAPL + [SPY, QQQ, IWM, DIA] + XLK + [MSFT, GOOGL, META] + [UVXY, GLD, TLT]

The model now sees:
- Broad market (SPY/QQQ/IWM/DIA)
- Tech sector (XLK)
- Tech peers (MSFT/GOOGL/META)
- Risk signals (UVXY/GLD/TLT)

This provides the context needed to learn patterns like:
- "AAPL up 2%, MSFT flat → AAPL leading tech"
- "XLK down, AAPL up → AAPL diverging from sector"
- "UVXY spiking, SPY flat → fear entering market"

---

## Files Created

### Configuration
- `config/symbol_universe.yaml` - Sector mappings and settings
- `config/tickers_new_context.txt` - 17 new symbols to ingest
- `config/tickers_full.txt` - All 69 symbols (52 existing + 17 new)

### Code
- `src/features/universe_builder.py` - Core builder class
- `tests/test_universe_builder.py` - Unit tests
- `tests/test_pipeline_with_universe.py` - Integration tests

### Documentation
- `docs/plans/PEERS_PLAN.md` - Full implementation plan (this document)
- `docs/guides/SYMBOL_UNIVERSE_USAGE.md` - Usage guide
- `docs/guides/SYMBOL_UNIVERSE_MIGRATION.md` - Migration guide

### Runtime Outputs
- `data/features/fitted_peers.json` - Peer selections per target

---

## Next Steps

1. **Review** the full plan in `docs/plans/PEERS_PLAN.md`
2. **Ingest** the 17 new symbols using commands above
3. **Implement** `SymbolUniverseBuilder` class
4. **Test** on a small subset (e.g., AAPL, NVDA, JPM)
5. **Validate** peer selections make sense
6. **Scale** to full universe

---

## Questions?

- **Why only 3 peers?** Prevents feature explosion while capturing key relationships
- **Why these specific ETFs?** Standard SPDR sector ETFs with high liquidity
- **Why UVXY not VIX?** VIX is an index, UVXY is tradeable ETF with OHLCV data
- **Can I add more sectors?** Yes, just update `symbol_universe.yaml`
- **What if a stock has no sector?** Falls back to SPY, logs warning

See full plan for detailed answers and implementation guidance.
