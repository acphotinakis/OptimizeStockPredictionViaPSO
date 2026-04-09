# Peer Selection Implementation Checklist

**Date Started:** 2026-04-08  
**Status:** Planning Complete, Ready to Implement  

---

## Phase 0: Pre-Implementation ✅

- [x] Identify existing ticker universe (51 stocks + SPY in `mass_tickers.txt`)
- [x] Identify missing symbols (17 ETFs/indices needed)
- [x] Create implementation plan (`PEERS_PLAN.md`)
- [x] Create architecture diagram (`PEERS_ARCHITECTURE.md`)
- [x] Create summary document (`PEERS_PLAN_SUMMARY.md`)

---

## Phase 1: Configuration Files

### Core Configuration
- [ ] Create `config/symbol_universe.yaml`
  - [ ] Define market_context (SPY, QQQ, IWM, DIA)
  - [ ] Map all 51 stocks to sector ETFs
  - [ ] Define market_internals (UVXY, GLD, TLT)
  - [ ] Set peer_selection parameters
  - [ ] List prediction_targets
  - [ ] List context_only symbols

### Ticker Lists
- [x] Create `config/tickers_new_context.txt` (17 new symbols)
- [x] Create `config/tickers_full.txt` (all 69 symbols)

### Schema Update
- [ ] Update `src/utils/config_schema.py`
  - [ ] Add `PeerSelectionConfig` dataclass
  - [ ] Add `SymbolUniverseConfig` dataclass
  - [ ] Add `symbol_universe` field to `Config` dataclass

**Estimated Time:** 2-3 hours  
**Dependencies:** None

---

## Phase 2: Data Ingestion (17 New Symbols)

### Ingest New Symbols
- [ ] Run ingestion for new context symbols
  ```bash
  python pipelines/ingest_data.py --mode ingest \
      --tickers config/tickers_new_context.txt \
      --config config/default_config.yaml
  ```
  - [ ] QQQ (Nasdaq 100)
  - [ ] IWM (Russell 2000)
  - [ ] DIA (Dow Jones)
  - [ ] XLK (Technology)
  - [ ] SOXX (Semiconductors)
  - [ ] XLF (Financials)
  - [ ] XLV (Healthcare)
  - [ ] XLE (Energy)
  - [ ] XLY (Consumer Discretionary)
  - [ ] XLP (Consumer Staples)
  - [ ] XLI (Industrials)
  - [ ] XLC (Communication)
  - [ ] XLB (Materials)
  - [ ] XLU (Utilities)
  - [ ] UVXY (Volatility)
  - [ ] GLD (Gold)
  - [ ] TLT (Treasuries)

### Clean New Symbols
- [ ] Run cleaning for new symbols
  ```bash
  python pipelines/ingest_data.py --mode clean \
      --tickers config/tickers_new_context.txt
  ```
- [ ] Verify cleaned data quality
  - [ ] Check for excessive gaps
  - [ ] Verify timestamp alignment
  - [ ] Check OHLCV columns present

### Re-Align Full Universe
- [ ] Run alignment for all 69 symbols
  ```bash
  python pipelines/ingest_data.py --mode align \
      --tickers config/tickers_full.txt
  ```
- [ ] Verify alignment quality
  - [ ] Check common timestamp count
  - [ ] Verify all 69 symbols present
  - [ ] Check for missing data patterns

**Estimated Time:** 4-6 hours (mostly data download)  
**Dependencies:** Phase 1 complete

---

## Phase 3: Core Implementation

### Universe Builder Module
- [ ] Create `src/features/universe_builder.py`
  - [ ] Implement `SymbolUniverseBuilder` class
    - [ ] `__init__` method
    - [ ] `from_config` class method
    - [ ] `get_universe` method
    - [ ] `_select_peers` method (correlation-based)
    - [ ] `get_fitted_peers` method
    - [ ] `save_peers` method
    - [ ] `load_peers` method
  - [ ] Implement `align_symbol_universe` function
  - [ ] Add comprehensive docstrings
  - [ ] Add type hints

**Estimated Time:** 4-6 hours  
**Dependencies:** Phase 1 complete

---

## Phase 4: Pipeline Integration

### Update FeaturePipeline
- [ ] Modify `src/features/pipeline.py`
  - [ ] Update `__init__` to accept `universe_builder`
  - [ ] Update `fit_transform` to use universe builder
  - [ ] Update `transform` to use stored universe
  - [ ] Add `_universe_tickers` storage
  - [ ] Update logging to show universe info

### Update Build Features Script
- [ ] Modify `pipelines/run_build_features.py`
  - [ ] Import `SymbolUniverseBuilder`
  - [ ] Load universe builder from config
  - [ ] Pass builder to `process_ticker`
  - [ ] Update `process_ticker` to use builder
  - [ ] Load prediction targets from config
  - [ ] Only process prediction targets (not all tickers)
  - [ ] Save fitted peers to JSON

### Update Data Ingestion Script
- [ ] Modify `pipelines/ingest_data.py`
  - [ ] Update default tickers to `tickers_full.txt`

**Estimated Time:** 3-4 hours  
**Dependencies:** Phase 3 complete

---

## Phase 5: Testing

### Unit Tests
- [ ] Create `tests/test_universe_builder.py`
  - [ ] Test builder initialization from config
  - [ ] Test universe construction for AAPL (tech)
  - [ ] Test universe construction for JPM (financials)
  - [ ] Test peer selection on training data
  - [ ] Test peer reuse on validation data (no re-selection)
  - [ ] Test alignment function (inner join)
  - [ ] Test save/load peers (reproducibility)
  - [ ] Test missing sector ETF fallback
  - [ ] Test minimum correlation threshold
  - [ ] Test max_peers limit

### Integration Tests
- [ ] Create `tests/test_pipeline_with_universe.py`
  - [ ] Test full pipeline with universe builder
  - [ ] Test fit_transform on training data
  - [ ] Test transform on validation data
  - [ ] Test feature matrix shapes
  - [ ] Test peer consistency across splits
  - [ ] Test feature names consistency

### Manual Validation
- [ ] Run feature building on test subset (3-5 stocks)
  - [ ] AAPL (technology)
  - [ ] NVDA (semiconductors)
  - [ ] JPM (financials)
  - [ ] XOM (energy)
  - [ ] JNJ (healthcare)
- [ ] Inspect `fitted_peers.json`
  - [ ] Verify peers are sector-appropriate
  - [ ] Check correlation values make sense
  - [ ] Ensure max 3 peers per target
- [ ] Check feature counts
  - [ ] Compare old vs. new feature counts
  - [ ] Verify cross-ticker features present
  - [ ] Check for feature name collisions

**Estimated Time:** 6-8 hours  
**Dependencies:** Phase 4 complete

---

## Phase 6: Validation & Debugging

### Data Quality Checks
- [ ] Verify ETF data completeness
  - [ ] Check date ranges match stocks
  - [ ] Verify no excessive gaps
  - [ ] Check volume is non-zero
- [ ] Verify alignment quality
  - [ ] Check timestamp overlap
  - [ ] Verify inner join correctness
  - [ ] Check for unexpected data loss

### Peer Selection Validation
- [ ] Review fitted peers for all targets
  - [ ] Technology stocks get tech peers
  - [ ] Financial stocks get financial peers
  - [ ] Energy stocks get energy peers
  - [ ] No cross-sector contamination
- [ ] Check correlation values
  - [ ] All above min_correlation threshold
  - [ ] Reasonable correlation magnitudes (0.3-0.8)
  - [ ] No perfect correlations (1.0)

### Feature Engineering Validation
- [ ] Check feature matrix shapes
  - [ ] Consistent across train/val/test
  - [ ] Reasonable feature count (80-120 raw, 40-50 selected)
  - [ ] No NaN/inf values
- [ ] Verify cross-ticker features
  - [ ] beta_spy present
  - [ ] peer_corr_* features present
  - [ ] sector ETF features present
  - [ ] market internals features present

### Memory Profiling
- [ ] Profile memory usage
  - [ ] Single target feature building
  - [ ] Multiple target feature building
  - [ ] Peak memory consumption
  - [ ] Memory leaks check

**Estimated Time:** 4-6 hours  
**Dependencies:** Phase 5 complete

---

## Phase 7: Documentation

### Code Documentation
- [ ] Add docstrings to all new functions
- [ ] Add type hints to all new code
- [ ] Add inline comments for complex logic
- [ ] Update existing docstrings if needed

### User Documentation
- [ ] Create `docs/guides/SYMBOL_UNIVERSE_USAGE.md`
  - [ ] Quick start guide
  - [ ] Adding new stocks
  - [ ] Adding new sectors
  - [ ] Troubleshooting section
- [ ] Create `docs/guides/SYMBOL_UNIVERSE_MIGRATION.md`
  - [ ] Migration steps for existing projects
  - [ ] Breaking changes
  - [ ] Backward compatibility notes
- [ ] Update main `README.md`
  - [ ] Add symbol universe section
  - [ ] Explain 4-tier architecture
  - [ ] Add configuration examples

### Example Notebooks
- [ ] Create example notebook: Basic usage
- [ ] Create example notebook: Adding custom sectors
- [ ] Create example notebook: Analyzing peer selections

**Estimated Time:** 4-6 hours  
**Dependencies:** Phase 6 complete

---

## Phase 8: Full Rollout

### Feature Building for All Targets
- [ ] Run feature building for all prediction targets
  ```bash
  python pipelines/run_build_features.py \
      --config config/default_config.yaml \
      --tickers config/tickers_full.txt
  ```
- [ ] Monitor progress and memory usage
- [ ] Verify all targets complete successfully
- [ ] Check fitted_peers.json for all targets

### Validation
- [ ] Compare old vs. new feature matrices
  - [ ] Feature count increase reasonable
  - [ ] No unexpected feature drops
  - [ ] Cross-ticker features present
- [ ] Spot-check feature values
  - [ ] Reasonable ranges
  - [ ] No constant features
  - [ ] No duplicate features

### Backup & Version Control
- [ ] Backup old feature matrices
- [ ] Commit all new code
- [ ] Tag release (e.g., v2.0-multivariate)
- [ ] Update changelog

**Estimated Time:** 6-8 hours (mostly compute time)  
**Dependencies:** Phase 7 complete

---

## Phase 9: Model Training & Evaluation

### Baseline Comparison
- [ ] Train LSTM on old features (1-2 stocks)
- [ ] Train LSTM on new features (same stocks)
- [ ] Compare metrics:
  - [ ] RMSE (should decrease)
  - [ ] Sharpe ratio (should increase)
  - [ ] Max drawdown (should decrease)
  - [ ] Feature importance (check cross-ticker features)

### Full Training
- [ ] Train models on new features for all targets
- [ ] Monitor training stability
- [ ] Check for overfitting
- [ ] Validate on test set

**Estimated Time:** Variable (depends on compute resources)  
**Dependencies:** Phase 8 complete

---

## Success Criteria

### Correctness ✓
- [ ] Each target gets sector-appropriate peers
  - AAPL → tech peers (MSFT, GOOGL, META)
  - JPM → financial peers (BAC, GS, MS)
  - XOM → energy peers (CVX, COP, SLB)
- [ ] No cross-sector contamination
- [ ] Peers consistent across train/val/test splits

### Performance ✓
- [ ] Feature count increase reasonable (50-100% increase)
- [ ] Memory usage acceptable (<16 GB for 51 targets)
- [ ] Feature building time reasonable (<2 hours for 51 targets)

### Model Improvement ✓
- [ ] RMSE reduction (5-15% improvement expected)
- [ ] Sharpe ratio increase (10-20% improvement expected)
- [ ] Drawdown reduction (5-10% improvement expected)

### Code Quality ✓
- [ ] All tests pass
- [ ] No linter errors
- [ ] Documentation complete
- [ ] Type hints present

---

## Risk Mitigation

### Data Availability Issues
- **Risk:** Some ETFs may not have full history
- **Mitigation:** 
  - [ ] Check ETF inception dates
  - [ ] Use fallback to SPY if sector ETF missing
  - [ ] Log warnings for missing symbols

### Peer Selection Instability
- **Risk:** Top-3 peers may vary between runs
- **Mitigation:**
  - [ ] Set random seed for reproducibility
  - [ ] Save fitted peers to JSON
  - [ ] Use correlation threshold to filter noise

### Feature Explosion
- **Risk:** Too many features → overfitting
- **Mitigation:**
  - [ ] Limit peers to 3 max
  - [ ] Use feature selector (already implemented)
  - [ ] Monitor feature count in metadata

### Memory Issues
- **Risk:** 69 symbols × 51 targets = large memory footprint
- **Mitigation:**
  - [ ] Process targets sequentially (not parallel)
  - [ ] Use float32 instead of float64
  - [ ] Free memory after each target
  - [ ] Monitor with memory profiler

---

## Rollback Plan

If issues arise, rollback steps:

1. **Revert to old feature matrices**
   ```bash
   cp -r data/features.backup data/features
   ```

2. **Revert code changes**
   ```bash
   git checkout main
   ```

3. **Use old ticker list**
   ```bash
   python pipelines/run_build_features.py \
       --tickers config/mass_tickers.txt
   ```

---

## Timeline Summary

| Phase | Description | Time | Dependencies |
|-------|-------------|------|--------------|
| 0 | Pre-implementation | Done | - |
| 1 | Configuration | 2-3h | - |
| 2 | Data ingestion | 4-6h | Phase 1 |
| 3 | Core implementation | 4-6h | Phase 1 |
| 4 | Pipeline integration | 3-4h | Phase 3 |
| 5 | Testing | 6-8h | Phase 4 |
| 6 | Validation | 4-6h | Phase 5 |
| 7 | Documentation | 4-6h | Phase 6 |
| 8 | Full rollout | 6-8h | Phase 7 |
| 9 | Model training | Variable | Phase 8 |

**Total Estimated Time:** 33-47 hours (excluding model training)

---

## Notes & Observations

### 2026-04-08
- Created comprehensive implementation plan
- Identified 51 existing stocks in mass_tickers.txt
- Determined only 17 new symbols needed (ETFs + indices)
- Created all configuration files
- Plan ready for implementation

### [Add your notes here as you progress]
