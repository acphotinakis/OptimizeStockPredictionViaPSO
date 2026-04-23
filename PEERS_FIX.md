# Peer Selection System - Issues and Fix Plan

**Date:** 2026-04-22  
**Severity:** HIGH  
**Status:** DESIGN COMPLETE - READY FOR IMPLEMENTATION

---

## Executive Summary

The peer selection system has a **critical hardcoded bug** in `src/features/universe_builder.py` where peers are manually specified for specific tickers (AAPL, MSFT, GOOGL, NVDA, AMD, META) instead of using the dynamic correlation-based selection logic that already exists in the codebase.

Additionally, there are **architectural issues** in how the peer selection configuration is structured and how the dynamic selection method (`_select_peers()`) is currently disabled.

---

## Issue #1: Hardcoded Peers (CRITICAL)

### Current State

In `src/features/universe_builder.py` lines 72-145, the `get_universe()` method contains hardcoded peer assignments:

```python
if fit:
    # self._fitted_peers[target_ticker] = self._select_peers(target_ticker, dfs)  # COMMENTED OUT!

    if target_ticker == "AAPL":
        self._fitted_peers[target_ticker] = ["MSFT", "GOOGL", "NVDA"]
    elif target_ticker == "MSFT":
        self._fitted_peers[target_ticker] = ["AAPL", "GOOGL", "META"]
    elif target_ticker == "GOOGL":
        self._fitted_peers[target_ticker] = ["AAPL", "MSFT", "META"]
    elif target_ticker == "NVDA":
        self._fitted_peers[target_ticker] = ["AAPL", "MSFT", "GOOGL"]
    elif target_ticker == "AMD":
        self._fitted_peers[target_ticker] = ["AAPL", "MSFT", "GOOGL"]
    elif target_ticker == "META":
        self._fitted_peers[target_ticker] = ["AAPL", "MSFT", "GOOGL"]
```

### Problems

1. **Non-scalable:** Only works for 6 specific tickers (AAPL, MSFT, GOOGL, NVDA, AMD, META)
2. **Non-data-driven:** Peers are manually chosen, not based on actual correlation
3. **Breaks for new tickers:** Any other ticker will have NO peers selected
4. **Commented-out logic:** The correct dynamic selection (`_select_peers()`) is commented out
5. **Inconsistent with TRD:** Violates data-driven principles
6. **Not reproducible:** Different training periods may warrant different peers

### Impact

- **High:** Cannot add new prediction targets without code changes
- **High:** Peer selection doesn't adapt to market regime changes
- **Medium:** Manual peers may not be optimal for the actual training data

---

## Issue #2: Incomplete Dynamic Selection Implementation

### Current State

The `_select_peers()` method (lines 182-257) exists and is **correctly implemented** but is **not being used**. It has:

✅ Sector-based filtering (peers must be in same sector)  
✅ Correlation computation on training data only  
✅ Top-N ranking by absolute correlation  
✅ Minimum correlation threshold  
✅ Proper logging

However, it's commented out in favor of hardcoded assignments.

### Problems

1. **Good code is unused:** The correct implementation exists but is disabled
2. **No fallback:** If a ticker doesn't have hardcoded peers, it fails
3. **Configuration mismatch:** `config/symbol_universe.yaml` has peer selection settings that are ignored

---

## Issue #3: Configuration Structure Issues

### Current Configuration (`config/symbol_universe.yaml`)

```yaml
peer_selection:
  max_peers: 3
  method: "pearson"
  min_correlation: 0.3
  lookback_days: 252
```

### Problems

1. **`method` is unused:** Always uses Pearson (no other methods implemented)
2. **`lookback_days` is unused:** Correlation uses all training data, not a sliding window
3. **Missing sector enforcement flag:** No way to disable sector restriction if needed
4. **Missing fallback strategy:** No configuration for what to do if no peers meet criteria

### Additional Issues

The configuration has **comments about missing data**:
```yaml
market_context:
  - SPY   # S&P 500 broad market (ALREADY HAVE)
  - QQQ   # Nasdaq 100 tech-heavy (NEED TO INGEST)
  - IWM   # Russell 2000 small caps (NEED TO INGEST)
  - DIA   # Dow Jones industrials (NEED TO INGEST)
```

This suggests incomplete data ingestion, which could affect peer selection quality.

---

## Issue #4: No Validation of Peer Quality

### Current State

Once peers are selected (whether hardcoded or dynamic), there's **no validation** that they:
- Actually exist in the data
- Have sufficient overlap with target ticker
- Meet minimum data quality standards

### Problems

1. **Silent failures:** Missing peer data could cause downstream errors
2. **No quality metrics:** Can't assess if selected peers are good
3. **No logging of correlation values:** Hard to debug why certain peers were chosen

---

## Issue #5: Pipeline Integration Issues

### Current State in `pipelines/run_build_features.py`

Lines 166-167:
```python
universe = universe_builder.get_universe(ticker, dfs_train, fit=True)
peers = universe_builder.get_fitted_peers(ticker)
```

### Problems

1. **No error handling:** If peer selection fails, pipeline crashes
2. **No validation:** Doesn't check if peers are empty
3. **No fallback:** If a ticker isn't in the hardcoded list, it silently gets no peers
4. **No logging of selection rationale:** Can't tell why certain peers were chosen

---

## Root Cause Analysis

### Why was dynamic selection disabled?

Looking at the code history (lines 71-73):
```python
if fit:
    # self._fitted_peers[target_ticker] = self._select_peers(target_ticker, dfs)
    
    if target_ticker == "AAPL":
```

The dynamic selection is commented out with no explanation. Likely reasons:
1. **Reproducibility concerns:** Dynamic selection might change with different data
2. **Testing/debugging:** Hardcoded peers for controlled experiments
3. **Performance issues:** Correlation computation might be slow
4. **Forgotten refactor:** Intended as temporary but never reverted

---

## Recommended Fix

### Strategy: Multi-Mode Selection with Configuration Control

Implement a **flexible peer selection system** that supports:
1. **Dynamic mode** (default): Use correlation-based selection
2. **Manual mode**: Use pre-specified peers from config
3. **Hybrid mode**: Dynamic with manual overrides for specific tickers

---

## Fix Plan

### Phase 1: Configuration Update

**File:** `config/symbol_universe.yaml`

Add a new `peer_selection_mode` configuration:

```yaml
# Peer Selection Configuration
peer_selection:
  # Selection mode: "dynamic", "manual", or "hybrid"
  mode: "dynamic"
  
  # Dynamic selection parameters
  max_peers: 3
  min_correlation: 0.3
  correlation_method: "pearson"  # pearson, spearman, kendall
  sector_constrained: true       # Restrict peers to same sector
  
  # Manual overrides (used in "manual" or "hybrid" mode)
  manual_peers:
    AAPL: ["MSFT", "GOOGL", "NVDA"]
    MSFT: ["AAPL", "GOOGL", "META"]
    GOOGL: ["AAPL", "MSFT", "META"]
    NVDA: ["AAPL", "MSFT", "GOOGL"]
    AMD: ["AAPL", "MSFT", "GOOGL"]
    META: ["AAPL", "MSFT", "GOOGL"]
  
  # Fallback strategy if no peers meet criteria
  fallback:
    allow_empty_peers: true      # Allow zero peers if none qualify
    min_peers_warning: 2         # Warn if fewer than this many peers
    cross_sector_fallback: false # Allow cross-sector if no in-sector peers
```

### Phase 2: Code Refactor

**File:** `src/features/universe_builder.py`

#### Change 1: Remove Hardcoded Logic

**Current (lines 71-145):**
```python
if fit:
    # self._fitted_peers[target_ticker] = self._select_peers(target_ticker, dfs)
    
    if target_ticker == "AAPL":
        self._fitted_peers[target_ticker] = ["MSFT", "GOOGL", "NVDA"]
    elif target_ticker == "MSFT":
        # ... more hardcoded assignments ...
```

**Fixed:**
```python
if fit:
    mode = self.peer_config.get("mode", "dynamic")
    
    if mode == "manual":
        self._fitted_peers[target_ticker] = self._get_manual_peers(target_ticker)
    elif mode == "hybrid":
        # Try manual first, fall back to dynamic
        manual = self._get_manual_peers(target_ticker)
        if manual:
            self._fitted_peers[target_ticker] = manual
        else:
            self._fitted_peers[target_ticker] = self._select_peers(target_ticker, dfs)
    else:  # dynamic
        self._fitted_peers[target_ticker] = self._select_peers(target_ticker, dfs)
    
    logger.info(
        f"[{target_ticker}] Peer selection mode='{mode}', "
        f"selected {len(self._fitted_peers[target_ticker])} peers: "
        f"{self._fitted_peers[target_ticker]}"
    )
```

#### Change 2: Add Manual Peer Retrieval Method

**Add new method:**
```python
def _get_manual_peers(self, target: str) -> List[str]:
    """Retrieve manually specified peers from config."""
    manual_peers = self.peer_config.get("manual_peers", {})
    
    if target in manual_peers:
        peers = manual_peers[target]
        logger.info(f"[{target}] Using manual peers: {peers}")
        return peers
    else:
        logger.warning(f"[{target}] No manual peers defined in config")
        return []
```

#### Change 3: Enhance `_select_peers()` with Validation

**Current:** Method exists but no validation

**Add at end of `_select_peers()`:**
```python
# Validation and fallback
if not peers:
    logger.warning(
        f"[{target}] No peers found meeting criteria "
        f"(min_corr={self.peer_config['min_correlation']})"
    )
    
    fallback_cfg = self.peer_config.get("fallback", {})
    
    if not fallback_cfg.get("allow_empty_peers", True):
        raise RuntimeError(
            f"[{target}] No peers found and allow_empty_peers=False"
        )
    
    if fallback_cfg.get("cross_sector_fallback", False):
        logger.info(f"[{target}] Attempting cross-sector peer selection...")
        # Fallback to original commented-out method (lines 259-292)
        peers = self._select_peers_cross_sector(target, dfs)

min_peers = self.peer_config.get("fallback", {}).get("min_peers_warning", 2)
if len(peers) < min_peers:
    logger.warning(
        f"[{target}] Only {len(peers)} peers found "
        f"(expected at least {min_peers})"
    )

return peers
```

#### Change 4: Add Cross-Sector Fallback Method

**Add new method (use commented-out logic from lines 259-292):**
```python
def _select_peers_cross_sector(
    self, target: str, dfs: Dict[str, pd.DataFrame]
) -> List[str]:
    """
    Select peers WITHOUT sector constraint (fallback).
    
    Use when sector-constrained selection finds no peers.
    """
    if target not in dfs:
        return []
    
    exclude = (
        set(self.market_context)
        | set(self.market_internals)
        | set(self.sector_map.values())
        | {target}
    )
    
    r_target = dfs[target]["log_return"]
    candidates = [
        t for t in dfs if t not in exclude and "log_return" in dfs[t].columns
    ]
    
    correlations: Dict[str, float] = {}
    for t in candidates:
        aligned = pd.concat([r_target, dfs[t]["log_return"]], axis=1).dropna()
        if len(aligned) < 100:
            continue
        corr = abs(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        if pd.notna(corr) and corr >= self.peer_config["min_correlation"]:
            correlations[t] = corr
    
    peers = sorted(correlations, key=correlations.get, reverse=True)[
        : self.peer_config["max_peers"]
    ]
    
    logger.info(
        f"[{target}] Cross-sector peers: {[(p, f'{correlations[p]:.3f}') for p in peers]}"
    )
    
    return peers
```

### Phase 3: Pipeline Enhancement

**File:** `pipelines/run_build_features.py`

**Current (lines 166-167):**
```python
universe = universe_builder.get_universe(ticker, dfs_train, fit=True)
peers = universe_builder.get_fitted_peers(ticker)
```

**Enhanced:**
```python
# Select universe and peers (with error handling)
try:
    universe = universe_builder.get_universe(ticker, dfs_train, fit=True)
    peers = universe_builder.get_fitted_peers(ticker)
    
    logger.info(f"[{ticker}] Universe selected: {len(universe)} symbols")
    logger.info(f"[{ticker}] Peers selected: {peers}")
    
    # Validate peer data availability
    missing_peers = [p for p in peers if p not in dfs_train]
    if missing_peers:
        logger.warning(
            f"[{ticker}] Missing peer data: {missing_peers}. "
            "These will be excluded from cross-ticker features."
        )
        # Remove missing peers
        peers = [p for p in peers if p in dfs_train]
        
    if not peers:
        logger.warning(
            f"[{ticker}] No valid peers available. "
            "Cross-ticker features will be limited."
        )

except Exception as e:
    logger.error(f"[{ticker}] Peer selection failed: {e}")
    raise
```

---

## Additional Bugs Found

### Bug #1: Sector Map Incomplete

**Issue:** `config/symbol_universe.yaml` only defines 2 sectors (technology, semiconductors) but has 6 prediction targets.

**Fix:** Add complete sector mappings in config:
```yaml
sector_etfs:
  technology:
    etf: XLK
    stocks: [AAPL, MSFT, GOOGL, META, CRM, ADBE]
  
  semiconductors:
    etf: SOXX
    stocks: [NVDA, AMD, INTC]
  
  # Add more sectors as needed
```

### Bug #2: Missing ETF Data Warnings

**Issue:** Config has "NEED TO INGEST" comments for QQQ, IWM, DIA, XLK, SOXX, UVXY, GLD, TLT.

**Impact:** Universe building will fail if these are missing.

**Fix:** 
1. Add data validation in pipeline startup
2. Log clear warnings for missing universe components
3. Allow pipeline to proceed with available data

**Code addition to `pipelines/run_build_features.py` (after line 523):**
```python
# Validate universe data availability
missing_symbols = [s for s in all_symbols if s not in dfs]
if missing_symbols:
    logger.warning(
        f"Missing {len(missing_symbols)} universe symbols: {missing_symbols}. "
        "This may affect feature quality. Consider ingesting missing data."
    )
```

### Bug #3: No Persistence of Selection Rationale

**Issue:** Peer selection rationale (correlations, why peers were chosen) is not saved.

**Fix:** Enhance `save_peers()` to include metadata:

```python
def save_peers(self, path: str | Path) -> None:
    """Save fitted peers with selection metadata."""
    metadata = {
        "fitted_peers": self._fitted_peers,
        "selection_mode": self.peer_config.get("mode", "dynamic"),
        "config": self.peer_config,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved peer metadata to {path}")
```

---

## Testing Strategy

### Test Case 1: Dynamic Mode
```bash
# Set mode to "dynamic" in config
python pipelines/run_build_features.py --prediction-target AAPL
```

**Expected:** 
- Peers selected based on training data correlation
- Logged correlation values
- Peers may differ from hardcoded values

### Test Case 2: Manual Mode
```bash
# Set mode to "manual" in config
python pipelines/run_build_features.py --prediction-target AAPL
```

**Expected:**
- Peers match manual_peers config exactly
- Log shows "Using manual peers"

### Test Case 3: Hybrid Mode
```bash
# Set mode to "hybrid" in config
# Test with ticker not in manual_peers
python pipelines/run_build_features.py --prediction-target CRM
```

**Expected:**
- Falls back to dynamic selection for CRM
- Uses manual for AAPL

### Test Case 4: New Ticker
```bash
# Add INTC to prediction_targets
python pipelines/run_build_features.py --prediction-target INTC
```

**Expected:**
- Dynamic selection works
- Selects AMD, NVDA as semiconductor peers

### Test Case 5: No Valid Peers
```bash
# Test with ticker in isolated sector
python pipelines/run_build_features.py --prediction-target <ISOLATED_TICKER>
```

**Expected:**
- Logs warning about no peers found
- Falls back to cross-sector if configured
- Continues pipeline with empty peers

---

## Migration Plan

### Phase 1: Configuration (Low Risk)
1. Update `config/symbol_universe.yaml` with new peer_selection structure
2. Set `mode: "manual"` initially (maintains current behavior)
3. Validate configuration loads correctly

### Phase 2: Code Refactor (Medium Risk)
1. Implement new methods in `universe_builder.py`
2. Replace hardcoded if/elif chain with mode-based selection
3. Add validation and fallback logic
4. Maintain backward compatibility

### Phase 3: Pipeline Enhancement (Low Risk)
1. Add error handling in `run_build_features.py`
2. Add data availability validation
3. Enhance logging

### Phase 4: Testing (High Priority)
1. Test all modes with existing tickers
2. Test with new tickers
3. Test edge cases (no peers, missing data)
4. Validate outputs match expected

### Phase 5: Migration (Low Risk)
1. Switch `mode: "dynamic"` in production config
2. Monitor logs for warnings
3. Compare peer selections with manual
4. Adjust thresholds if needed

---

## Risk Assessment

### High Risk
- ❌ None (changes are backward compatible)

### Medium Risk
- ⚠️ Dynamic selection may choose different peers than hardcoded
  - **Mitigation:** Use "hybrid" mode during transition
  - **Mitigation:** Compare peer quality metrics

### Low Risk
- ⚠️ Configuration parsing errors
  - **Mitigation:** Add schema validation
  - **Mitigation:** Comprehensive testing

---

## Success Criteria

1. ✅ Can add new prediction targets without code changes
2. ✅ Peer selection is data-driven and reproducible
3. ✅ All three modes (dynamic, manual, hybrid) work correctly
4. ✅ Proper error handling and validation
5. ✅ Backward compatible with existing behavior
6. ✅ Comprehensive logging of selection rationale
7. ✅ Graceful handling of missing data

---

## Implementation Checklist

### Configuration
- [ ] Update `config/symbol_universe.yaml` with new structure
- [ ] Add `mode`, `manual_peers`, `fallback` sections
- [ ] Complete sector mappings
- [ ] Validate YAML syntax

### Code Changes
- [ ] Refactor `get_universe()` method
- [ ] Add `_get_manual_peers()` method
- [ ] Enhance `_select_peers()` with validation
- [ ] Add `_select_peers_cross_sector()` method
- [ ] Update `save_peers()` to include metadata
- [ ] Add error handling in pipeline

### Testing
- [ ] Test dynamic mode with all targets
- [ ] Test manual mode with all targets
- [ ] Test hybrid mode
- [ ] Test new ticker (not in config)
- [ ] Test missing data handling
- [ ] Test no valid peers case

### Documentation
- [ ] Update README with peer selection modes
- [ ] Document configuration options
- [ ] Add troubleshooting guide
- [ ] Update pipeline documentation

---

## Estimated Effort

- **Configuration Update:** 30 minutes
- **Code Refactor:** 2-3 hours
- **Testing:** 1-2 hours
- **Documentation:** 1 hour

**Total:** 4-6 hours

---

## Conclusion

The peer selection system has a **critical architectural flaw** where hardcoded peer assignments override a correctly implemented dynamic selection method. The fix is straightforward:

1. **Enable dynamic selection** by uncommenting the correct logic
2. **Add configuration control** to support multiple selection modes
3. **Enhance validation** to handle edge cases gracefully
4. **Improve logging** for better debugging and auditability

The recommended solution is **backward compatible** (can maintain current behavior via "manual" mode) while enabling **flexible, data-driven peer selection** for new tickers and evolving market conditions.

---

**Priority:** HIGH  
**Complexity:** MEDIUM  
**Impact:** HIGH  
**Backward Compatibility:** YES  
**Ready for Implementation:** YES
