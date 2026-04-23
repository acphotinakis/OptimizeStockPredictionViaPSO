# Peer Selection and Universe Configuration Audit

This document identifies bugs and architectural issues related to peer selection and symbol universe management in the `ClaudePaper` project, specifically within `src/features/**` and `pipelines/run_build_features.py`.

## 1. Identified Issues & Bugs

### 1.1 Hardcoded Peers in `SymbolUniverseBuilder`
- **Location:** `src/features/universe_builder.py`, `get_universe()` method (Lines 76-131).
- **Bug:** The method contains explicit `if target_ticker == "AAPL": ...` blocks that manually assign peers.
- **Impact:** The dynamic selection logic (`_select_peers`) is commented out, making it impossible to add new tickers or change peer selection without modifying source code.

### 1.2 Hardcoded Universe Constants in `cross_ticker_strict.py`
- **Location:** `src/features/cross_ticker_strict.py` (Lines 22-38).
- **Bug:** `MARKET_CONTEXT_TICKERS`, `MARKET_INTERNAL_TICKERS`, and `SECTOR_MAP` are hardcoded as module constants.
- **Impact:** These values duplicate information in `config/symbol_universe.yaml`. Any change to the YAML configuration is ignored by the feature generation logic, leading to inconsistent features and potential runtime errors (e.g., trying to access a ticker that wasn't ingested).

### 1.3 Ignored Peer Selection Configuration
- **Location:** `src/features/universe_builder.py`, `_select_peers()` method.
- **Bug:** The implementation ignores `lookback_days` and `method` from the `peer_selection` block in `symbol_universe.yaml`.
- **Impact:** It uses the entire training set for correlation, even if the user specifies a shorter lookback window (e.g., 252 days).

### 1.4 Fixed Peer Limit in Feature Generation
- **Location:** `src/features/cross_ticker_strict.py`, `compute_cross_ticker_features_strict()` (Line 268).
- **Bug:** The code contains `peers = peers[:3]`, hard-limiting peers to 3 regardless of the `max_peers` setting in the configuration.
- **Impact:** Prevents the model from scaling to more peers even if the configuration allows it.

### 1.5 Brittle Data Pipeline (Context Loss)
- **Location:** `pipelines/run_build_features.py` and `src/features/feature_generators.py`.
- **Bug:** `generate_raw_features` only accepts a list of `peer_tickers`.
- **Impact:** Downstream functions like `compute_cross_ticker_features_strict` do not know which tickers are market benchmarks vs. sector ETFs vs. internals, forcing them to rely on the hardcoded constants identified in 1.2.

## 2. Proposed Fixes

### 2.1 Refactor `SymbolUniverseBuilder`
- Remove the hardcoded `if/else` blocks in `get_universe`.
- Uncomment and activate `self._select_peers(target_ticker, dfs)`.
- Update `_select_peers` to respect `lookback_days` by slicing the input DataFrames before computing correlation.
- Remove redundant/commented-out implementations of `_select_peers` at the bottom of the file.

### 2.2 Dynamic Configuration Injection
- Update `compute_cross_ticker_features_strict` to accept `market_context`, `sector_etf`, and `market_internals` as parameters instead of using module-level constants.
- Update `generate_raw_features` to accept these parameters and pass them down.
- Modify `pipelines/run_build_features.py` to extract this information from the `SymbolUniverseBuilder` instance and pass it to `generate_raw_features`.

### 2.3 Decouple Sector Mapping
- Remove `SECTOR_MAP` from `cross_ticker_strict.py`.
- Ensure the `SymbolUniverseBuilder` is the "Single Source of Truth" for all symbol relationships, derived directly from `config/symbol_universe.yaml`.

## 3. Implementation Plan (Summary)

1.  **Cleanup `universe_builder.py`**: Enable dynamic peer selection and fix configuration usage.
2.  **Refactor `cross_ticker_strict.py`**: Remove constants and update function signatures.
3.  **Update `feature_generators.py`**: Extend `generate_raw_features` signature.
4.  **Update `run_build_features.py`**: Connect the `UniverseBuilder` state to the feature generation calls.
