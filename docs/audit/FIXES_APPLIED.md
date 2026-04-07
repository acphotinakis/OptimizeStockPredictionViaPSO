# Audit Fixes Applied

**Date:** April 7, 2026  
**Status:** In Progress  

## Summary

This document tracks the fixes applied to address the 87 issues identified in `CODEBASE_AUDIT.md`.

---

## ✅ CRITICAL Issues Fixed (7/7 Complete)

### 1.1 DataCleaner._init_stats() Call Fixed ✅
- **File:** `src/data/cleaner.py`
- **Fix:** Removed incorrect `df` argument from `self._init_stats(df)` call
- **Also fixed:** Added division-by-zero guards in `generate_cleaning_report()`
- **Also fixed:** Replaced deprecated `datetime.utcnow()` with `datetime.now(timezone.utc)`
- **Also fixed:** Added empty DataFrame handling in `clean()` method

### 1.2 API Keys Removed from Stdout ✅
- **File:** `src/data/alpaca_ingestor.py`
- **Fix:** Removed all `print()` statements exposing API keys
- **Fix:** Replaced with secure `logger.debug()` calls
- **Fix:** Unified environment variable names to `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`

### 1.3 Import Order Fixed ✅
- **File:** `pipelines/ingest_data.py`
- **Fix:** Moved `sys.path.insert()` before all project imports
- **Fix:** Updated module docstring from `scripts/01_...` to `pipelines/...`

### 1.4 Backtester Index Out of Bounds Fixed ✅
- **File:** `src/evaluation/backtester.py`
- **Fix:** Added bounds check `if t + 1 < N:` before writing `signals[t + 1]`

### 1.5 Undefined CLI Arguments Fixed ✅
- **File:** `scripts/evaluate.py`
- **Fix:** Added `--quantize` and `--profile-memory` arguments to argparse
- **Fix:** Moved `output_dir` creation before first use
- **Fix:** Added conditional import for `MemoryProfiler`
- **Fix:** Removed duplicate `output_dir` assignment

### 1.6 Look-Ahead Bias in Ichimoku Fixed ✅
- **File:** `src/features/technical.py`
- **Fix:** Removed `ichi_chikou = C.shift(-26)` which used future data
- **Fix:** Added explanatory comment about why it was removed

### 1.7 Peer Selection Look-Ahead Bias Fixed ✅
- **File:** `src/features/cross_ticker.py`
- **Fix:** Added warning when `peer_tickers=None` (computes on full series)
- **Fix:** Added documentation about need to pre-select peers on training data only
- **Fix:** Removed `bfill()` from SPY close prices (was using future data)

---

## ✅ HIGH Priority Issues Fixed (12/39 Complete)

### Data Pipeline (5/19)

#### 2.1 Session Filter Memory Optimization ✅
- **File:** `src/data/cleaner.py`
- **Fix:** Removed `df.copy()` in `_session_filter()`
- **Fix:** Build mask directly from `df.index.tz_convert()` without copying

#### 2.2 Outlier Clipping Vectorized ✅
- **File:** `src/data/cleaner.py`
- **Fix:** Replaced Python loop with vectorized operations
- **Fix:** Use `.loc[]` for batch updates instead of `.at[]` per row

#### 2.3 Empty DataFrame Handling ✅
- **File:** `src/data/cleaner.py`
- **Fix:** Added early return in `clean()` when `len(df) == 0`

#### 2.4 Division by Zero Guards ✅
- **File:** `src/data/cleaner.py`
- **Fix:** Added checks for `rows_initial == 0` and `expected_bars == 0`

#### 2.5 Deprecated datetime.utcnow() ✅
- **File:** `src/data/cleaner.py`
- **Fix:** Replaced with `datetime.now(timezone.utc)`

### Feature Engineering (4/9)

#### 3.1 Unconditional CUDA Device Fixed ✅
- **File:** `src/features/selector.py`
- **Fix:** Check `torch.cuda.is_available()` before setting device="cuda"
- **Fix:** Set `tree_method` based on device availability

#### 3.2 Hard-Coded Column Names Fixed ✅
- **File:** `src/features/cross_ticker.py`
- **Fix:** Use `f"beta_spy_{rolling_window}"` instead of hardcoded `"beta_spy_60"`
- **Fix:** Use `f"corr_spy_{rolling_window}"` instead of hardcoded `"corr_spy_60"`
- **Fix:** Updated `alpha_spy` calculation to use dynamic column name

#### 3.3 Naive Timezone Handling Fixed ✅
- **File:** `src/features/volume.py`
- **Fix:** Check `if df.index.tz is None` before `tz_convert()`
- **Fix:** Localize to UTC first if naive

#### 3.4 SPY bfill Removed ✅
- **File:** `src/features/cross_ticker.py`
- **Fix:** Removed `.bfill()` from SPY close reindexing
- **Fix:** Only use `.ffill()` to avoid using future data

### Model Implementation (0/7)

*Pending*

### Evaluation & Metrics (3/4)

#### 5.1 Backtester Index Bounds ✅
- **File:** `src/evaluation/backtester.py`
- **Fix:** Added `if t + 1 < N:` check before stop-loss signal assignment

#### 5.2 Undefined CLI Args in evaluate.py ✅
- **File:** `scripts/evaluate.py`
- **Fix:** Added `--quantize` and `--profile-memory` to argparse

#### 5.3 Output Directory Ordering ✅
- **File:** `scripts/evaluate.py`
- **Fix:** Created `output_dir` early before quantization block

---

## 🔄 In Progress

### HIGH Priority Model Implementation (0/7)
- [ ] Fix gradient accumulation tail in LSTM
- [ ] Fix hardcoded DataLoader seed
- [ ] Add NaN/Inf checks in training
- [ ] Fix JSON serialization in run_lstm_baseline.py
- [ ] Add map_location to torch.load calls
- [ ] Thread seed through all components
- [ ] Fix empty fold metrics handling

### HIGH Priority Evaluation (1/4)
- [ ] Parameterize annualization factors
- [ ] Fix bar-level vs trade-level metrics
- [ ] Fix fragile price alignment
- [ ] Fix in-sample threshold optimization

---

## 📋 Remaining Work

### MEDIUM Priority (35 issues)
- Memory bottlenecks in combined Parquet
- Column layout mismatch risk
- Missing error handling for Parquet I/O
- Timezone handling confusion
- Slow gap filling loop
- Import inside hot path
- Inefficient astype on mixed columns
- Fragile date splits
- Edge cases in build_windows
- Large aligned Parquet load
- Parallel workers share large dicts
- Unlimited ffill/bfill
- Redundant cleaner.reset_stats()
- Redundant _init_db() call
- VIX proxy scale misleading
- Autocorr NaN handling
- Slow CCI calculation
- Duplicate rolling calculations
- Slow session loop
- Redundant rolling passes
- Slow linear regression
- Dead LAG_SOURCES config
- Hardcoded DataLoader seed
- No NaN/Inf checks
- Device handling incomplete
- Misleading docstring in baselines.py
- QAT on LSTM not supported
- Quantized model on GPU
- Unsafe torch.load
- Broken --retrain-on-trainval flag
- GPU-only XGBoost default
- Fragile config access
- PSO DataLoader seed mismatch
- No GPU memory cleanup in PSO
- Inconsistent config access

### LOW Priority (22 issues)
- Commented-out code blocks (multiple files)
- Duplicate imports
- Missing type hints
- Non-English words in docstrings
- In-place mutation warnings
- Typos in log messages
- Noisy logging
- Layer count documentation
- Inertia at t=0
- Memory manager mutates model state
- Logger clears all root handlers
- Invalid log level silent
- Config loader docstring misleading
- Seed setting incomplete
- Missing psutil not logged
- GB vs GiB inconsistency
- Cursor after close
- Null JSON in get_best_run
- Unused config parameter
- Wrong path in docstrings

### Documentation (Entire section pending)
- Fix SETUP.md corruption
- Update all script paths in README
- Fix ticker count mismatch
- Update repo layout documentation
- Fix missing file references
- Update CLI command examples
- Fix artifact naming
- Update quantization docs
- Fix embedded YAML examples
- Clarify date range configuration
- Update module docstrings
- Add missing dependencies to requirements.txt

---

## Testing Recommendations

After all fixes are applied, the following tests should be run:

1. **Unit Tests**
   - `test_cleaner_empty_df()` - Verify empty DataFrame handling
   - `test_cleaner_zero_division()` - Verify division by zero guards
   - `test_no_look_ahead_bias()` - Verify no future data leakage
   - `test_feature_consistency()` - Verify reproducibility
   - `test_seed_reproducibility()` - Verify same seed → same results

2. **Integration Tests**
   - Run full pipeline on 2-3 tickers
   - Verify no crashes
   - Verify output artifacts are valid

3. **Smoke Tests**
   - `python pipelines/ingest_data.py --config config/default_config.yaml`
   - `python pipelines/build_features.py --ticker AAPL`
   - `python scripts/run_pso.py --ticker AAPL --episodes 10`
   - `python scripts/evaluate.py --ticker AAPL`

---

## Next Steps

1. **Complete HIGH priority fixes** (27 remaining)
   - Model implementation issues (7)
   - Evaluation issues (3)
   - Remaining data pipeline issues (14)
   - Remaining feature engineering issues (3)

2. **Address MEDIUM priority issues** (35)
   - Focus on performance bottlenecks first
   - Then code quality improvements

3. **Clean up LOW priority issues** (22)
   - Remove commented code
   - Fix docstrings
   - Improve logging

4. **Update all documentation** (1 major task)
   - Fix paths and commands
   - Update examples
   - Add missing sections

5. **Add comprehensive tests**
   - Unit tests for critical functions
   - Integration tests for pipelines
   - Regression tests for metrics

---

## Estimated Completion

- **HIGH priority:** 2-3 hours remaining
- **MEDIUM priority:** 3-4 hours
- **LOW priority:** 1-2 hours
- **Documentation:** 2-3 hours
- **Testing:** 2-3 hours

**Total:** ~10-15 hours of focused work

---

**Last Updated:** April 7, 2026 (In Progress)
