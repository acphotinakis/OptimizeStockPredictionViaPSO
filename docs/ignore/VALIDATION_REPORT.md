# PSO-LSTM Codebase Validation Report
**Generated:** 2026-04-01  
**Status:** Pre-Training Audit Complete

---

## Executive Summary

This report documents the systematic validation of the PSO-LSTM Stock Tuner codebase against the comprehensive audit checklist. The codebase demonstrates strong architectural foundations but requires several critical fixes before full-scale training.

**Overall Status:** ⚠️ **REQUIRES FIXES** (14 Critical, 8 High Priority, 12 Medium Priority)

---

## A. General Checks

| Check                           | Status | Notes                                                                 |
| ------------------------------- | ------ | --------------------------------------------------------------------- |
| **Environment reproducibility** | ✅ PASS | `.env` keys loaded via `os.environ` in `data_ingestion.py:33-34`     |
| **Random seed consistency**     | ✅ PASS | All seeds set in `utils/seed.py:19-25` (torch, numpy, random, cuda)  |
| **Hydra config overrides**      | ⚠️ PARTIAL | Config loading works, CLI overrides untested                       |
| **Logging**                     | ✅ PASS | Dual handlers (file+console) with timestamps in `utils/logger.py`    |
| **Git & .gitignore**            | ✅ PASS | All sensitive paths ignored (`.env`, `logs/`, `data/`, `models/`)    |

**Issues Found:**
- ⚠️ **MEDIUM**: Hydra CLI overrides not tested (need integration tests)

---

## B. Module-Level Checks

### 1. Ingestion ✅ MOSTLY PASS

**Strengths:**
- ✅ Alpaca API credentials loaded from environment variables
- ✅ Data validation checks (DatetimeIndex, required columns, monotonic timestamps)
- ✅ Proper error handling and logging
- ✅ Parquet persistence with pyarrow engine

**Issues Found:**
- ❌ **CRITICAL**: `main.py:8` imports `from src.data.ingest import AlpacaIngestor` but file is named `data_ingestion.py`
- ❌ **CRITICAL**: `data_ingestion.py:72` references `self.cfg.timeframe` but should be `self.cfg.data.timeframe`
- ⚠️ **HIGH**: Missing unit test for single ticker fetch
- ⚠️ **MEDIUM**: No retry logic for API failures (relies on alpaca-py internal retries)

### 2. Validation + Cleaning ✅ MOSTLY PASS

**Strengths:**
- ✅ Comprehensive validation suite (missing timestamps, duplicates, nulls, outliers)
- ✅ Two-tiered missing data strategy (forward-fill short gaps, drop long gaps)
- ✅ Winsorization for outlier handling (preserves sample size)
- ✅ Duplicate removal with logging

**Issues Found:**
- ❌ **CRITICAL**: `main.py:9` imports `from src.data.validation import run_validation_suite` but file is named `data_validation.py`
- ❌ **CRITICAL**: `main.py:10` imports `from src.data.cleaning import run_cleaning_pipeline` but file is named `data_cleaning.py`
- ⚠️ **HIGH**: `data_validation.py:108` uses `cfg.data.validation.max_missing_pct` but needs validation this exists
- ⚠️ **MEDIUM**: Missing unit test for synthetic data with known gaps/outliers
- ⚠️ **MEDIUM**: No check for negative volumes in integrity validation

### 3. Feature Engineering ⚠️ NEEDS WORK

**Strengths:**
- ✅ All required indicators implemented (RSI, MACD, EMA, SMA, Bollinger Bands)
- ✅ Vectorized calculations for performance
- ✅ NaN handling after rolling windows
- ✅ Wavelet denoising with proper error handling

**Issues Found:**
- ❌ **CRITICAL**: `main.py:11` imports `from src.features.build_features import calculate_indicators` - correct
- ❌ **CRITICAL**: `main.py:12` imports `from src.features.denoising import apply_denoising_pipeline` but file is named `wavelet_denoising.py`
- ❌ **CRITICAL**: `main.py:13` imports `from src.features.selection import run_selection_pipeline` - correct
- ⚠️ **HIGH**: Feature engineering happens BEFORE denoising in `main.py:68-69`, should denoise first
- ⚠️ **HIGH**: Missing unit test comparing output against known input dataset
- ⚠️ **MEDIUM**: No validation that config paths exist (e.g., `cfg.features.indicators.rsi.window`)

### 4. Scaling ❌ CRITICAL ISSUES

**Strengths:**
- ✅ MinMaxScaler with correct range [-1, 1] for LSTM activations
- ✅ Separate fit and transform functions
- ✅ Scaler persistence with joblib
- ✅ DataFrame structure preservation (index, columns)

**Issues Found:**
- ❌ **CRITICAL**: No inverse transform function implemented
- ❌ **CRITICAL**: No validation that scaler is fit only on training data
- ❌ **CRITICAL**: Missing unit test for fit-transform-inverse check
- ⚠️ **HIGH**: No tolerance check for inverse transform accuracy
- ⚠️ **MEDIUM**: Scaler not integrated into main pipeline (missing from `main.py`)

### 5. Dataset / Data Loader ⚠️ NEEDS VALIDATION

**Strengths:**
- ✅ Sliding window implementation correct
- ✅ Proper sequence length calculation (`len(X) - lookback`)
- ✅ Correct indexing for target (next time step)
- ✅ Tensor conversion with float32 dtype

**Issues Found:**
- ⚠️ **HIGH**: No validation of sequence creation correctness
- ⚠️ **HIGH**: Missing unit test for batch shape and first/last sequence values
- ⚠️ **MEDIUM**: No device allocation testing
- ⚠️ **MEDIUM**: Dataset not integrated into main pipeline

### 6. Model (LSTM) ✅ PASS

**Strengths:**
- ✅ Architecture matches config (hidden size, layers, bidirectional, dropout)
- ✅ Correct handling of bidirectional output dimensions
- ✅ Proper hidden state initialization
- ✅ Many-to-one architecture (last time step prediction)

**Issues Found:**
- ⚠️ **MEDIUM**: Missing unit test for forward pass with dummy input
- ⚠️ **MEDIUM**: No gradient flow validation
- ⚠️ **LOW**: No check for NaN in forward pass output

### 7. Training ⚠️ NEEDS WORK

**Strengths:**
- ✅ Early stopping implemented correctly
- ✅ MSE loss function appropriate for regression
- ✅ Model checkpointing on validation improvement
- ✅ Best weights loaded after training

**Issues Found:**
- ❌ **CRITICAL**: No gradient NaN checks during training
- ❌ **CRITICAL**: No gradient clipping for exploding gradients
- ⚠️ **HIGH**: Missing unit test for 1-2 mini-batch training
- ⚠️ **HIGH**: No validation that loss decreases over mini-batches
- ⚠️ **MEDIUM**: `train.py:132` imports Path but not at top of file
- ⚠️ **MEDIUM**: No learning rate scheduler implemented

### 8. Optimization (PSO) ⚠️ NEEDS VALIDATION

**Strengths:**
- ✅ Swarm initialization matches config
- ✅ Non-linear inertia decay implemented
- ✅ Mutation for diversity maintenance
- ✅ Proper personal best and global best tracking

**Issues Found:**
- ❌ **CRITICAL**: Fitness evaluation returns `-val_loss` instead of Sharpe ratio (line 125)
- ❌ **CRITICAL**: Categorical parameter velocity updates not handled correctly (line 157)
- ⚠️ **HIGH**: No unit test for small swarm on dummy function
- ⚠️ **HIGH**: No boundary enforcement for categorical types
- ⚠️ **MEDIUM**: Search space config uses DictConfig, needs proper access pattern
- ⚠️ **MEDIUM**: No convergence criteria (runs all iterations regardless)

### 9. Evaluation ✅ MOSTLY PASS

**Strengths:**
- ✅ All required metrics implemented (RMSE, MAE, Sharpe, Max Drawdown, Directional Accuracy)
- ✅ Proper Sharpe ratio calculation with frequency adjustment
- ✅ Cumulative returns tracking

**Issues Found:**
- ⚠️ **HIGH**: Missing unit test for synthetic predictions
- ⚠️ **MEDIUM**: No visualization outputs implemented
- ⚠️ **MEDIUM**: Results not saved to CSV automatically
- ⚠️ **LOW**: Risk-free rate hardcoded to 0.0

### 10. Utils ✅ PASS

**Strengths:**
- ✅ Logger setup with dual handlers (file + console)
- ✅ Timestamp formatting correct
- ✅ Seed setting comprehensive (random, numpy, torch, cuda)
- ✅ Path handling with proper directory creation

**Issues Found:**
- ⚠️ **LOW**: No unit test for logger output validation

---

## C. Integration & End-to-End Checks

| Check                                | Status      | Notes                                                |
| ------------------------------------ | ----------- | ---------------------------------------------------- |
| **Full pipeline execution**          | ❌ BLOCKED  | Import errors prevent execution                      |
| **Hydra CLI overrides**              | ⚠️ UNTESTED | Need to test with actual overrides                   |
| **Mini-run with subset**             | ⚠️ UNTESTED | Requires fixing import errors first                  |
| **Output artifacts in correct dirs** | ⚠️ UNTESTED | Directory structure exists but pipeline not runnable |

**Critical Blockers:**
1. Import name mismatches in `main.py` (lines 8, 9, 10, 12)
2. Missing scaling integration in main pipeline
3. Missing data loader creation in optimize command
4. Feature engineering order incorrect (should denoise before calculating indicators)

---

## D. Static Analysis & CI Checks

| Tool              | Status      | Notes                                     |
| ----------------- | ----------- | ----------------------------------------- |
| **Ruff / Flake8** | ⚠️ NOT RUN  | No `.ruff.toml` or configuration found    |
| **Black**         | ⚠️ NOT RUN  | No `pyproject.toml` configuration         |
| **Mypy**          | ⚠️ NOT RUN  | No type checking configuration            |
| **pytest**        | ❌ BLOCKED  | No test files exist in `tests/` directory |
| **Coverage.py**   | ❌ BLOCKED  | No tests to measure coverage              |

**Issues Found:**
- ❌ **CRITICAL**: `pyproject.toml` is empty (no dependencies defined)
- ❌ **CRITICAL**: No test files exist in `tests/` directory
- ❌ **CRITICAL**: No CI/CD configuration (GitHub Actions, etc.)
- ⚠️ **HIGH**: No linting configuration files
- ⚠️ **HIGH**: No pre-commit hooks configured

---

## E. Performance & Profiling

| Check                           | Status      | Notes                                  |
| ------------------------------- | ----------- | -------------------------------------- |
| **Feature engineering timing**  | ⚠️ UNTESTED | No profiling implemented               |
| **PSO loop profiling**          | ⚠️ UNTESTED | No performance metrics collected       |
| **GPU/CPU utilization**         | ⚠️ UNTESTED | No monitoring implemented              |

**Recommendations:**
- Add `@profile` decorators or `cProfile` for bottleneck identification
- Implement tqdm progress bars for long-running operations
- Add GPU memory monitoring for large batch sizes

---

## F. Notebooks & Reproducibility

**Status:** ⚠️ UNTESTED (6 notebooks exist but not validated)

**Notebooks Found:**
1. `01_data_overview.py`
2. `02_data_validation.py`
3. `03_feature_exploration.py`
4. `04_feature_engineering_validation.py`
5. `05_model_results_analysis.py`
6. `06_pso_optimization_analysis.py`

**Issues:**
- ⚠️ **HIGH**: Notebooks not tested for execution
- ⚠️ **MEDIUM**: Unknown if notebooks read Hydra configs correctly
- ⚠️ **MEDIUM**: No validation of plot outputs

---

## G. Security & Secrets

| Check                        | Status  | Notes                                              |
| ---------------------------- | ------- | -------------------------------------------------- |
| **`.env` not committed**     | ✅ PASS | `.env` in `.gitignore` (line 45)                   |
| **No API keys in code**      | ✅ PASS | Keys loaded via `os.getenv()` in `data_ingestion` |
| **File permissions**         | ⚠️ N/A  | Not applicable for development environment         |

**Security Note:**
- ⚠️ **MEDIUM**: `.env` file currently contains real API keys (should use `.env.example` template)

---

## Critical Issues Summary (Must Fix Before Training)

### 🔴 Blocking Issues (14)

1. **Import Errors in main.py:**
   - Line 8: `ingest` → should be `data_ingestion`
   - Line 9: `validation` → should be `data_validation`
   - Line 10: `cleaning` → should be `data_cleaning`
   - Line 12: `denoising` → should be `wavelet_denoising`

2. **Missing pyproject.toml:** Empty file, no dependencies defined

3. **No Test Suite:** `tests/` directory is empty

4. **Data Leakage Risks:**
   - Feature selection using XGBoost may leak validation data
   - No explicit validation that scaler is fit only on training data

5. **Missing Pipeline Components:**
   - Scaling not integrated in `main.py` process command
   - Data loaders not created in optimize command (lines 97-98 are placeholders)

6. **PSO Fitness Function:** Returns `-val_loss` instead of Sharpe ratio (line 125)

7. **No Inverse Transform:** Cannot recover original price scale for evaluation

8. **No Gradient Validation:** Missing NaN checks and gradient clipping

9. **Feature Engineering Order:** Should denoise before calculating indicators

10. **Configuration Access:** PSO search_space uses DictConfig, needs proper attribute access

### 🟡 High Priority Issues (8)

1. Missing unit tests for all modules
2. No configuration validation (missing key checks)
3. No integration tests for full pipeline
4. Missing visualization outputs in evaluation
5. No convergence criteria in PSO
6. Missing static analysis tool configurations
7. Notebooks not validated for execution
8. No CI/CD pipeline

### 🟢 Medium Priority Issues (12)

1. No retry logic for API failures
2. Missing negative volume checks
3. No learning rate scheduler
4. No performance profiling
5. No GPU memory monitoring
6. Missing progress bars for long operations
7. Risk-free rate hardcoded
8. No walk-forward validation implementation
9. Missing pre-commit hooks
10. No type hints validation (mypy)
11. Missing documentation for complex functions
12. No integration with MLflow or experiment tracking

---

## Recommended Fix Priority

### Phase 1: Critical Fixes (Required for Basic Execution)
1. Fix import errors in `main.py`
2. Add dependencies to `pyproject.toml`
3. Fix PSO fitness function to use proper metric
4. Integrate scaling into main pipeline
5. Add inverse transform function
6. Fix feature engineering order (denoise first)
7. Create data loaders in optimize command

### Phase 2: Data Integrity (Required for Valid Results)
1. Add data leakage prevention checks
2. Add gradient NaN validation
3. Add gradient clipping
4. Fix configuration access patterns
5. Add negative volume checks

### Phase 3: Testing & Validation (Required for Confidence)
1. Create unit test suite for all modules
2. Add integration tests
3. Validate notebooks execution
4. Add configuration validation

### Phase 4: Quality & Maintainability (Recommended)
1. Setup static analysis tools (ruff, black, mypy)
2. Add pre-commit hooks
3. Configure CI/CD pipeline
4. Add performance profiling
5. Implement visualization outputs

---

## Module-Specific Validation Details

### Data Ingestion (`src/data/data_ingestion.py`)

**Validation Checks:**
- ✅ Raw data downloads with correct tickers
- ✅ Data shape validation (required OHLCV columns)
- ✅ DatetimeIndex validation
- ✅ Monotonic timestamp enforcement
- ⚠️ Missing: Duplicate row handling test
- ⚠️ Missing: Unit test for single ticker

**Code Quality:**
- Lines of Code: 174
- Logging: Comprehensive
- Error Handling: Good
- Type Hints: Missing

### Data Validation (`src/data/data_validation.py`)

**Validation Checks:**
- ✅ Missing timestamp detection with frequency awareness
- ✅ Duplicate detection
- ✅ Null value counting
- ✅ Outlier detection (IQR and Z-score methods)
- ⚠️ Missing: Negative volume checks
- ⚠️ Missing: Unit test with synthetic data

**Code Quality:**
- Lines of Code: 144
- Logging: Good
- Error Handling: Good
- Type Hints: Partial

### Data Cleaning (`src/data/data_cleaning.py`)

**Validation Checks:**
- ✅ Duplicate removal
- ✅ Two-tiered missing data strategy
- ✅ Winsorization for outliers
- ✅ Proper logging of actions
- ⚠️ Missing: Unit test for cleaning pipeline

**Code Quality:**
- Lines of Code: 131
- Logging: Good
- Error Handling: Good
- Type Hints: Partial

### Feature Engineering (`src/features/build_features.py`)

**Validation Checks:**
- ✅ RSI calculation (vectorized)
- ✅ MACD calculation (fast, slow, signal, histogram)
- ✅ Bollinger Bands (mid, high, low)
- ✅ EMA and SMA for multiple windows
- ✅ Rolling statistics (mean, std)
- ⚠️ Missing: Unit test against known values
- ⚠️ Missing: Validation of indicator correctness

**Code Quality:**
- Lines of Code: 93
- Logging: Good
- Error Handling: Minimal
- Type Hints: Good

### Wavelet Denoising (`src/features/wavelet_denoising.py`)

**Validation Checks:**
- ✅ DWT decomposition with PyWavelets
- ✅ MAD-based threshold calculation
- ✅ Soft thresholding on detail coefficients
- ✅ Signal reconstruction with length matching
- ✅ Fallback handling on error
- ⚠️ Missing: Unit test for denoising effectiveness

**Code Quality:**
- Lines of Code: 83
- Logging: Good
- Error Handling: Excellent
- Type Hints: Good

### Feature Selection (`src/features/selection.py`)

**Validation Checks:**
- ✅ Pearson correlation filtering
- ✅ XGBoost importance ranking
- ✅ Top-N feature selection
- ❌ **DATA LEAKAGE**: XGBoost fit on full dataset, not just training
- ⚠️ Missing: Unit test for selection pipeline

**Code Quality:**
- Lines of Code: 97
- Logging: Good
- Error Handling: Minimal
- Type Hints: Good

### Scaling (`src/features/scaling.py`)

**Validation Checks:**
- ✅ Scaler fitting on training data only (by design)
- ✅ Transform preserves DataFrame structure
- ✅ Scaler persistence
- ❌ **MISSING**: Inverse transform function
- ❌ **MISSING**: Unit test for inverse transform

**Code Quality:**
- Lines of Code: 104
- Logging: Good
- Error Handling: Good
- Type Hints: Good

### Dataset (`src/data/dataset.py`)

**Validation Checks:**
- ✅ Sliding window implementation
- ✅ Correct sequence length calculation
- ✅ Proper target alignment
- ⚠️ Missing: Shape validation test
- ⚠️ Missing: Boundary condition tests

**Code Quality:**
- Lines of Code: 62
- Logging: None
- Error Handling: Minimal
- Type Hints: Minimal

### Data Split (`src/data/split.py`)

**Validation Checks:**
- ✅ Chronological ordering enforced
- ✅ Year-based splitting
- ✅ No overlap between splits
- ✅ Empty dataset validation
- ✅ Comprehensive logging of date ranges
- ⚠️ Missing: Unit test for split correctness

**Code Quality:**
- Lines of Code: 84
- Logging: Excellent
- Error Handling: Good
- Type Hints: Good

### LSTM Model (`src/models/lstm.py`)

**Validation Checks:**
- ✅ Correct architecture initialization
- ✅ Bidirectional support
- ✅ Dropout between layers
- ✅ Proper hidden state initialization
- ✅ Device-aware tensor creation
- ⚠️ Missing: Forward pass unit test
- ⚠️ Missing: Output shape validation

**Code Quality:**
- Lines of Code: 87
- Logging: Minimal
- Error Handling: None
- Type Hints: Minimal

### Training (`src/training/train.py`)

**Validation Checks:**
- ✅ Early stopping logic correct
- ✅ Checkpoint saving on improvement
- ✅ Train/eval mode switching
- ✅ Gradient zeroing
- ❌ **MISSING**: Gradient clipping
- ❌ **MISSING**: NaN detection
- ⚠️ Missing: Loss decrease validation test

**Code Quality:**
- Lines of Code: 164
- Logging: Good
- Error Handling: Minimal
- Type Hints: Minimal

### PSO Optimization (`src/optimization/pso.py`)

**Validation Checks:**
- ✅ Particle initialization
- ✅ Velocity update formula
- ✅ Boundary enforcement for continuous parameters
- ✅ Non-linear inertia decay
- ✅ Mutation implementation
- ❌ **WRONG METRIC**: Uses validation loss instead of Sharpe ratio
- ⚠️ Missing: Categorical parameter handling
- ⚠️ Missing: Unit test on dummy function

**Code Quality:**
- Lines of Code: 197
- Logging: Good
- Error Handling: Minimal
- Type Hints: Minimal

### Evaluation Metrics (`src/evaluation/metrics.py`)

**Validation Checks:**
- ✅ RMSE calculation correct
- ✅ MAE calculation correct
- ✅ Sharpe ratio with frequency adjustment
- ✅ Max drawdown calculation
- ✅ Directional accuracy
- ⚠️ Missing: Unit test with known values

**Code Quality:**
- Lines of Code: 106
- Logging: Good
- Error Handling: Minimal
- Type Hints: Minimal

---

## Configuration Validation

### Main Config (`configs/config.yaml`)
- ✅ Proper defaults structure
- ✅ Hydra interpolation for paths
- ✅ Timestamp in log directory
- ⚠️ Missing: Validation that all referenced configs exist

### Data Config (`configs/data/data.yaml`)
- ✅ Environment variable interpolation (`${oc.env:ALPACA_API_KEY}`)
- ✅ Proper ticker list
- ✅ Validation and cleaning thresholds defined
- ⚠️ Issue: `data_ingestion.py` accesses `cfg.timeframe` instead of `cfg.data.timeframe`

### Features Config (`configs/features/features.yaml`)
- ✅ All indicator parameters defined
- ✅ Denoising configuration
- ✅ Selection thresholds
- ✅ Well-structured and complete

### Model Config (`configs/model/model.yaml`)
- ✅ Architecture parameters
- ✅ Training hyperparameters
- ✅ Early stopping patience
- ⚠️ Issue: `input_size: null` needs runtime determination

### PSO Config (`configs/optimization/pso.yaml`)
- ✅ Swarm parameters defined
- ✅ Search space with proper types
- ✅ Fitness strategy defined
- ⚠️ Issue: Search space access pattern needs validation

---

## Test Coverage Analysis

**Current Coverage:** 0% (no tests exist)

**Required Test Files (Minimum):**
1. `tests/test_data_ingestion.py` - API fetch, validation, persistence
2. `tests/test_data_validation.py` - All validation checks
3. `tests/test_data_cleaning.py` - Missing data, duplicates, outliers
4. `tests/test_feature_engineering.py` - Indicator calculations
5. `tests/test_wavelet_denoising.py` - Signal reconstruction
6. `tests/test_feature_selection.py` - Correlation and importance filtering
7. `tests/test_scaling.py` - Fit, transform, inverse transform
8. `tests/test_dataset.py` - Sequence creation, shapes
9. `tests/test_model.py` - Forward pass, output shapes
10. `tests/test_training.py` - Mini-batch training, early stopping
11. `tests/test_pso.py` - Swarm initialization, particle updates
12. `tests/test_evaluation.py` - Metric calculations
13. `tests/test_utils.py` - Logger, seed, paths
14. `tests/test_integration.py` - End-to-end pipeline

**Target Coverage:** >80% for production readiness

---

## Action Items (Prioritized)

### 🔴 IMMEDIATE (Blocking Execution)

1. **Fix import errors in main.py** (4 errors)
2. **Create pyproject.toml with dependencies**
3. **Fix config access patterns** (cfg.timeframe → cfg.data.timeframe)
4. **Add inverse_transform function to scaling.py**
5. **Integrate scaling into main.py process command**
6. **Create data loaders in optimize command**
7. **Fix feature engineering order** (denoise before indicators)

### 🟡 HIGH PRIORITY (Required for Valid Training)

8. **Fix PSO fitness function** (use Sharpe ratio, not -val_loss)
9. **Add gradient clipping in training loop**
10. **Add NaN detection in training**
11. **Fix data leakage in feature selection** (split before XGBoost)
12. **Add configuration validation checks**
13. **Create basic unit test suite** (at least 5 critical tests)

### 🟢 MEDIUM PRIORITY (Quality & Reliability)

14. **Add negative volume validation**
15. **Setup static analysis tools** (ruff, black, mypy configs)
16. **Add performance profiling**
17. **Validate notebook execution**
18. **Add visualization outputs**
19. **Create integration tests**

### 🔵 LOW PRIORITY (Nice to Have)

20. **Add pre-commit hooks**
21. **Setup CI/CD pipeline**
22. **Add experiment tracking (MLflow)**
23. **Implement learning rate scheduler**
24. **Add convergence criteria to PSO**

---

## Conclusion

The codebase demonstrates solid architectural design with proper separation of concerns, comprehensive logging, and good reproducibility foundations. However, **14 critical issues must be resolved before training can begin**, primarily:

1. Import name mismatches preventing execution
2. Missing dependency management
3. Missing test suite
4. Incomplete pipeline integration
5. Data leakage risks

**Estimated Effort:**
- Critical fixes: 2-3 hours
- High priority fixes: 4-6 hours
- Test suite creation: 6-8 hours
- Total to production-ready: 12-17 hours

**Recommendation:** Address all 🔴 IMMEDIATE and 🟡 HIGH PRIORITY items before attempting full-scale training on 51 tickers.
