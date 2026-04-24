# Codebase Architecture Analysis
**Date:** 2026-04-22  
**Purpose:** Foundation for Gemini Skills Customization

---

## 1. Primary Domain

**Financial Machine Learning & Quantitative Trading Research System**

- **Focus:** Stock price prediction using LSTM neural networks and XGBoost
- **Approach:** PSO-optimized hyperparameter search for LSTM models
- **Data:** Daily OHLCV data with cross-ticker features and technical indicators
- **Evaluation:** Walk-forward validation, backtesting with transaction costs

---

## 2. Architecture Pattern

**Modular ML Pipeline with Strict Temporal Causality**

```
Data Ingestion → Feature Engineering → Temporal Split → 
Model Training (LSTM/PSO-LSTM/XGBoost) → Evaluation → Backtesting
```

**Key Characteristics:**
- **Leakage-Prevention Architecture:** All operations are strictly causal
- **Config-Driven:** YAML-based configuration system
- **Frozen Pipeline:** Fit-once-freeze-forever for scalers/selectors/wavelets
- **Multi-Model:** LSTM, PSO-LSTM, XGBoost with unified interfaces

---

## 3. Technology Stack

**Primary Language:** Python 3.10+

**Core Dependencies:**
- **Deep Learning:** PyTorch (LSTM implementation)
- **ML:** XGBoost, scikit-learn
- **Optimization:** Custom PSO/IPSO implementation
- **Data:** Pandas, NumPy
- **Signal Processing:** PyWavelets (wavelet denoising)
- **Visualization:** Matplotlib
- **Config:** PyYAML

---

## 4. Module Structure & Responsibilities

### `src/data/` - Data Pipeline
- **Purpose:** Ingestion, cleaning, alignment, splitting
- **Key Files:** `alpaca_ingestor.py`, `cleaner.py`, `aligner.py`, `splitter.py`, `windowing.py`
- **Responsibility:** SPY-aligned timestamps, gap handling, temporal windowing
- **Critical:** Must never leak future data

### `src/features/` - Feature Engineering
- **Purpose:** Technical indicators, cross-ticker features, wavelets, selection, scaling
- **Key Files:** `feature_generators.py`, `cross_ticker_strict.py`, `wavelet.py`, `selector.py`, `scaler.py`, `universe_builder.py`
- **Responsibility:** Generate 45+ features, fit-once-freeze-forever paradigm
- **Critical:** Backward-looking only, training-only fitting

### `src/models/` - Model Implementations
- **Purpose:** LSTM and XGBoost model architectures + training
- **Key Files:** `baseline_lstm_model.py`, `pso_lstm_model.py`, `xgboost_model.py`, trainers
- **Responsibility:** PyTorch LSTM, XGBoost wrapper, training loops
- **Critical:** Currently has ~70% duplication (being refactored)

### `src/optimizer/` - PSO/IPSO Optimization
- **Purpose:** Particle swarm optimization for LSTM hyperparameters
- **Key Files:** `pso_core.py`, `ipso.py`, `particle.py`, `fitness.py`
- **Responsibility:** 6D search space (units_1, units_2, dropout, lr, batch, epochs)
- **Critical:** Fitness = 0.9×MSE + 0.1×MSW

### `src/evaluation/` - Evaluation & Backtesting
- **Purpose:** Walk-forward validation, metrics, backtesting, plotting
- **Key Files:** `walk_forward_pso.py`, `backtest.py`, `metrics.py`, `model_loader.py`, `plotting.py`
- **Responsibility:** Expanding-window validation, transaction cost simulation, 15+ metrics
- **Critical:** Inverse transform before metrics, per-fold isolation

### `pipelines/` - CLI Entry Points
- **Purpose:** Orchestration scripts for end-to-end workflows
- **Key Files:** `run_build_features.py`, `train_*.py`, `run_backtest.py`, `walk_forward_evaluation.py`
- **Responsibility:** Execute complete workflows with logging and error handling
- **Critical:** Single entry points for reproducible execution

### `config/` - Configuration
- **Purpose:** YAML-based system configuration
- **Key Files:** `default_config.yaml`, `symbol_universe.yaml`
- **Responsibility:** Hyperparameters, paths, PSO settings, universe definition
- **Critical:** Single source of truth for all parameters

---

## 5. Data Flow

### Primary Workflow:
1. **Ingestion** (`pipelines/data_ingest_data.py`)
   - Raw OHLCV → Cleaned, SPY-aligned data
   
2. **Feature Engineering** (`pipelines/run_build_features.py`)
   - Cleaned data → Technical indicators + cross-ticker features
   - **Split:** 70/10/20 (train/val/test)
   - **Fit:** Scalers, selectors, wavelet thresholds (train only)
   - **Output:** `train_features.pkl`, `val_features.pkl`, `test_features.pkl`

3. **Training** (`pipelines/train_*.py`)
   - Features → Trained model + metadata
   - **LSTM:** PyTorch 2-layer LSTM (units_1, units_2, dropout)
   - **PSO-LSTM:** Two-phase (PSO search → final training)
   - **XGBoost:** Gradient boosting with early stopping

4. **Evaluation** (`pipelines/walk_forward_evaluation.py` or `run_backtest.py`)
   - Model + test features → Metrics + plots
   - **Metrics:** Statistical (RMSE, R², DA) + Trading (Sharpe, MDD, CAGR)
   - **Output:** JSON metrics, CSV results, PNG plots

---

## 6. Critical Constraints (TRD Requirements)

### Temporal Causality (MANDATORY)
- **Rule:** No future data access at any stage
- **Enforcement:** Split-first architecture, training-only fitting
- **Validation:** Test set never seen during training/feature-fitting

### Cross-Ticker Alignment (STRICT)
- **Rule:** All tickers must share common timestamp index (SPY-canonical)
- **Enforcement:** `TickerAligner` with strict reindex validation
- **Tolerance:** ≤0.1% isolated missing values, max 5-bar forward-fill

### Scaling Protocol (STRICT)
- **Rule:** Separate scalers for features and targets
- **Enforcement:** `FrozenMinMaxScaler` fit on train, frozen for val/test
- **Requirement:** Inverse transform before all evaluation metrics

### PSO Protocol (STRICT)
- **Rule:** Two-phase (Phase 1: PSO search, Phase 2: final training)
- **Enforcement:** 20 particles × 50 iterations, fitness = 0.9×MSE + 0.1×MSW
- **Validation:** Test set isolation, validation-only fitness

---

## 7. Current State Assessment

### Strengths
- Well-documented TRD specifications
- Comprehensive audit trail (5+ audit documents)
- Strict leakage prevention architecture
- Unified backtesting system (recently implemented)

### Active Work
- Model layer refactor in progress (reducing 40% duplication)
- Peer selection system recently refactored (dynamic selection)
- Backtesting system recently completed (unified entry point)

### Technical Debt
- LSTM layer has ~70% duplication (PSO vs Baseline)
- Some hardcoded constants being phased out
- Legacy implementations being deprecated

---

## 8. Skill Mapping to Repository

### `architect-reviewer` → **System Design & TRD Compliance**
- Review pipeline architecture for leakage risks
- Validate temporal causality
- Check split-first paradigm adherence
- Assess cross-ticker alignment correctness

### `code_reviewer` → **Python Code Quality & ML Correctness**
- Enforce TRD requirements in code
- Check for data leakage patterns
- Validate scaling protocols
- Review NumPy/Pandas/PyTorch usage

### `debugger` → **ML Pipeline Debugging & Root Cause Analysis**
- Diagnose alignment errors
- Debug PSO convergence issues
- Trace data leakage sources
- Fix scaling/inverse-transform bugs

### `performance-engineer` → **Model Training & Inference Optimization**
- Optimize LSTM training loops
- Profile PSO iteration performance
- Reduce feature generation bottlenecks
- GPU utilization analysis

### `python-pro` → **Python Implementation & ML Best Practices**
- Implement feature engineering logic
- Write PyTorch model architectures
- Implement PSO algorithms
- Refactor duplicate implementations

### `api-designer` → **Pipeline Interface & Config Design**
- Design pipeline CLI interfaces
- Structure YAML configuration schemas
- Define model save/load formats
- Specify evaluation result schemas

---

## 9. Key Terminology

- **TRD:** Technical Requirements Document (design specs)
- **PSO:** Particle Swarm Optimization
- **IPSO:** Improved PSO (adaptive inertia)
- **MSW:** Mean Squared Weights (model complexity penalty)
- **DA:** Directional Accuracy (sign correctness metric)
- **MDD:** Maximum Drawdown
- **SPY:** S&P 500 ETF (benchmark for alignment)
- **Lookback:** 20-day temporal window
- **Split-First:** Architecture pattern (split before fitting)

---

**Status:** Analysis complete. Ready for skill customization.
