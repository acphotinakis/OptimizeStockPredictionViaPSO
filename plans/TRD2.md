I'll now consolidate these two high-level design documents into a single, unified, production-grade Technical Requirements Document (TRD). Let me analyze the conflicts and resolve them systematically.
Now I'll produce the final consolidated TRD:

---

# MASTER FEATURE ENGINEERING PIPELINE
## Technical Requirements Document (TRD)
### Quantitative Finance System — LSTM & XGBoost Production Pipeline

**Classification:** Technical Requirements Document — Production Release  
**Version:** 1.0 — Consolidated Canonical Specification  
**Authority:** Synthesized from four peer-reviewed papers (2020–2025)  
**Date:** 2026-04-16

---

## 1. EXECUTIVE SYSTEM OVERVIEW

### 1.1 System Purpose

This document specifies a unified, production-grade feature engineering pipeline for financial time-series forecasting. The pipeline ingests OHLCV market data, applies wavelet-based denoising, generates technical indicators, selects features via correlation analysis, and produces model-ready inputs for both LSTM (sequence) and XGBoost (tabular) models.

### 1.2 Model Targets

| Model | Input Format | Output | Primary Use Case |
|-------|--------------|--------|----------------|
| **LSTM** | 3D tensor `(samples, 20, F)` | Next-period normalized closing price | Sequence learning, temporal dependency capture |
| **XGBoost** | 2D matrix `(samples, 20×F)` | Next-period normalized closing price | Tabular learning, feature importance analysis |

### 1.3 Data Flow

```
Raw OHLCV → Cleaning → Feature Generation → Wavelet Denoising → 
Correlation Selection → Normalization → Windowing → 
    ├─→ LSTM Tensor (3D)
    └─→ XGBoost Matrix (2D)
         ↓
    PSO Optimization (LSTM hyperparameters only)
         ↓
    Trained Model Artifact
```

---

## 2. CANONICAL PIPELINE ARCHITECTURE

### Stage 1: Data Ingestion

**Purpose:** Load and validate raw market data with strict temporal ordering.

**Inputs:**
- `open`, `high`, `low`, `close`, `volume` (OHLCV) — mandatory
- `turnover_rate`, `pe_ratio`, `pb_ratio` — optional (Deng & Peng 2025)
- `exchange_rate`, `interest_rate` — optional (Zeng et al. 2025)

**Transformations:**
- Parse timestamps; enforce ascending chronological order
- Assert `high ≥ low` for all rows; assert `open, high, low, close, volume > 0`
- Log data provenance: source, date range, row count, ticker symbol

**Outputs:** DataFrame with monotonic DatetimeIndex, float64 dtype, standardized column names

**Constraints:**
- Missing values preserved as NaN; no imputation at this stage
- Must support daily and intraday (minute) frequencies

---

### Stage 2: Data Cleaning

**Purpose:** Handle missing values and invalid observations without look-ahead bias.

**Transformations:**
1. **Gap Classification:**
   - Gaps of 1–5 consecutive observations: forward-fill imputation (use last valid prior value)
   - Gaps > 5 consecutive observations: row discard
2. **Range Validation:** Remove rows where `high < low` or any price ≤ 0

**Outputs:** Cleaned DataFrame with no NaN in core OHLCV columns

**Constraints:**
- Imputation uses only prior observations; interpolation using future data is PROHIBITED
- Discard threshold of 5 observations is fixed per Lanbouri & Achchab (2020)

---

### Stage 3: Feature Generation

**Purpose:** Compute all research-mandated technical indicators and derived quantities.

**Mandatory Features (All Papers):**

| Feature | Formula | Source |
|---------|---------|--------|
| `ema12` | 12-period EMA of close | Lanbouri & Achchab 2020 |
| `ema20` | 20-period EMA of close | Zeng et al. 2025 |
| `ema25` | 25-period EMA of close | Lanbouri & Achchab 2020 |
| `macd` | EMA12 − EMA26 | Lanbouri & Achchab 2020; Zeng et al. 2025 |
| `boll_upper` | 20-period SMA + 2σ | Lanbouri & Achchab 2020 |
| `boll_lower` | 20-period SMA − 2σ | Lanbouri & Achchab 2020 |
| `cci` | Commodity Channel Index | Zeng et al. 2025 |
| `atr` | Average True Range | Zeng et al. 2025 |
| `ma5` | 5-period SMA of close | Zeng et al. 2025 |
| `ma10` | 10-period SMA of close | Zeng et al. 2025 |
| `mtm6` | Close_t − Close_{t-6} | Zeng et al. 2025 |
| `mtm12` | Close_t − Close_{t-12} | Zeng et al. 2025 |
| `roc` | (Close_t − Close_{t-n}) / Close_{t-n} × 100 | Zeng et al. 2025 |
| `smi` | Stochastic Momentum Index | Zeng et al. 2025 |
| `wvad` | Williams Variable Accumulation/Distribution | Zeng et al. 2025 |
| `earn_rate` | (Close_t − Close_{t-1}) / Close_{t-1} | Deng & Peng 2025 |

**Indicator Parameters (Fixed):**
- EMA/SMA periods: 5, 10, 12, 20, 25 as specified above
- Bollinger Bands: 20-period SMA, 2σ bandwidth
- MACD: 12-period and 26-period EMAs (standard definition)
- ATR: 14-period (standard definition)
- CCI: 20-period (standard definition)
- SMI: Default parameters (K=13, D=25, smoothing=2)

**Outputs:** DataFrame with OHLCV + 16 computed features; NaN rows from indicator warm-up dropped

**Constraints:**
- All indicators computed causally using data ≤ time t
- First 20 rows dropped due to EMA20/Bollinger warm-up

---

### Stage 4: Wavelet Denoising

**Purpose:** Reduce noise in price series while preserving signal structure.

**Method:** Discrete Wavelet Transform (DWT)
- **Wavelet:** Haar function
- **Decomposition levels:** 3
- **Thresholding:** Soft thresholding on detail coefficients (levels 1–3)
- **Threshold estimation:** Median absolute deviation (MAD) rule on training data only

**Transformations:**
1. Apply 3-level DWT to `close` price series
2. Estimate threshold λ = σ × √(2 × log(N)) where σ = MAD / 0.6745
3. Apply soft thresholding: d_thresholded = sign(d) × max(0, |d| − λ)
4. Reconstruct denoised series via inverse DWT

**Outputs:**
- `close_denoised`: DWT-filtered closing price
- `dwt_approx`: Approximation coefficients (level 3)
- `dwt_detail_l1`, `dwt_detail_l2`, `dwt_detail_l3`: Detail coefficients (optional features)

**Constraints:**
- Threshold λ estimated on training split ONLY; same λ applied to validation and test
- DWT parameters (wavelet, levels) fixed; no adaptive selection per-split

---

### Stage 5: Correlation-Based Feature Selection

**Purpose:** Remove multicollinear features to reduce redundancy.

**Method:** Pearson correlation analysis

**Procedure:**
1. Compute Pearson correlation matrix on **training data only**
2. For each feature, compute |r| with target (`close_{t+1}`)
3. Drop features where |r| ≥ 0.95 with target (p < 0.01, 2-tailed)
4. Among remaining features, drop one from each pair with |r| ≥ 0.95 (higher variance retained)

**Outputs:**
- Reduced feature set (typically 12–17 features from original 20+)
- Feature retention mask (binary vector, versioned)

**Constraints:**
- Correlation computed on training split ONLY
- Same mask applied to all splits; no recomputation for validation/test

---

### Stage 6: Normalization

**Purpose:** Scale features to bounded range for neural network convergence.

**Method:** MinMaxScaler

**Canonical Range:** [−1, 1]

**Formula:**  
`x_norm = 2 × (x − x_min) / (x_max − x_min) − 1`

**Alternative Range:** [0, 1] — permitted ONLY for pipelines excluding signed indicators (pure OHLCV)

**Formula:**  
`x_norm = (x − x_min) / (x_max − x_min)`

**Scaling Rules:**
- Fit scaler on **training split only**: compute `x_min`, `x_max` per feature
- Apply fitted scaler to validation and test splits
- Store scaler parameters (JSON serialization) for inference-time denormalization

**Constraints:**
- StandardScaler (z-score) is **PROHIBITED**
- Scaler refit on full dataset is **PROHIBITED**
- Per-feature min/max stored with model artifact

---

### Stage 7: Temporal Windowing

**Purpose:** Construct sequences for LSTM and flattened arrays for XGBoost.

**Look-back Period:** L = 20 days (canonical), configurable to {7, 50}

**LSTM Windowing:**
```
For t from L to N-1:
    X_lstm[i] = features[t-L : t, :]    # shape: (20, F)
    y[i] = close_normalized[t+1]       # next-period target
```
Output shape: `(N-L, 20, F)`

**XGBoost Windowing:**
```
For t from L to N-1:
    X_xgb[i] = flatten(features[t-L : t, :])  # shape: (20 × F,)
    y[i] = close_normalized[t+1]
```
Output shape: `(N-L, 20×F)`

**Split Enforcement:**
- Temporal split BEFORE windowing: 80% training, 20% test
- Windows must not cross split boundary
- First valid window index = L (ensures full history)

---

### Stage 8: PSO Hyperparameter Optimization

**Purpose:** Automatically optimize LSTM architecture and training parameters.

**Search Space (6-dimensional):**

| Dimension | Parameter | Range | Type |
|-----------|-----------|-------|------|
| 1 | Training epochs | [50, 300] | Integer |
| 2 | Hidden layer 1 neurons | [50, 300] | Integer |
| 3 | Hidden layer 2 neurons | [20, 200] | Integer |
| 4 | Learning rate | [0.001, 0.01] | Float |
| 5 | Dropout rate | [0.0, 0.5] | Float |
| 6 | Batch size | {32, 64} | Categorical |

**Algorithm:** Improved PSO (IPSO) per Ji et al. 2021

**Parameters:**
- Swarm size: N = 20 particles
- Max iterations: 50
- Learning factors: c1 = c2 = 1.5
- Inertia weight: ω_t = 0.8 − 0.2 × tanh(4 × t / 50)
- Mutation factor: μ_f = 0.3 × (t / 50) + 0.7

**Objective Function (Single):**
```
fitness = 0.9 × MSE + 0.1 × MSW
where:
    MSE = mean((y_pred − y_val)²)
    MSW = mean(network_weights²)
```

**Validation Protocol:**
- Training set: 72% of total data (90% of 80% training split)
- PSO validation set: 8% of total data (10% of 80% training split)
- Test set: 20% of total data (held out, never seen by PSO)

**Outputs:**
- Optimal hyperparameter tuple: `{epochs*, node1*, node2*, lr*, dropout*, batch_size*}`
- PSO convergence log (fitness trajectory per iteration)

**Constraints:**
- PSO evaluates fitness on validation set ONLY
- Test set access during PSO is **PROHIBITED**
- Final model trained on combined train+validation (80%) with optimal parameters

---

## 3. FINAL FEATURE SCHEMA (CANONICAL)

### Core Input Features (Post-Selection)

| Index | Feature | Category | Type | Notes |
|-------|---------|----------|------|-------|
| 1 | `open` | Price | Input | Raw OHLCV |
| 2 | `high` | Price | Input | Raw OHLCV |
| 3 | `low` | Price | Input | Raw OHLCV |
| 4 | `close` | Price | Input | Raw OHLCV |
| 5 | `volume` | Volume | Input | Raw OHLCV |
| 6 | `close_denoised` | Price | Input | Post-DWT |
| 7 | `ema12` | Trend | Input | Derived |
| 8 | `ema20` | Trend | Input | Derived |
| 9 | `macd` | Momentum | Input | Derived |
| 10 | `boll_upper` | Volatility | Input | Derived |
| 11 | `boll_lower` | Volatility | Input | Derived |
| 12 | `cci` | Momentum | Input | Derived |
| 13 | `atr` | Volatility | Input | Derived |
| 14 | `ma5` | Trend | Input | Derived |
| 15 | `mtm6` | Momentum | Input | Derived |
| 16 | `earn_rate` | Return | Input | Derived, signed |
| 17 | `wvad` | Volume | Input | Derived |

**Target Variable:**
- `close_next`: Normalized closing price at t+1 (regression target, shape `(samples,)`)

**Optional Features (If Available):**
- `pe_ratio`, `pb_ratio`, `turnover_rate`, `exchange_rate`, `interest_rate`

**Post-Selection Feature Count:** F = 12–17 (depending on correlation screening results)

---

## 4. DATA PROCESSING RULES

### 4.1 Missing Value Handling

| Scenario | Action | Constraint |
|----------|--------|------------|
| OHLCV gap ≤ 5 periods | Forward-fill | Causal only |
| OHLCV gap > 5 periods | Row discard | No interpolation |
| Indicator warm-up (first 20 rows) | Row discard | No backfill |
| Optional feature missing | Feature exclusion | Per-column decision |

### 4.2 Scaling Rules

| Aspect | Specification |
|--------|-------------|
| Scaler type | MinMaxScaler ONLY |
| Range | [−1, 1] (canonical), [0, 1] (price-only alternative) |
| Fit scope | Training split exclusively |
| Transform scope | Training, validation, test splits |
| Storage | JSON serialization of `data_min`, `data_max`, `feature_names` |

### 4.3 Causality Constraints

- All rolling computations (EMA, SMA, Bollinger, ATR, etc.) use `min_periods` parameter equal to window size
- No computation uses data from timestamp > current timestamp
- Target variable (`close_{t+1}`) never appears in feature matrix for sample at time t

---

## 5. LSTM SPECIFICATION

### 5.1 Architecture Constraints

| Parameter | Specification | Source |
|-----------|---------------|--------|
| Input shape | `(batch, 20, F)` | Ji et al. 2021 |
| Hidden layers | 2 (default), 1 or 3 (PSO-searchable) | Zeng et al. 2025 |
| Layer 1 neurons | 50–300 (PSO-optimized) | All papers |
| Layer 2 neurons | 20–200 (PSO-optimized) | All papers |
| Output layer | 1 neuron (linear activation) | All papers |
| Activation | ReLU (hidden layers) | Zeng et al. 2025 |
| Regularization | Dropout (rate PSO-optimized) | Deng & Peng 2025 |
| Optimizer | Adam (learning rate PSO-optimized) | Zeng et al. 2025 |
| Loss function | MSE | All papers |

### 5.2 Training Protocol

| Aspect | Specification |
|--------|-------------|
| Epochs | PSO-optimized ∈ [50, 300] |
| Batch size | 32 or 64 (PSO-selected) |
| Early stopping | Patience = 10 epochs on validation MSE |
| Validation split | 10% of training data (temporal) |
| Shuffle | PROHIBITED (maintain temporal order) |

### 5.3 Tensor Structure

```
Shape: (samples, timesteps, features)
      = (N - 20, 20, F)
      
Where:
    N = total observations after cleaning
    20 = look-back periods (canonical)
    F = retained features after Stage 5 (12–17)
    
Dtype: float32
```

---

## 6. XGBOOST SPECIFICATION

### 6.1 Feature Representation

**Canonical Method:** Flattened sequence (exclusively)

```
X_xgb[i] = [f_1(t-20), f_2(t-20), ..., f_F(t-20),
            f_1(t-19), ..., f_F(t-19),
            ...,
            f_1(t-1), ..., f_F(t-1)]
            
Shape: (samples, 20 × F)
```

**Column Naming:** `{feature_name}_lag_{k}` where k ∈ {0, 1, ..., 19} (0 = most recent)

### 6.2 Input Schema

| Property | Specification |
|----------|---------------|
| Feature matrix | Dense float32 array or DataFrame |
| Target vector | float32 1D array (`close_next`) |
| Sample count | N − 20 (same as LSTM) |
| Feature count | 20 × F (240–340 for F=12–17) |

### 6.3 XGBoost Hyperparameters (External to PSO)

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| `max_depth` | 6–8 | Prevent overfitting on high-dimensional flattened features |
| `learning_rate` | 0.01–0.1 | Conservative update for time-series stability |
| `n_estimators` | 500–1000 | Sufficient for convergence |
| `subsample` | 0.8 | Row sampling for regularization |
| `colsample_bytree` | 0.8 | Column sampling for regularization |
| `objective` | `reg:squarederror` | Match LSTM MSE objective |

---

## 7. PSO SYSTEM DESIGN

### 7.1 Unified Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Algorithm | IPSO (Improved PSO) | Superior performance per Ji et al. 2021 |
| Swarm size | 20 particles | Balance exploration/exploitation |
| Max iterations | 50 | Convergence observed within 20–50 iterations |
| Dimensions | 6 | Epochs, nodes×2, lr, dropout, batch_size |
| Inertia | Nonlinear tanh | ω_t = 0.8 − 0.2×tanh(4t/50) |
| Mutation | Adaptive | μ_f = 0.3×(t/50) + 0.7 |
| c1, c2 | 1.5 | Standard cognitive/social balance |
| Bounds | See Section 7.2 | Enforced via position clipping |

### 7.2 Search Space Bounds

| Dimension | Min | Max | Type | Initialization |
|-----------|-----|-----|------|----------------|
| Epochs | 50 | 300 | Integer | Uniform |
| Hidden 1 | 50 | 300 | Integer | Uniform |
| Hidden 2 | 20 | 200 | Integer | Uniform |
| Learning rate | 0.001 | 0.01 | Float | Log-uniform |
| Dropout | 0.0 | 0.5 | Float | Uniform |
| Batch size | 32 | 64 | Categorical | Random choice |

### 7.3 Objective Function (Single Canonical)

```
fitness(particle) = 0.9 × MSE_validation + 0.1 × MSW_network

where:
    MSE_validation = (1/N_val) × Σ (ŷ_i − y_i)²
    MSW_network = mean(flatten(all_layer_weights)²)
```

### 7.4 Evaluation Protocol

1. Decode particle to hyperparameter tuple
2. Construct LSTM with specified architecture
3. Train on training split (72% of data)
4. Evaluate MSE on validation split (8% of data)
5. Compute MSW from network weights
6. Return composite fitness
7. Update pbest/gbest per standard PSO velocity equations
8. Apply adaptive mutation if random < μ_f

### 7.5 Constraints

- **Test set isolation:** PSO never accesses test data (20% holdout)
- **Determinism:** Random seed fixed for reproducible particle initialization
- **Resource limits:** Maximum 50 LSTM training runs per optimization

---

## 8. DATA LEAKAGE PREVENTION RULES

### 8.1 Absolute Prohibitions

| Rule ID | Rule | Violation Consequence |
|---------|------|----------------------|
| L-1 | **No temporal shuffling:** Dataset must remain chronologically ordered at all stages | Model unlearnable; spurious performance |
| L-2 | **No future data in features:** All indicators computed causally (t ≤ current) | Look-ahead bias; inflated backtest performance |
| L-3 | **No test data in scaler fit:** MinMax parameters from training only | Distribution leakage; overfitting |
| L-4 | **No test data in correlation:** Feature selection mask from training only | Target leakage; feature selection bias |
| L-5 | **No test data in wavelet threshold:** DWT threshold from training only | Noise structure leakage |
| L-6 | **No test data in PSO:** PSO validation split from training only; test set isolated | Hyperparameter overfitting |
| L-7 | **No window boundary crossing:** Look-back windows contained within single split | Temporal contamination |
| L-8 | **No target in features:** `close_{t+1}` never appears in X[t] | Perfect prediction artifact |
| L-9 | **No indicator backfill:** Warm-up NaN rows discarded, not imputed | Artificial history |

### 8.2 Enforcement Mechanisms

| Mechanism | Implementation |
|-----------|----------------|
| Timestamp audit | All DataFrames carry `timestamp` column; operations verified for `max(timestamp) ≤ current_boundary` |
| Split isolation | Train/validation/test stored as separate DataFrames; no concatenation until final model training |
| Scaler serialization | `data_min`, `data_max` stored as JSON; hash verification on load |
| Feature mask versioning | Retention mask versioned with model artifact; mismatch raises `ValueError` |
| Window boundary check | Assertion: `window_start ≥ split_start` for all windows in split |

---

## 9. PRODUCTION CONSTRAINTS

### 9.1 Reproducibility Requirements

| Aspect | Specification |
|--------|---------------|
| Random seeds | Fixed across: NumPy, Python random, TensorFlow, XGBoost |
| Software versions | Pinned: Python 3.9+, TensorFlow 2.12+, PyWavelets 1.4+, pandas 1.5+, XGBoost 1.7+ |
| Docker image | Hash-tagged container with locked dependency tree |
| Pipeline version | Semantic versioning (MAJOR.MINOR.PATCH) tied to feature schema |

### 9.2 Versioning Requirements

| Artifact | Versioning Method | Storage |
|----------|-------------------|---------|
| Feature schema | Semantic version (e.g., v1.3.0) | Git + MLflow |
| Scaler parameters | Hash of (data_min, data_max, feature_names) | S3/MinIO |
| Feature retention mask | Hash of binary mask vector | S3/MinIO |
| PSO configuration | JSON with all hyperparameters | Git |
| Trained model | (Schema version, PSO config hash, timestamp) | MLflow Model Registry |

### 9.3 Monitoring Requirements

| Metric | Frequency | Alert Threshold |
|--------|-----------|---------------|
| Feature distribution drift (PSI) | Daily | PSI > 0.2 |
| Prediction accuracy (MAPE) | Daily | MAPE > 5% (relative to baseline) |
| Inference latency | Per-request | p99 > 100ms |
| Data freshness | Per-bar | Delay > 5 minutes |
| Scaler parameter validation | Per-model-load | Hash mismatch |

### 9.4 Deployment Assumptions

| Assumption | Validation |
|------------|------------|
| Market data feed | Real-time OHLCV via WebSocket or REST API |
| Historical data | Minimum 252 trading days (1 year) for warm-up |
| Compute | GPU for LSTM training (PSO), CPU for inference |
| Storage | Time-series database (InfluxDB/TimescaleDB) for features |
| Feature store | Feast or custom registry for serving-time consistency |

---

## 10. CONFIGURATION PROFILES

### Profile: Production Daily (Canonical)

```yaml
pipeline:
  frequency: daily
  look_back: 20
  split_ratio: [0.72, 0.08, 0.20]  # train, pso_val, test
  
features:
  core: [open, high, low, close, volume]
  derived: [ema12, ema20, macd, boll_upper, boll_lower, cci, atr, ma5, mtm6, earn_rate, wvad]
  optional: [pe_ratio, pb_ratio, turnover_rate, exchange_rate, interest_rate]
  
preprocessing:
  wavelet: {enabled: true, wavelet: haar, levels: 3, threshold: mad}
  selection: {method: pearson, threshold: 0.95, p_value: 0.01}
  scaling: {method: minmax, range: [-1, 1]}
  
pso:
  algorithm: ipso
  particles: 20
  iterations: 50
  dimensions: [epochs, hidden_1, hidden_2, learning_rate, dropout, batch_size]
  objective: {mse_weight: 0.9, msw_weight: 0.1}
  
lstm:
  layers: 2
  activation: relu
  optimizer: adam
  
xgboost:
  representation: flattened
  max_depth: 6
  learning_rate: 0.05
  n_estimators: 1000
```

### Profile: High-Frequency (Intraday)

```yaml
pipeline:
  frequency: 1min
  look_back: 20  # 20 minutes
  split_ratio: [0.72, 0.08, 0.20]
  
features:
  core: [open, high, low, close, volume]
  derived: [ema12, ema25, macd, boll_upper, boll_lower]
  optional: []
  
preprocessing:
  wavelet: {enabled: true, wavelet: haar, levels: 3}
  selection: {method: pearson, threshold: 0.95}
  scaling: {method: minmax, range: [0, 1]}  # Price-only, no signed indicators
  
pso: {particles: 20, iterations: 30}  # Faster convergence for smaller data
```

---

**END OF TECHNICAL REQUIREMENTS DOCUMENT**

*This TRD is the single source of truth for production implementation. All specifications are canonical and non-optional unless explicitly marked as configurable.*