Now I have sufficient information to resolve the ambiguities in the technical indicator calculations. Let me consolidate all three documents into a single, unified Technical Requirements Document (TRD) for the feature engineering pipeline.

---

# MASTER FEATURE ENGINEERING PIPELINE
## Technical Requirements Document (TRD)
### Quantitative Finance System — LSTM & XGBoost
**Version:** 1.0 FINAL  
**Classification:** Production System Specification

---

## 1. EXECUTIVE SYSTEM OVERVIEW

### 1.1 System Purpose
This document specifies a unified, production-grade feature engineering pipeline for financial time-series forecasting. The pipeline produces model-ready inputs for dual-model architecture:

- **LSTM Model**: Sequence-based deep learning for temporal pattern recognition
- **XGBoost Model**: Tabular gradient boosting for feature interaction modeling

### 1.2 Data Flow Architecture
```
Raw OHLCV → Cleaning → Feature Generation → Wavelet Denoising → 
Correlation Selection → Normalization → Windowing → Model Formatting
                                    ↓
                              PSO Optimization (LSTM only)
```

### 1.3 Canonical Target Definition
**Target Variable IS DEFINED AS:** Next-period closing price return  
**Formula:** `r_t = (Close_t+1 - Close_t) / Close_t`  
**Rationale:** Return-based targets are stationary and scale-invariant. Price-level targets are PROHIBITED due to non-stationarity.

---

## 2. CANONICAL PIPELINE ARCHITECTURE

### Stage 1: Data Ingestion
**Purpose:** Acquire and validate raw market data with strict temporal ordering.

**Inputs:**
- Required: OHLCV columns (Open, High, Low, Close, Volume)
- Optional: Fundamental (PE Ratio, PB Ratio, Turnover Rate, Market Value)
- Optional: Macroeconomic (Exchange Rate, Interest Rate)

**Transformations:**
- Parse timestamps to UTC-aware DatetimeIndex
- Enforce strict chronological monotonicity (ascending)
- Assert temporal constraint: `max(timestamp) ≤ current_execution_time`
- Validate OHLCV consistency: `High ≥ Low`, `Open > 0`, `Close > 0`, `Volume ≥ 0`

**Outputs:**
- Pandas DataFrame with monotonic DatetimeIndex
- Data provenance log: source, date range, row count, ticker identifier

**Constraints:**
- Missing values SHALL be preserved as NaN for Stage 2 handling
- No imputation SHALL occur at this stage

---

### Stage 2: Data Cleaning
**Purpose:** Handle missing values and invalid observations without forward-looking bias.

**Transformations:**
1. **Gap Classification (Lanbouri & Achchab 2020):**
   - Gaps of 1–5 consecutive observations → Forward-fill imputation
   - Gaps > 5 consecutive observations → Row discard
2. **Range Validation:**
   - Remove rows where `High < Low` or `Close ∉ [Low, High]`
   - Remove rows where `Volume < 0`

**Outputs:**
- Cleaned DataFrame with no NaN values in core OHLCV columns

**Constraints:**
- Imputation SHALL use only prior observations (causal forward-fill)
- No interpolation across gaps exceeding 5 observations

---

### Stage 3: Feature Generation
**Purpose:** Compute complete feature set with deterministic, causal calculations.

#### 3.1 Price-Based Features (Category A)
| Feature | Formula | Source |
|---------|---------|--------|
| Open | Raw opening price | OHLCV |
| High | Raw highest price | OHLCV |
| Low | Raw lowest price | OHLCV |
| Close | Raw closing price | OHLCV |
| Volume | Raw trading volume | OHLCV |
| Daily_Return | `(Close_t - Close_{t-1}) / Close_{t-1}` | Deng & Peng 2025 |

#### 3.2 Trend-Following Indicators (Category B)
| Feature | Formula | Periods | Source |
|---------|---------|---------|--------|
| EMA12 | `EMA(Close, 12)` | 12 | Lanbouri & Achchab 2020 |
| EMA25 | `EMA(Close, 25)` | 25 | Lanbouri & Achchab 2020 |
| EMA20 | `EMA(Close, 20)` | 20 | Zeng et al. 2025 |
| MA5 | `SMA(Close, 5)` | 5 | Zeng et al. 2025 |
| MA10 | `SMA(Close, 10)` | 10 | Zeng et al. 2025 |
| MACD | `EMA12 - EMA26` | 12/26 | Lanbouri & Achchab 2020 |

*Note:* MACD signal line (9-period EMA of MACD) IS EXCLUDED per research paper ambiguity. Only MACD line IS INCLUDED.

#### 3.3 Volatility Indicators (Category C)
| Feature | Formula | Periods | Source |
|---------|---------|---------|--------|
| BOLL_upper | `SMA20 + 2 * σ(20)` | 20 | Lanbouri & Achchab 2020 |
| BOLL_lower | `SMA20 - 2 * σ(20)` | 20 | Lanbouri & Achchab 2020 |
| BOLL_mid | `SMA20` | 20 | Zeng et al. 2025 |
| ATR | `SMA(TR, 14)` | 14 | Zeng et al. 2025 |

**ATR Calculation (Standard 14-period Wilder):**
```
TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)
ATR_t = (13 * ATR_{t-1} + TR_t) / 14  [smoothed]
Initial ATR = mean(TR[1:14])
```

#### 3.4 Momentum/Oscillator Indicators (Category D)
| Feature | Formula | Periods | Source |
|---------|---------|---------|--------|
| CCI | `(TP - SMA20_TP) / (0.015 * MeanDev)` | 20 | Zeng et al. 2025 |
| MTM6 | `Close_t - Close_{t-6}` | 6 | Zeng et al. 2025 |
| MTM12 | `Close_t - Close_{t-12}` | 12 | Zeng et al. 2025 |
| ROC | `(Close_t - Close_{t-12}) / Close_{t-12} * 100` | 12 | Zeng et al. 2025 |
| SMI | `200 * (D_S / HLD_S)` | 10/3 | Zeng et al. 2025 |
| WVAD | `Σ[(Close-Open)/(High-Low) * Volume]` | Cumulative | Zeng et al. 2025 |

**CCI Calculation (Standard 20-period Lambert):**
```
TP = (High + Low + Close) / 3
SMA20_TP = SMA(TP, 20)
MeanDev = mean(|TP - SMA20_TP| over 20 periods)
CCI = (TP - SMA20_TP) / (0.015 * MeanDev)
```

**SMI Calculation (Standard 10/3-period):**
```
%K Length = 10, %D Length = 3
HH = highest(High, 10), LL = lowest(Low, 10)
Midpoint = (HH + LL) / 2
RelativeRange = Close - Midpoint
Range = HH - LL
D_S = EMA(EMA(RelativeRange, 3), 3)
HLD_S = EMA(EMA(Range, 3), 3)
SMI = 200 * (D_S / HLD_S)
```

**WVAD Calculation (Williams Variable AD):**
```
If Close > Open: WVAD += (Close - Open)/(High - Low) * Volume
If Close < Open: WVAD -= (Open - Close)/(High - Low) * Volume
Cumulative sum over time
```

#### 3.5 Optional Features (Category E & F)
| Feature | Type | Condition |
|---------|------|-----------|
| PE_Ratio | Fundamental | If available in source data |
| PB_Ratio | Fundamental | If available in source data |
| Turnover_Rate | Market | If available in source data |
| Exchange_Rate | Macro | If available in source data |
| Interest_Rate | Macro | If available in source data |

**Constraints:**
- All indicators SHALL use causal computation (data ≤ time t only)
- Warm-up rows (NaN due to indicator initialization) SHALL be dropped
- Longest warm-up period: 20 periods (Bollinger Bands, EMA20)

---

### Stage 4: Feature Transformation

#### 4.1 Wavelet Denoising (Zeng et al. 2025)
**Status:** ENABLED with strict constraints

**Method:** Discrete Wavelet Transform (DWT)
- Wavelet: Haar
- Decomposition levels: 3
- Thresholding: Soft threshold on detail coefficients (D1, D2, D3)

**Procedure:**
1. Apply DWT to Close price series: `coeffs = pywt.wavedec(Close, 'haar', level=3)`
2. Estimate threshold on training data only using universal threshold: `σ * sqrt(2 * log(N))`
3. Apply soft thresholding to detail coefficients
4. Reconstruct: `Close_denoised = pywt.waverec(coeffs, 'haar')`

**Output:** Denoised closing price series (replaces raw Close for downstream processing)

**Constraints:**
- Threshold estimation SHALL use training data only
- Same threshold SHALL apply to validation and test sets
- Boundary handling: symmetric padding to prevent edge effects

#### 4.2 Normalization
**Canonical Rule:** MinMaxScaler to [-1, 1] range

**Formula:** `x_norm = 2 * (x - x_min) / (x_max - x_min) - 1`

**Scaling Discipline:**
1. Fit scaler parameters (`x_min`, `x_max`) on training split ONLY
2. Apply fitted scaler to validation and test splits
3. Store scaler parameters for inference-time inverse transformation

**Rationale:** [-1, 1] range IS SELECTED because:
- Includes return-based features (can be negative)
- Compatible with tanh/sigmoid activations in LSTM
- Used by majority of research papers (Deng & Peng 2025, Zeng et al. 2025)

---

### Stage 5: Correlation-Based Feature Selection
**Purpose:** Remove multicollinear features to reduce redundancy.

**Method (Zeng et al. 2025):**
1. Compute Pearson correlation between each feature and target (Daily_Return) on training data
2. Compute pairwise Pearson correlation between features
3. Feature removal rules:
   - If |r(feature, target)| ≥ 0.95: REMOVE (collinear with target)
   - If pairwise |r(feature_i, feature_j)| ≥ 0.95: REMOVE feature with lower target correlation
4. Significance threshold: p < 0.01 (two-tailed)

**Output:**
- Reduced feature DataFrame
- Feature retention mask (binary vector)
- Correlation matrix (logged for audit)

**Constraints:**
- Correlation computation SHALL use training data only
- Retention mask SHALL be frozen and applied identically to validation/test sets

---

### Stage 6: Temporal Windowing

#### 6.1 LSTM Sequence Construction
**Look-back Period:** 20 days (canonical default per Ji et al. 2021)

**Windowing Logic:**
```
For each index t from (L) to (N-1):
    X[t] = feature_matrix[t - L : t, :]    # Shape: (L, F)
    y[t] = target[t]                        # Next-period return

Output shapes:
- X_train: (samples_train, 20, F)
- X_val: (samples_val, 20, F)  
- X_test: (samples_test, 20, F)
- y: (samples,) for each split
```

**Split Enforcement:**
- Windows SHALL NOT cross train/validation/test boundaries
- First valid window in each split starts at index L within that split

#### 6.2 XGBoost Feature Representation
**Strategy:** Lag-based feature engineering (NOT flattened sequences)

**Rationale:** XGBoost operates on tabular data. Flattening sequences creates high-dimensional sparse features that violate tree-based model assumptions.

**Feature Structure:**
```
For each sample at time t:
    Features = [
        # Current values (t-1)
        Open_{t-1}, High_{t-1}, Low_{t-1}, Close_{t-1}, Volume_{t-1},
        EMA12_{t-1}, EMA25_{t-1}, ...,  # All technicals at t-1
        
        # Lagged values (t-2, t-3, ..., t-L)
        Open_{t-2}, High_{t-2}, ..., 
        ...
        Open_{t-L}, High_{t-L}, ...,  # L = 20
        
        # Rolling statistics over window
        SMA20_Close_{t-1},  # Redundant with MA20, deduplicated in selection
        Volatility20_{t-1}
    ]
```

**Output Shape:** `(samples, F')` where F' depends on lag depth and feature count

**Target:** Same as LSTM: `r_t = (Close_t+1 - Close_t) / Close_t`

---

### Stage 7: PSO Optimization (LSTM Only)

#### 7.1 Configuration
**Algorithm:** Improved PSO (IPSO) per Ji et al. 2021

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Particle count | 20 |
| Max iterations | 50 |
| Search dimension | 4 |
| Learning factors | c1 = 1.5, c2 = 1.5 |
| Inertia weight | Nonlinear tanh schedule |

**Inertia Schedule:**
```
ω_t = ω_max - (ω_max - ω_min) * tanh(4 * t / max_iter)
where ω_max = 0.8, ω_min = 0.6
```

**Adaptive Mutation:**
```
μ_f = 0.3 * (t / max_iter) + 0.7
If random ξ ∈ [0,1] < μ_f: reinitialize particle position
```

#### 7.2 Search Space (4-Dimensional)
| Dimension | Parameter | Range | Type |
|-----------|-----------|-------|------|
| 1 | Training epochs | [50, 300] | Integer |
| 2 | Hidden layer 1 neurons | [50, 300] | Integer |
| 3 | Hidden layer 2 neurons | [50, 300] | Integer |
| 4 | Learning rate | [0.001, 0.01] | Float |

**Architecture:** Fixed 2-layer LSTM (depth not optimized per consolidated design)

#### 7.3 Objective Function
**Canonical Selection:** MSE-only (simpler, more stable than composite)

**Formula:** `fitness = MSE(y_pred, y_true) on validation set`

**Validation Protocol:**
- Training set: 70% of data (chronological)
- Validation set: 10% of data (immediately following training)
- Test set: 20% of data (final evaluation only, never used in PSO)

#### 7.4 PSO Execution Flow
1. Initialize swarm with random positions within bounds
2. For each particle: build LSTM, train on training set, evaluate on validation set
3. Update personal best (pbest) and global best (gbest)
4. Update velocities and positions using IPSO equations
5. Apply adaptive mutation
6. Iterate until max iterations or convergence (gbest unchanged for 10 iterations)
7. Return optimal hyperparameters: `{epochs*, node1*, node2*, lr*}`

---

## 3. FINAL FEATURE SCHEMA (CANONICAL)

### Core Feature Set (17 features minimum)
| ID | Feature | Category | Formula | Periods |
|----|---------|----------|---------|---------|
| 1 | Open | Price | Raw | - |
| 2 | High | Price | Raw | - |
| 3 | Low | Price | Raw | - |
| 4 | Close | Price | Raw | - |
| 5 | Volume | Price | Raw | - |
| 6 | Daily_Return | Price | `(C_t - C_{t-1})/C_{t-1}` | 1 |
| 7 | EMA12 | Trend | `EMA(Close, 12)` | 12 |
| 8 | EMA25 | Trend | `EMA(Close, 25)` | 25 |
| 9 | EMA20 | Trend | `EMA(Close, 20)` | 20 |
| 10 | MA5 | Trend | `SMA(Close, 5)` | 5 |
| 11 | MA10 | Trend | `SMA(Close, 10)` | 10 |
| 12 | MACD | Trend | `EMA12 - EMA26` | 12/26 |
| 13 | BOLL_upper | Volatility | `SMA20 + 2σ` | 20 |
| 14 | BOLL_lower | Volatility | `SMA20 - 2σ` | 20 |
| 15 | BOLL_mid | Volatility | `SMA20` | 20 |
| 16 | ATR | Volatility | `SMA(TR, 14)` | 14 |
| 17 | CCI | Momentum | `(TP - SMA20_TP)/(0.015*MD)` | 20 |
| 18 | MTM6 | Momentum | `Close_t - Close_{t-6}` | 6 |
| 19 | MTM12 | Momentum | `Close_t - Close_{t-12}` | 12 |
| 20 | ROC | Momentum | `(C_t - C_{t-12})/C_{t-12}*100` | 12 |
| 21 | SMI | Momentum | `200*(D_S/HLD_S)` | 10/3 |
| 22 | WVAD | Volume | `Σ[(C-O)/(H-L)*V]` | Cumulative |
| 23 | Close_Denoised | Decomposition | `DWT_Haar_3level(Close)` | - |

### Optional Features (if data available)
| ID | Feature | Category |
|----|---------|----------|
| 24 | PE_Ratio | Fundamental |
| 25 | PB_Ratio | Fundamental |
| 26 | Turnover_Rate | Market |
| 27 | Exchange_Rate | Macro |
| 28 | Interest_Rate | Macro |

---

## 4. DATA PROCESSING RULES

### 4.1 Missing Value Handling
| Scenario | Action |
|----------|--------|
| 1-5 consecutive NaN | Forward-fill from last valid observation |
| >5 consecutive NaN | Discard rows |
| Leading NaN (indicator warm-up) | Drop rows (do not impute) |
| Isolated NaN in optional features | Forward-fill if possible, else drop |

### 4.2 Scaling Rules
- **Method:** MinMaxScaler to [-1, 1]
- **Fit:** Training data only
- **Transform:** All splits using fitted parameters
- **Inverse:** Store scaler parameters for prediction denormalization

### 4.3 Causality Constraints
**ALL feature calculations SHALL satisfy:**
```
feature[t] = f(data[0:t])  # Inclusive of t, exclusive of t+1
```
- No future prices in moving averages
- No future volumes in indicators
- No interpolation across gaps using future data

---

## 5. LSTM SPECIFICATION

### 5.1 Input Tensor Structure
```
Shape: (batch_size, timesteps, features)
       (N, 20, F)
       
Where:
- N = number of samples (variable)
- 20 = look-back period (fixed)
- F = number of features after selection (variable, ≤ 23)
```

### 5.2 Target Definition
```
y[t] = (Close_{t+1} - Close_t) / Close_t
```

### 5.3 Architecture Constraints
- **Type:** Stateful LSTM (optional) or Stateless LSTM
- **Layers:** 2 hidden layers (fixed)
- **Output:** Single neuron with linear activation
- **Loss:** Mean Squared Error
- **Optimizer:** Adam (learning rate from PSO)

### 5.4 Training Protocol
- **Batch size:** 32 (fixed, not optimized)
- **Validation:** 10% temporal holdout
- **Early stopping:** Patience = 10 epochs (monitor validation loss)
- **Max epochs:** From PSO optimization (50-300)

---

## 6. XGBOOST SPECIFICATION

### 6.1 Input Structure
```
Shape: (samples, features)
Type: Dense matrix (NumPy array or Pandas DataFrame)

Feature naming: {feature_name}_lag_{k}
Where k = 0 (t-1), 1 (t-2), ..., 19 (t-20)
```

### 6.2 Feature Engineering Strategy
- **Primary:** Lag-based representation (current + 19 lags = 20 values per feature)
- **Secondary:** Rolling statistics (mean, std, min, max over 20-period window)
- **Excluded:** Flattened sequence representation (prohibited)

### 6.3 Hyperparameters (Fixed, Not PSO-Optimized)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_depth | 6 | Prevent overfitting |
| learning_rate | 0.05 | Conservative convergence |
| n_estimators | 500 | Early stopping determines final |
| subsample | 0.8 | Row sampling |
| colsample_bytree | 0.8 | Feature sampling |
| objective | reg:squarederror | MSE for regression |

### 6.4 Training Protocol
- **Validation:** 10% temporal holdout
- **Early stopping:** Patience = 50 rounds
- **Evaluation metric:** RMSE

---

## 7. PSO SYSTEM DESIGN

### 7.1 Unified Configuration
```
Algorithm: IPSO (Improved Particle Swarm Optimization)
Population: 20 particles
Iterations: 50 max
Dimensions: 4 (epochs, node1, node2, learning_rate)

Cognitive coefficient (c1): 1.5
Social coefficient (c2): 1.5
Inertia: Adaptive tanh schedule [0.8 → 0.6]
Mutation: Adaptive factor [0.7 → 1.0]
```

### 7.2 Search Space Bounds
| Dimension | Min | Max | Type |
|-----------|-----|-----|------|
| Epochs | 50 | 300 | Integer |
| Hidden Layer 1 | 50 | 300 | Integer |
| Hidden Layer 2 | 50 | 300 | Integer |
| Learning Rate | 0.001 | 0.01 | Float |

### 7.3 Objective Function
**Function:** Mean Squared Error on validation set  
**Formula:** `fitness(particle) = mean((y_pred - y_val)^2)`

**Evaluation Protocol:**
1. Construct LSTM with particle parameters
2. Train on training set (70% chronological)
3. Evaluate on validation set (10% chronological)
4. Return MSE as fitness score

### 7.4 Constraints
- Test set SHALL NOT be accessed during PSO
- Each particle evaluation SHALL use identical data splits
- Random seeds SHALL be fixed for reproducibility

---

## 8. DATA LEAKAGE PREVENTION RULES

### 8.1 Absolute Constraints

| Rule ID | Rule Statement | Enforcement |
|---------|---------------|-------------|
| L-1 | **Temporal Ordering:** Dataset SHALL be sorted chronologically before any processing. Shuffling is PROHIBITED. | Pre-processing check |
| L-2 | **Causal Feature Computation:** Features at time t SHALL use only data with timestamp ≤ t. | Code audit all indicators |
| L-3 | **Scaler Training:** MinMaxScaler SHALL be fit on training split ONLY. Refitting on full dataset is PROHIBITED. | Store scaler parameters |
| L-4 | **Correlation Computation:** Pearson correlation for feature selection SHALL be computed on training data ONLY. | Log correlation matrix |
| L-5 | **Wavelet Threshold:** DWT threshold estimation SHALL use training data ONLY. | Store threshold value |
| L-6 | **PSO Validation:** PSO fitness SHALL be evaluated on validation set, NEVER on test set. | Three-way split enforcement |
| L-7 | **Window Boundaries:** Sliding windows SHALL NOT cross train/validation/test boundaries. | Index validation |
| L-8 | **Target Separation:** Target variable SHALL be future return (t+1), NEVER included in feature window at time t. | Target construction audit |
| L-9 | **Warm-up Exclusion:** Indicator warm-up rows (NaN) SHALL be dropped before splitting. | Pre-split validation |

### 8.2 Split Ratios (Chronological)
```
Training:   70% (earliest data)
Validation: 10% (immediately following training)
Test:       20% (most recent data, final evaluation only)
```

### 8.3 Prohibited Operations
- Random train/test split (PROHIBITED)
- Whole-dataset normalization (PROHIBITED)
- Target encoding with global statistics (PROHIBITED)
- Feature selection on full dataset (PROHIBITED)
- PSO using test set for fitness (PROHIBITED)

---

## 9. PRODUCTION CONSTRAINTS

### 9.1 Reproducibility Requirements
- **Random seeds:** Fixed across Python, NumPy, TensorFlow
- **Version control:** Feature schema version SHALL be logged with each model artifact
- **Determinism:** All operations SHALL be deterministic (no stochasticity except PSO initialization)

### 9.2 Versioning
- **Feature schema:** Semantic versioning (MAJOR.MINOR.PATCH)
- **Schema changes:** Increment MAJOR for breaking changes, MINOR for feature additions
- **Retention masks:** Versioned and stored with model artifacts

### 9.3 Monitoring
- **Feature drift:** Track distribution shifts in production features vs. training
- **Correlation stability:** Monitor target-feature correlation stability
- **Prediction drift:** Track model output distribution over time

### 9.4 Deployment Assumptions
- **Inference latency:** Feature computation MUST complete within 1 minute for daily predictions
- **Historical buffer:** Minimum 40 periods (2x look-back) required for warm-up
- **Scalability:** Pipeline SHALL handle single-ticker inference efficiently

### 9.5 State Management
- **Scaler persistence:** MinMaxScaler parameters SHALL be serialized (JSON/pickle) at training time
- **Feature mask persistence:** Retention mask SHALL be versioned and loaded at inference
- **DWT consistency:** Same wavelet library and configuration across training and inference

---

## 10. CONFLICT RESOLUTION SUMMARY

| Conflict | Documents | Resolution | Justification |
|----------|-----------|------------|---------------|
| Normalization range | [0,1] vs [-1,1] | **[-1, 1]** | Supports return features; majority of papers |
| PSO objective | MSE-only vs MSE+MSW | **MSE-only** | Simpler, more stable, sufficient for production |
| XGBoost representation | Flattened vs Lag-based | **Lag-based** | Tree-appropriate, lower dimensionality |
| Wavelet transform | Enabled vs Disabled | **Enabled** | Zeng et al. 2025 demonstrates value |
| Feature selection | Correlation-only vs PSO+correlation | **Correlation-only** | Deterministic, interpretable, faster |
| Target variable | Price vs Return | **Return** | Stationary, scale-invariant, preferred by modern finance |
| PSO algorithm | Standard vs IPSO | **IPSO** | Superior performance per Ji et al. 2021 |
| Look-back window | Multiple options | **20 days** | Optimal per Ji et al. 2021 experiments |
| MACD signal line | Include vs Exclude | **Exclude** | Papers mention MACD ambiguously; line only is standard |
| LSTM depth | Fixed 2-layer vs Optimized 1-3 | **Fixed 2-layer** | Simpler, sufficient, reduces search space |

---

**END OF TECHNICAL REQUIREMENTS DOCUMENT**

*This TRD is ready for implementation. All specifications are deterministic, production-safe, and traceable to research foundations.*