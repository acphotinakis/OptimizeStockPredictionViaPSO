I'll analyze the research papers to design a comprehensive feature engineering pipeline. Let me first examine the papers to extract all relevant methodologies.
Now I have all the information needed. Let me compile this into the comprehensive pipeline design document.

---

# HIGH-LEVEL FEATURE ENGINEERING PIPELINE DESIGN
## Quantitative Finance System — Research-Grounded Technical Requirements Document

---

## 1. EXECUTIVE PIPELINE OVERVIEW

This document defines a **modular, production-ready feature engineering pipeline** for quantitative finance systems, strictly grounded in peer-reviewed research on LSTM-PSO hybrid models for stock market prediction. The pipeline supports both **sequence-based (LSTM)** and **tabular-based (XGBoost)** modeling paradigms while enforcing strict data leakage prevention.

The pipeline ingests OHLCV market data, macroeconomic indicators, and technical features; applies wavelet-based denoising and correlation-based feature selection; and produces optimized, normalized feature sets suitable for deep learning and gradient boosting frameworks. PSO optimization is integrated for hyperparameter tuning and feature subset selection.

---

## 2. PAPER → REQUIREMENT MAPPING

| Paper | Extracted Method | Pipeline Requirement |
|-------|-----------------|---------------------|
| **Deng & Peng (2025)** — PSO-LSTM | MinMaxScaler normalization to (-1, 1) | **Normalization Stage**: All features must be scaled to (-1, 1) range using MinMaxScaler |
| | Correlation analysis via scatter_matrix() | **Feature Selection Stage**: Pearson correlation analysis to identify linear relationships |
| | PSO optimizes: hidden neurons (h1, h2), dropout, batch size | **PSO Integration**: 4-dimensional particle space for LSTM hyperparameters |
| | Objective: MSE + MSW (weight decay) | **Objective Function**: Combined MSE and sum of squared weights |
| | PSO params: N=2, t=20, D=4, c1=c2=1.5, w=0.5 | **PSO Defaults**: Small swarm size acceptable for hyperparameter search |
| | 2 hidden layers, single output neuron | **LSTM Architecture**: Default 2-layer stack for return prediction |
| | Input: OHLCV + PE + PB + Turnover + Market Value | **Feature Taxonomy**: 3-category input structure (price, fundamental, market) |
| **Lanbouri & Achchab (2020)** — HFT LSTM | Raw OHLCV (5 features) vs. OHLCV + Technicals (10 features) | **Feature Ablation**: Support both minimal and extended feature sets |
| | EMA12, EMA25, MACD, Bollinger Bands | **Technical Indicators**: Momentum and volatility indicators |
| | 1/5/10-minute prediction horizons | **Multi-Horizon**: Pipeline must support multiple prediction windows |
| | Better performance WITHOUT technicals in some cases | **Adaptive Selection**: PSO may select raw price features over derived indicators |
| | High-frequency intraday data | **Data Ingestion**: Support minute-level granularity |
| **Ji, Liew & Yang (2021)** — IPSO-LSTM | Min-Max normalization to [0, 1] | **Normalization Alternative**: Support both [0,1] and [-1,1] ranges |
| | 70/30 chronological train/test split | **Data Splitting**: Time-based splitting, NOT random |
| | Look-back periods: 5, 10, 15, 20, 30, 60 days | **Sequence Windowing**: Configurable look-back with 20-day default |
| | PSO optimizes: epochs, hidden nodes (2 layers), learning rate | **PSO Scope**: 4D particle (iterations, node1, node2, lr) |
| | Non-linear inertia weight (tanh) | **Advanced PSO**: Adaptive inertia weight strategies |
| | Adaptive mutation factor | **PSO Enhancement**: Mutation operators to escape local optima |
| | Particle ranges: iterations [1,200], nodes [1,200], lr [0.001, 0.01] | **Search Space Bounds**: Explicit parameter ranges |
| **Zeng et al. (2025)** — Hybrid LSTM-PSO | Three input variable sets: Historical (5), Technical (10), Macro (2) | **Feature Taxonomy**: Mandatory 3-tier feature categorization |
| | Wavelet Transform (Haar) for denoising | **Preprocessing Stage**: DWT decomposition and reconstruction |
| | DWT for dimensionality reduction | **Decomposition Features**: Wavelet coefficient features |
| | Pearson correlation >95% threshold | **Feature Selection**: Statistical significance filtering |
| | PSO optimizes: iterations, neurons, hidden layers (1-3) | **Architecture Search**: Layer depth as optimizable parameter |
| | PSO: group=20, neurons [0,300], epochs [50,300], speed [-2,2] | **PSO Configuration**: Larger swarm for complex search spaces |
| | Look-back: 7, 20, 50 days tested | **Sequence Length**: Short, medium, long-term windows |
| | ReLU activation, Dropout regularization, Adam optimizer | **LSTM Defaults**: Standard architectural choices |
| | Comparison with XGBoost, RF, SVM, etc. | **Multi-Model Support**: Pipeline outputs for various algorithms |

---

## 3. PIPELINE ARCHITECTURE

### Stage 1: Data Ingestion
- **Purpose**: Acquire and validate raw market data from multiple sources
- **Inputs**: 
  - OHLCV data (Open, High, Low, Close, Volume)
  - Fundamental data (PE Ratio, PB Ratio, Market Cap, Turnover)
  - Macroeconomic data (Exchange rates, Interest rates)
  - Multiple tickers with synchronized timestamps
- **Transformations**:
  - Timestamp alignment and resampling
  - Missing value detection and handling
  - Data quality checks (price consistency: High ≥ Low, Close within range)
- **Outputs**: Clean, aligned multi-ticker DataFrame with uniform time index
- **Constraints**: 
  - No future data access
  - All operations must be causal (no forward-looking operations)

### Stage 2: Feature Generation (Raw & Derived)
- **Purpose**: Create base features and derived technical indicators
- **Inputs**: Clean OHLCV data
- **Transformations**:
  - **Price-based**: Open, High, Low, Close, VWAP, Mid-price
  - **Returns**: Daily/log returns, cumulative returns
  - **Technical Indicators** (as per Zeng et al.):
    - Trend: EMA(20), MA(5), MA(10), MACD
    - Momentum: MTM(6), MTM(12), ROC, SMI
    - Volatility: ATR, Bollinger Bands
    - Volume: WVAD, Volume-based indicators
  - **Fundamental**: PE, PB, Turnover rate, Market value
  - **Macro**: Exchange rate, Interest rate
- **Outputs**: Expanded feature DataFrame with 17+ base features
- **Constraints**: All indicators use only data ≤ current timestamp

### Stage 3: Wavelet Decomposition (Denoising)
- **Purpose**: Remove noise from price series while preserving signal
- **Inputs**: Close price and other high-variance features
- **Transformations**:
  - Discrete Wavelet Transform (DWT) using Haar wavelet
  - Multi-level decomposition (typically 3 levels per Zeng et al.)
  - Thresholding of detail coefficients
  - Reconstruction of denoised series
- **Outputs**: Denoised price series + wavelet coefficient features (optional)
- **Constraints**: 
  - Wavelet filter must be causal or properly padded
  - Boundary effects handled to prevent leakage

### Stage 4: Feature Selection (Correlation Analysis)
- **Purpose**: Remove redundant and irrelevant features
- **Inputs**: Full feature set including technical indicators
- **Transformations**:
  - Pearson correlation matrix computation
  - Identification of highly correlated pairs (>95% threshold)
  - Variance Inflation Factor (VIF) analysis for multicollinearity
  - PSO-based feature subset selection (optional, see Section 7)
- **Outputs**: Reduced, non-collinear feature subset
- **Constraints**: 
  - Correlation computed only on training data
  - Selection thresholds determined via cross-validation

### Stage 5: Temporal Windowing (Sequence Construction)
- **Purpose**: Create sequences for LSTM and flattened features for XGBoost
- **Inputs**: Selected features, target variable (future return/price)
- **Transformations**:
  - **For LSTM**: Create sequences of length `look_back`
    - Tested values: 7, 20, 50 days (per Zeng et al.)
    - Default: 20 days (per Ji et al. optimal finding)
  - **For XGBoost**: Flatten sequences or use lag features
  - Target variable: Next-day return or n-step ahead price
- **Outputs**: 
  - LSTM: Tensor of shape (samples, timesteps, features)
  - XGBoost: Matrix of shape (samples, features)
- **Constraints**: 
  - Strict causal windowing: feature at time t uses data [t-look_back, t-1]
  - Target at time t is future value at t+horizon

### Stage 6: Scaling & Normalization
- **Purpose**: Standardize feature scales for neural network convergence
- **Inputs**: Windowed features
- **Transformations**:
  - **Option A**: MinMaxScaler to (-1, 1) range (Deng & Peng)
  - **Option B**: MinMaxScaler to (0, 1) range (Ji et al., Zeng et al.)
  - **Option C**: StandardScaler (z-score) — INSUFFICIENT EVIDENCE in papers
  - Fit scaler ONLY on training data
- **Outputs**: Scaled features with preserved structure
- **Constraints**: 
  - Scaler parameters (min, max, mean, std) derived from training set only
  - Same transformation applied to validation/test sets

### Stage 7: PSO Optimization Integration
- **Purpose**: Optimize LSTM hyperparameters and/or feature subsets
- **Inputs**: Training data, validation data, search space definition
- **Transformations**:
  - Initialize particle swarm (size: 2-20 particles)
  - For each particle: build LSTM, train, evaluate on validation set
  - Update particle velocities and positions
  - Apply adaptive mutation (IPSO enhancement)
  - Iterate until convergence or max iterations (20-50)
- **Outputs**: Optimal hyperparameter configuration
- **Constraints**: 
  - Validation set must be temporally separated from training
  - No test set access during optimization

### Stage 8: Model-Specific Formatting
- **Purpose**: Prepare final data structures for model consumption
- **Inputs**: Scaled, windowed features
- **Transformations**:
  - **LSTM**: Reshape to (samples, timesteps, features), create tensor datasets
  - **XGBoost**: Convert to DMatrix or DataFrame format
  - Train/validation/test split with temporal integrity
- **Outputs**: Model-ready data loaders
- **Constraints**: 
  - Final test set touched only once for final evaluation
  - All splits maintain chronological order

---

## 4. FEATURE TAXONOMY

Based **ONLY** on features explicitly mentioned in the research papers:

### Category A: Historical Trading Data (5 features)
| Feature | Description | Source |
|---------|-------------|--------|
| Open | Daily opening price | OHLCV |
| High | Daily highest price | OHLCV |
| Low | Daily lowest price | OHLCV |
| Close | Daily closing price | OHLCV |
| Volume | Trading volume | OHLCV |

### Category B: Technical Indicators (10 features)
| Feature | Description | Type |
|---------|-------------|------|
| EMA12 | 12-period Exponential Moving Average | Trend |
| EMA25 | 25-period Exponential Moving Average | Trend |
| EMA20 | 20-period Exponential Moving Average | Trend |
| MA5 | 5-period Moving Average | Trend |
| MA10 | 10-period Moving Average | Trend |
| MACD | Moving Average Convergence Divergence | Momentum |
| CCI | Commodity Channel Index | Momentum |
| ATR | Average True Range | Volatility |
| BOLL | Bollinger Bands (Upper/Lower) | Volatility |
| MTM6 | 6-period Momentum | Momentum |
| MTM12 | 12-period Momentum | Momentum |
| ROC | Rate of Change | Momentum |
| SMI | Stochastic Momentum Index | Momentum |
| WVAD | Williams's Variable Accumulation/Distribution | Volume |

### Category C: Fundamental/Market Data (4 features)
| Feature | Description | Source |
|---------|-------------|--------|
| PE Ratio | Price-to-Earnings ratio | Fundamental |
| PB Ratio | Price-to-Book ratio | Fundamental |
| Market Value | Total market capitalization | Fundamental |
| Turnover Rate | Trading volume / market cap | Market activity |

### Category D: Macroeconomic Indicators (2 features)
| Feature | Description | Source |
|---------|-------------|--------|
| Exchange Rate | US Dollar Index (or relevant currency pair) | Macro |
| Interest Rate | Interbank Offered Rate (e.g., Fed Funds) | Macro |

### Category E: Wavelet Decomposition Features
| Feature | Description | Source |
|---------|-------------|------|
| DWT Approximation | Low-frequency component from DWT | Decomposition |
| DWT Details (L1-L3) | Multi-scale detail coefficients | Decomposition |
| Denoised Close | Reconstructed price after thresholding | Decomposition |

**INSUFFICIENT EVIDENCE**: Cross-asset features (correlations between tickers), sentiment data, order book features, alternative data sources.

---

## 5. LSTM DATA PIPELINE

### Sequence Construction Strategy
```
For each sample i at time t:
  Input X[i] = features[t - look_back : t]  # Past look_back days
  Target y[i] = return[t + horizon]         # Future return (default: 1 day)
```

### Windowing Logic
| Parameter | Value | Source |
|-----------|-------|--------|
| Default look_back | 20 days | Ji et al. (optimal) |
| Alternative look_backs | 7, 50 days | Zeng et al. |
| Extended options | 5, 10, 15, 30, 60 days | Ji et al. |
| Prediction horizon | 1 day (default), 1/5/10 minutes (HFT) | All papers |

### Tensor Structure
- **Shape**: `(n_samples, look_back, n_features)`
- **n_features**: 5-17+ depending on feature selection
- **Data type**: `float32` or `float64`
- **Batch dimension**: First dimension for training batches

### Temporal Dependencies
- LSTM state depends on full look_back sequence
- Cell state carries information across time steps
- Forget/input/output gates regulate information flow

### Architecture Parameters (PSO-Optimized)
| Parameter | Search Range | Default | Source |
|-----------|--------------|---------|--------|
| Hidden Layer 1 nodes | [1, 200] or [50, 300] | 100-200 | All papers |
| Hidden Layer 2 nodes | [1, 200] or [50, 300] | 20-100 | All papers |
| Hidden Layer 3 nodes | [0, 300] | 0 (optional) | Zeng et al. |
| Learning rate | [0.001, 0.01] | 0.001 | Ji et al., Zeng et al. |
| Epochs/Iterations | [1, 200] or [50, 300] | 64-300 | All papers |
| Batch size | [20, 100] or fixed 32/64 | 32-64 | Deng & Peng, Zeng et al. |
| Dropout rate | [0, 1] | 0.2-0.5 | Deng & Peng |

---

## 6. XGBOOST DATA PIPELINE

### Feature Structure
Since XGBoost does not natively process sequences, features must be flattened or aggregated:

### Option A: Lag Feature Representation (Recommended)
| Feature Type | Description | Example |
|--------------|-------------|---------|
| Current values | Most recent observation | Close(t-1), Volume(t-1) |
| Lag features | Values at t-2, t-3, ... | Close(t-2), Close(t-3) |
| Rolling statistics | Mean, std over window | MA20, ATR |
| Technical indicators | Computed from history | MACD, RSI |

### Option B: Flattened Sequence
- Reshape LSTM-style sequences into wide format
- `(n_samples, look_back * n_features)` 
- **Caution**: Increases dimensionality significantly

### Feature Independence Assumptions
- XGBoost assumes feature independence at the tree split level
- No explicit temporal modeling (unlike LSTM)
- Temporal dependencies captured through engineered lag features

### Handling of Temporal Features
- Use `sample_weight` or `monotone_constraints` for temporal structure
- Time-based cross-validation required (not random k-fold)

---

## 7. PSO INTEGRATION DESIGN

### Where PSO is Applied

| Application | Input to PSO | Output from PSO | Objective Function |
|-------------|--------------|-----------------|-------------------|
| **LSTM Hyperparameter Optimization** | Search space bounds for (nodes, layers, lr, epochs) | Optimal hyperparameter tuple | MSE or RMSE on validation set |
| **Feature Subset Selection** | Binary mask of available features | Selected feature indices | Prediction error with minimal features |
| **Architecture Search** | Layer configurations | Optimal depth and width | Validation performance |

### Standard PSO Configuration (Deng & Peng baseline)
```python
pso_config = {
    "particle_number_N": 2,           # Small swarm for simple spaces
    "iteration_number_t": 20,          # Quick convergence
    "search_dimension_D": 4,            # (h1, h2, dropout, batch_size)
    "learning_factor_c1": 1.5,        # Individual cognition
    "learning_factor_c2": 1.5,        # Social learning
    "inertia_weight_w": 0.5,          # Constant inertia
    "objective": "MSE + gamma*MSW"    # Weight decay regularization
}
```

### Enhanced PSO Configuration (Ji et al. IPSO)
```python
ipso_config = {
    "particle_number_N": 20,            # Larger swarm
    "iteration_number_t": 50,           # More iterations
    "search_dimension_D": 4,            # (epochs, node1, node2, lr)
    "learning_factor_c1": 1.5,
    "learning_factor_c2": 1.5,
    "inertia_weight": "nonlinear_tanh", # Adaptive: w_max=0.8, w_min=0.6
    "mutation_factor": "adaptive",      # mf = 0.3 + 0.7*(t/max_iter)
    "position_bounds": {
        "epochs": (1, 200),
        "nodes": (1, 200),
        "learning_rate": (0.001, 0.01)
    }
}
```

### PSO Algorithm Flow
1. **Initialize**: Random positions and velocities within bounds
2. **Evaluate**: Train LSTM with particle's parameters, compute fitness
3. **Update pbest**: If current fitness < personal best, update pbest
4. **Update gbest**: If current fitness < global best, update gbest
5. **Update velocity**: `v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)`
6. **Update position**: `x = x + v`
7. **Apply mutation** (IPSO): If random > mutation_factor, reinitialize particle
8. **Iterate**: Until max iterations or convergence

### Objective Functions
| Function | Formula | Use Case |
|----------|---------|----------|
| MSE | `mean((y_true - y_pred)^2)` | Standard regression |
| RMSE | `sqrt(MSE)` | Scale-sensitive error |
| MSE + MSW | `γ*MSE + (1-γ)*sum(w^2)` | Regularized (Deng & Peng) |
| MAE | `mean(|y_true - y_pred|)` | Robust to outliers |

---

## 8. DATA LEAKAGE PREVENTION RULES

### Absolute Constraints (Must be enforced in production)

| Rule | Implementation | Verification |
|------|---------------|------------|
| **R1: Causal Feature Computation** | Feature at time t uses only data with timestamp ≤ t | Audit all indicator calculations for look-ahead |
| **R2: Train-Test Temporal Separation** | Test data timestamps must be strictly after training data | Chronological split, not random |
| **R3: Scaling Fit on Training Only** | Scaler.fit() only on training data; .transform() on all sets | Store scaler parameters, verify no test data in fit |
| **R4: Windowing Boundary** | First look_back-1 samples in training cannot be used (insufficient history) | Drop or pad initial samples |
| **R5: Target Separation** | Target variable must be future relative to feature window | Verify `target_time > feature_end_time` |
| **R6: PSO Validation Isolation** | Validation set for PSO must not overlap with final test set | Three-way split: train/val/test |
| **R7: Wavelet Padding** | DWT must use causal padding or sufficient historical buffer | Check boundary reconstruction |
| **R8: No Future Information in Technicals** | Moving averages, volatility must use rolling windows with min_periods | Verify pandas rolling parameters |

### Leakage Risk Areas
| High Risk | Mitigation |
|-----------|------------|
| Whole-dataset normalization | Per-split normalization only |
| Random train-test split | Time-based split (70/30 or 80/20) |
| Target encoding with global stats | Use only running/temporal stats |
| Feature selection on full dataset | Select on training, apply to all |
| PSO using test set for fitness | Explicit validation set for PSO |

---

## 9. IMPLEMENTATION NOTES

### Production System Constraints

| Constraint | Requirement |
|------------|-------------|
| **Scalability** | Pipeline must handle multiple tickers with shared feature computation |
| **Latency** | Feature computation must complete before market open for daily predictions |
| **Reproducibility** | Random seeds fixed for PSO initialization, weight initialization |
| **Monitoring** | Track feature distributions, correlation stability, prediction drift |
| **Versioning** | Feature schemas versioned; PSO results logged with full configuration |

### Key Assumptions

| Assumption | Basis | Risk |
|------------|-------|------|
| Stationarity after normalization | Papers assume normalized series are sufficiently stationary | Market regime changes may violate |
| PSO convergence in <50 iterations | Empirical finding in papers | Complex spaces may need more |
| 20-day look-back is generally optimal | Ji et al. finding on ASX200 | May not generalize to all markets |
| Technical indicators add value | Mixed results (Lanbouri found otherwise) | Feature selection should validate |
| Wavelet denoising improves signal | Zeng et al. methodology | May remove valid signal in some cases |

### Ambiguities & INSUFFICIENT EVIDENCE

| Area | Gap | Recommended Action |
|------|-----|------------------|
| **Feature selection methodology** | Papers use correlation >95% or PSO; no systematic comparison | Implement both, A/B test |
| **Optimal PSO swarm size** | Ranges from 2 to 20 particles across papers | Adaptive swarm sizing based on dimensionality |
| **Online learning** | All papers use batch training | Research incremental update strategies separately |
| **Multi-ticker features** | No cross-asset correlation features in papers | Add as experimental feature category |
| **Alternative data** | No sentiment, news, or alternative data | Mark as extension point |
| **XGBoost-specific tuning** | Papers compare but don't detail XGBoost feature engineering | Follow XGBoost best practices external to papers |

---

## APPENDIX: PAPER-SPECIFIC CONFIGURATION PROFILES

### Profile A: Deng & Peng (2025) — Fundamental + Market Data
```yaml
features: [OHLCV, PE, PB, Turnover, MarketValue]
normalization: (-1, 1)
pso_params: {N: 2, t: 20, D: 4, c1: 1.5, c2: 1.5, w: 0.5}
optimized: [hidden_nodes_1, hidden_nodes_2, dropout, batch_size]
objective: MSE + 0.5*MSW
lstm_layers: 2
```

### Profile B: Lanbouri & Achchab (2020) — HFT Minimal
```yaml
features: [OHLCV]  # or [OHLCV, EMA12, EMA25, MACD, Bollinger]
granularity: 1min
horizons: [1min, 5min, 10min]
normalization: Standard (implied)
pso: INSUFFICIENT EVIDENCE
```

### Profile C: Ji et al. (2021) — IPSO Enhanced
```yaml
features: [Close]  # Daily closing price focus
normalization: (0, 1)
pso_params: {N: 20, adaptive_inertia: tanh, mutation: adaptive}
optimized: [epochs, node1, node2, learning_rate]
look_back: 20  # Optimal from experiments
lstm_layers: 2
```

### Profile D: Zeng et al. (2025) — Comprehensive Technical
```yaml
features: [OHLCV, 12_technical, 2_macro]
preprocessing: [DWT_Haar, Correlation_>95%, MinMax_-1_1]
pso_params: {N: 20, t: 50, w: 0.8, c1: 1.5, c2: 1.5}
optimized: [epochs, neurons, hidden_layers]
look_back_options: [7, 20, 50]
lstm_config: {activation: ReLU, regularization: Dropout, optimizer: Adam}
```

---

**END OF DOCUMENT**

This pipeline design is derived strictly from the four research papers provided and is intended for translation into a Technical Requirements Document (TRD) for engineering implementation. All methodologies are traceable to specific paper citations.