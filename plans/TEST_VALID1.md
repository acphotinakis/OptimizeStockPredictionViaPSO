I'll analyze the research papers to extract the specific validation, walk-forward, and backtesting methodologies, then synthesize a comprehensive protocol.

Based on my analysis of the four papers and financial ML best practices, here is the definitive specification:

---

# VALIDATION, WALK-FORWARD & BACKTESTING PROTOCOL
## PSO-LSTM Stock Prediction System — v1.0

---

## 1. EXECUTIVE SUMMARY

| Model | Training Strategy | Validation Strategy | Walk-Forward | Hyperparameter Handling |
|-------|-----------------|---------------------|--------------|------------------------|
| **Baseline LSTM** | Static 80% train | 10% chronological validation | Expanding window | Fixed defaults, never tuned |
| **PSO-LSTM** | Static 72% train | 8% PSO validation + 10% early stopping | Expanding window | PSO-optimized once on initial train, fixed thereafter |
| **XGBoost** | Static 80% train | 10% chronological validation | Expanding window | Grid search on initial train, fixed thereafter |

**Core Principle:** All models use **expanding window walk-forward** (not rolling). Models are **fully retrained** at each step. No incremental updates. No hyperparameter re-optimization after initial calibration.

---

## 2. DATA SPLIT ARCHITECTURE

### 2.1 Temporal Split Structure (All Models)

```
Timeline:  [---- T1 ----|---- T2 ----|---- T3 ----|---- T4 ----]
           2008-07       2012-07       2014-07       2016-07
           
           [========= TRAIN (80%) ==========][===== TEST (20%) =====]
           
For PSO-LSTM:
           [==== E1-1 (72%) ====][E1-2 (8%)][======== E2 (20%) ========]
                                ↑ PSO validation    ↑ Final holdout
           
For Baseline/XGBoost:
           [======= Train (80%) ========][=== Val (from train) ===][==== Test (20%) ===]
                                        ↑ Early stopping/grid      ↑ Final holdout
```

**Critical Rules:**
- All splits are **chronological** — no random shuffling at any stage
- Test set (E2) is **touched exactly once** for final metrics
- Validation set is used for **early stopping** (LSTM) or **grid search** (XGBoost)
- PSO validation (E1-2) is used **only during PSO optimization**, never for final evaluation

### 2.2 Split Percentages

| Model | Training | Validation (PSO/Early Stop) | Test |
|-------|----------|----------------------------|------|
| Baseline LSTM | 80% | 10% of training (early stopping) | 20% |
| PSO-LSTM | 72% | 8% of total (PSO fitness) | 20% |
| XGBoost | 80% | 10% of training (grid search) | 20% |

---

## 3. WALK-FORWARD VALIDATION PROTOCOL

### 3.1 Window Strategy: Expanding Window (All Models)

```
Step 1:  [Train: t_0..t_100] → Predict: t_101..t_110 → Evaluate
Step 2:  [Train: t_0..t_110] → Predict: t_111..t_120 → Evaluate
Step 3:  [Train: t_0..t_120] → Predict: t_121..t_130 → Evaluate
...
```

**Why Expanding (Not Rolling):**
- Financial time series exhibit **regime shifts** and **structural breaks**
- Expanding window preserves **all historical information**, allowing model to adapt to new patterns while retaining old context
- Rolling window discards old data that may contain valuable regime information
- Aligned with Zeng et al. (2025) multi-year evaluation approach

### 3.2 Walk-Forward Step Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Initial train size | Minimum 252 days (1 year) | Sufficient for LSTM warm-up and indicator calculation |
| Step size (prediction horizon) | 1 day (default), 5 days, 20 days | Matches Ji et al. (2021) look-back and trading horizons |
| Retraining frequency | **Every step** — full retrain | No incremental updates; full model rebuild |
| Feature pipeline | Re-fit on expanding train at each step | Scalers, selectors, wavelet thresholds re-computed |

### 3.3 Walk-Forward Algorithm (Pseudocode)

```python
def walk_forward_evaluate(model_type, data, config, step_size=1):
    """
    Expanding window walk-forward evaluation.
    All models retrained from scratch at each step.
    """
    results = []
    initial_train_size = 252  # Minimum 1 year
    total_samples = len(data)
    
    # Pointer starts after initial training period
    current_idx = initial_train_size
    
    while current_idx + step_size <= total_samples:
        # Define expanding train set (grows at each step)
        train_data = data.iloc[:current_idx]
        
        # Define test/prediction window (next step_size days)
        test_data = data.iloc[current_idx:current_idx + step_size]
        
        # Apply feature pipeline FITTED ON TRAIN ONLY
        features_train, features_test = run_feature_pipeline(
            train_data, test_data, fit_on_train_only=True
        )
        
        if model_type == "pso_lstm":
            # PSO hyperparameters: optimized ONCE on first window, fixed thereafter
            if current_idx == initial_train_size:
                pso_params = run_pso_optimization(features_train, config)
            
            model = train_lstm(features_train, pso_params, shuffle=False)
            
        elif model_type == "baseline_lstm":
            # Fixed hyperparameters, never tuned
            model = train_lstm(features_train, config["baseline_params"], shuffle=False)
            
        elif model_type == "xgboost":
            # Grid search on first window, fixed thereafter
            if current_idx == initial_train_size:
                xgb_params = grid_search_xgboost(features_train, config)
            
            model = train_xgboost(features_train, xgb_params)
        
        # Generate predictions (NO test data leakage)
        predictions = model.predict(features_test)
        
        # Evaluate
        metrics = compute_metrics(predictions, test_data["target"])
        results.append({
            "step": len(results),
            "train_end_date": train_data.index[-1],
            "test_start_date": test_data.index[0],
            "test_end_date": test_data.index[-1],
            "predictions": predictions,
            "actuals": test_data["target"].values,
            "metrics": metrics
        })
        
        # Advance pointer
        current_idx += step_size
    
    # Aggregate metrics across all steps
    aggregated = aggregate_walk_forward_results(results)
    return results, aggregated
```

---

## 4. MODEL-SPECIFIC TRAINING & VALIDATION

### 4.1 Baseline LSTM

| Aspect | Specification |
|--------|--------------|
| **Architecture** | Fixed: 128 units (L1), 64 units (L2), dropout 0.2, ReLU activation |
| **Training data** | Expanding window train set |
| **Validation** | Last 10% of expanding train (chronological) for early stopping |
| **Early stopping** | Patience=10, monitor val_loss, restore best weights |
| **Shuffle** | **FALSE** (mandatory) |
| **Epochs** | 100 (fixed, or early stopped) |
| **Batch size** | 32 (fixed) |
| **Learning rate** | 0.001 (fixed) |
| **Optimizer** | Adam |
| **Loss** | MSE |
| **Retraining** | Full retrain from scratch at each walk-forward step |
| **Hyperparameters** | **Never tuned** — fixed for entire walk-forward |

**Why Fixed Hyperparameters?**
- Baseline serves as **lower-bound reference**
- Any tuning would contaminate the comparison
- Allows isolation of PSO benefit by comparing to truly naive configuration

### 4.2 PSO-Optimized LSTM

| Aspect | Specification |
|--------|--------------|
| **Architecture** | PSO-determined: units_1 ∈ [50,300], units_2 ∈ [20,200], dropout ∈ [0.0,0.5] |
| **Training data** | Expanding window train set (72% of available data up to current point) |
| **PSO validation** | 8% of total data (chronological, from end of training period) |
| **PSO objective** | `0.9 × MSE(E1-2) + 0.1 × MSW(network_weights)` |
| **PSO configuration** | IPSO: 20 particles, 50 iterations, nonlinear tanh inertia, adaptive mutation |
| **Early stopping** | Patience=10 on separate 10% validation split |
| **Shuffle** | **FALSE** (mandatory) |
| **Epochs** | PSO-optimized ∈ [50, 300] |
| **Batch size** | PSO-selected ∈ {32, 64} |
| **Learning rate** | PSO-optimized ∈ [0.001, 0.01] |
| **Optimizer** | Adam |
| **Loss** | MSE |
| **Retraining** | Full retrain from scratch at each walk-forward step |
| **Hyperparameters** | **Optimized once on first window, fixed for all subsequent steps** |

**PSO Optimization Protocol:**

```
Step 0 (Initial calibration window only):
    1. Split train_data into E1-1 (72%) and E1-2 (8%)
    2. Initialize PSO swarm: 20 particles, random positions in search space
    3. For each iteration (max 50):
        a. For each particle:
            - Decode position → hyperparameters
            - Train LSTM on E1-1 with these params (no shuffle)
            - Evaluate MSE on E1-2
            - Compute MSW = mean(weight²)
            - Fitness = 0.9 × MSE + 0.1 × MSW
        b. Update pbest/gbest
        c. Update velocities/positions with IPSO rules
        d. Apply adaptive mutation
    4. Return gbest hyperparameters

Steps 1..N (All walk-forward steps):
    - Use hyperparameters from Step 0 (FIXED, no re-optimization)
    - Train LSTM on expanding window with these fixed params
    - Early stopping on 10% validation split
```

**Why No PSO Re-optimization?**
- PSO is computationally expensive (50 LSTM trainings per optimization)
- Re-optimizing at every step would make walk-forward infeasible
- Hyperparameters represent **model architecture preferences**, which should be stable across similar market regimes
- Re-optimization would risk **overfitting to recent volatility**

### 4.3 XGBoost

| Aspect | Specification |
|--------|--------------|
| **Representation** | Flattened sequence: (samples, 20 × F) |
| **Training data** | Expanding window train set |
| **Validation** | 10% chronological validation for early stopping |
| **Early stopping** | Patience=10, monitor validation RMSE |
| **Hyperparameters** | Grid search on first window, fixed thereafter |
| **Search space** | max_depth: [3, 6, 9], learning_rate: [0.01, 0.05, 0.1], n_estimators: [100, 500, 1000] |
| **Subsample** | 0.8 |
| **Colsample_bytree** | 0.8 |
| **Objective** | reg:squarederror |
| **Retraining** | Full retrain from scratch at each walk-forward step |
| **Feature importance** | Computed post-hoc for analysis only (not used for selection) |

**Why Grid Search (Not PSO) for XGBoost?**
- XGBoost trains much faster than LSTM; grid search is feasible
- Maintains consistency with Zeng et al. (2025) comparison methodology
- Simpler hyperparameter space (tree depth, learning rate, estimators)

---

## 5. FEATURE PIPELINE CONSISTENCY ACROSS MODELS

### 5.1 Pipeline Execution at Each Walk-Forward Step

```
For each walk-forward step:
    1. Define expanding train_data and current test_data
    2. Compute technical indicators on FULL data (causal only)
    3. Apply DWT denoising:
        - Fit threshold on train_data only
        - Transform train_data and test_data
    4. Run FeatureSelector.fit_transform():
        - Fit ALL statistics on train_data only
        - Apply selection mask to train_data and test_data
    5. Apply MinMaxScaler:
        - Fit on train_data selected features
        - Transform train_data and test_data
    6. Window sequences (L=20) for LSTM
    7. Flatten sequences for XGBoost
    8. Train model on processed train_data
    9. Predict on processed test_data
```

### 5.2 Critical Consistency Rules

| Rule | Enforcement |
|------|-------------|
| **Feature selection mask** | Computed on train at step 0, **re-computed at each step** on expanding train |
| **Scaler parameters** | Fit on train at each step; **different scaler per step** |
| **Wavelet threshold** | Estimated on train at each step; **different threshold per step** |
| **Indicator parameters** | Fixed (EMA periods, etc.); **never data-dependent** |
| **Target definition** | `y[t] = (Close_{t+1} - Close_t) / Close_t` for **all models** |
| **Window length** | L=20 for **all models** (LSTM native, XGBoost flattened) |

### 5.3 Why Re-fit Pipeline at Each Step?

- **Expanding window** means distribution shifts over time
- Scaler min/max from 2008 data is invalid for 2015 predictions
- Feature correlations change across market regimes
- Re-fitting ensures **locally adaptive normalization** without leakage

---

## 6. BACKTESTING PROTOCOL

### 6.1 Simulation Objective

Backtesting simulates **real-time trading execution** where:
- At time t, only information ≤ t is available
- Position is held based on prediction for t+1
- Returns are computed using actual realized prices

### 6.2 Signal Generation

| Model Output | Signal Conversion |
|-------------|-------------------|
| LSTM/XGBoost predicts `ŷ[t] = return_{t+1}` | **Directional signal**: sign(ŷ[t]) |
| | If ŷ[t] > 0: Long (+1 position) |
| | If ŷ[t] < 0: Short (-1 position) |
| | If ŷ[t] == 0: Flat (0 position) — rare |

**Alternative (Confidence-Weighted):**
```
position_size = tanh(k × ŷ[t])  # k = scaling factor (e.g., 10)
```
- Limits position size to [-1, 1]
- Larger predicted returns → larger positions
- Requires calibration of k on validation data (not test)

### 6.3 Position Sizing

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Fixed unit** | ±1 share per signal | Baseline comparison |
| **Normalized** | Position = sign(ŷ[t]) / σ_recent | Volatility targeting |
| **Kelly fraction** | f = μ/σ² (simplified) | Theoretically optimal, requires variance estimate |

**Default for Fair Comparison:** Fixed unit position (±1) across all models. No leverage, no volatility scaling.

### 6.4 Transaction Cost Assumption

| Cost Component | Value | Application |
|---------------|-------|-------------|
| Commission | 0.1% per trade | Applied to absolute position change |
| Slippage | 0.05% per trade | Applied to entry/exit |
| **Total round-trip** | **0.3%** | Industry-standard baseline |

**Cost Application:**
```
net_return = raw_return − (commission + slippage) × |position_change|
```

### 6.5 Performance Metrics (Backtest)

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Cumulative Return** | `Π(1 + r_t) - 1` | Total strategy performance |
| **Annualized Return** | `(1 + total_return)^(252/n_days) - 1` | Return per year |
| **Annualized Volatility** | `std(daily_returns) × √252` | Risk per year |
| **Sharpe Ratio** | `(annualized_return − r_f) / annualized_volatility` | Risk-adjusted return (r_f = 0 for simplicity) |
| **Maximum Drawdown** | `max_t (peak_t − current_t) / peak_t` | Worst peak-to-trough decline |
| **Win Rate** | `count(r_t > 0) / total_trades` | Directional accuracy |
| **Profit Factor** | `sum(gains) / sum(\|losses\|)` | Gross profit / gross loss |
| **Calmar Ratio** | `annualized_return / max_drawdown` | Return per unit of drawdown |

### 6.6 Benchmark Comparison

All models evaluated against:

| Benchmark | Description |
|-----------|-------------|
| **Buy & Hold** | Long position from day 1, never trade |
| **Random Walk** | Prediction = last observed return (naive forecast) |
| **Moving Average** | Trade based on price vs. 20-day MA crossover |

**Fair Comparison Rules:**
- Same transaction costs applied to all strategies
- Same evaluation period (test set only)
- Same position sizing methodology
- Metrics computed on **out-of-sample** predictions only

---

## 7. METRIC AGGREGATION OVER WALK-FORWARD STEPS

### 7.1 Per-Step Metrics

At each walk-forward step, compute:
- RMSE, MAE, MAPE, R² (prediction accuracy)
- Directional accuracy (sign match)
- Cumulative return (trading performance)

### 7.2 Aggregation Methods

| Aggregation | Method | Use Case |
|-------------|--------|----------|
| **Simple Average** | `mean(metric across all steps)` | Stable metrics (RMSE, MAE) |
| **Weighted Average** | Weight by step sample size | Unequal step sizes |
| **Cumulative** | Concatenate all predictions, then compute | Final reported metrics |
| **Rolling** | 252-day rolling window | Regime analysis |

**Canonical Aggregation:** Concatenate all out-of-sample predictions into a single series, then compute metrics once. This matches Zeng et al. (2025) evaluation methodology.

### 7.3 Statistical Significance

| Test | Application |
|------|-------------|
| **Diebold-Mariano** | Compare forecast accuracy between two models |
| **t-test (paired)** | Significance of return difference vs. benchmark |
| **Sharpe ratio difference** | Risk-adjusted performance comparison |

---

## 8. COMPLETE WORKFLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INITIAL SETUP (Step 0)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │  Baseline   │  │    PSO      │  │   XGBoost   │                         │
│  │   LSTM      │  │  Optimize   │  │ Grid Search │                         │
│  │  (fixed)    │  │  (50 iter)  │  │  (search)   │                         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                         │
│         │                │                │                                   │
│         ▼                ▼                ▼                                   │
│    params_baseline   params_pso      params_xgb                              │
│    (default YAML)    (optimized)    (searched)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WALK-FORWARD LOOP (Steps 1..N)                           │
│                                                                              │
│  For each step:                                                              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1. EXPANDING WINDOW DEFINITION                                     │    │
│  │     train_data = data[0 : current_idx]                              │    │
│  │     test_data  = data[current_idx : current_idx + step_size]        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  2. FEATURE PIPELINE (fit on train, transform both)                 │    │
│  │     → Indicators (causal)                                           │    │
│  │     → DWT denoising (threshold from train)                          │    │
│  │     → Feature selection (mask from train)                           │    │
│  │     → MinMax scaling (params from train)                            │    │
│  │     → Windowing (L=20)                                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3. MODEL TRAINING (full retrain, fixed hyperparameters)            │    │
│  │     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              │    │
│  │     │ Baseline LSTM│ │  PSO-LSTM    │ │   XGBoost    │              │    │
│  │     │ params_baseline│ │ params_pso   │ │ params_xgb   │              │    │
│  │     │ epochs=100   │ │ epochs=*     │ │ n_est=*      │              │    │
│  │     │ shuffle=False│ │ shuffle=False│ │              │              │    │
│  │     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  4. PREDICTION & BACKTEST                                           │    │
│  │     → Predict returns for test_data                                 │    │
│  │     → Convert to trading signals (sign)                             │    │
│  │     → Apply transaction costs (0.3% round-trip)                     │    │
│  │     → Compute step metrics (RMSE, return, Sharpe)                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  5. ADVANCE POINTER                                                 │    │
│  │     current_idx += step_size                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼ (loop until end of data)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FINAL EVALUATION (After Loop)                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  6. AGGREGATE RESULTS                                               │    │
│  │     → Concatenate all out-of-sample predictions                     │    │
│  │     → Compute final metrics (RMSE, MAE, MAPE, R²)                   │    │
│  │     → Compute trading performance (return, Sharpe, max drawdown)    │    │
│  │     → Statistical tests (Diebold-Mariano, t-test)                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  7. REPORT & COMPARE                                                │    │
│  │     ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐    │    │
│  │     │  Baseline  │ │   PSO      │ │  XGBoost   │ │  Buy &     │    │    │
│  │     │   LSTM     │ │  LSTM      │ │            │ │   Hold     │    │    │
│  │     │            │ │  ★ BEST    │ │            │ │            │    │    │
│  │     └────────────┘ └────────────┘ └────────────┘ └────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. SUMMARY TABLE: MODEL COMPARISON PROTOCOL

| Dimension | Baseline LSTM | PSO-LSTM | XGBoost |
|-----------|--------------|----------|---------|
| **Architecture** | Fixed 128/64 | PSO-optimized | Tree ensemble |
| **Feature input** | 3D (N,20,F) | 3D (N,20,F) | 2D (N,20×F) |
| **Hyperparameter source** | YAML defaults | PSO (Step 0 only) | Grid search (Step 0 only) |
| **Retraining** | Full retrain each step | Full retrain each step | Full retrain each step |
| **Incremental update** | ❌ No | ❌ No | ❌ No |
| **Shuffle** | ❌ No | ❌ No | N/A (trees) |
| **Validation use** | Early stopping | Early stopping + PSO fitness | Early stopping + grid search |
| **PSO re-optimization** | N/A | ❌ No (fixed after Step 0) | N/A |
| **Feature pipeline** | Re-fit per step | Re-fit per step | Re-fit per step |
| **Signal generation** | sign(ŷ) | sign(ŷ) | sign(ŷ) |
| **Position sizing** | Fixed unit | Fixed unit | Fixed unit |
| **Transaction costs** | 0.3% round-trip | 0.3% round-trip | 0.3% round-trip |
| **Benchmark** | Buy & Hold | Buy & Hold | Buy & Hold |
| **Final metric** | Cumulative OOS | Cumulative OOS | Cumulative OOS |

---

**Key Takeaway:** All three models follow identical walk-forward and backtesting protocols. The only differences are (1) architecture, (2) hyperparameter source, and (3) feature representation. This ensures **fair, unbiased comparison** where performance differences are attributable to model design, not methodological variation.