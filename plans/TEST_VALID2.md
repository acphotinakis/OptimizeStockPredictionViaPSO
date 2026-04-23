I'll define a comprehensive, production-grade validation and backtesting protocol for all three models. This specification ensures strict temporal integrity, fair comparison, and realistic trading simulation.

---

# VALIDATION, WALK-FORWARD & BACKTESTING PROTOCOL
## Quantitative Finance System — LSTM & XGBoost

**Version:** 1.0 FINAL  
**Classification:** Production Evaluation Specification

---

## 1. EXECUTIVE OVERVIEW

This document defines the **canonical evaluation framework** for three models:

| Model | Training | Hyperparameters | Architecture |
|-------|----------|-----------------|--------------|
| **Baseline LSTM** | Single fit on initial train | Fixed (YAML baseline) | 2-layer LSTM, fixed units |
| **PSO-Optimized LSTM** | Single fit on initial train | PSO-optimized once | 2-layer LSTM, PSO-tuned units |
| **XGBoost** | Single fit on initial train | Grid-search or default | Gradient boosting trees |

**Core Principle:** All models share **identical feature pipelines** and **identical temporal splits**. The only variation is the model architecture and hyperparameter source.

---

## 2. TEMPORAL SPLIT ARCHITECTURE

### 2.1 Canonical Split Structure

```
Timeline:  [============================================================>]
           |<--- Train --->|<-- Val -->|<------- Test/Walk-Forward ----->|
           |     70%      |    10%    |             20%                 |
           
           T_0          T_train_end  T_val_end                      T_final
```

**Split Rules (MANDATORY):**
- **Train:** 70% earliest data — used for feature fitting, model training, PSO optimization
- **Validation:** 10% next chronological block — used for early stopping, PSO fitness evaluation
- **Test/Walk-Forward:** 20% most recent data — **never touched during training or hyperparameter search**

### 2.2 Walk-Forward Window Configuration

```
Walk-Forward Test Period (20% of data)
├─ Window 1:  [T_val_end : T_val_end + w]     → Predict T_val_end + w + 1
├─ Window 2:  [T_val_end + 1 : T_val_end + w + 1] → Predict T_val_end + w + 2
├─ Window 3:  [T_val_end + 2 : T_val_end + w + 2] → Predict T_val_end + w + 3
├─ ...
└─ Window N:  [... : T_final - 1]             → Predict T_final
```

Where:
- `w` = look-back window = **20 days** (fixed)
- Each window contains exactly 20 timesteps of features
- Prediction is always **1-step ahead** (next-day return)
- Windows advance by **1 day** (rolling, not expanding)

---

## 3. MODEL-SPECIFIC TRAINING & VALIDATION PROTOCOLS

### 3.1 BASELINE LSTM

#### Training Phase
```
Input:  X_train (70%), y_train (70%)
Val:    X_val (10%), y_val (10%)
Config: Fixed YAML baseline
```

| Parameter | Value | Source |
|-----------|-------|--------|
| lstm_units_1 | 128 | Baseline YAML |
| lstm_units_2 | 64 | Baseline YAML |
| dropout_rate | 0.2 | Baseline YAML |
| learning_rate | 0.001 | Baseline YAML |
| batch_size | 32 | Baseline YAML |
| epochs | 100 | Baseline YAML (not PSO range) |
| early_stopping_patience | 10 | Fixed |
| shuffle | FALSE | Mandatory |

**Procedure:**
1. Build model with fixed architecture
2. Train on `X_train`, validate on `X_val`
3. Early stopping monitors `val_loss`
4. Restore best weights at end
5. **Model is frozen after this single training run**

#### Walk-Forward Evaluation
```
FOR each day t in Test Period:
    features_t = X_test[t - 20 : t, :]    # Last 20 days
    prediction_t = model.predict(features_t)  # Single forward pass
    actual_t = y_test[t]                    # Next-day return
    
    # NO retraining
    # NO weight updates
    # Model remains in eval mode throughout
```

**Retraining Policy:** **NEVER** — Baseline LSTM is trained once and evaluated statically on the entire test period.

---

### 3.2 PSO-OPTIMIZED LSTM

#### Phase 1: PSO Hyperparameter Search (Training Data Only)
```
Search Space:
    epochs ∈ [50, 300]
    lstm_units_1 ∈ [50, 300]
    lstm_units_2 ∈ [20, 200]
    learning_rate ∈ [0.001, 0.01]
    
PSO Validation: Uses X_val (10%) for fitness evaluation
PSO Test Access: PROHIBITED
```

**IPSO Configuration:**
- Particles: 20
- Iterations: 50
- Inertia: Adaptive tanh schedule
- Mutation: Adaptive factor

**Procedure:**
1. PSO proposes hyperparameter particle
2. Build LSTM with proposed config
3. Train on `X_train`, early-stop on `X_val`
4. Return validation MSE as fitness
5. IPSO updates particle positions
6. After convergence: select global best hyperparameters

#### Phase 2: Final Model Training
```
Optimal Config: {epochs*, units_1*, units_2*, lr*} from PSO
Training Data:  X_train + X_val (combined 80%)
Validation:     None (use all data for final fit, early stopping disabled)
```

**Rationale:** Once optimal hyperparameters are found, retrain on full 80% with exact epoch count from PSO (no early stopping) to maximize data utilization.

#### Walk-Forward Evaluation
```
Identical to Baseline LSTM:
    FOR each day t in Test Period:
        features_t = X_test[t - 20 : t, :]
        prediction_t = model.predict(features_t)
        actual_t = y_test[t]
        
    # NO retraining at each step
    # NO incremental updates
    # Model frozen after single Phase 2 training
```

**Retraining Policy:** **NEVER** during walk-forward. PSO-optimized LSTM is trained once (Phase 2) and evaluated statically.

---

### 3.3 XGBOOST MODEL

#### Training Phase
```
Input:  X_train_flat (70%), y_train (70%)
Val:    X_val_flat (10%), y_val (10%)
```

**Feature Representation (Lag-Based, NOT Flattened Sequences):**
```
For each sample at time t:
    Features = [
        # Current values (t-1)
        Open_{t-1}, High_{t-1}, Low_{t-1}, Close_{t-1}, Volume_{t-1},
        EMA12_{t-1}, EMA25_{t-1}, ...,
        
        # Lagged values (t-2, ..., t-20)
        Open_{t-2}, ..., Open_{t-20},
        High_{t-2}, ..., High_{t-20},
        ...
    ]
```

**Hyperparameters (Fixed or Grid-Searched):**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_depth | 6 | Prevent overfitting |
| learning_rate | 0.05 | Conservative |
| n_estimators | 500 | Early stopping determines final |
| subsample | 0.8 | Row sampling |
| colsample_bytree | 0.8 | Feature sampling |
| objective | reg:squarederror | MSE regression |

**Procedure:**
1. Flatten features to tabular format (if using sequence data, use lag representation)
2. Train on `X_train_flat`, validate on `X_val_flat`
3. Early stopping: patience=50 rounds, monitor RMSE
4. **Model is frozen after this single training run**

#### Walk-Forward Evaluation
```
FOR each day t in Test Period:
    features_t = X_test_flat[t, :]          # Pre-computed lag features
    prediction_t = model.predict(features_t)  # Single inference
    actual_t = y_test[t]
    
    # NO retraining
    # Trees are static after initial fit
```

**Retraining Policy:** **NEVER** — XGBoost is trained once and evaluated statically.

---

## 4. WALK-FORWARD VALIDATION (DETAILED)

### 4.1 Window Mechanics

```
Test Period: Days T_val_end+1 through T_final

For prediction at day t (where t > T_val_end + 20):
    
    Feature Window: [t-20, t-1]  (exactly 20 days)
    
    Target at t:    y[t] = (Close_t - Close_{t-1}) / Close_{t-1}
    
    Prediction:     ŷ[t] = model.predict(X[t-20:t, :])
    
    Error:          e[t] = y[t] - ŷ[t]
```

### 4.2 Critical Constraints

| Constraint | Rule | Violation |
|------------|------|-----------|
| No future features | `X[t]` uses only data ≤ `t-1` | Leakage |
| No future targets | `y[t]` is return from `t-1` to `t` | Leakage |
| Fixed model | Weights frozen after initial training | Data snooping |
| No look-ahead | Predictions generated sequentially | Temporal integrity |

### 4.3 Rolling vs. Expanding Windows

**DECISION: Fixed Rolling Window (NOT Expanding)**

| Aspect | Rolling Window | Expanding Window |
|--------|---------------|------------------|
| Feature window | Fixed at 20 days | Grows over time |
| Model | Static (trained once) | Would require retraining |
| Memory | Constant | Increasing |
| Stationarity | Better (recent data only) | Degrades (old data dominates) |

**Rationale:** Rolling window maintains constant memory and stationarity. Expanding windows would require periodic retraining, which violates the "train once, evaluate statically" principle for fair comparison.

---

## 5. BACKTESTING PROTOCOL

### 5.1 Prediction-to-Signal Conversion

```
Raw Prediction: ŷ[t] = predicted next-day return

Signal Generation:
    IF ŷ[t] > +threshold  →  LONG signal (+1)
    IF ŷ[t] < -threshold  →  SHORT signal (-1)
    ELSE                  →  NEUTRAL (0)
    
Default threshold: 0.0 (directional only)
Alternative: 0.001 (0.1% minimum predicted return)
```

### 5.2 Position Sizing (Equal Weight Baseline)

```
Position Size: Fixed fractional (e.g., 100% of available capital)

For each day t with active signal:
    If LONG:  Invest 100% in asset
    If SHORT: Invest -100% (short sell, if allowed)
    If NEUTRAL: 0% (cash)
```

**No leverage, no dynamic position sizing** for baseline comparison. Advanced sizing (Kelly, volatility targeting) is an extension.

### 5.3 Transaction Cost Assumption

```
Transaction Cost: 0.1% per trade (10 bps)
Slippage: 0.05% per trade (5 bps)
Total Round-Trip Cost: 0.3% (entry + exit)

Applied on every position change:
    If signal changes from LONG → SHORT: pay 0.3%
    If signal changes from LONG → NEUTRAL: pay 0.15%
    If signal stays LONG: no cost
```

### 5.4 Backtest Simulation Loop

```
Initialize: capital = 1.0, position = 0, costs = 0

FOR each day t in Test Period:
    
    # Step 1: Generate prediction (before market open)
    ŷ[t] = model.predict(X[t-20:t, :])
    
    # Step 2: Convert to signal
    signal[t] = sign(ŷ[t])  # -1, 0, +1
    
    # Step 3: Execute position change (if any)
    IF signal[t] != position[t-1]:
        trade_cost = |signal[t] - position[t-1]| * 0.0015  # Half round-trip
        costs += trade_cost
        position[t] = signal[t]
    ELSE:
        position[t] = position[t-1]
    
    # Step 4: Compute strategy return
    strategy_return[t] = position[t-1] * actual_return[t] - trade_cost
    
    # Step 5: Update capital
    capital[t] = capital[t-1] * (1 + strategy_return[t])

END FOR
```

---

## 6. PERFORMANCE METRICS

### 6.1 Statistical Metrics (Model-Level)

| Metric | Formula | Purpose |
|--------|---------|---------|
| MSE | `mean((y - ŷ)^2)` | Primary loss (PSO fitness) |
| RMSE | `sqrt(MSE)` | Scale-sensitive error |
| MAE | `mean(\|y - ŷ\|)` | Robust error measure |
| MAPE | `mean(\|y - ŷ\| / \|y\|)` | Percentage error |
| R² | `1 - SS_res/SS_tot` | Variance explained |
| Directional Accuracy | `mean(sign(y) == sign(ŷ))` | Correct sign prediction |

### 6.2 Financial Metrics (Strategy-Level)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Total Return | `capital_final - 1` | Cumulative profit |
| Annualized Return | `(1 + total_return)^(252/n_days) - 1` | Return per year |
| Volatility | `std(strategy_return) * sqrt(252)` | Risk per year |
| Sharpe Ratio | `(ann_return - risk_free) / ann_volatility` | Risk-adjusted return |
| Max Drawdown | `max_t (peak_t - capital_t) / peak_t` | Worst peak-to-trough |
| Calmar Ratio | `ann_return / max_drawdown` | Return per unit drawdown |
| Win Rate | `count(strategy_return > 0) / total_trades` | % profitable trades |
| Profit Factor | `sum(gains) / sum(\|losses\|)` | Gross profit / gross loss |

### 6.3 Benchmark Comparison

```
Benchmark: Buy-and-Hold (always long, no trading costs)

Metrics computed for both strategy and benchmark:
    - Total Return
    - Sharpe Ratio
    - Max Drawdown
    
Excess Return = Strategy Return - Benchmark Return
Information Ratio = Excess Return / Tracking Error
```

---

## 7. RETRAINING & UPDATE POLICY

### 7.1 Canonical Protocol: NO RETRAINING

All three models follow **static evaluation**:

| Model | Training Events | Update Policy |
|-------|----------------|---------------|
| Baseline LSTM | 1 (initial fit) | None |
| PSO LSTM | 2 (PSO search + final fit) | None |
| XGBoost | 1 (initial fit) | None |

**Rationale:** 
- Ensures fair comparison (same information access)
- Prevents data leakage from future information
- Simulates "deploy once" production scenario
- Avoids look-ahead bias from rolling retraining

### 7.2 Alternative: Periodic Retraining (Extension Only)

If business requirements mandate model refresh:

```
Retraining Trigger: Every 252 trading days (1 year)

Procedure:
    1. Expand training window: Train + Val + first year of Test
    2. Refit feature pipeline (scaler, selector) on expanded train
    3. Retrain model on expanded train
    4. Evaluate on subsequent year only
    
Constraint: Retraining uses ONLY data available up to retrain point
```

**This is NOT the canonical protocol and must be explicitly documented as an extension.**

---

## 8. FEATURE PIPELINE CONSISTENCY

### 8.1 Shared Pipeline (All Models)

```
Stage 1: Raw OHLCV Ingestion
    ↓
Stage 2: Data Cleaning (causal forward-fill, range validation)
    ↓
Stage 3: Feature Generation (technical indicators, causal only)
    ↓
Stage 4: Feature Transformation
    ├── Wavelet Denoising (threshold from train only)
    └── MinMax Normalization [-1, 1] (params from train only)
    ↓
Stage 5: Feature Selection (correlation/VIF/MI on train only)
    ↓
Stage 6: Windowing
    ├── LSTM: (N, 20, F) tensor
    └── XGBoost: Lag-based tabular features
```

### 8.2 Pipeline State Persistence

```python
pipeline_state = {
    "wavelet": {
        "threshold": float,      # From train
        "wavelet": "haar",
        "level": 3,
        "mode": "symmetric"
    },
    "scaler": {
        col: {"min": float, "max": float}  # From train per column
    },
    "selector": {
        "selected_features": [...],         # From train
        "dropped_features": {...},
        "thresholds": {...}
    }
}
```

**Rule:** `pipeline_state` is computed **once** on the initial 70% training data and **frozen** for all subsequent validation and test processing. No refitting, no adaptation.

---

## 9. FAIR COMPARISON FRAMEWORK

### 9.1 Controlled Variables

| Aspect | Fixed Value | Variation |
|--------|-------------|-----------|
| Feature pipeline | Shared | None |
| Temporal splits | 70/10/20 chronological | None |
| Target variable | Next-day return | None |
| Look-back window | 20 days | None |
| Walk-forward step | 1 day | None |
| Transaction costs | 0.1% + 0.05% slippage | None |
| Seed | 42 | None |
| Evaluation period | Test set only | None |

### 9.2 Independent Variables

| Aspect | Baseline LSTM | PSO LSTM | XGBoost |
|--------|--------------|----------|---------|
| Architecture | 2-layer LSTM | 2-layer LSTM | Gradient boosting |
| Units/layers | Fixed (128, 64) | PSO-optimized | Tree-based |
| Dropout | 0.2 | PSO-optimized | N/A (subsample) |
| Learning rate | 0.001 | PSO-optimized | 0.05 |
| Batch size | 32 | PSO-optimized | N/A |
| Epochs | 100 | PSO-optimized | Early stopping |
| Hyperparameter source | YAML file | IPSO search | Grid search or default |

### 9.3 Evaluation Order

```
1. Run shared feature pipeline → produce X_train, X_val, X_test
2. Train Baseline LSTM → evaluate on walk-forward test
3. Run PSO optimization → find optimal hyperparameters
4. Train PSO LSTM (final fit) → evaluate on walk-forward test
5. Train XGBoost → evaluate on walk-forward test
6. Compute all metrics for all three models
7. Statistical significance testing (if applicable)
```

---

## 10. BACKTEST OUTPUT SPECIFICATION

### 10.1 Daily Log

```python
backtest_log = pd.DataFrame({
    "date": [...],                    # Timestamp
    "actual_return": [...],           # y[t]
    "predicted_return": [...],        # ŷ[t]
    "signal": [...],                  # -1, 0, +1
    "position": [...],                # -1, 0, +1 (after execution)
    "strategy_return": [...],         # Net of costs
    "cumulative_return": [...],       # Compounded
    "trade_cost": [...],              # Transaction cost paid
    "capital": [...]                  # Portfolio value
})
```

### 10.2 Summary Report

```python
evaluation_report = {
    "model_name": str,
    "hyperparameters": dict,
    "pipeline_state": dict,
    
    "statistical_metrics": {
        "mse": float,
        "rmse": float,
        "mae": float,
        "mape": float,
        "r2": float,
        "directional_accuracy": float
    },
    
    "financial_metrics": {
        "total_return": float,
        "annualized_return": float,
        "annualized_volatility": float,
        "sharpe_ratio": float,
        "max_drawdown": float,
        "calmar_ratio": float,
        "win_rate": float,
        "profit_factor": float
    },
    
    "benchmark_comparison": {
        "benchmark_return": float,
        "excess_return": float,
        "information_ratio": float
    },
    
    "trade_statistics": {
        "total_trades": int,
        "avg_trade_return": float,
        "avg_win": float,
        "avg_loss": float,
        "largest_win": float,
        "largest_loss": float
    }
}
```

---

## 11. FAILURE CONDITIONS

Any of the following invalidates the evaluation:

| Violation | Consequence |
|-----------|-------------|
| Test data used in training | Data leakage, invalid results |
| Feature pipeline refit on test | Leakage, invalid results |
| Model retrained during walk-forward | Data snooping, invalid results |
| Shuffling enabled in training | Temporal leakage |
| Different splits across models | Unfair comparison |
| Transaction costs omitted | Unrealistic performance |
| Look-ahead in signal generation | Future information leak |

---

## 12. SUMMARY TABLE: MODEL BEHAVIOR MATRIX

| Aspect | Baseline LSTM | PSO LSTM | XGBoost |
|--------|--------------|----------|---------|
| **Training Data** | 70% train | 70% train (PSO) + 80% train+val (final) | 70% train |
| **Validation Data** | 10% val | 10% val (PSO fitness) | 10% val |
| **Test Data** | 20% walk-forward | 20% walk-forward | 20% walk-forward |
| **Hyperparameters** | Fixed YAML | IPSO-optimized once | Fixed or grid search |
| **Retraining** | Never | Never | Never |
| **Window Type** | Rolling (20-day) | Rolling (20-day) | Lag features (20-day) |
| **Prediction** | 1-step ahead | 1-step ahead | 1-step ahead |
| **Signal** | sign(ŷ) | sign(ŷ) | sign(ŷ) |
| **Position Size** | Equal weight | Equal weight | Equal weight |
| **Costs** | 0.15% per trade | 0.15% per trade | 0.15% per trade |
| **Metrics** | MSE, Sharpe, Drawdown | MSE, Sharpe, Drawdown | MSE, Sharpe, Drawdown |

---

**END OF VALIDATION & BACKTESTING PROTOCOL**

*This specification ensures deterministic, leakage-free, and fair evaluation across all three models in a production quantitative trading system.*