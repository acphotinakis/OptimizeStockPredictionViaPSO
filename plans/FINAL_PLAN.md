# UNIFIED VALIDATION & BACKTESTING PROTOCOL
## PSO-LSTM Stock Prediction System — CANONICAL SPECIFICATION v1.0

**Date:** April 21, 2026  
**Status:** PRODUCTION AUTHORITATIVE  
**Supersedes:** TEST_VALID1.md, TEST_VALID2.md

---

## EXECUTIVE SUMMARY

This document defines the **single canonical evaluation framework** for the PSO-LSTM stock prediction system. All conflicts between prior specifications have been resolved. This protocol is **mandatory** for all model evaluation, walk-forward validation, and backtesting.

### Core Design Decisions

| Dimension | Canonical Choice | Rationale |
|-----------|-----------------|-----------|
| **Retraining Policy** | Static (train once) | Fair comparison, computational feasibility, "deploy once" realism |
| **Window Strategy** | Rolling 20-day | Maintains stationarity, constant memory, no retraining required |
| **Data Splits** | 70% train / 10% val / 20% test | Standard ML practice, sufficient test coverage |
| **Feature Pipeline** | Fit once, freeze forever | Prevents leakage, ensures reproducibility |
| **PSO Protocol** | Two-phase (search + final fit) | Maximizes data utilization after hyperparameter search |

---

## 1. CONFLICT RESOLUTION ANALYSIS

### 1.1 Conflict 1: Retraining Policy

**TEST_VALID1 Position:** Models retrained at every walk-forward step with expanding windows.

**TEST_VALID2 Position:** Models trained once and frozen for entire evaluation period.

**CANONICAL RESOLUTION:** **Static evaluation (TEST_VALID2 approach)**

**Rationale:**
1. **Fair Comparison:** All models must have identical information access. Retraining creates asymmetry if hyperparameters aren't re-optimized (which would be computationally infeasible for PSO).
2. **Production Realism:** "Deploy once and monitor" is a realistic production scenario. Continuous retraining requires separate deployment infrastructure and governance.
3. **Computational Feasibility:** Retraining PSO-LSTM at every step would require 50 LSTM trainings per step, making walk-forward evaluation infeasible.
4. **Methodological Clarity:** Static evaluation isolates model quality from retraining schedule decisions.

**Implementation:** Models are trained once on the initial training set and frozen. Walk-forward evaluation uses the frozen model for all predictions.

---

### 1.2 Conflict 2: Data Split Ratios

**TEST_VALID1 Position:** 80/10/20 (Baseline/XGBoost), 72/8/20 (PSO-LSTM)

**TEST_VALID2 Position:** 70/10/20 (all models)

**CANONICAL RESOLUTION:** **70/10/20 for all models (TEST_VALID2 approach)**

**Rationale:**
1. **Consistency:** All models must use identical splits for fair comparison.
2. **Standard Practice:** 70/10/20 is widely recognized in ML literature.
3. **Test Coverage:** 20% test set provides sufficient statistical power for walk-forward evaluation.
4. **PSO Simplification:** Using the same split for PSO as other models simplifies implementation and ensures no advantage from larger training sets.

**Implementation:**
```
[============ 70% Train ============][== 10% Val ==][====== 20% Test ======]
T_0                            T_train_end     T_val_end              T_final
```

All models use this exact split. No exceptions.

---

### 1.3 Conflict 3: Window Strategy

**TEST_VALID1 Position:** Expanding window (train window grows at each step)

**TEST_VALID2 Position:** Rolling 20-day window (fixed size)

**CANONICAL RESOLUTION:** **Rolling 20-day window (TEST_VALID2 approach)**

**Rationale:**
1. **Stationarity:** Rolling windows maintain local stationarity by not including ancient data.
2. **Constant Memory:** Fixed window size ensures consistent model input dimensionality.
3. **Consistency with Static Models:** Expanding windows logically require retraining; rolling windows are compatible with frozen models.
4. **Look-back Alignment:** 20-day window matches the LSTM architecture input requirement and technical indicator calculation periods.

**Implementation:**
```
For prediction at time t:
    Input: X[t-20:t, :]  (exactly 20 timesteps)
    Target: y[t] = (Close_t - Close_{t-1}) / Close_{t-1}
    Prediction: ŷ[t] = model.predict(X[t-20:t, :])
```

The 20-day window slides forward one day at a time. No expansion.

---

### 1.4 Conflict 4: Feature Pipeline Behavior

**TEST_VALID1 Position:** Re-fit scalers, selectors, and wavelet thresholds at each step

**TEST_VALID2 Position:** Fit once on training data, freeze forever

**CANONICAL RESOLUTION:** **Fit once and freeze (TEST_VALID2 approach)**

**Rationale:**
1. **Leakage Prevention:** Re-fitting on expanding windows would use future information not available at deployment.
2. **Reproducibility:** Frozen pipeline state ensures deterministic, reproducible results.
3. **Consistency with Static Models:** If models aren't retrained, pipeline shouldn't be refitted either.
4. **Production Realism:** Deployed models use fixed normalization parameters computed at training time.

**Implementation:**
```python
# Phase 1: Fit pipeline ONCE on 70% train data
pipeline_state = {
    "wavelet_threshold": fit_wavelet_threshold(X_train),
    "scaler_params": fit_minmax_scaler(X_train),
    "selected_features": fit_feature_selector(X_train),
}

# Phase 2: Transform all splits using FROZEN state
X_train_transformed = transform(X_train, pipeline_state)
X_val_transformed = transform(X_val, pipeline_state)
X_test_transformed = transform(X_test, pipeline_state)  # Same pipeline_state!

# Phase 3: Walk-forward uses same pipeline_state forever
for t in test_period:
    X_t = transform(X_raw[t-20:t], pipeline_state)  # FROZEN
    prediction = model.predict(X_t)
```

---

## 2. CANONICAL TEMPORAL SPLIT ARCHITECTURE

### 2.1 Single Universal Split

**All models use identical temporal splits:**

```
Timeline: [================================================================]
          |<------- 70% Train -------->|<-- 10% Val -->|<----- 20% Test ---->|
          T_0                      T_train_end    T_val_end               T_final
          
Split Indices (example for 1000 days):
    Train: [0, 700)
    Val:   [700, 800)
    Test:  [800, 1000]
```

### 2.2 Split Computation

```python
def compute_canonical_split(data):
    """
    Compute 70/10/20 temporal split.
    
    Returns:
        train_data: [0:70%]
        val_data:   [70%:80%]
        test_data:  [80%:100%]
    """
    n = len(data)
    train_end = int(0.70 * n)
    val_end = int(0.80 * n)
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    return train_data, val_data, test_data
```

### 2.3 Mandatory Split Properties

| Property | Rule | Violation Type |
|----------|------|---------------|
| **Chronological ordering** | Splits must be contiguous temporal blocks | Data leakage |
| **No shuffling** | Data order must be preserved | Temporal leakage |
| **No overlap** | Train/val/test must be disjoint | Data leakage |
| **Fixed boundaries** | Split points computed once, never changed | Reproducibility failure |

---

## 3. FEATURE PIPELINE SPECIFICATION

### 3.1 Pipeline Stages (Canonical Order)

```
Stage 1: Raw OHLCV Ingestion
    ↓
Stage 2: Data Cleaning
    - Forward-fill (max 5 bars)
    - Range validation
    - Outlier detection
    ↓
Stage 3: Technical Indicator Generation
    - 36 indicators (causal only)
    - Cross-ticker features (SPY, sectors, peers)
    ↓
Stage 4: Wavelet Denoising
    - 3-level Haar DWT
    - Soft thresholding
    - Threshold computed on TRAIN only
    ↓
Stage 5: Feature Selection
    - Variance threshold
    - Pearson correlation (≥95%)
    - VIF (>10)
    - Mutual Information (bottom quartile)
    - Mask computed on TRAIN only
    ↓
Stage 6: MinMax Normalization
    - Scale to [-1, 1]
    - Parameters fit on TRAIN only
    ↓
Stage 7: Windowing
    - LSTM: (N, 20, F) sequences
    - XGBoost: Lag-based tabular (N, F×21)
```

### 3.2 Pipeline Lifecycle (MANDATORY)

```python
# ====================================================================
# PHASE 1: FIT PIPELINE (EXECUTED ONCE)
# ====================================================================
def fit_pipeline(train_data):
    """
    Fit all pipeline components on training data ONLY.
    This is executed EXACTLY ONCE at the beginning.
    """
    # Stage 4: Wavelet threshold
    threshold = compute_wavelet_threshold(train_data["close"])
    
    # Stage 5: Feature selection mask
    feature_mask = fit_feature_selector(train_data)
    
    # Stage 6: Scaler parameters
    scaler_params = fit_minmax_scaler(train_data[feature_mask])
    
    # Save state
    pipeline_state = {
        "wavelet_threshold": threshold,
        "selected_features": feature_mask,
        "scaler_params": scaler_params,
        "fit_date": train_data.index[-1],
        "n_features_in": len(train_data.columns),
        "n_features_out": sum(feature_mask),
    }
    
    return pipeline_state


# ====================================================================
# PHASE 2: TRANSFORM (EXECUTED ON ALL SPLITS)
# ====================================================================
def transform_pipeline(data, pipeline_state):
    """
    Transform data using FROZEN pipeline state.
    pipeline_state MUST NOT be modified.
    """
    # Apply wavelet denoising with TRAINING threshold
    data_denoised = apply_wavelet(
        data, 
        threshold=pipeline_state["wavelet_threshold"]  # FROZEN
    )
    
    # Apply feature selection with TRAINING mask
    data_selected = data_denoised[:, pipeline_state["selected_features"]]
    
    # Apply normalization with TRAINING scaler
    data_normalized = apply_minmax(
        data_selected,
        params=pipeline_state["scaler_params"]  # FROZEN
    )
    
    return data_normalized


# ====================================================================
# CANONICAL USAGE
# ====================================================================
# Fit once
pipeline_state = fit_pipeline(train_data)

# Transform all splits with SAME state
X_train = transform_pipeline(train_data, pipeline_state)
X_val = transform_pipeline(val_data, pipeline_state)
X_test = transform_pipeline(test_data, pipeline_state)

# Walk-forward: SAME state forever
for t in range(len(test_data) - 20):
    X_t = transform_pipeline(test_data[t:t+20], pipeline_state)
    # pipeline_state NEVER changes
```

### 3.3 Forbidden Operations

| Operation | Status | Reason |
|-----------|--------|--------|
| Refit scaler on validation/test | **FORBIDDEN** | Data leakage |
| Refit feature selector on validation/test | **FORBIDDEN** | Data leakage |
| Recompute wavelet threshold on validation/test | **FORBIDDEN** | Data leakage |
| Modify pipeline_state after initial fit | **FORBIDDEN** | Reproducibility |
| Use different pipeline_state per model | **FORBIDDEN** | Unfair comparison |

---

## 4. MODEL TRAINING PROTOCOLS

### 4.1 Baseline LSTM

#### Training Phase (Execute Once)

```python
def train_baseline_lstm(X_train, y_train, X_val, y_val, config):
    """
    Train Baseline LSTM with fixed hyperparameters.
    NO hyperparameter search. NO PSO. NO grid search.
    """
    # Fixed architecture from YAML
    model = LSTMModel(
        units_1=128,           # FIXED
        units_2=64,            # FIXED
        dropout=0.2,           # FIXED
        activation="relu",     # FIXED
        learning_rate=0.001,   # FIXED
        batch_size=32,         # FIXED
    )
    
    # Train with early stopping
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        shuffle=False,  # MANDATORY
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # Model is now FROZEN
    return model, history
```

#### Walk-Forward Evaluation

```python
def evaluate_baseline_lstm(model, X_test, y_test):
    """
    Evaluate frozen model on test set.
    Model weights NEVER change.
    """
    predictions = []
    
    for t in range(20, len(X_test)):
        # Rolling 20-day window
        X_window = X_test[t-20:t, :, :]  # Shape: (20, F)
        X_input = X_window.reshape(1, 20, -1)  # Shape: (1, 20, F)
        
        # Predict with FROZEN model
        pred = model.predict(X_input, verbose=0)
        predictions.append(pred[0, 0])
    
    predictions = np.array(predictions)
    actuals = y_test[20:]
    
    return predictions, actuals
```

---

### 4.2 PSO-Optimized LSTM

#### Phase 1: PSO Hyperparameter Search

```python
def pso_optimize_lstm(X_train, y_train, X_val, y_val, config_pso):
    """
    Run IPSO to find optimal hyperparameters.
    Uses ONLY train (70%) and val (10%) data.
    Test data (20%) is NEVER accessed.
    """
    search_space = {
        "epochs": [50, 300],           # Continuous
        "units_1": [50, 300],          # Continuous (rounded to int)
        "units_2": [20, 200],          # Continuous (rounded to int)
        "learning_rate": [0.001, 0.01],  # Log scale
        "dropout": [0.0, 0.5],         # Continuous
        "batch_size": [32, 64],        # Discrete
    }
    
    # Initialize PSO
    pso = IPSO(
        n_particles=20,
        n_iterations=50,
        search_space=search_space,
        fitness_func=evaluate_lstm_fitness,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    
    # Run optimization
    best_params, best_fitness = pso.optimize()
    
    return best_params


def evaluate_lstm_fitness(params, X_train, y_train, X_val, y_val):
    """
    Fitness function for PSO.
    Train on X_train, evaluate MSE on X_val.
    """
    model = LSTMModel(**params)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        shuffle=False,  # MANDATORY
        verbose=0,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # Fitness = validation MSE
    y_pred = model.predict(X_val, verbose=0)
    mse = np.mean((y_val - y_pred) ** 2)
    
    return mse
```

#### Phase 2: Final Model Training

```python
def train_pso_lstm_final(X_train, y_train, X_val, y_val, best_params):
    """
    Train final model using PSO-optimized hyperparameters.
    
    CRITICAL: Train on COMBINED train+val (80% total) to maximize
    data utilization after hyperparameter search is complete.
    """
    # Combine train and val
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    
    # Build model with PSO parameters
    model = LSTMModel(
        units_1=best_params["units_1"],
        units_2=best_params["units_2"],
        dropout=best_params["dropout"],
        activation="relu",
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
    )
    
    # Train on combined data
    # Use exact epoch count from PSO (no early stopping in final fit)
    history = model.fit(
        X_combined, y_combined,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        shuffle=False,  # MANDATORY
        verbose=1,
    )
    
    # Model is now FROZEN
    return model, history
```

#### Walk-Forward Evaluation

```python
# IDENTICAL to Baseline LSTM
# Use evaluate_baseline_lstm() function
predictions, actuals = evaluate_baseline_lstm(pso_model, X_test, y_test)
```

---

### 4.3 XGBoost Model

#### Feature Representation (Lag-Based)

```python
def build_xgboost_features(X, lookback=20):
    """
    Convert tabular features to lag-based representation for XGBoost.
    
    Input: X (N, F) - tabular features
    Output: X_lag (N-lookback, F * (lookback+1))
    
    Each sample contains:
        [X[t], X[t-1], X[t-2], ..., X[t-20]]
    """
    N, F = X.shape
    n_samples = N - lookback
    n_features = F * (lookback + 1)
    
    X_lag = np.zeros((n_samples, n_features), dtype=np.float32)
    
    for i in range(n_samples):
        # Current time: i + lookback
        # Window: [i : i+lookback+1]
        window = X[i:i+lookback+1, :]  # (21, F)
        X_lag[i, :] = window.flatten()  # (F * 21,)
    
    return X_lag
```

#### Training Phase (Execute Once)

```python
def train_xgboost(X_train, y_train, X_val, y_val, config):
    """
    Train XGBoost with fixed or grid-searched hyperparameters.
    """
    # Convert to lag-based features
    X_train_lag = build_xgboost_features(X_train, lookback=20)
    X_val_lag = build_xgboost_features(X_val, lookback=20)
    y_train_lag = y_train[20:]  # Align with lag features
    y_val_lag = y_val[20:]
    
    # Fixed hyperparameters
    model = XGBRegressor(
        objective="reg:squarederror",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    
    # Train with early stopping
    model.fit(
        X_train_lag, y_train_lag,
        eval_set=[(X_train_lag, y_train_lag), (X_val_lag, y_val_lag)],
        early_stopping_rounds=50,
        verbose=False,
    )
    
    # Model is now FROZEN
    return model
```

#### Walk-Forward Evaluation

```python
def evaluate_xgboost(model, X_test, y_test):
    """
    Evaluate frozen XGBoost model on test set.
    Model NEVER changes.
    """
    # Convert entire test set to lag features
    X_test_lag = build_xgboost_features(X_test, lookback=20)
    y_test_lag = y_test[20:]  # Align
    
    # Predict with FROZEN model
    predictions = model.predict(X_test_lag)
    
    return predictions, y_test_lag
```

---

## 5. WALK-FORWARD EVALUATION PROTOCOL

### 5.1 Canonical Walk-Forward Algorithm

```python
def walk_forward_evaluate(model, X_test, y_test, model_type):
    """
    Walk-forward evaluation with frozen model and rolling windows.
    
    Args:
        model: Trained model (FROZEN, never modified)
        X_test: Test features (20% of data)
        y_test: Test targets
        model_type: "lstm" or "xgboost"
    
    Returns:
        predictions: Out-of-sample predictions for entire test period
        actuals: Corresponding actual returns
    """
    if model_type == "lstm":
        # LSTM expects 3D input: (batch, 20, F)
        predictions = []
        
        for t in range(20, len(X_test)):
            # Rolling 20-day window
            X_window = X_test[t-20:t, :]  # (20, F)
            X_input = X_window.reshape(1, 20, -1)  # (1, 20, F)
            
            # Predict with frozen model
            pred = model.predict(X_input, verbose=0)
            predictions.append(pred[0, 0])
        
        predictions = np.array(predictions)
        actuals = y_test[20:]
    
    elif model_type == "xgboost":
        # XGBoost uses lag-based features
        X_test_lag = build_xgboost_features(X_test, lookback=20)
        predictions = model.predict(X_test_lag)
        actuals = y_test[20:]
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return predictions, actuals
```

### 5.2 Walk-Forward Properties

| Property | Value | Rationale |
|----------|-------|-----------|
| **Window size** | 20 days (fixed) | Matches LSTM input, indicator periods |
| **Step size** | 1 day | Standard next-day prediction |
| **Window type** | Rolling (not expanding) | Maintains stationarity |
| **Model state** | Frozen (never updated) | Fair comparison, no retraining |
| **Pipeline state** | Frozen (never refitted) | Leakage prevention |
| **Prediction horizon** | 1 step ahead (t+1) | Standard financial forecasting |

### 5.3 Forbidden Operations During Walk-Forward

| Operation | Status | Enforcement |
|-----------|--------|-------------|
| Model weight updates | **FORBIDDEN** | Model in eval mode |
| Model retraining | **FORBIDDEN** | No `.fit()` calls |
| Pipeline refitting | **FORBIDDEN** | Pipeline_state frozen |
| Using future data in features | **FORBIDDEN** | X[t] uses only data ≤ t-1 |
| Shuffling data | **FORBIDDEN** | Temporal order preserved |

---

## 6. BACKTESTING PROTOCOL

### 6.1 Signal Generation

```python
def generate_signals(predictions, threshold=0.0):
    """
    Convert return predictions to trading signals.
    
    Args:
        predictions: Predicted returns (ŷ[t])
        threshold: Minimum return to trigger signal (default: 0.0)
    
    Returns:
        signals: +1 (long), -1 (short), 0 (neutral)
    """
    signals = np.zeros_like(predictions)
    signals[predictions > threshold] = 1   # LONG
    signals[predictions < -threshold] = -1  # SHORT
    # predictions == 0 → signal = 0 (NEUTRAL)
    
    return signals
```

### 6.2 Transaction Cost Application

```python
def apply_transaction_costs(signals, actual_returns):
    """
    Apply transaction costs to strategy returns.
    
    Costs:
        - Commission: 0.10% per trade
        - Slippage: 0.05% per trade
        - Total one-way: 0.15%
        - Round-trip: 0.30%
    
    Cost applied on position changes only.
    """
    position = np.zeros(len(signals) + 1)
    position[0] = 0  # Start neutral
    position[1:] = signals
    
    # Detect position changes
    position_changes = np.abs(np.diff(position))
    
    # Cost per unit change (0.15% one-way)
    trade_costs = position_changes * 0.0015
    
    # Strategy return = position × actual_return - trade_cost
    strategy_returns = signals * actual_returns - trade_costs
    
    return strategy_returns, trade_costs
```

### 6.3 Backtest Execution

```python
def run_backtest(predictions, actual_returns, initial_capital=1.0):
    """
    Execute full backtest with transaction costs.
    
    Returns:
        backtest_results: DataFrame with daily performance
    """
    # Generate signals
    signals = generate_signals(predictions, threshold=0.0)
    
    # Apply transaction costs
    strategy_returns, trade_costs = apply_transaction_costs(
        signals, actual_returns
    )
    
    # Compute cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    capital = initial_capital * cumulative_returns
    
    # Build results DataFrame
    results = pd.DataFrame({
        "prediction": predictions,
        "actual_return": actual_returns,
        "signal": signals,
        "strategy_return": strategy_returns,
        "trade_cost": trade_costs,
        "cumulative_return": cumulative_returns - 1,
        "capital": capital,
    })
    
    return results
```

### 6.4 Performance Metrics

```python
def compute_performance_metrics(backtest_results, risk_free_rate=0.0):
    """
    Compute financial performance metrics.
    
    Returns:
        metrics: Dictionary of performance statistics
    """
    returns = backtest_results["strategy_return"].values
    capital = backtest_results["capital"].values
    
    # Total return
    total_return = capital[-1] / capital[0] - 1
    
    # Annualized return (252 trading days)
    n_days = len(returns)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    
    # Volatility
    daily_vol = np.std(returns)
    annualized_vol = daily_vol * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = (annualized_return - risk_free_rate) / annualized_vol
    
    # Maximum drawdown
    peak = np.maximum.accumulate(capital)
    drawdown = (capital - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    win_rate = np.sum(returns > 0) / len(returns)
    
    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses != 0 else np.inf
    
    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": np.sum(np.diff(backtest_results["signal"].values) != 0),
    }
    
    return metrics
```

---

## 7. BENCHMARK COMPARISON

### 7.1 Buy-and-Hold Benchmark

```python
def compute_buy_and_hold_benchmark(actual_returns, initial_capital=1.0):
    """
    Compute buy-and-hold (always long) benchmark.
    No transaction costs for benchmark (holding continuously).
    """
    cumulative_returns = (1 + actual_returns).cumprod()
    capital = initial_capital * cumulative_returns
    
    # Metrics
    total_return = capital.iloc[-1] / capital.iloc[0] - 1
    n_days = len(actual_returns)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    annualized_vol = actual_returns.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol
    
    peak = capital.expanding().max()
    drawdown = (capital - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }
```

### 7.2 Model Comparison

```python
def compare_models(baseline_results, pso_results, xgb_results, benchmark):
    """
    Generate comparative performance table.
    """
    comparison = pd.DataFrame({
        "Baseline LSTM": baseline_results,
        "PSO-LSTM": pso_results,
        "XGBoost": xgb_results,
        "Buy & Hold": benchmark,
    }).T
    
    # Highlight best performer
    comparison["Rank"] = comparison["sharpe_ratio"].rank(ascending=False)
    
    return comparison
```

---

## 8. COMPLETE EVALUATION WORKFLOW

### 8.1 End-to-End Pipeline

```python
def canonical_evaluation_pipeline(data, config):
    """
    Execute complete canonical evaluation protocol.
    
    Steps:
        1. Temporal split (70/10/20)
        2. Feature pipeline (fit once, freeze)
        3. Train all models (once each)
        4. Walk-forward evaluation (frozen models)
        5. Backtesting (with transaction costs)
        6. Performance comparison
    
    Returns:
        results: Complete evaluation results for all models
    """
    # ================================================================
    # STEP 1: TEMPORAL SPLIT
    # ================================================================
    train_data, val_data, test_data = compute_canonical_split(data)
    
    logger.info(f"Train: {len(train_data)} days")
    logger.info(f"Val:   {len(val_data)} days")
    logger.info(f"Test:  {len(test_data)} days")
    
    # ================================================================
    # STEP 2: FEATURE PIPELINE (FIT ONCE)
    # ================================================================
    pipeline_state = fit_pipeline(train_data)
    
    X_train = transform_pipeline(train_data, pipeline_state)
    X_val = transform_pipeline(val_data, pipeline_state)
    X_test = transform_pipeline(test_data, pipeline_state)
    
    y_train = compute_returns(train_data["close"])
    y_val = compute_returns(val_data["close"])
    y_test = compute_returns(test_data["close"])
    
    logger.info("Feature pipeline fitted and frozen")
    logger.info(f"Features: {X_train.shape[1]}")
    
    # ================================================================
    # STEP 3: TRAIN MODELS (ONCE EACH)
    # ================================================================
    logger.info("Training Baseline LSTM...")
    baseline_model, _ = train_baseline_lstm(
        X_train, y_train, X_val, y_val, config
    )
    
    logger.info("Running PSO optimization...")
    pso_params = pso_optimize_lstm(
        X_train, y_train, X_val, y_val, config["pso"]
    )
    
    logger.info("Training PSO-LSTM (final fit)...")
    pso_model, _ = train_pso_lstm_final(
        X_train, y_train, X_val, y_val, pso_params
    )
    
    logger.info("Training XGBoost...")
    xgb_model = train_xgboost(
        X_train, y_train, X_val, y_val, config["xgboost"]
    )
    
    logger.info("All models trained and frozen")
    
    # ================================================================
    # STEP 4: WALK-FORWARD EVALUATION (FROZEN MODELS)
    # ================================================================
    logger.info("Walk-forward evaluation...")
    
    baseline_preds, actuals = walk_forward_evaluate(
        baseline_model, X_test, y_test, model_type="lstm"
    )
    
    pso_preds, _ = walk_forward_evaluate(
        pso_model, X_test, y_test, model_type="lstm"
    )
    
    xgb_preds, _ = walk_forward_evaluate(
        xgb_model, X_test, y_test, model_type="xgboost"
    )
    
    # ================================================================
    # STEP 5: BACKTESTING (WITH TRANSACTION COSTS)
    # ================================================================
    logger.info("Running backtests...")
    
    baseline_backtest = run_backtest(baseline_preds, actuals)
    pso_backtest = run_backtest(pso_preds, actuals)
    xgb_backtest = run_backtest(xgb_preds, actuals)
    
    # ================================================================
    # STEP 6: COMPUTE METRICS & COMPARISON
    # ================================================================
    logger.info("Computing performance metrics...")
    
    baseline_metrics = compute_performance_metrics(baseline_backtest)
    pso_metrics = compute_performance_metrics(pso_backtest)
    xgb_metrics = compute_performance_metrics(xgb_backtest)
    
    benchmark = compute_buy_and_hold_benchmark(actuals)
    
    comparison = compare_models(
        baseline_metrics, pso_metrics, xgb_metrics, benchmark
    )
    
    # ================================================================
    # RETURN RESULTS
    # ================================================================
    results = {
        "predictions": {
            "baseline": baseline_preds,
            "pso": pso_preds,
            "xgboost": xgb_preds,
        },
        "backtests": {
            "baseline": baseline_backtest,
            "pso": pso_backtest,
            "xgboost": xgb_backtest,
        },
        "metrics": {
            "baseline": baseline_metrics,
            "pso": pso_metrics,
            "xgboost": xgb_metrics,
            "benchmark": benchmark,
        },
        "comparison": comparison,
        "pipeline_state": pipeline_state,
    }
    
    return results
```

---

## 9. LEAKAGE PREVENTION CHECKLIST

### 9.1 Mandatory Verifications

Before reporting any results, verify ALL of the following:

- [ ] **Temporal integrity:** Train/val/test are chronological, non-overlapping blocks
- [ ] **No shuffling:** Data order preserved in all training and evaluation
- [ ] **Test isolation:** Test data never used for training, validation, or hyperparameter search
- [ ] **Pipeline freeze:** Scaler, selector, wavelet threshold fit ONLY on train, frozen forever
- [ ] **Model freeze:** Models trained once, weights never updated during walk-forward
- [ ] **Feature causality:** X[t] uses only data ≤ t-1
- [ ] **Target causality:** y[t] is return from t-1 to t (no future data)
- [ ] **Consistent splits:** All models use identical 70/10/20 split
- [ ] **Consistent pipeline:** All models use same pipeline_state
- [ ] **Transaction costs:** Applied to all strategies, not to benchmark

### 9.2 Common Leakage Patterns (FORBIDDEN)

| Leakage Type | Example | Detection |
|--------------|---------|-----------|
| **Future features** | Using Close[t] to predict return[t] | Check feature generation timestamps |
| **Pipeline refitting** | Fitting scaler on test data | Verify pipeline_state never changes |
| **Test contamination** | Using test data for early stopping | Check validation set source |
| **Look-ahead bias** | Sorting data before split | Verify chronological ordering |
| **Model updates** | Retraining during walk-forward | Check model.trainable_variables |

---

## 10. SUMMARY: CANONICAL DECISIONS

### 10.1 Resolution Table

| Dimension | TEST_VALID1 | TEST_VALID2 | **CANONICAL CHOICE** |
|-----------|-------------|-------------|---------------------|
| **Retraining** | Every step | Never | **Never (static)** |
| **Window** | Expanding | Rolling 20-day | **Rolling 20-day** |
| **Splits** | 80/10/20 (72/8/20 PSO) | 70/10/20 | **70/10/20 all models** |
| **Pipeline** | Refit per step | Fit once, freeze | **Fit once, freeze** |
| **PSO training** | Phase 1 only | Phase 1 + Phase 2 | **Phase 1 + Phase 2** |

### 10.2 Model Behavior Matrix

| Model | Training Data | Validation Data | Hyperparameters | Retraining | Walk-Forward Window |
|-------|--------------|----------------|----------------|------------|-------------------|
| **Baseline LSTM** | 70% train | 10% val (early stop) | Fixed YAML | Never | Rolling 20-day |
| **PSO-LSTM** | 70% train (PSO)<br>80% train+val (final) | 10% val (PSO fitness) | IPSO-optimized once | Never | Rolling 20-day |
| **XGBoost** | 70% train | 10% val (early stop) | Fixed or grid search | Never | Lag-based (20-day) |

### 10.3 Feature Pipeline Behavior

| Component | Fit On | Transform | State Persistence |
|-----------|--------|-----------|------------------|
| **Wavelet threshold** | 70% train | All splits | Frozen forever |
| **Feature selector** | 70% train | All splits | Frozen forever |
| **MinMax scaler** | 70% train | All splits | Frozen forever |
| **Windowing** | N/A (deterministic) | All splits | Fixed 20-day |

---

## 11. IMPLEMENTATION REQUIREMENTS

### 11.1 Required Scripts

```
pipelines/
├── canonical_feature_pipeline.py     # Fit pipeline once on train
├── canonical_train_baseline.py       # Train Baseline LSTM
├── canonical_train_pso.py            # PSO Phase 1 + 2
├── canonical_train_xgboost.py        # Train XGBoost
├── canonical_walk_forward.py         # Walk-forward evaluation
├── canonical_backtest.py             # Backtesting execution
└── canonical_compare.py              # Model comparison report
```

### 11.2 Configuration Requirements

```yaml
# config/canonical_config.yaml

evaluation:
  protocol_version: "1.0_CANONICAL"
  
  splits:
    train_pct: 0.70
    val_pct: 0.10
    test_pct: 0.20
    
  walk_forward:
    window_size: 20        # Fixed 20-day rolling window
    step_size: 1           # 1-day ahead prediction
    retraining: false      # NEVER retrain
    
  pipeline:
    fit_once: true         # Fit on train, freeze forever
    refit_per_step: false  # FORBIDDEN
    
  backtesting:
    transaction_cost: 0.0015  # 0.15% per trade
    slippage: 0.0             # Included in transaction_cost
    initial_capital: 1.0
    
baseline_lstm:
  # Fixed hyperparameters (NEVER tuned)
  units_1: 128
  units_2: 64
  dropout: 0.2
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  early_stopping_patience: 10
  
pso:
  # PSO configuration
  n_particles: 20
  n_iterations: 50
  search_space:
    epochs: [50, 300]
    units_1: [50, 300]
    units_2: [20, 200]
    learning_rate: [0.001, 0.01]
    dropout: [0.0, 0.5]
    batch_size: [32, 64]
  
  # Final training
  use_combined_data: true  # Train+Val after PSO
  
xgboost:
  # Fixed or grid-searched hyperparameters
  max_depth: 6
  learning_rate: 0.05
  n_estimators: 500
  subsample: 0.8
  colsample_bytree: 0.8
  early_stopping_rounds: 50
```

---

## 12. VALIDATION & COMPLIANCE

### 12.1 Protocol Compliance Verification

```python
def verify_protocol_compliance(evaluation_results):
    """
    Verify that evaluation followed canonical protocol.
    Raises exceptions if violations detected.
    """
    checks = []
    
    # Check 1: Split ratios
    train_pct = len(evaluation_results["train"]) / len(evaluation_results["data"])
    val_pct = len(evaluation_results["val"]) / len(evaluation_results["data"])
    test_pct = len(evaluation_results["test"]) / len(evaluation_results["data"])
    
    assert abs(train_pct - 0.70) < 0.01, "Train split must be 70%"
    assert abs(val_pct - 0.10) < 0.01, "Val split must be 10%"
    assert abs(test_pct - 0.20) < 0.01, "Test split must be 20%"
    checks.append(" Split ratios: 70/10/20")
    
    # Check 2: Pipeline state frozen
    pipeline_state = evaluation_results["pipeline_state"]
    assert pipeline_state["fit_date"] <= evaluation_results["val_start_date"], \
        "Pipeline must be fit before validation"
    checks.append(" Pipeline fit only on train")
    
    # Check 3: No model retraining
    assert evaluation_results["n_model_fits"] == 1, \
        "Model must be trained exactly once"
    checks.append(" Single model training (no retraining)")
    
    # Check 4: Rolling window
    assert evaluation_results["window_type"] == "rolling", \
        "Must use rolling window"
    assert evaluation_results["window_size"] == 20, \
        "Window size must be 20 days"
    checks.append(" Rolling 20-day window")
    
    # Check 5: Transaction costs applied
    assert "transaction_costs" in evaluation_results["backtest"], \
        "Transaction costs must be applied"
    checks.append(" Transaction costs applied")
    
    # Check 6: Test isolation
    assert evaluation_results["test_accessed_during_training"] == False, \
        "Test data must not be accessed during training"
    checks.append(" Test data isolated")
    
    logger.info("Protocol Compliance Verification:")
    for check in checks:
        logger.info(f"  {check}")
    
    return True
```

---

## 13. FINAL AUTHORITY

### 13.1 Precedence

This document (**FINAL_PLAN.md**) is the **single source of truth** for all evaluation, validation, and backtesting in the PSO-LSTM system.

**Superseded Documents:**
- TEST_VALID1.md
- TEST_VALID2.md
- Any other prior evaluation specifications

**In Case of Conflict:**
If any code, configuration, or documentation conflicts with this specification, **this document takes absolute precedence**.

### 13.2 Modification Policy

Changes to this protocol require:
1. Written justification
2. Impact analysis on prior results
3. Version increment
4. Re-evaluation of all models under new protocol

### 13.3 Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-04-21 | Initial canonical specification | System Architect |

---

**END OF CANONICAL SPECIFICATION**

*This document defines the complete, unambiguous, production-grade evaluation protocol for the PSO-LSTM stock prediction system. All implementations must conform to this specification without exception.*
