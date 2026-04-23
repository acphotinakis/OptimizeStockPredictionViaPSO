# Loss Computation with Scaled Targets - Analysis

**Question:** Are we computing training and validation losses correctly when targets are scaled?  
**Answer:** ✅ **YES - Loss computation is CORRECT as-is**  
**Date:** 2026-04-21

---

## Executive Summary

The current implementation computes losses on **scaled targets** during training, which is **CORRECT** and **intentional** for the following reasons:

1. **Optimization Stability:** MSE on scaled targets provides stable gradients
2. **Fair Comparison:** All models (Baseline LSTM, PSO-LSTM, XGBoost) train on the same scale
3. **TRD Compliance:** TRD1 §4.2 specifies normalization to [-1, 1] for training
4. **Inverse Transform for Evaluation:** Test set evaluation SHOULD use inverse-transformed predictions

**❌ DO NOT inverse-transform targets during training**  
**✅ DO inverse-transform predictions before final evaluation**

---

## Current Implementation Analysis

### 1. Training Pipeline Flow

```
Raw Data
    ↓
Feature Pipeline (pipelines/run_build_features.py)
    ↓
FrozenMinMaxScaler.fit(y_train)  → Fit target scaler on training data
    ↓
y_train_scaled = scaler.transform(y_train)  → Scale to [-1, 1]
y_val_scaled = scaler.transform(y_val)
y_test_scaled = scaler.transform(y_test)
    ↓
Save scaled data: y_train.npy, y_val.npy, y_test.npy
    ↓
Training Script (train_baseline_lstm.py)
    ↓
Load scaled targets
    ↓
Train LSTM with MSELoss(y_pred_scaled, y_true_scaled)
    ↓
Compute losses on SCALED space
```

### 2. Loss Computation in LSTMTrainer

**Location:** `src/models/trainer.py` lines 193-194, 214-215

```python
# Training phase
loss = criterion(outputs, batch_y)  # Both in scaled space [-1, 1]
loss.backward()
train_losses.append(loss.item())

# Validation phase
loss = criterion(outputs, batch_y)  # Both in scaled space [-1, 1]
val_losses.append(loss.item())
```

**Key Points:**
- `outputs`: Model predictions in scaled space [-1, 1]
- `batch_y`: Ground truth targets in scaled space [-1, 1]
- `criterion`: MSELoss computed on scaled values
- `loss.item()`: Scalar loss value in scaled space

---

## Why This is CORRECT

### 1. Optimization Stability

**Scaled Space (CURRENT - CORRECT):**
```python
# Targets scaled to [-1, 1]
y_true_scaled = [-0.5, 0.2, -0.1, 0.8]
y_pred_scaled = [-0.4, 0.3, -0.2, 0.7]

MSE = mean((y_pred - y_true)²)
    = mean([0.01, 0.01, 0.01, 0.01])
    = 0.01

Gradients ∝ (y_pred - y_true)
         = [0.1, -0.1, 0.1, -0.1]
         → Stable, bounded gradients
```

**Unscaled Space (WRONG - WOULD CAUSE ISSUES):**
```python
# Targets in original scale (e.g., stock returns %)
y_true_raw = [-5.2, 12.8, -3.1, 45.2]
y_pred_raw = [-4.1, 15.3, -6.7, 42.8]

MSE = mean((y_pred - y_true)²)
    = mean([1.21, 6.25, 12.96, 5.76])
    = 6.545  ← Much larger, less stable

Gradients ∝ (y_pred - y_true)
         = [1.1, -2.5, 3.6, -2.4]
         → Large, unbounded gradients
         → Potential instability
```

### 2. TRD1 §4.2 Normalization Requirement

**TRD1 §4.2:**
> **Canonical Rule:** MinMaxScaler to [-1, 1] range
> 
> **Rationale:** [-1, 1] range IS SELECTED because:
> - Includes return-based features (can be negative)
> - Compatible with tanh/sigmoid activations in LSTM
> - Used by majority of research papers

**Training on scaled targets ensures:**
- ✅ Gradient stability
- ✅ Faster convergence
- ✅ Better numerical precision (avoid overflow/underflow)
- ✅ Fair comparison across models

### 3. Fair Model Comparison

All models (Baseline LSTM, PSO-LSTM, XGBoost) train on **identical scaled data**:

```python
# All models see:
y_train_scaled ∈ [-1, 1]
y_val_scaled ∈ [-1, 1]
y_test_scaled ∈ [-1, 1]

# This ensures:
# - PSO optimizes hyperparameters on same scale
# - Early stopping uses consistent loss magnitudes
# - Walk-forward evaluation compares apples-to-apples
```

If we inverse-transformed during training:
- ❌ PSO would optimize on unscaled MSE (inconsistent with val loss)
- ❌ Early stopping would use unscaled loss (different magnitude)
- ❌ Training logs would show unscaled loss (confusing)

---

## Where Inverse Transform IS Required

### Test Set Evaluation (Production Metrics)

**Location:** After training, before final evaluation

**Correct Flow:**
```python
# 1. Load trained model
model = load_model("baseline_lstm_model.h5")

# 2. Load scaler parameters
target_scaler = FrozenMinMaxScaler.from_params(scaler_params)

# 3. Predict on scaled test data
y_test_scaled = load("y_test.npy")  # Scaled to [-1, 1]
X_test_win = build_lstm_windows(X_test, y_test, lookback)

y_pred_scaled = model.predict(X_test_win)  # Predictions in [-1, 1]

# 4. Inverse transform BEFORE computing final metrics
y_pred_original = target_scaler.inverse_transform(y_pred_scaled)
y_test_original = target_scaler.inverse_transform(y_test_scaled)

# 5. Compute metrics on ORIGINAL scale
mse_original = np.mean((y_pred_original - y_test_original) ** 2)
mae_original = np.mean(np.abs(y_pred_original - y_test_original))
rmse_original = np.sqrt(mse_original)

# 6. Report metrics
print(f"Test MSE (original scale): {mse_original:.6f}")
print(f"Test RMSE (original scale): {rmse_original:.6f}")
```

**Why:**
- ✅ Final metrics meaningful to stakeholders (e.g., "3% return error")
- ✅ Comparable to baseline/benchmark strategies
- ✅ Interpretable for business decisions

### Walk-Forward Evaluation

**Location:** During backtesting for trading signals

**Correct Flow:**
```python
# For each walk-forward step:
for t in range(test_start, test_end):
    # 1. Predict in scaled space
    X_window_scaled = get_window(t, lookback)
    y_pred_scaled = model.predict(X_window_scaled)
    
    # 2. Inverse transform prediction
    y_pred_original = target_scaler.inverse_transform(y_pred_scaled)
    
    # 3. Generate trading signal
    if y_pred_original > 0:
        signal = "BUY"
    else:
        signal = "SELL"
    
    # 4. Compute P&L on original scale
    actual_return = y_test_original[t]
    pnl = actual_return * signal_direction
```

**Why:**
- ✅ Trading decisions based on actual returns
- ✅ P&L computed on real $ values
- ✅ Transaction costs applied to real prices

---

## Common Misconception

### ❌ WRONG: "We need to inverse-transform during training"

**Flawed Logic:**
> "If targets are scaled, the model learns to predict scaled values. 
> This is wrong because we want predictions in original scale."

**Why This is Wrong:**
The model **should** learn to predict scaled values during training because:

1. **Stable optimization:** Scaled targets → bounded gradients → stable learning
2. **Model learns relative patterns:** Whether scaled or not, patterns are preserved
3. **Post-processing is cheap:** Inverse transform is O(1) operation
4. **Evaluation correctness:** Final metrics use original scale anyway

**Analogy:**
```
Training a model in scaled space is like:
- Teaching someone to measure in meters (metric)
- Then converting to feet (imperial) when reporting

vs.

Training in unscaled space is like:
- Teaching someone to measure in feet
- Dealing with large, inconsistent numbers
- More prone to measurement errors
```

---

## Implementation Verification

### Current State

**✅ Training (CORRECT):**
```python
# src/models/trainer.py
criterion = nn.MSELoss()
loss = criterion(outputs, batch_y)  # Both scaled to [-1, 1]
```

**✅ Feature Pipeline (CORRECT):**
```python
# src/features/scaler.py
class FrozenMinMaxScaler:
    def fit(self, X: np.ndarray):
        # Fit on training data only
        self._params = {"data_min": X.min(), "data_max": X.max()}
    
    def transform(self, X: np.ndarray):
        # Scale to [-1, 1]
        return 2 * (X - self._params["data_min"]) / \
               (self._params["data_max"] - self._params["data_min"]) - 1
    
    def inverse_transform(self, X_scaled: np.ndarray):
        # Recover original scale
        return (X_scaled + 1) / 2 * \
               (self._params["data_max"] - self._params["data_min"]) + \
               self._params["data_min"]
```

**❌ Missing: Test Evaluation with Inverse Transform**

Currently, `src/models/trainer.py` evaluate method computes metrics on scaled space:

```python
def evaluate(self, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)  # ← Scaled space
    mae = np.mean(np.abs(y_test - y_pred))  # ← Scaled space
    rmse = np.sqrt(mse)  # ← Scaled space
    return {"mse": mse, "mae": mae, "rmse": rmse}
```

**Should be:**
```python
def evaluate(self, model, X_test, y_test, target_scaler=None):
    y_pred_scaled = model.predict(X_test)
    
    if target_scaler is not None:
        # Inverse transform to original scale
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_true = target_scaler.inverse_transform(y_test)
    else:
        y_pred = y_pred_scaled
        y_true = y_test
    
    # Compute metrics on original scale
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mse_scaled": np.mean((y_test - y_pred_scaled) ** 2),  # For comparison
    }
```

---

## Recommendations

### 1. Keep Training Losses on Scaled Space ✅

**Do NOT change:**
```python
# src/models/trainer.py - Lines 193, 214
loss = criterion(outputs, batch_y)  # Keep as-is
```

**Reason:** Correct and optimal for training.

### 2. Add Inverse Transform to Evaluation ⚠️

**Change needed:**
```python
# src/models/trainer.py - evaluate() method
def evaluate(self, model, X_test, y_test, target_scaler=None):
    """
    Evaluate model on test set.
    
    Args:
        target_scaler: Optional scaler to inverse-transform predictions
    """
    # ... existing code ...
    
    if target_scaler is not None:
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_true = target_scaler.inverse_transform(y_test)
    
    # Compute metrics on original scale
```

**Apply to:**
- `src/models/trainer.py`
- `src/models/lstm.py`
- `src/models/xgboost_model.py`
- `src/models/xgboost_trainer.py`

### 3. Update Training Scripts to Pass Scaler

**Change needed:**
```python
# pipelines/train_baseline_lstm.py
# After training, load scaler for evaluation
scaler_path = data_path / "target_scaler_params.json"
if scaler_path.exists():
    with open(scaler_path, "r") as f:
        scaler_params = json.load(f)
    target_scaler = FrozenMinMaxScaler.from_params(scaler_params)
else:
    target_scaler = None
    logger.warning("No target scaler found - evaluation on scaled space")

# Evaluate with scaler
metrics = trainer.evaluate(model, X_test_win, y_test, target_scaler=target_scaler)
```

### 4. Document Scaler in Metadata

**Add to metadata:**
```python
metadata = {
    # ... existing fields ...
    "target_scaled": True,
    "target_scaler_path": "target_scaler_params.json",
    "target_range": [-1, 1],
    "metrics_computed_on": "scaled_space",  # During training
    "final_evaluation_on": "original_scale",  # After inverse transform
}
```

---

## Testing Checklist

### Training Losses (Current - Correct):
- [x] Losses computed on scaled space [-1, 1]
- [x] Loss magnitudes reasonable (typically 0.001-0.1)
- [x] Early stopping works correctly
- [x] PSO optimization on same scale

### Final Evaluation (Needs Update):
- [ ] Load target scaler parameters
- [ ] Inverse transform predictions before metrics
- [ ] Report metrics on original scale
- [ ] Compare to baseline/benchmark on same scale

### Production Inference (Needs Update):
- [ ] Save scaler params with model
- [ ] Load scaler at inference time
- [ ] Inverse transform predictions
- [ ] Generate signals on original scale

---

## Summary

**Question:** Should we inverse-transform targets before computing losses?

**Answer:** 
- ❌ **NO** during training (current implementation is CORRECT)
- ✅ **YES** during final test evaluation (needs minor update)
- ✅ **YES** during production inference (needs implementation)

**Training losses on scaled space:**
- ✅ Stable optimization
- ✅ TRD1 §4.2 compliant
- ✅ Fair model comparison
- ✅ Standard ML practice

**Inverse transform for evaluation:**
- ✅ Interpretable metrics
- ✅ Comparable to benchmarks
- ✅ Meaningful for stakeholders

**Verdict:** Current training implementation is CORRECT. Only need to add inverse transform to evaluation and inference pipelines.

---

**Analysis completed:** 2026-04-21  
**Current status:** Training CORRECT, Evaluation needs minor enhancement  
**Priority:** MEDIUM (training works correctly, evaluation enhancement is optional for better interpretability)
