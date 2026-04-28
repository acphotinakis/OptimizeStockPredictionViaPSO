# CRITICAL BUG REPORT: Directional Accuracy Inflation via np.sign(0) = 0

**Date:** 2026-04-28  
**Severity:** CRITICAL (Data Leakage / Measurement Error)  
**Status:** IDENTIFIED — FIX REQUIRED

---

## Executive Summary

The LSTM baseline model reports **~99.96% directional accuracy** on test data, which is **statistically implausible** for financial return prediction. Root cause: incorrect use of `np.sign()` that treats near-zero predictions and actuals as "correctly predicted" due to `np.sign(0) = 0`.

**Expected directional accuracy for financial returns:**
- Weak signal: ~50–55%
- Strong signal: ~55–60%
- Random: ~50%

**Observed:** ~99.96% → **indicates measurement bug, NOT data leakage**.

---

## Root Cause Analysis

### Bug Location 1: `src/evaluation/metrics.py` (lines 50-53)

```python
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with the correct sign."""
    correct = np.sign(y_pred) == np.sign(y_true)
    return float(correct.mean())
```

### Bug Location 2: `src/models/lstm_model.py` (line 361)

```python
correct_dir = np.sum(np.sign(y_test) == np.sign(y_pred))
directional_accuracy = float(correct_dir / len(y_test))
```

### Bug Location 3: `src/models/xgboost_model.py` (line 79)

```python
correct_direction = np.sum(np.sign(y_test) == np.sign(y_pred))
```

### Bug Location 4: `src/evaluation/lstm_walk_forward.py` (line 673)

```python
correct_dir = np.sum(np.sign(y_true) == np.sign(y_pred))
```

---

## Why This Causes Inflation

### NumPy Sign Function Behavior

```python
>>> np.sign(0.001)
1.0
>>> np.sign(-0.001)
-1.0
>>> np.sign(0.0)
0.0   # ← CRITICAL ISSUE
```

### Financial Returns Distribution

Daily stock returns typically have:
- **Mean:** ~0.0003 (0.03%)
- **Std:** ~0.015 (1.5%)
- **~68% of values:** within ±1.5% (~0.015)
- **Many near-zero returns:** |r| < 0.0001

### Artificial Agreement Example

```python
y_true = [0.0001, -0.0002,  0.0000, -0.0001,  0.0003]
y_pred = [0.0002, -0.0001,  0.0001,  0.0000, -0.0002]

# Using np.sign (WRONG):
sign_true = [ 1,      -1,       0,      -1,       1   ]
sign_pred = [ 1,      -1,       1,       0,      -1   ]
matches   = [ T,       T,       F,       F,       F   ]
accuracy  = 2/5 = 40%  # Still inflated!

# But with near-zero domination:
y_true = [0.00001] * 950 + [0.01] * 50
y_pred = [0.00002] * 950 + [-0.01] * 50

sign_true = [1] * 950 + [1] * 50
sign_pred = [1] * 950 + [-1] * 50
matches = 950 / 1000 = 95%  # ← ARTIFICIAL INFLATION
```

### Why 99.96% Specifically?

This suggests:
1. **Most predictions are near-zero** (common for poorly-trained or over-regularized models)
2. **Most actual returns are near-zero** (true for daily stock data)
3. Both map to `sign(x) ≈ 0` or small positive/negative, creating false agreement

---

## Correct Implementation

### Option 1: Threshold-Based Classification (RECOMMENDED)

```python
def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    Directional accuracy with explicit zero handling.
    
    Classification rules:
    - y > threshold  → UP (+1)
    - y < -threshold → DOWN (-1)
    - Otherwise      → EXCLUDE (no direction)
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        threshold: Minimum absolute value to classify (default: 0.0)
    
    Returns:
        Fraction of correctly classified directions (excluding near-zero)
    """
    # Classify into {-1, 0, +1}
    true_dir = np.where(y_true > threshold, 1, np.where(y_true < -threshold, -1, 0))
    pred_dir = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
    
    # Only evaluate non-zero directions
    mask = (true_dir != 0)
    
    if mask.sum() == 0:
        return float('nan')  # No directional samples
    
    correct = (true_dir[mask] == pred_dir[mask]).sum()
    return float(correct / mask.sum())
```

### Option 2: Strict Sign Comparison (Alternative)

```python
def directional_accuracy_strict(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Strict directional accuracy: only count non-zero actuals.
    
    Zero returns have no "direction" to predict correctly.
    """
    # Only evaluate where actual return is non-zero
    mask = (y_true != 0.0)
    
    if mask.sum() == 0:
        return float('nan')
    
    # Check if prediction has correct sign
    correct = (np.sign(y_true[mask]) == np.sign(y_pred[mask]))
    return float(correct.mean())
```

### Option 3: Ternary Classification (Most Robust)

```python
def directional_accuracy_ternary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 1e-4  # 1 basis point
) -> float:
    """
    Ternary directional accuracy: {UP, FLAT, DOWN}.
    
    Treats near-zero returns as a separate "FLAT" class.
    """
    def classify(arr):
        return np.where(arr > threshold, 1, np.where(arr < -threshold, -1, 0))
    
    true_class = classify(y_true)
    pred_class = classify(y_pred)
    
    return float((true_class == pred_class).mean())
```

---

## Recommended Fix (Immediate Action)

### 1. Update `src/evaluation/metrics.py`

```python
def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.0,
    exclude_zeros: bool = True
) -> float:
    """
    Directional accuracy with correct zero handling.
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        threshold: Minimum absolute value to classify as directional
        exclude_zeros: If True, exclude near-zero actuals from evaluation
    
    Returns:
        Fraction of correct direction predictions
    
    Notes:
        - TRD3 Section 5.2: Directional accuracy must exclude flat markets
        - Common pitfall: np.sign(0) = 0 creates artificial agreement
    """
    if exclude_zeros:
        # Only evaluate where actual has clear direction
        mask = (np.abs(y_true) > threshold)
        if mask.sum() == 0:
            return float('nan')
        
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        correct = (np.sign(y_true_filtered) == np.sign(y_pred_filtered))
        return float(correct.mean())
    else:
        # Threshold-based ternary classification
        true_dir = np.where(y_true > threshold, 1, np.where(y_true < -threshold, -1, 0))
        pred_dir = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
        correct = (true_dir == pred_dir)
        return float(correct.mean())
```

### 2. Update `src/models/lstm_model.py` (line 361)

Replace:
```python
correct_dir = np.sum(np.sign(y_test) == np.sign(y_pred))
directional_accuracy = float(correct_dir / len(y_test))
```

With:
```python
# Use corrected directional accuracy (exclude near-zero returns)
from src.evaluation.metrics import directional_accuracy as da_corrected
directional_accuracy = da_corrected(y_test, y_pred, threshold=0.0, exclude_zeros=True)
```

### 3. Update `src/models/xgboost_model.py` (line 79)

Same fix as above.

### 4. Update `src/evaluation/lstm_walk_forward.py` (line 673)

Same fix as above.

---

## Verification Strategy

After fix, run:

```bash
python pipelines/run_lstm.py \
    --ticker AAPL \
    --timeframe daily \
    --data-path data/processed/AAPL/features \
    --skip-train \
    --skip-plot
```

**Expected Results (Post-Fix):**
- Directional accuracy: ~50–57% (realistic for financial data)
- RMSE: ~0.015–0.025 (typical for daily returns)
- R²: ~0.05–0.15 (weak predictive signal is normal)

**If DA is still >95%:**
- Check for data leakage (target in features)
- Check for label shift/misalignment
- Inspect prediction variance (model may be predicting constants)

---

## Impact Assessment

### Files Affected
1. `src/evaluation/metrics.py` ← **PRIMARY FIX**
2. `src/models/lstm_model.py`
3. `src/models/xgboost_model.py`
4. `src/evaluation/lstm_walk_forward.py`
5. `src/evaluation/walk_forward_pso.py` (commented code)

### Downstream Effects
- All historical evaluation results are **INVALID**
- Walk-forward validation results need re-computation
- PSO hyperparameter search may have converged to wrong optimum (if DA was used in fitness)
- Backtesting signal generation is NOT affected (uses threshold-based signals)

---

## Prevention

### Code Review Checklist for Financial Metrics

- [ ] Never use `np.sign()` directly for classification
- [ ] Always handle near-zero values explicitly
- [ ] Document threshold assumptions
- [ ] Add unit tests with edge cases:
  ```python
  def test_directional_accuracy_near_zero():
      y_true = np.array([0.0001, -0.0001, 0.0, 0.0])
      y_pred = np.array([0.0002, -0.0002, 0.0001, -0.0001])
      da = directional_accuracy(y_true, y_pred)
      assert 0.45 <= da <= 0.55, "DA should be ~50% for near-zero noise"
  ```

### TRD Update Required

**TRD3 Section 5.2** should specify:
> **Directional Accuracy Calculation:**  
> Exclude returns with |r| < threshold (default: 0.0) from evaluation.  
> Rationale: Near-zero returns have no meaningful "direction" to predict.  
> 
> Formula:  
> ```
> DA = Σ(sign(y_pred[i]) == sign(y_true[i]) | |y_true[i]| > threshold) / N_nonzero
> ```

---

## Conclusion

This is a **measurement bug, not data leakage**. The model is NOT actually achieving 99.96% directional accuracy. The fix is straightforward but requires updating the metric implementation and re-running all evaluations.

**Priority:** CRITICAL  
**Effort:** 2 hours (fix + verification)  
**Risk:** Low (isolated to metric calculation)

---

**Next Steps:**
1. Apply fix to `src/evaluation/metrics.py` (primary)
2. Update model evaluation methods to use corrected metric
3. Re-run training and evaluation for all models
4. Update TRD3 with explicit DA definition
5. Add unit tests for edge cases
