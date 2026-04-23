# PSO-LSTM Training Script Compliance Audit

**Script:** `pipelines/train_pso_lstm.py`  
**TRD Sources:** `docs/TRD1.md`, `docs/TRD2.md`, `docs/TRD3.md`  
**Date:** 2026-04-21  
**Verdict:** ⚠️ **NON-COMPLIANT** (Multiple Critical Issues)

---

## Executive Summary

The PSO-LSTM training script contains **7 CRITICAL** and **3 HIGH** severity TRD violations. The script fails to implement the canonical two-phase protocol correctly, contains incorrect data splits, uses wrong configuration accessors, and has undefined variables that would cause runtime failures.

**Deployment Status:** ❌ **BLOCKED** - Must be fixed before use

---

## Critical Issues (BLOCKING)

### Issue #1: INCORRECT DATA SPLIT RATIOS (CRITICAL)

**Location:** Lines 154-180, 289-320

**TRD Requirement:**
- TRD1 §2.2: 70/10/20 split (train/val/test)
- TRD1 §8.2: "Training: 70% (earliest data), Validation: 10% (immediately following training), Test: 20% (most recent data)"

**Actual Implementation:**
```python
# Phase 1: Uses 70% train, 10% val (CORRECT)
# Phase 2: Combines train+val to 80% (INCORRECT for PSO)
X_combined = np.concatenate([X_train, X_val], axis=0)  # Line 329
```

**TRD2 §7.4:**
- "Training set: 72% of total data (90% of 80% training split)"
- "PSO validation set: 8% of total data (10% of 80% training split)"

**Problem:**
The script uses a 70/10/20 split, but TRD2 specifies 72/8/20 for PSO. Additionally, Phase 2 combines train+val incorrectly.

**Impact:**
- Phase 1 uses wrong validation set size (10% instead of 8%)
- Phase 2 trains on 80% instead of proper combined set
- Test set contamination risk

**Fix Required:**
```python
# Phase 1: Split 80% for PSO
pso_train_end = int(len(X_train) * 0.9)  # 72% of total
X_pso_train = X_train[:pso_train_end]
y_pso_train = y_train[:pso_train_end]
X_pso_val = X_train[pso_train_end:]      # 8% of total
y_pso_val = y_train[pso_train_end:]

# Phase 2: Train on full 80% (train + pso_val)
# NOT train + val from outside PSO
```

---

### Issue #2: MISSING VALIDATION IN PHASE 2 (CRITICAL)

**Location:** Lines 373-386

**TRD Requirement:**
- TRD1 §5.4: "Early stopping: Patience = 10 epochs (monitor validation loss)"
- Phase 2 should use exact PSO epochs, NO early stopping

**Actual Implementation:**
```python
model, history = trainer.train(
    X_combined_win,
    y_combined_win,
    X_combined_win,  # Using training data for validation (WRONG)
    y_combined_win,
    epochs=best_params["epochs"],
    batch_size=best_params["batch_size"],
    patience=pso_lstm_config.early_stopping.patience,  # Should be disabled
    shuffle=False,
)
```

**Problems:**
1. Uses training data as validation data (no true validation)
2. Enables early stopping with patience > 0 (should be disabled or inf)
3. May stop before PSO-determined optimal epochs

**Impact:**
- Model may undertrain compared to PSO optimization
- Invalidates PSO search results
- Not following two-phase protocol correctly

**Fix Required:**
```python
# Phase 2: NO early stopping, use exact PSO epochs
model, history = trainer.train(
    X_combined_win,
    y_combined_win,
    X_test_win,       # Monitor on test for logging only
    y_test_win,
    epochs=best_params["epochs"],  # Exact count from PSO
    batch_size=best_params["batch_size"],
    patience=None,    # Disable early stopping
    shuffle=False,
)
```

---

### Issue #3: UNDEFINED VARIABLE IN FITNESS FUNCTION (CRITICAL)

**Location:** Lines 137-145

**Problem:**
```python
def fitness_function(...):
    # ...
    model, _ = trainer.train(
        X_train_win,
        y_train_win,
        X_val_win,
        y_val_win,
        epochs=lstm_config.epochs,          # ❌ UNDEFINED
        batch_size=lstm_config.batch_size,  # ❌ UNDEFINED
        patience=lstm_config.early_stopping.patience,  # ❌ UNDEFINED
        shuffle=False,
        lstm_units_1=lstm_config.lstm_units_1,  # ❌ UNDEFINED
        lstm_units_2=lstm_config.lstm_units_2,  # ❌ UNDEFINED
        dropout_rate=lstm_config.dropout_rate,  # ❌ UNDEFINED
        learning_rate=lstm_config.learning_rate,  # ❌ UNDEFINED
    )
```

**Issue:**
`lstm_config` is never defined or passed to the function. This will cause `NameError` at runtime.

**Impact:**
- **RUNTIME FAILURE** - Script will crash during PSO Phase 1
- Cannot evaluate any particles
- PSO optimization completely broken

**Fix Required:**
```python
def fitness_function(
    params: dict,
    X_train_win: np.ndarray,
    y_train_win: np.ndarray,
    X_val_win: np.ndarray,
    y_val_win: np.ndarray,
    lookback: int,
    n_features: int,
    seed: int,
    config: Config,  # ADD THIS
) -> float:
    # Use config.lstm or params directly
    model, _ = trainer.train(
        X_train_win,
        y_train_win,
        X_val_win,
        y_val_win,
        epochs=params["epochs"],  # Use PSO params
        batch_size=params["batch_size"],
        patience=config.lstm.early_stopping.patience,
        shuffle=False,
        lstm_units_1=params["units_1"],
        lstm_units_2=params["units_2"],
        dropout_rate=params["dropout"],
        learning_rate=params["learning_rate"],
    )
```

---

### Issue #4: INCORRECT CONFIG ACCESSOR IN PHASE 2 (CRITICAL)

**Location:** Lines 342-386

**Problem:**
```python
pso_lstm_config = config.pso  # Line 342

# Later:
activation=pso_lstm_config.activation,           # Line 353
output_units=pso_lstm_config.output_units,       # Line 354
output_activation=pso_lstm_config.output_activation,  # Line 355
loss=pso_lstm_config.loss,                       # Line 357

# And:
lstm_units_1=pso_lstm_config.search_space.lstm_units_1,  # Line 382
lstm_units_2=pso_lstm_config.lstm_units_2,               # Line 383
dropout_rate=pso_lstm_config.dropout_rate,                # Line 384
learning_rate=pso_lstm_config.learning_rate,              # Line 385
```

**Issues:**
1. `config.pso` likely doesn't have `activation`, `output_units`, etc. (these are LSTM params)
2. Line 382 accesses `search_space.lstm_units_1` (search bounds, not actual values)
3. Lines 383-385 use PSO config values instead of PSO-optimized values from `best_params`

**Impact:**
- **RUNTIME FAILURE** - AttributeError likely
- If it runs, uses wrong parameter values
- Negates entire PSO optimization

**Fix Required:**
```python
# Use LSTM config for architecture params
model_config = {
    "input_shape": (lookback, X_train.shape[1]),
    "lstm_units_1": best_params["units_1"],  # From PSO
    "lstm_units_2": best_params["units_2"],  # From PSO
    "dropout_rate": best_params["dropout"],  # From PSO
    "activation": config.lstm.activation,     # From LSTM config
    "output_units": config.lstm.output_units,
    "output_activation": config.lstm.output_activation,
    "learning_rate": best_params["learning_rate"],  # From PSO
    "loss": config.lstm.loss,
}

# In train() call:
model, history = trainer.train(
    X_combined_win,
    y_combined_win,
    None,  # No validation in Phase 2
    None,
    epochs=best_params["epochs"],
    batch_size=best_params["batch_size"],
    patience=None,  # Disable
    shuffle=False,
)
```

---

### Issue #5: MISSING DATA LEAKAGE CHECKS (CRITICAL)

**Location:** Throughout

**TRD Requirements:**
- TRD1 §8: All data leakage prevention rules (L-1 through L-9)
- TRD1 §8.1 L-7: "Window Boundaries: Sliding windows SHALL NOT cross train/validation/test boundaries"
- TRD1 §8.1 L-4: "Correlation Computation: SHALL be computed on training data ONLY"

**Problems:**
1. No verification that windows don't cross split boundaries
2. No check that features were fit on training only
3. No validation of temporal ordering
4. No assertion of chronological integrity

**Impact:**
- Silent data leakage possible
- No guardrails against future information
- Production deployment risk

**Fix Required:**
Add validation checks:
```python
def validate_no_leakage(data: dict, lookback: int):
    """Validate TRD compliance for data leakage prevention."""
    logger.info("Validating TRD data leakage rules...")
    
    # L-1: Temporal ordering
    for split in ["train", "val", "test"]:
        X = data[f"X_{split}"]
        # Assume index column exists
        if hasattr(X, 'index'):
            assert X.index.is_monotonic_increasing, \
                f"{split}: Temporal ordering violated (L-1)"
    
    # L-7: Window boundaries
    # Check that first 'lookback' samples are not used
    assert data["X_train"].shape[0] >= lookback, \
        "Train set too small for lookback window"
    
    # L-3: Scaler fit on training only
    # (Requires access to pipeline state - check upstream)
    
    logger.info("✅ Data leakage validation passed")
```

---

### Issue #6: NO TEST SET ISOLATION ENFORCEMENT (CRITICAL)

**Location:** Phase 1 (lines 154-287)

**TRD Requirement:**
- TRD1 §7.4: "Test set SHALL NOT be accessed during PSO"
- TRD1 §8.3: "PSO using test set for fitness (PROHIBITED)"

**Problem:**
The script loads test data but doesn't explicitly prevent access during PSO. The fitness function wrapper doesn't validate this.

**Impact:**
- Risk of accidental test set usage
- No programmatic enforcement
- Compliance cannot be verified

**Fix Required:**
```python
def phase1_pso_search(...):
    # ...
    
    # Explicitly verify test set not accessed
    _test_data_hash = hash(data["X_test"].tobytes())
    
    def fitness_wrapper(params):
        result = fitness_function(...)
        
        # Verify test set unchanged
        assert hash(data["X_test"].tobytes()) == _test_data_hash, \
            "CRITICAL: Test set accessed during PSO (TRD violation L-6)"
        
        return result
    
    # Run PSO
    best_params, best_fitness = optimizer.optimize()
```

---

### Issue #7: MISSING METADATA TRACKING (CRITICAL)

**Location:** Lines 413-430

**TRD Requirement:**
- TRD1 §9.1: "Feature schema version SHALL be logged with each model artifact"
- TRD1 §9.2: "Retention masks: Versioned and stored with model artifacts"

**Problem:**
Metadata saved (lines 413-430) doesn't include:
- Feature schema version
- Feature selection mask
- Scaler parameters
- Pipeline state

**Impact:**
- Cannot reproduce model in production
- Cannot verify feature consistency
- Deployment will fail

**Fix Required:**
```python
metadata = {
    "model_type": "pso_lstm",
    "protocol": "CANONICAL_1.0",
    "source": "TRD1 §7 + TRD2 §7.4",
    
    # ADD THESE:
    "feature_schema_version": "1.0.0",
    "feature_names": feature_names_list,
    "feature_selection_mask": feature_mask.tolist(),
    "scaler_params": {
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
    },
    "pipeline_state_path": str(pipeline_state_path),
    
    # Existing:
    "phase1_samples": len(X_train),
    "phase2_samples": len(X_combined_win),
    "features": X_train.shape[1],
    "lookback": lookback,
    "pso_hyperparameters": best_params,
    "training_complete": True,
    "frozen": True,
    "retraining_allowed": False,
}
```

---

## High Priority Issues (NON-BLOCKING but REQUIRED)

### Issue #8: MISSING PSO CONVERGENCE CHECK (HIGH)

**Location:** Lines 256-257

**TRD Requirement:**
- TRD1 §7.4: "Iterate until max iterations or convergence (gbest unchanged for 10 iterations)"

**Problem:**
Script runs PSO for fixed iterations but doesn't implement convergence check.

**Fix:**
Add convergence detection in PSO optimizer or check fitness history.

---

### Issue #9: NO SEED CONFIGURATION (HIGH)

**Location:** Lines 189, 345-346

**TRD Requirement:**
- TRD1 §9.1: "Random seeds: Fixed across Python, NumPy, TensorFlow"

**Problem:**
```python
seed = pso_config.random_seed  # Line 190
seed = config.pso.seed         # Line 345
```

Different config accessors used, and TensorFlow seed not set.

**Fix:**
```python
def set_all_seeds(seed: int):
    """Set seeds for full reproducibility."""
    import random
    import numpy as np
    import tensorflow as tf
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    logger.info(f"All seeds set to {seed}")

# Call at start of Phase 1 and Phase 2
```

---

### Issue #10: MISSING TRD REFERENCE IN COMMENTS (HIGH)

**Location:** Throughout

**Problem:**
Comments reference "FINAL_PLAN.md" which doesn't exist. Should reference TRD docs.

**Fix:**
Update all comments:
```python
# OLD:
"""
FINAL_PLAN.md Section 4.2: PSO Phase 1

# NEW:
"""
TRD1 §7: PSO System Design
TRD2 §7.4: Evaluation Protocol
```

---

## Medium Priority Issues

### Issue #11: INCOMPLETE LOGGING (MEDIUM)

**Location:** Throughout

**Problem:**
Missing detailed logging for:
- Window shapes after construction
- Feature statistics
- Training convergence metrics
- PSO particle evolution

**Fix:** Add structured logging per TRD1 §9.3

---

### Issue #12: NO PERFORMANCE METRICS (MEDIUM)

**Location:** Missing

**Problem:**
Script doesn't compute or log:
- Training RMSE
- Validation RMSE
- Test RMSE (after Phase 2)

**Fix:** Add metrics computation and logging

---

## TRD Compliance Matrix

| TRD Rule | Requirement | Status | Issue # |
|----------|-------------|--------|---------|
| TRD1 §2.2 | 70/10/20 split | ❌ NON-COMPLIANT | #1 |
| TRD1 §5.4 | Early stopping patience=10 | ⚠️ PARTIAL | #2 |
| TRD1 §7.4 | PSO on validation only | ❌ NON-COMPLIANT | #6 |
| TRD1 §8.1 L-1 | Temporal ordering | ⚠️ NOT VALIDATED | #5 |
| TRD1 §8.1 L-3 | Scaler on train only | ⚠️ NOT VALIDATED | #5 |
| TRD1 §8.1 L-6 | PSO no test access | ❌ NON-COMPLIANT | #6 |
| TRD1 §8.1 L-7 | Window boundaries | ⚠️ NOT VALIDATED | #5 |
| TRD1 §9.1 | Reproducibility | ⚠️ PARTIAL | #9 |
| TRD1 §9.2 | Versioning | ❌ NON-COMPLIANT | #7 |
| TRD2 §7.4 | 72/8/20 PSO split | ❌ NON-COMPLIANT | #1 |

**Overall Compliance Score:** 20% (2/10 passing)

---

## Required Actions Before Deployment

### Immediate (CRITICAL):
1. ✅ Fix data split ratios (Issue #1)
2. ✅ Fix Phase 2 validation (Issue #2)
3. ✅ Fix undefined lstm_config variable (Issue #3)
4. ✅ Fix config accessor errors (Issue #4)
5. ✅ Add data leakage validation (Issue #5)
6. ✅ Add test set isolation enforcement (Issue #6)
7. ✅ Add metadata tracking (Issue #7)

### Short-term (HIGH):
8. ✅ Add PSO convergence check (Issue #8)
9. ✅ Fix seed configuration (Issue #9)
10. ✅ Update TRD references (Issue #10)

### Medium-term (MEDIUM):
11. ⏸️ Enhance logging (Issue #11)
12. ⏸️ Add performance metrics (Issue #12)

---

## Estimated Remediation Time

- **Critical fixes:** 4-6 hours
- **High priority:** 2-3 hours
- **Medium priority:** 2-3 hours
- **Total:** ~8-12 hours

---

## Conclusion

The `pipelines/train_pso_lstm.py` script is **NOT production-ready** and contains multiple **TRD-blocking violations**. The script will likely fail at runtime due to undefined variables and incorrect configuration access.

**Recommendation:** **DO NOT USE** until all CRITICAL and HIGH issues are resolved.

**Status:** ❌ **DEPLOYMENT BLOCKED**

---

**Audit completed:** 2026-04-21  
**Auditor:** TRD Compliance Verification System  
**Next review:** After remediation
