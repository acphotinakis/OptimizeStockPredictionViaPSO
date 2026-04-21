# Validation Protocol Alignment Analysis

## Critical Discrepancies Between TEST_VALID1.md and TEST_VALID2.md

### 1. RETRAINING POLICY (MAJOR CONFLICT)

#### TEST_VALID1.md (Walk-Forward with Retraining)
- **Models retrained at EVERY walk-forward step**
- **Expanding window** strategy
- Feature pipeline re-fit at each step
- Scalers, selectors, wavelet thresholds re-computed per step

```python
# TEST_VALID1 approach
while current_idx + step_size <= total_samples:
    train_data = data.iloc[:current_idx]  # EXPANDING
    test_data = data.iloc[current_idx:current_idx + step_size]
    
    # RE-FIT pipeline on expanding train
    features_train, features_test = run_feature_pipeline(
        train_data, test_data, fit_on_train_only=True
    )
    
    # RETRAIN model from scratch
    model = train_lstm(features_train, params, shuffle=False)
    
    predictions = model.predict(features_test)
    current_idx += step_size
```

#### TEST_VALID2.md (Static Evaluation)
- **Models trained ONCE, never retrained**
- **Rolling window** (20-day fixed)
- Feature pipeline fit ONCE on initial 70% train
- Pipeline state frozen for all subsequent evaluation

```python
# TEST_VALID2 approach
# Phase 1: Train once
pipeline_state = fit_pipeline(X_train)  # 70% data, ONCE
model = train_model(X_train, X_val)    # Train ONCE

# Phase 2: Walk-forward with fixed model
for t in test_period:
    features_t = X_test[t-20:t, :]      # ROLLING window
    prediction_t = model.predict(features_t)  # NO retraining
    # Model frozen
```

### 2. DATA SPLIT PERCENTAGES

| Split | TEST_VALID1.md | TEST_VALID2.md |
|-------|---------------|---------------|
| **Train** | 80% (Baseline/XGBoost)<br>72% (PSO-LSTM) | 70% |
| **Validation** | 10% of training<br>8% (PSO-LSTM) | 10% |
| **Test** | 20% | 20% |

### 3. WINDOW STRATEGY

| Aspect | TEST_VALID1.md | TEST_VALID2.md |
|--------|---------------|---------------|
| **Type** | Expanding | Rolling (fixed 20-day) |
| **Train window** | Grows: [0:t] | Fixed: [0:70%] |
| **Prediction window** | Next step_size days | Next 1 day |
| **Memory** | Increasing | Constant |

### 4. PSO TRAINING PROTOCOL

#### TEST_VALID1.md
```
PSO Phase:
  - Split: 72% train (E1-1), 8% PSO validation (E1-2)
  - Optimize on E1-1, evaluate fitness on E1-2
  - Hyperparameters optimized ONCE on first window
  - Fixed for all subsequent walk-forward steps
```

#### TEST_VALID2.md
```
PSO Phase 1: Hyperparameter Search
  - Use 70% train for PSO fitting
  - Use 10% val for PSO fitness

PSO Phase 2: Final Model Training
  - Retrain on combined 80% (train + val)
  - Use exact epoch count from PSO
  - No early stopping in final fit
```

### 5. HYPERPARAMETER RE-OPTIMIZATION

| Model | TEST_VALID1.md | TEST_VALID2.md |
|-------|---------------|---------------|
| **PSO-LSTM** | Optimized ONCE on first window,<br>fixed for all walk-forward steps | Optimized ONCE on initial train,<br>fixed forever |
| **XGBoost** | Grid search ONCE on first window,<br>fixed thereafter | Fixed or grid search ONCE,<br>fixed forever |

Both agree: **NO re-optimization during walk-forward**

### 6. FEATURE PIPELINE RE-FITTING

| Component | TEST_VALID1.md | TEST_VALID2.md |
|-----------|---------------|---------------|
| **Scaler (MinMax)** | Re-fit at each step | Fit ONCE, frozen |
| **Feature Selector** | Re-fit at each step | Fit ONCE, frozen |
| **Wavelet Threshold** | Re-computed at each step | Computed ONCE, frozen |

---

## CANONICAL DECISION (Required for Code Alignment)

### Which Protocol to Implement?

**DECISION REQUIRED FROM USER:**

**Option A: TEST_VALID1.md (Expanding + Retraining)**
- Pros: Adapts to regime shifts, realistic production scenario
- Cons: Computationally expensive, more complex implementation

**Option B: TEST_VALID2.md (Static + Rolling)**
- Pros: Fair comparison, simpler, "deploy once" scenario
- Cons: May degrade over time without adaptation

---

## RECOMMENDED HYBRID APPROACH (If User Allows)

Combine best practices from both:

```
1. DATA SPLITS: Use TEST_VALID2 (70/10/20)
   - Cleaner, simpler

2. PSO PROTOCOL: Use TEST_VALID2 Phase 1+2
   - More robust (use all available data for final fit)

3. WALK-FORWARD: Use TEST_VALID1 (Expanding + Retraining)
   - More realistic for production
   - Adapts to distribution shifts
   
4. FEATURE PIPELINE: Use TEST_VALID1 (Re-fit per step)
   - Prevents stale normalization
   - Maintains local stationarity
```

---

## CODE CHANGES REQUIRED (Per Protocol)

### If TEST_VALID1.md is canonical:

#### pipelines/run_feature_pipeline.py
- [ ] Implement expanding window logic
- [ ] Re-fit pipeline at each walk-forward step
- [ ] Use 80% initial train (72% for PSO)

#### pipelines/unified_train.py
- [ ] Implement walk-forward loop
- [ ] Full model retraining at each step
- [ ] Expanding train window

#### src/features/pipeline.py
- [ ] Ensure fit_transform() can be called multiple times
- [ ] Save/load pipeline state per step

---

### If TEST_VALID2.md is canonical:

#### pipelines/run_feature_pipeline.py
- [ ] Use 70/10/20 split (FIXED)
- [ ] Fit pipeline ONCE on 70% train
- [ ] Freeze pipeline state

#### pipelines/unified_train.py
- [ ] Train model ONCE on 70% train + 10% val
- [ ] PSO Phase 2: Retrain on combined 80%
- [ ] NO retraining during walk-forward

#### Walk-forward evaluation (NEW SCRIPT NEEDED)
- [ ] Create `pipelines/walk_forward_evaluate.py`
- [ ] Use frozen model for all predictions
- [ ] Rolling 20-day windows

---

## ALIGNMENT PRIORITY

**MUST ALIGN:**
1. Retraining policy (YES/NO)
2. Window strategy (Expanding/Rolling)
3. Pipeline re-fitting (YES/NO)
4. Data splits (70/10/20 or 80/10/20)
5. PSO protocol (Phase 1 only or Phase 1+2)

**NICE TO ALIGN:**
6. XGBoost feature representation (lag-based confirmed in both)
7. Transaction costs (0.3% vs 0.15% - minor)
8. Metrics (both documents agree on core metrics)

---

## NEXT STEPS

**User must specify:**
1. Which document is canonical? (TEST_VALID1 or TEST_VALID2)
2. OR: Should we implement hybrid approach?
3. OR: Should we implement BOTH as separate evaluation modes?

**Then:**
- Update all pipeline code to match chosen protocol
- Create walk-forward evaluation script
- Update configuration YAML
- Update documentation

---

**Waiting for user decision before proceeding with code changes.**
