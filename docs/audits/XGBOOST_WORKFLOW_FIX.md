# XGBoost Workflow Fix

**Date:** April 6, 2026  
**Issue:** FileNotFoundError when running validation/test modes  
**Status:** FIXED

---

## Problem

When running validation or test modes, the scripts were looking for models trained in those modes:

```bash
# This failed:
python scripts/xgboost/run_xgboost.py --ticker AAPL --mode val

# Error:
FileNotFoundError: Booster not found: results/xgb_model_AAPL_val_seed42.ubj
```

### Root Cause

The validation and test scripts were incorrectly using `args.mode` to load the model:

**Before (WRONG):**
```python
# xgboost_val.py line 126
model, meta, _ = load_artefacts(results_dir, ticker, args.mode, args.seed)
# When args.mode="val", this looks for xgb_model_AAPL_val_seed42.ubj

# xgboost_test.py line 32
model, meta = load_model(results_dir, ticker, args.mode, args.seed)
# When args.mode="test", this looks for xgb_model_AAPL_test_seed42.ubj
```

This is incorrect because:
- Models are **trained** in `mode="train"`
- Validation **evaluates** the trained model on the validation set
- Testing **evaluates** the trained model on the test set

---

## Solution

### Files Modified

1. **`scripts/xgboost/xgboost_val.py`** (line 126)
2. **`scripts/xgboost/xgboost_test.py`** (lines 32-33)

### Changes Made

**xgboost_val.py:**
```python
# BEFORE:
model, meta, _ = load_artefacts(results_dir, ticker, args.mode, args.seed)

# AFTER:
# Load model trained in 'train' mode, not 'val' mode
model, meta, _ = load_artefacts(results_dir, ticker, "train", args.seed)
```

**xgboost_test.py:**
```python
# BEFORE:
model, meta = load_model(results_dir, ticker, args.mode, args.seed)
theta = load_optimal_threshold(results_dir, ticker, args.mode, args.seed)

# AFTER:
# Load model trained in 'train' mode, not 'test' mode
model, meta = load_model(results_dir, ticker, "train", args.seed)
# Load threshold from 'val' mode (determined during validation)
theta = load_optimal_threshold(results_dir, ticker, "val", args.seed)
```

---

## Correct Workflow

### 1. Training Phase
```bash
python scripts/xgboost/run_xgboost.py \
    --ticker AAPL \
    --mode train \
    --train-mode default
```

**Creates:**
- `results/xgb_model_AAPL_train_seed42.ubj` (model)
- `results/xgb_params_AAPL_train_seed42.json` (hyperparameters)
- `results/xgb_history_AAPL_train_seed42.json` (training curves)
- `results/xgb_importances_AAPL_train_seed42.npy` (feature importances)

### 2. Validation Phase
```bash
python scripts/xgboost/run_xgboost.py \
    --ticker AAPL \
    --mode val \
    --wfv-fold-size 8190 \
    --wfv-folds 6
```

**Loads:**
- `results/xgb_model_AAPL_train_seed42.ubj` (trained model)

**Creates:**
- `results/xgb_val_metrics_AAPL_val_seed42.json` (validation metrics)
- `results/xgb_val_threshold_AAPL_val_seed42.json` (optimal threshold)
- `results/xgb_val_importances_AAPL_val_seed42.json` (importances)

### 3. Testing Phase
```bash
python scripts/xgboost/run_xgboost.py \
    --ticker AAPL \
    --mode test \
    --initial-capital 100000 \
    --position-fraction 0.02
```

**Loads:**
- `results/xgb_model_AAPL_train_seed42.ubj` (trained model)
- `results/xgb_val_threshold_AAPL_val_seed42.json` (optimal threshold from validation)

**Creates:**
- `results/xgb_test_metrics_AAPL_test_seed42.json` (test metrics)
- `results/xgb_test_backtest_AAPL_test_seed42.json` (backtest results)

---

## File Naming Convention

The file naming convention follows this pattern:
```
xgb_{artifact}_{ticker}_{phase}_seed{seed}.{ext}
```

Where:
- `{artifact}`: model, params, history, importances, metrics, etc.
- `{ticker}`: Stock symbol (AAPL, MSFT, etc.)
- `{phase}`: **Phase where the artifact was created** (train, val, test)
- `{seed}`: Random seed (default: 42)

### Examples

| File | Created In | Used By |
|------|-----------|---------|
| `xgb_model_AAPL_train_seed42.ubj` | Training | Val, Test |
| `xgb_params_AAPL_train_seed42.json` | Training | Plotting |
| `xgb_history_AAPL_train_seed42.json` | Training | Plotting |
| `xgb_val_threshold_AAPL_val_seed42.json` | Validation | Test |
| `xgb_test_backtest_AAPL_test_seed42.json` | Testing | Analysis |

---

## Complete Example

```bash
# 1. Train model
python scripts/xgboost/run_xgboost.py \
    --ticker AAPL \
    --mode train \
    --train-mode default \
    --seed 42

# 2. Validate model (walk-forward validation)
python scripts/xgboost/run_xgboost.py \
    --ticker AAPL \
    --mode val \
    --seed 42

# 3. Test model (out-of-sample backtest)
python scripts/xgboost/run_xgboost.py \
    --ticker AAPL \
    --mode test \
    --seed 42

# 4. Generate plots
python scripts/plot_xgboost_results.py \
    --ticker AAPL \
    --dashboard
```

---

## Batch Processing

Process multiple tickers:

```bash
#!/bin/bash
# run_xgboost_pipeline.sh

TICKERS=("AAPL" "MSFT" "GOOGL" "NVDA" "TSLA")

for ticker in "${TICKERS[@]}"; do
    echo "Processing $ticker..."
    
    # Train
    python scripts/xgboost/run_xgboost.py \
        --ticker $ticker \
        --mode train \
        --train-mode default
    
    # Validate
    python scripts/xgboost/run_xgboost.py \
        --ticker $ticker \
        --mode val
    
    # Test
    python scripts/xgboost/run_xgboost.py \
        --ticker $ticker \
        --mode test
    
    # Plot
    python scripts/plot_xgboost_results.py \
        --ticker $ticker \
        --dashboard
    
    echo "✓ $ticker complete"
done
```

---

## Troubleshooting

### Issue: "Booster not found: xgb_model_TICKER_val_seed42.ubj"

**Cause:** Running validation before training.

**Solution:** Run training first:
```bash
python scripts/xgboost/run_xgboost.py --ticker AAPL --mode train
```

### Issue: "Threshold file not found"

**Cause:** Running test before validation.

**Solution:** Run validation first:
```bash
python scripts/xgboost/run_xgboost.py --ticker AAPL --mode val
```

### Issue: Different seeds

If you trained with a different seed, specify it in all commands:
```bash
python scripts/xgboost/run_xgboost.py --ticker AAPL --mode train --seed 123
python scripts/xgboost/run_xgboost.py --ticker AAPL --mode val --seed 123
python scripts/xgboost/run_xgboost.py --ticker AAPL --mode test --seed 123
```

---

## Why This Design?

### Single Model, Multiple Evaluations

The design follows the principle of **train once, evaluate multiple times**:

1. **Training** creates the model artifact
2. **Validation** evaluates on validation set and finds optimal threshold
3. **Testing** evaluates on test set with optimal threshold

This ensures:
- No data leakage (test set never seen during training/validation)
- Consistent model across all evaluation phases
- Proper out-of-sample testing

### Alternative: Separate Models per Phase

Some frameworks train separate models for each phase. This is **NOT** recommended because:
- ❌ Wastes computation (training multiple identical models)
- ❌ Inconsistent results (different random initializations)
- ❌ Violates ML best practices (test set should never be used for training)

---

## Summary

**What was fixed:**
- Validation script now loads model from `mode="train"` instead of `mode="val"`
- Test script now loads model from `mode="train"` instead of `mode="test"`
- Test script loads threshold from `mode="val"` (determined during validation)

**Correct workflow:**
1. Train → creates model
2. Validate → loads model, creates threshold
3. Test → loads model and threshold
4. Plot → loads training artifacts

**Key principle:** Models are trained once in `mode="train"`, then evaluated in `mode="val"` and `mode="test"`.
