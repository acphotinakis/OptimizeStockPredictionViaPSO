# XGBoost Quick Start Guide

## TL;DR

```bash
# 1. Train
python scripts/xgboost/run_xgboost.py --ticker AAPL --mode train

# 2. Validate
python scripts/xgboost/run_xgboost.py --ticker AAPL --mode val

# 3. Test
python scripts/xgboost/run_xgboost.py --ticker AAPL --mode test

# 4. Plot
python scripts/plot_xgboost_results.py --ticker AAPL --dashboard
```

---

## Recent Fixes (April 6, 2026)

### ✓ Fixed: Validation/Test Mode Errors

**Issue:** `FileNotFoundError: Booster not found: results/xgb_model_AAPL_val_seed42.ubj`

**Fix:** Validation and test scripts now correctly load models trained in "train" mode.

**Files Modified:**
- `scripts/xgboost/xgboost_val.py`
- `scripts/xgboost/xgboost_test.py`

### ✓ Memory Issues Documented

See `docs/XGBOOST_MODS.md` for:
- Critical memory issues identified
- 75% memory reduction possible
- Detailed fixes for all issues

---

## Complete Workflow

### Step 1: Train Model

```bash
python scripts/xgboost/run_xgboost.py \
    --ticker AAPL \
    --mode train \
    --train-mode default \
    --seed 42
```

**Output:**
- `results/xgb_model_AAPL_train_seed42.ubj`
- `results/xgb_params_AAPL_train_seed42.json`
- `results/xgb_history_AAPL_train_seed42.json`
- `results/xgb_importances_AAPL_train_seed42.npy`

### Step 2: Validate Model

```bash
python scripts/xgboost/run_xgboost.py \
    --ticker AAPL \
    --mode val \
    --wfv-fold-size 8190 \
    --wfv-folds 6
```

**Output:**
- `results/xgb_val_metrics_AAPL_val_seed42.json`
- `results/xgb_val_threshold_AAPL_val_seed42.json`
- `results/xgb_val_importances_AAPL_val_seed42.json`

### Step 3: Test Model

```bash
python scripts/xgboost/run_xgboost.py \
    --ticker AAPL \
    --mode test \
    --initial-capital 100000 \
    --position-fraction 0.02
```

**Output:**
- `results/xgb_test_metrics_AAPL_test_seed42.json`
- `results/xgb_test_backtest_AAPL_test_seed42.json`

### Step 4: Generate Plots

```bash
# Dashboard (recommended)
python scripts/plot_xgboost_results.py --ticker AAPL --dashboard

# Or all individual plots
python scripts/plot_xgboost_results.py --ticker AAPL
```

**Output:**
- `plots/xgb_dashboard_AAPL_train_seed42.png` (dashboard)
- Or 4 individual plot files

---

## Training Options

### Default Training (Fixed Hyperparameters)

```bash
python scripts/xgboost/run_xgboost.py \
    --ticker AAPL \
    --mode train \
    --train-mode default
```

Uses hyperparameters from `config/default_config.yaml`.

### Hyperparameter Tuning

```bash
python scripts/xgboost/run_xgboost.py \
    --ticker AAPL \
    --mode train \
    --train-mode tune \
    --n-trials 20
```

Performs random search over hyperparameter grid.

---

## Batch Processing

### Process Multiple Tickers

```bash
#!/bin/bash
for ticker in AAPL MSFT GOOGL NVDA; do
    echo "=== $ticker ==="
    python scripts/xgboost/run_xgboost.py --ticker $ticker --mode train
    python scripts/xgboost/run_xgboost.py --ticker $ticker --mode val
    python scripts/xgboost/run_xgboost.py --ticker $ticker --mode test
    python scripts/plot_xgboost_results.py --ticker $ticker --dashboard
done
```

---

## Common Issues

### 1. FileNotFoundError during validation/test

**Error:** `Booster not found: results/xgb_model_AAPL_val_seed42.ubj`

**Solution:** Run training first:
```bash
python scripts/xgboost/run_xgboost.py --ticker AAPL --mode train
```

### 2. Memory errors during training

**Error:** Process killed or OOM

**Solution:** See `docs/XGBOOST_MODS.md` for fixes. Quick fix:
- Use `config/memory_optimized.yaml`
- Reduce lookback window in config

### 3. Missing feature files

**Error:** `Feature directory not found: data/features/AAPL`

**Solution:** Run feature engineering first:
```bash
python scripts/02_build_features.py --target AAPL
```

---

## Configuration Files

### default_config.yaml
Standard configuration for most use cases.

### memory_optimized.yaml
Reduced memory footprint for systems with limited RAM/GPU.

**Usage:**
```bash
python scripts/xgboost/run_xgboost.py \
    --ticker AAPL \
    --mode train \
    --config config/memory_optimized.yaml
```

---

## Documentation

- **`docs/XGBOOST_MODS.md`** - Memory issues and fixes
- **`docs/XGBOOST_WORKFLOW_FIX.md`** - Workflow explanation
- **`docs/PLOTTING_SYSTEM.md`** - Plotting system docs
- **`plots/README.md`** - Plotting module reference
- **`plots/USAGE_EXAMPLES.md`** - Plotting examples

---

## Results Files Reference

| File Pattern | Created By | Used By |
|-------------|-----------|---------|
| `xgb_model_{ticker}_train_seed{seed}.ubj` | train | val, test, plot |
| `xgb_params_{ticker}_train_seed{seed}.json` | train | plot |
| `xgb_history_{ticker}_train_seed{seed}.json` | train | plot |
| `xgb_importances_{ticker}_train_seed{seed}.npy` | train | plot |
| `xgb_val_threshold_{ticker}_val_seed{seed}.json` | val | test |
| `xgb_test_backtest_{ticker}_test_seed{seed}.json` | test | analysis |

---

## Next Steps

1. **Fix memory issues** (if needed): See `docs/XGBOOST_MODS.md`
2. **Run training**: `python scripts/xgboost/run_xgboost.py --ticker AAPL --mode train`
3. **Generate plots**: `python scripts/plot_xgboost_results.py --ticker AAPL --dashboard`
4. **Batch process**: Run pipeline for all tickers

---

## Support

For detailed information:
- Memory issues → `docs/XGBOOST_MODS.md`
- Workflow → `docs/XGBOOST_WORKFLOW_FIX.md`
- Plotting → `plots/README.md`
- Examples → `plots/USAGE_EXAMPLES.md`
