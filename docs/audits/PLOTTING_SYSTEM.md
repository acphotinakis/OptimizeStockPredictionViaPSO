# XGBoost Plotting System Documentation

**Created:** April 6, 2026  
**Status:** Complete and Ready to Use

---

## Overview

A comprehensive plotting system for visualizing XGBoost model training results. Generates publication-quality plots for model analysis, diagnostics, and reporting.

## Files Created

### Core Module
- **`plots/xgboost_plots.py`** (570 lines)
  - Main plotting module with all visualization functions
  - Supports individual plots and comprehensive dashboards
  - Publication-quality styling with seaborn

### Supporting Files
- **`plots/__init__.py`**
  - Module initialization with public API exports
  
- **`plots/README.md`**
  - Module documentation and quick reference
  
- **`plots/USAGE_EXAMPLES.md`**
  - Comprehensive usage examples and tutorials
  - Covers CLI, Python API, and advanced customization
  
- **`scripts/plot_xgboost_results.py`**
  - Convenience wrapper script for command-line usage
  - Executable with proper error handling

---

## Features

### 1. Training Curves Plot
- Plots train/validation RMSE over boosting rounds
- Marks best iteration with vertical line and star
- Includes hyperparameter summary box
- Shows runtime and convergence information

### 2. Feature Importances Plot
- Horizontal bar chart of top K features
- Aggregates importance across time steps for windowed features
- Color-coded by importance score
- Configurable number of features to display (default: 20)

### 3. Metrics Summary Plot
- Bar chart of validation metrics
- Automatically separates small-scale and large-scale metrics
- Includes: RMSE, MAE, R², Directional Accuracy, F1, AUC
- Value labels on each bar

### 4. Hyperparameters Table
- Formatted table of all model hyperparameters
- Color-coded header and alternating row colors
- Automatic scientific notation for small values

### 5. Summary Dashboard
- Comprehensive 6-panel overview
- Combines all key visualizations in one figure
- Includes training info and key metrics
- Ideal for quick model assessment

---

## Usage

### Command Line Interface

```bash
# Generate all plots for AAPL
python scripts/plot_xgboost_results.py --ticker AAPL

# Create dashboard only
python scripts/plot_xgboost_results.py --ticker AAPL --dashboard

# Show plots interactively
python scripts/plot_xgboost_results.py --ticker AAPL --show

# Custom output directory
python scripts/plot_xgboost_results.py --ticker AAPL --output-dir figures/

# Batch process multiple tickers
for ticker in AAPL MSFT GOOGL; do
    python scripts/plot_xgboost_results.py --ticker $ticker --dashboard
done
```

### Python API

```python
from plots.xgboost_plots import plot_all_results, create_summary_dashboard

# Generate all individual plots
figures = plot_all_results(
    results_dir="results",
    ticker="AAPL",
    mode="train",
    seed=42,
    output_dir="plots",
    show=False
)

# Create comprehensive dashboard
fig = create_summary_dashboard(
    results_dir="results",
    ticker="AAPL",
    save_path="plots/AAPL_dashboard.png",
    show=True
)
```

---

## Input Files Required

For a given ticker (e.g., AAPL) with mode=train and seed=42:

1. **`results/xgb_history_AAPL_train_seed42.json`**
   - Training/validation loss curves
   - Format: `{"train_rmse": [...], "val_rmse": [...]}`

2. **`results/xgb_params_AAPL_train_seed42.json`**
   - Hyperparameters and metadata
   - Includes validation metrics and runtime info

3. **`results/xgb_importances_AAPL_train_seed42.npy`**
   - Feature importance scores (numpy array)
   - Shape: `[lookback × n_features]`

4. **`data/features/AAPL/metadata.pkl`** (optional)
   - Feature names for better labels
   - If missing, uses generic "Feature 0", "Feature 1", etc.

---

## Output Files Generated

### Individual Plots Mode (default)
- `xgb_training_curves_AAPL_train_seed42.png`
- `xgb_feature_importances_AAPL_train_seed42.png`
- `xgb_metrics_summary_AAPL_train_seed42.png`
- `xgb_hyperparameters_AAPL_train_seed42.png`

### Dashboard Mode (`--dashboard`)
- `xgb_dashboard_AAPL_train_seed42.png`

All plots saved at 300 DPI for publication quality.

---

## API Reference

### `load_xgboost_results(results_dir, ticker, mode='train', seed=42)`
Load all result files for a ticker.

**Returns:** Dictionary with keys:
- `history`: Training curves
- `params`: Hyperparameters and metrics
- `importances`: Feature importance array
- `feature_names`: List of feature names (if available)

### `plot_training_curves(history, params, save_path=None, show=True)`
Plot training/validation RMSE curves.

**Args:**
- `history`: Dict with 'train_rmse' and 'val_rmse'
- `params`: Parameters dict with 'best_iteration'
- `save_path`: Optional path to save figure
- `show`: Whether to display plot

**Returns:** Matplotlib figure object

### `plot_feature_importances(importances, feature_names, lookback=30, top_k=20, ...)`
Plot feature importance bar chart.

**Args:**
- `importances`: Numpy array of importance scores
- `feature_names`: List of feature names
- `lookback`: Window size for aggregation
- `top_k`: Number of top features to show

**Returns:** Matplotlib figure object

### `plot_metrics_summary(params, save_path=None, show=True)`
Plot validation metrics bar chart.

**Args:**
- `params`: Parameters dict containing 'val_metrics'

**Returns:** Matplotlib figure object

### `plot_hyperparameters(params, save_path=None, show=True)`
Display hyperparameters as formatted table.

**Args:**
- `params`: Parameters dict containing 'hyperparameters'

**Returns:** Matplotlib figure object

### `plot_all_results(results_dir, ticker, mode='train', seed=42, output_dir='plots', show=False)`
Generate all plots for a ticker.

**Returns:** Dictionary of figure objects keyed by plot type

### `create_summary_dashboard(results_dir, ticker, mode='train', seed=42, save_path=None, show=True)`
Create comprehensive 6-panel dashboard.

**Returns:** Matplotlib figure object

---

## Styling and Customization

### Default Style
- Uses seaborn darkgrid style
- 150 DPI for display, 300 DPI for saved files
- Color palette: "husl" (colorblind-friendly)
- Font sizes optimized for readability

### Customization Example

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set custom style before importing plotting functions
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12

# Now import and use plotting functions
from plots.xgboost_plots import plot_all_results
plot_all_results("results", "AAPL")
```

---

## Integration with Training Pipeline

Add to the end of your training script:

```python
# After saving model and results
print("\nGenerating plots...")
try:
    from plots.xgboost_plots import plot_all_results
    
    plot_all_results(
        results_dir=results_dir,
        ticker=ticker,
        mode=args.mode,
        seed=args.seed,
        output_dir="plots",
        show=False
    )
    print("✓ Plots generated successfully")
except ImportError:
    print("Warning: plotting module not available")
except Exception as e:
    print(f"Warning: Could not generate plots: {e}")
```

---

## Dependencies

Required packages (already in `requirements.txt`):
- `matplotlib >= 3.8.0`
- `seaborn >= 0.13.0`
- `numpy >= 1.26.0`

Optional:
- `pandas` (for DataFrame export of tuning results)

---

## Examples

### Example 1: Quick Dashboard
```bash
python scripts/plot_xgboost_results.py --ticker AAPL --dashboard
```

### Example 2: Batch Processing
```bash
#!/bin/bash
# Generate dashboards for all trained models
for ticker in AAPL MSFT GOOGL NVDA TSLA AMD INTC; do
    echo "Processing $ticker..."
    python scripts/plot_xgboost_results.py \
        --ticker $ticker \
        --dashboard \
        --output-dir plots/dashboards/
done
```

### Example 3: Custom Analysis
```python
from plots.xgboost_plots import load_xgboost_results
import matplotlib.pyplot as plt

# Load results for multiple tickers
tickers = ["AAPL", "MSFT", "GOOGL"]
results_list = [load_xgboost_results("results", t) for t in tickers]

# Compare validation RMSE
fig, ax = plt.subplots(figsize=(10, 6))

for results in results_list:
    ticker = results["ticker"]
    val_rmse = results["history"]["val_rmse"]
    epochs = range(1, len(val_rmse) + 1)
    ax.plot(epochs, val_rmse, label=ticker, linewidth=2)

ax.set_xlabel('Boosting Round')
ax.set_ylabel('Validation RMSE')
ax.set_title('XGBoost Training Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('plots/comparison.png')
```

---

## Troubleshooting

### Issue: FileNotFoundError
**Cause:** Results files not found for the specified ticker.

**Solution:** Run training first:
```bash
python scripts/xgboost/run_xgboost.py --ticker AAPL --mode train
```

### Issue: Missing feature names
**Cause:** `metadata.pkl` not found or doesn't contain `feature_names`.

**Solution:** Plots will use generic labels. To fix, ensure feature engineering script saves metadata properly.

### Issue: Import errors
**Cause:** Required packages not installed.

**Solution:**
```bash
pip install matplotlib seaborn numpy
# Or install all requirements
pip install -r requirements.txt
```

### Issue: Plots look wrong
**Cause:** Incorrect lookback parameter or mismatched array shapes.

**Solution:** Check that `lookback` in config matches the value used during training.

---

## Testing

Validate syntax:
```bash
python3 -m py_compile plots/xgboost_plots.py
python3 -m py_compile scripts/plot_xgboost_results.py
```

Test with actual data (requires trained model):
```bash
# Test individual plots
python scripts/plot_xgboost_results.py --ticker AAPL

# Test dashboard
python scripts/plot_xgboost_results.py --ticker AAPL --dashboard

# Test interactive display (requires X11/display)
python scripts/plot_xgboost_results.py --ticker AAPL --show
```

---

## Future Enhancements

Possible additions:
1. **Prediction plots**: Actual vs predicted scatter plots
2. **Residual analysis**: Distribution and autocorrelation of residuals
3. **Learning curves**: Performance vs training set size
4. **Hyperparameter sensitivity**: Grid of metric vs hyperparameter
5. **Multi-model comparison**: Side-by-side comparison of different models
6. **Interactive plots**: Plotly/Bokeh versions for web dashboards
7. **PDF reports**: Automated LaTeX/PDF report generation

---

## Notes

- All plots use publication-quality settings (300 DPI)
- Color schemes are colorblind-friendly
- Plots are optimized for both screen display and printing
- File naming convention ensures no overwrites across different runs
- Module is fully documented with docstrings
- Code follows project style guidelines

---

## Summary

The XGBoost plotting system is **complete and ready to use**. It provides:

✓ 5 distinct plot types  
✓ Comprehensive dashboard view  
✓ Command-line and Python API  
✓ Publication-quality output  
✓ Extensive documentation  
✓ Error handling and validation  
✓ Batch processing support  

**Next steps:**
1. Install dependencies: `pip install matplotlib seaborn`
2. Run training: `python scripts/xgboost/run_xgboost.py --ticker AAPL --mode train`
3. Generate plots: `python scripts/plot_xgboost_results.py --ticker AAPL --dashboard`
