# XGBoost Plotting - Usage Examples

## Quick Start

### 1. Generate All Plots for a Ticker

```bash
# Generate all individual plots
python scripts/plot_xgboost_results.py --ticker AAPL

# Or use the module directly
python plots/xgboost_plots.py --ticker AAPL
```

This creates 4 plots in the `plots/` directory:
- `xgb_training_curves_AAPL_train_seed42.png`
- `xgb_feature_importances_AAPL_train_seed42.png`
- `xgb_metrics_summary_AAPL_train_seed42.png`
- `xgb_hyperparameters_AAPL_train_seed42.png`

### 2. Create a Single Dashboard

```bash
# Create comprehensive dashboard
python scripts/plot_xgboost_results.py --ticker AAPL --dashboard

# Show interactively (requires display)
python scripts/plot_xgboost_results.py --ticker AAPL --dashboard --show
```

This creates a single file:
- `xgb_dashboard_AAPL_train_seed42.png`

### 3. Batch Process Multiple Tickers

```bash
# Generate plots for all trained models
for ticker in AAPL MSFT GOOGL NVDA; do
    python scripts/plot_xgboost_results.py --ticker $ticker
done
```

### 4. Custom Output Directory

```bash
# Save to custom directory
python scripts/plot_xgboost_results.py \
    --ticker AAPL \
    --output-dir figures/xgboost \
    --dashboard
```

## Python API Examples

### Example 1: Generate All Plots

```python
from plots.xgboost_plots import plot_all_results

# Generate all plots for AAPL
figures = plot_all_results(
    results_dir="results",
    ticker="AAPL",
    mode="train",
    seed=42,
    output_dir="plots",
    show=False  # Set to True to display interactively
)

# Access individual figures
training_fig = figures["training_curves"]
importance_fig = figures["feature_importances"]
metrics_fig = figures["metrics_summary"]
params_fig = figures["hyperparameters"]
```

### Example 2: Create Dashboard

```python
from plots.xgboost_plots import create_summary_dashboard

# Create comprehensive dashboard
fig = create_summary_dashboard(
    results_dir="results",
    ticker="AAPL",
    mode="train",
    seed=42,
    save_path="plots/AAPL_dashboard.png",
    show=True  # Display interactively
)
```

### Example 3: Individual Plots with Customization

```python
from plots.xgboost_plots import (
    load_xgboost_results,
    plot_training_curves,
    plot_feature_importances,
)

# Load results
results = load_xgboost_results("results", "AAPL", mode="train", seed=42)

# Plot training curves with custom styling
fig1 = plot_training_curves(
    results["history"],
    results["params"],
    save_path="plots/custom_training.png",
    show=False
)

# Plot top 15 features instead of default 20
fig2 = plot_feature_importances(
    results["importances"],
    results["feature_names"],
    lookback=30,
    top_k=15,  # Show only top 15
    save_path="plots/custom_importances.png",
    show=False
)
```

### Example 4: Programmatic Batch Processing

```python
from pathlib import Path
from plots.xgboost_plots import plot_all_results

# Define tickers
tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

# Generate plots for each
for ticker in tickers:
    try:
        print(f"Processing {ticker}...")
        figures = plot_all_results(
            results_dir="results",
            ticker=ticker,
            mode="train",
            seed=42,
            output_dir=f"plots/{ticker}",
            show=False
        )
        print(f"✓ {ticker} complete ({len(figures)} plots)")
    except FileNotFoundError:
        print(f"✗ {ticker} - no results found")
    except Exception as e:
        print(f"✗ {ticker} - error: {e}")
```

### Example 5: Compare Multiple Seeds

```python
from plots.xgboost_plots import load_xgboost_results, plot_training_curves
import matplotlib.pyplot as plt

ticker = "AAPL"
seeds = [42, 123, 456]

fig, axes = plt.subplots(1, len(seeds), figsize=(15, 5))

for i, seed in enumerate(seeds):
    results = load_xgboost_results("results", ticker, mode="train", seed=seed)
    
    ax = axes[i]
    val_rmse = results["history"]["val_rmse"]
    epochs = range(1, len(val_rmse) + 1)
    
    ax.plot(epochs, val_rmse, linewidth=2)
    ax.set_title(f'Seed {seed}')
    ax.set_xlabel('Boosting Round')
    ax.set_ylabel('Validation RMSE')
    ax.grid(True, alpha=0.3)

plt.suptitle(f'{ticker} - Training Curves Across Seeds', fontweight='bold')
plt.tight_layout()
plt.savefig(f'plots/{ticker}_seed_comparison.png')
```

### Example 6: Extract Specific Metrics

```python
from plots.xgboost_plots import load_xgboost_results

# Load results
results = load_xgboost_results("results", "AAPL", mode="train", seed=42)

# Extract metrics
val_metrics = results["params"]["val_metrics"]

print(f"Ticker: {results['ticker']}")
print(f"Best Iteration: {results['params']['best_iteration']}")
print(f"Runtime: {results['params']['runtime_seconds']:.2f}s")
print("\nValidation Metrics:")
print(f"  RMSE: {val_metrics['rmse']:.6f}")
print(f"  MAE: {val_metrics['mae']:.6f}")
print(f"  R²: {val_metrics['r2']:.6f}")
print(f"  Directional Accuracy: {val_metrics['directional_accuracy']:.4f}")
print(f"  F1 (Ternary): {val_metrics['f1_ternary']:.4f}")

# Get top features
importances = results["importances"]
feature_names = results["feature_names"]
lookback = results["params"]["hyperparameters"]["lookback"]

# Aggregate importances
n_features = len(feature_names)
imp_reshaped = importances.reshape(lookback, n_features)
imp_aggregated = imp_reshaped.mean(axis=0)

# Sort and display
sorted_idx = imp_aggregated.argsort()[::-1]
print("\nTop 5 Features:")
for i, idx in enumerate(sorted_idx[:5], 1):
    print(f"  {i}. {feature_names[idx]}: {imp_aggregated[idx]:.2f}")
```

## Integration with Training Scripts

Add plotting to your training pipeline:

```python
# At the end of your training script
from plots.xgboost_plots import plot_all_results

# After model.save() and saving params/history
print("\nGenerating plots...")
try:
    plot_all_results(
        results_dir=results_dir,
        ticker=ticker,
        mode="train",
        seed=seed,
        output_dir="plots",
        show=False
    )
    print("✓ Plots generated successfully")
except Exception as e:
    print(f"Warning: Could not generate plots: {e}")
```

## Troubleshooting

### FileNotFoundError

```
FileNotFoundError: Missing results/xgb_history_AAPL_train_seed42.json
```

**Solution:** Run training first:
```bash
python scripts/xgboost/run_xgboost.py --ticker AAPL --mode train
```

### Missing Feature Names

If feature names are not found, the plots will use generic labels like "Feature 0", "Feature 1", etc.

**Solution:** Ensure `data/features/TICKER/metadata.pkl` exists with a `feature_names` key.

### Import Errors

```
ModuleNotFoundError: No module named 'matplotlib'
```

**Solution:** Install required packages:
```bash
pip install matplotlib seaborn numpy
# Or install all requirements
pip install -r requirements.txt
```

## Advanced Customization

### Custom Color Schemes

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set custom color palette before plotting
sns.set_palette("Set2")
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)

# Then create plots as usual
from plots.xgboost_plots import plot_all_results
plot_all_results("results", "AAPL")
```

### Custom Figure Sizes

Modify the plotting functions or use matplotlib directly:

```python
from plots.xgboost_plots import load_xgboost_results
import matplotlib.pyplot as plt

results = load_xgboost_results("results", "AAPL")

# Create custom-sized figure
fig, ax = plt.subplots(figsize=(12, 8))

val_rmse = results["history"]["val_rmse"]
epochs = range(1, len(val_rmse) + 1)

ax.plot(epochs, val_rmse, linewidth=3, color='darkblue')
ax.set_xlabel('Boosting Round', fontsize=14)
ax.set_ylabel('Validation RMSE', fontsize=14)
ax.set_title('Custom Training Curve', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.savefig('plots/custom_figure.png', dpi=300, bbox_inches='tight')
```

## Tips

1. **Use `--dashboard` for quick overview**: Single comprehensive plot is faster than 4 separate plots
2. **Batch process at night**: Generate plots for all tickers in a loop
3. **Version control plots**: Add `plots/*.png` to `.gitignore` if plots are regenerated frequently
4. **High-DPI displays**: Plots are saved at 300 DPI by default for publication quality
5. **Interactive exploration**: Use `--show` flag during development to see plots immediately
