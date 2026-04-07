# XGBoost Plotting - Quick Reference

## One-Liners

```bash
# Generate all plots
python scripts/plot_xgboost_results.py --ticker AAPL

# Dashboard only
python scripts/plot_xgboost_results.py --ticker AAPL --dashboard

# Show interactively
python scripts/plot_xgboost_results.py --ticker AAPL --dashboard --show

# Batch process
for t in AAPL MSFT GOOGL; do python scripts/plot_xgboost_results.py --ticker $t --dashboard; done
```

## Python Quick Start

```python
# All plots
from plots.xgboost_plots import plot_all_results
plot_all_results("results", "AAPL")

# Dashboard
from plots.xgboost_plots import create_summary_dashboard
create_summary_dashboard("results", "AAPL", save_path="plots/dashboard.png")

# Load data
from plots.xgboost_plots import load_xgboost_results
results = load_xgboost_results("results", "AAPL")
```

## File Locations

### Input (from training)
- `results/xgb_history_TICKER_train_seed42.json`
- `results/xgb_params_TICKER_train_seed42.json`
- `results/xgb_importances_TICKER_train_seed42.npy`
- `data/features/TICKER/metadata.pkl` (optional)

### Output (generated)
- `plots/xgb_training_curves_TICKER_train_seed42.png`
- `plots/xgb_feature_importances_TICKER_train_seed42.png`
- `plots/xgb_metrics_summary_TICKER_train_seed42.png`
- `plots/xgb_hyperparameters_TICKER_train_seed42.png`
- `plots/xgb_dashboard_TICKER_train_seed42.png` (with --dashboard)

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ticker` | Stock symbol (required) | - |
| `--results-dir` | Results directory | `results` |
| `--output-dir` | Output directory | `plots` |
| `--mode` | train/val/test | `train` |
| `--seed` | Random seed | `42` |
| `--dashboard` | Single dashboard plot | `False` |
| `--show` | Display interactively | `False` |

## Plot Types

1. **Training Curves** - RMSE over boosting rounds
2. **Feature Importances** - Top K features bar chart
3. **Metrics Summary** - Validation metrics
4. **Hyperparameters** - Model configuration table
5. **Dashboard** - All-in-one comprehensive view

## Troubleshooting

| Error | Solution |
|-------|----------|
| `FileNotFoundError` | Run training first: `python scripts/xgboost/run_xgboost.py --ticker AAPL --mode train` |
| `ModuleNotFoundError` | Install deps: `pip install matplotlib seaborn numpy` |
| Missing feature names | Optional - plots will use generic labels |

## Integration

Add to training script:
```python
from plots.xgboost_plots import plot_all_results
plot_all_results(results_dir, ticker, mode, seed, "plots", show=False)
```

## Full Documentation

- **Module docs**: `plots/README.md`
- **Examples**: `plots/USAGE_EXAMPLES.md`
- **System docs**: `docs/PLOTTING_SYSTEM.md`
