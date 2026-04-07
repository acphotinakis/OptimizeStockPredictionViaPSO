# XGBoost Plotting Module

Publication-quality visualization tools for XGBoost model training results.

## Features

- **Training Curves**: Plot train/validation RMSE over boosting rounds
- **Feature Importances**: Visualize top contributing features
- **Metrics Summary**: Bar charts of validation performance metrics
- **Hyperparameters Table**: Formatted display of model configuration
- **Summary Dashboard**: Comprehensive single-page overview

## Usage

### Command Line Interface

Generate all plots for a ticker:
```bash
python plots/xgboost_plots.py --ticker AAPL --results-dir results --output-dir plots
```

Create a single dashboard:
```bash
python plots/xgboost_plots.py --ticker AAPL --dashboard
```

Show plots interactively:
```bash
python plots/xgboost_plots.py --ticker AAPL --show
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

# Or create a single dashboard
fig = create_summary_dashboard(
    results_dir="results",
    ticker="AAPL",
    mode="train",
    seed=42,
    save_path="plots/AAPL_dashboard.png",
    show=True
)
```

### Individual Plot Functions

```python
from plots.xgboost_plots import (
    load_xgboost_results,
    plot_training_curves,
    plot_feature_importances,
    plot_metrics_summary,
    plot_hyperparameters,
)

# Load results
results = load_xgboost_results("results", "AAPL", mode="train", seed=42)

# Plot training curves
plot_training_curves(
    results["history"],
    results["params"],
    save_path="plots/training_curves.png"
)

# Plot feature importances
plot_feature_importances(
    results["importances"],
    results["feature_names"],
    lookback=30,
    top_k=20,
    save_path="plots/importances.png"
)

# Plot metrics summary
plot_metrics_summary(
    results["params"],
    save_path="plots/metrics.png"
)

# Plot hyperparameters
plot_hyperparameters(
    results["params"],
    save_path="plots/hyperparams.png"
)
```

## Output Files

For a given ticker (e.g., AAPL) with mode=train and seed=42:

- `xgb_training_curves_AAPL_train_seed42.png` - Training/validation loss curves
- `xgb_feature_importances_AAPL_train_seed42.png` - Top feature importances
- `xgb_metrics_summary_AAPL_train_seed42.png` - Validation metrics bar chart
- `xgb_hyperparameters_AAPL_train_seed42.png` - Hyperparameters table
- `xgb_dashboard_AAPL_train_seed42.png` - Comprehensive dashboard (if using --dashboard)

## Requirements

- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- numpy >= 1.20.0

## Plot Descriptions

### Training Curves
Shows RMSE over boosting rounds for both training and validation sets. Marks the best iteration with a vertical line and star marker.

### Feature Importances
Horizontal bar chart of the top K most important features. For windowed features, aggregates importance scores across time steps.

### Metrics Summary
Bar chart visualization of key validation metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- Directional Accuracy
- F1 Score (Ternary)
- AUC (Ternary)

Automatically separates small-scale and large-scale metrics for better visualization.

### Hyperparameters Table
Formatted table displaying all XGBoost hyperparameters used in training.

### Summary Dashboard
Comprehensive 6-panel dashboard combining:
1. Training curves
2. Top 10 feature importances
3. Validation metrics
4. Key hyperparameters
5. Training information

## Customization

All plotting functions accept optional parameters for customization:

```python
plot_training_curves(
    history,
    params,
    save_path="custom_path.png",
    show=True  # Display interactively
)

plot_feature_importances(
    importances,
    feature_names,
    lookback=30,
    top_k=15,  # Show top 15 instead of 20
    save_path="importances.png"
)
```

## Examples

See `scripts/xgboost/` for example training scripts that generate the required result files.
