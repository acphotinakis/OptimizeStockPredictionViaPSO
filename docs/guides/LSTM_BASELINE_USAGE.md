# LSTM Baseline Usage Guide

This guide explains how to use the `run_lstm_baseline.py` script to train and evaluate the vanilla LSTM baseline model.

## Overview

The LSTM baseline provides a fixed-hyperparameter LSTM model for comparison against the PSO-optimized LSTM. It uses manually chosen "sensible" hyperparameters based on literature recommendations.

## Quick Start

### Training

Train the LSTM baseline on a ticker:

```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --seed 42
```

### Validation

Run walk-forward validation on the trained model:

```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode val \
    --seed 42
```

### Testing

Evaluate on the test set:

```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode test \
    --seed 42
```

## Command-Line Arguments

### Common Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ticker` | str | **required** | Target ticker symbol |
| `--mode` | str | **required** | Execution mode: train, val, or test |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--features-dir` | str | data/features | Directory containing feature arrays |
| `--results-dir` | str | results | Output directory for results |
| `--log-file` | str | logs/run_lstm_baseline.log | Log file path |
| `--config` | str | config/default_config.yaml | Config file path |

### Model Hyperparameters (Training Mode)

Override default hyperparameters:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-layers` | int | 2 | Number of LSTM layers |
| `--hidden-units` | int | 128 | Hidden units per layer |
| `--dropout` | float | 0.2 | Dropout probability |
| `--learning-rate` | float | 0.001 | Adam learning rate |
| `--lookback` | int | 30 | Sequence length (timesteps) |
| `--max-epochs` | int | 100 | Maximum training epochs |
| `--patience` | int | 10 | Early stopping patience |
| `--batch-size` | int | 256 | Training batch size |

### Validation Mode Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--wfv-fold-size` | int | 252 | Samples per fold (~1 trading year) |
| `--wfv-folds` | int | 10 | Maximum number of folds |

### Testing Mode Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--initial-capital` | float | 100000.0 | Starting portfolio value |
| `--position-fraction` | float | 0.02 | Position size as fraction of capital |
| `--transaction-cost` | float | 0.001 | Transaction cost (0.1%) |
| `--slippage` | float | 0.0005 | Slippage (0.05%) |
| `--stop-loss` | float | 0.02 | Stop loss threshold (2%) |
| `--daily-loss-limit` | float | 0.05 | Daily loss limit (5%) |

## Usage Examples

### Example 1: Basic Training

Train with default hyperparameters:

```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --seed 42
```

**Output:**
- `results/lstm_baseline_model_AAPL_train_seed42.pth` - Model weights
- `results/lstm_baseline_params_AAPL_train_seed42.json` - Hyperparameters and metrics
- `results/lstm_baseline_history_AAPL_train_seed42.json` - Training history
- `results/lstm_baseline_predictions_AAPL_train_seed42.npy` - Validation predictions
- `results/plots/lstm_baseline_train_AAPL_train_seed42.png` - Training plots

### Example 2: Custom Hyperparameters

Train with custom settings:

```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --num-layers 3 \
    --hidden-units 256 \
    --dropout 0.3 \
    --learning-rate 0.0005 \
    --lookback 60 \
    --batch-size 128 \
    --max-epochs 200
```

### Example 3: Walk-Forward Validation

Validate model stability over time:

```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode val \
    --seed 42 \
    --wfv-fold-size 252 \
    --wfv-folds 10
```

**Output:**
- `results/lstm_baseline_wfv_AAPL_val_seed42.json` - Fold metrics and aggregates

### Example 4: Test Set Evaluation

Final evaluation on held-out test set:

```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode test \
    --seed 42
```

**Output:**
- `results/lstm_baseline_test_AAPL_test_seed42.json` - Test metrics
- `results/lstm_baseline_test_predictions_AAPL_test_seed42.npy` - Test predictions
- `results/plots/lstm_baseline_test_AAPL_test_seed42.png` - Test predictions plot

### Example 5: Multiple Tickers

Train on multiple tickers:

```bash
for ticker in AAPL MSFT GOOGL TSLA; do
    python scripts/run_lstm_baseline.py \
        --ticker $ticker \
        --mode train \
        --seed 42
done
```

### Example 6: Different Seeds

Train with multiple seeds for robustness:

```bash
for seed in 42 123 456 789; do
    python scripts/run_lstm_baseline.py \
        --ticker AAPL \
        --mode train \
        --seed $seed
done
```

## Output Files

### Training Mode

After training, the following files are created:

```
results/
├── lstm_baseline_model_{ticker}_train_seed{seed}.pth
├── lstm_baseline_params_{ticker}_train_seed{seed}.json
├── lstm_baseline_history_{ticker}_train_seed{seed}.json
├── lstm_baseline_predictions_{ticker}_train_seed{seed}.npy
└── plots/
    └── lstm_baseline_train_{ticker}_train_seed{seed}.png
```

### Validation Mode

```
results/
└── lstm_baseline_wfv_{ticker}_val_seed{seed}.json
```

### Test Mode

```
results/
├── lstm_baseline_test_{ticker}_test_seed{seed}.json
├── lstm_baseline_test_predictions_{ticker}_test_seed{seed}.npy
└── plots/
    └── lstm_baseline_test_{ticker}_test_seed{seed}.png
```

## Configuration File

Default hyperparameters are defined in `config/default_config.yaml`:

```yaml
lstm_baseline:
  # Fixed hyperparameters
  num_layers: 2
  hidden_units: 128
  dropout: 0.2
  learning_rate: 0.001
  lookback: 30
  max_epochs: 100
  patience: 10
  batch_size: 256
  
  # Training settings
  grad_clip: 1.0
  use_amp: true
  accumulation_steps: 1
  
  # Walk-forward validation
  wfv_fold_size: 252
  wfv_folds: 10
  
  # Backtesting
  initial_capital: 100000.0
  position_fraction: 0.02
  transaction_cost: 0.001
  slippage: 0.0005
  stop_loss: 0.02
  daily_loss_limit: 0.05
```

## Reading Results

### Load Training Parameters

```python
import json

with open("results/lstm_baseline_params_AAPL_train_seed42.json") as f:
    params = json.load(f)

print(f"Hyperparameters: {params['hyperparameters']}")
print(f"Validation RMSE: {params['val_metrics']['rmse']:.6f}")
print(f"Training time: {params['runtime_seconds']:.1f}s")
```

### Load Training History

```python
import json
import matplotlib.pyplot as plt

with open("results/lstm_baseline_history_AAPL_train_seed42.json") as f:
    history = json.load(f)

plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()
```

### Load Predictions

```python
import numpy as np

y_pred = np.load("results/lstm_baseline_predictions_AAPL_train_seed42.npy")
print(f"Predictions shape: {y_pred.shape}")
print(f"Mean prediction: {y_pred.mean():.6f}")
print(f"Std prediction: {y_pred.std():.6f}")
```

### Load Validation Results

```python
import json

with open("results/lstm_baseline_wfv_AAPL_val_seed42.json") as f:
    wfv = json.load(f)

# Aggregate metrics
agg = wfv['aggregate_metrics']
print(f"RMSE: {agg['rmse']['mean']:.6f} ± {agg['rmse']['std']:.6f}")
print(f"Sharpe: {agg['sharpe']['mean']:.4f} ± {agg['sharpe']['std']:.4f}")

# Per-fold metrics
for fold in wfv['fold_metrics']:
    print(f"Fold {fold['fold']}: RMSE={fold['metrics']['rmse']:.6f}")
```

## Comparison with PSO-Optimized LSTM

### Key Differences

| Aspect | LSTM Baseline | PSO-Optimized LSTM |
|--------|---------------|-------------------|
| **Hyperparameters** | Fixed (manually chosen) | Optimized via IPSO |
| **Training Time** | Fast (~10-30 min) | Slow (~6-12 hours) |
| **Performance** | Good baseline | Better (optimized) |
| **Use Case** | Quick baseline | Production model |
| **Script** | `run_lstm_baseline.py` | `run_pso.py` |
| **Reproducibility** | High (fixed params) | Medium (stochastic) |
| **Interpretability** | High | Medium |

### When to Use Each

**Use LSTM Baseline:**
- Quick experiments and prototyping
- Baseline comparisons for research
- Resource-constrained environments
- Initial model validation
- When interpretability is important

**Use PSO-Optimized LSTM:**
- Production deployment
- Maximum performance needed
- Sufficient compute resources available
- Final model selection
- When performance > interpretability

### Running Both for Comparison

```bash
# Train baseline
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --seed 42

# Train PSO-optimized
python scripts/run_pso.py \
    --ticker AAPL \
    --config config/default_config.yaml \
    --seed 42

# Compare results
python scripts/compare_models.py \
    --ticker AAPL \
    --models lstm_baseline pso_lstm
```

## Troubleshooting

### Issue: Model not found during validation/testing

**Error:**
```
FileNotFoundError: Model not found: results/lstm_baseline_model_AAPL_train_seed42.pth
```

**Solution:**
Run training first:
```bash
python scripts/run_lstm_baseline.py --ticker AAPL --mode train --seed 42
```

### Issue: Out of memory during training

**Solution 1:** Reduce batch size:
```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --batch-size 64
```

**Solution 2:** Reduce model size:
```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --num-layers 1 \
    --hidden-units 64
```

### Issue: Training is too slow

**Solution 1:** Reduce max epochs:
```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --max-epochs 50
```

**Solution 2:** Increase batch size (if memory allows):
```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --batch-size 512
```

### Issue: Model not converging

**Solution 1:** Adjust learning rate:
```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --learning-rate 0.0001
```

**Solution 2:** Increase patience:
```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --patience 20
```

## Best Practices

1. **Always set a seed** for reproducibility:
   ```bash
   --seed 42
   ```

2. **Run validation** after training to check model stability:
   ```bash
   # Train
   python scripts/run_lstm_baseline.py --ticker AAPL --mode train --seed 42
   # Validate
   python scripts/run_lstm_baseline.py --ticker AAPL --mode val --seed 42
   ```

3. **Use multiple seeds** for robust evaluation:
   ```bash
   for seed in 42 123 456; do
       python scripts/run_lstm_baseline.py --ticker AAPL --mode train --seed $seed
   done
   ```

4. **Monitor training** via logs:
   ```bash
   tail -f logs/run_lstm_baseline.log
   ```

5. **Check plots** to diagnose issues:
   - Training/validation loss curves
   - Prediction vs true values
   - Look for overfitting, underfitting, or instability

## Advanced Usage

### Custom Config File

Create a custom config file:

```yaml
# config/lstm_baseline_custom.yaml
lstm_baseline:
  num_layers: 3
  hidden_units: 256
  dropout: 0.3
  learning_rate: 0.0005
  lookback: 60
  max_epochs: 200
  patience: 20
  batch_size: 128
```

Use it:

```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --config config/lstm_baseline_custom.yaml
```

### Batch Processing

Process multiple tickers in parallel:

```bash
# Create a script: batch_train.sh
#!/bin/bash
TICKERS=("AAPL" "MSFT" "GOOGL" "TSLA" "AMZN")

for ticker in "${TICKERS[@]}"; do
    python scripts/run_lstm_baseline.py \
        --ticker $ticker \
        --mode train \
        --seed 42 &
done

wait
echo "All training jobs complete!"
```

Run it:
```bash
chmod +x batch_train.sh
./batch_train.sh
```

## Related Documentation

- [LSTM Baseline Implementation Plan](../plans/LSTM_BASELINE_PLAN.md)
- [Model Architecture Overview](../overviews/model_architecture.md)
- [PSO-LSTM Usage Guide](PSO_LSTM_USAGE.md)
- [Experiment Plan](../overviews/experiment_plan.md)

## Support

For issues or questions:
1. Check the logs: `logs/run_lstm_baseline.log`
2. Review the implementation plan: `docs/plans/LSTM_BASELINE_PLAN.md`
3. Check existing issues in the project repository
4. Create a new issue with:
   - Command used
   - Error message
   - Log file excerpt
   - System information

---

**Last Updated:** April 6, 2026
