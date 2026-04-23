# Unified Backtesting System - Quick Usage Guide

**Version:** PRODUCTION_2.1  
**Date:** 2026-04-21

---

## Quick Start

### Basic Command
```bash
python pipelines/run_backtest.py \
    --model_type <MODEL_TYPE> \
    --model_path <PATH_TO_MODEL> \
    --ticker <TICKER>
```

---

## Command-Line Arguments

### Required Arguments

| Argument | Description | Choices | Example |
|----------|-------------|---------|---------|
| `--model_type` | Model type to backtest | `pso_lstm`, `lstm_baseline`, `xgboost` | `pso_lstm` |
| `--model_path` | Path to trained model | Any valid path | `results/pso_lstm/best_model.pt` |
| `--ticker` | Ticker symbol | Any valid ticker | `AAPL` |

### Optional Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--config` | Config file path | `config/default_config.yaml` | `config/my_config.yaml` |
| `--output_dir` | Output directory | `results/backtest/{model_type}/{ticker}` | `custom_results/` |
| `--split` | Data split to use | `test` | `val`, `train` |

---

## Usage Examples

### Example 1: PSO-LSTM on AAPL (Test Set)
```bash
python pipelines/run_backtest.py \
    --model_type pso_lstm \
    --model_path results/pso_lstm/best_model.pt \
    --ticker AAPL
```

**Output:** `results/backtest/pso_lstm/AAPL/test/`

---

### Example 2: Baseline LSTM on MSFT (Validation Set)
```bash
python pipelines/run_backtest.py \
    --model_type lstm_baseline \
    --model_path results/lstm_baseline/best_model.pt \
    --ticker MSFT \
    --split val
```

**Output:** `results/backtest/lstm_baseline/MSFT/val/`

---

### Example 3: XGBoost on GOOGL (Custom Output)
```bash
python pipelines/run_backtest.py \
    --model_type xgboost \
    --model_path results/xgboost/model.pkl \
    --ticker GOOGL \
    --output_dir custom_results/my_backtest
```

**Output:** `custom_results/my_backtest/`

---

## Output Files

After running the backtest, you'll find the following files in the output directory:

### 1. Metrics and Data

| File | Description | Format |
|------|-------------|--------|
| `backtest_results.json` | Complete metrics + summary statistics | JSON |
| `equity_curve.csv` | Daily time series (equity, returns, signals) | CSV |
| `predictions.csv` | Model predictions vs actual returns | CSV |
| `metadata.yaml` | Model info + backtest configuration | YAML |

### 2. Report

| File | Description |
|------|-------------|
| `backtest_report.md` | Human-readable summary with metrics tables |

### 3. Visualizations

| File | Description |
|------|-------------|
| `equity_curve.png` | Portfolio value vs buy-and-hold benchmark |
| `drawdown.png` | Drawdown chart over time |
| `returns_distribution.png` | Histogram of strategy returns |
| `signal_analysis.png` | Signal distribution and trading activity |

---

## Typical Workflow

### Step 1: Train Model
```bash
# Train baseline LSTM
python pipelines/train_baseline_lstm.py

# OR train PSO-LSTM
python pipelines/train_pso_lstm.py

# OR train XGBoost
python pipelines/train_xgboost.py
```

### Step 2: Run Backtest
```bash
python pipelines/run_backtest.py \
    --model_type lstm_baseline \
    --model_path results/lstm_baseline/best_model.pt \
    --ticker AAPL
```

### Step 3: Review Results
```bash
# Read the report
cat results/backtest/lstm_baseline/AAPL/test/backtest_report.md

# View metrics
cat results/backtest/lstm_baseline/AAPL/test/backtest_results.json

# Open visualizations
open results/backtest/lstm_baseline/AAPL/test/*.png
```

---

## Metrics Explained

### Statistical Metrics (Prediction Quality)

| Metric | Full Name | Interpretation | Good Value |
|--------|-----------|----------------|------------|
| RMSE | Root Mean Squared Error | Prediction accuracy | Lower is better |
| MAE | Mean Absolute Error | Average error magnitude | Lower is better |
| MAPE | Mean Absolute Percentage Error | Relative error (%) | Lower is better |
| R² | Coefficient of Determination | Variance explained | Higher is better (max 1.0) |
| DA | Directional Accuracy | Sign match rate | Higher is better (>0.5) |
| F1 | F1 Score (Ternary) | Classification quality | Higher is better (max 1.0) |

### Trading Metrics (Portfolio Performance)

| Metric | Full Name | Interpretation | Good Value |
|--------|-----------|----------------|------------|
| Sharpe | Sharpe Ratio | Risk-adjusted return | >1 good, >2 excellent |
| Sortino | Sortino Ratio | Downside risk-adjusted return | >1 good, >2 excellent |
| MDD | Maximum Drawdown | Worst peak-to-trough decline | Lower is better |
| CAGR | Compound Annual Growth Rate | Annualized return | Higher is better |
| Calmar | Calmar Ratio | CAGR / Max Drawdown | Higher is better |
| PF | Profit Factor | Gross profit / gross loss | >1 profitable, >2 strong |
| WR | Win Rate | Fraction of profitable trades | >0.5 good |
| IR | Information Ratio | Active return / active risk | Higher is better |

---

## Troubleshooting

### Error: Model file not found
```
FileNotFoundError: Model file not found: results/pso_lstm/best_model.pt
```

**Solution:** Ensure you've trained the model first:
```bash
python pipelines/train_pso_lstm.py
```

---

### Error: Features file not found
```
FileNotFoundError: Features file not found: data/processed/test_features.pkl
```

**Solution:** Run feature generation pipeline:
```bash
python pipelines/run_build_features.py
```

---

### Error: Ticker not found
```
ValueError: Ticker 'XYZ' not found in test split.
```

**Solution:** Ensure ticker was included in data ingestion and feature generation.

---

### Error: Model type mismatch
```
ValueError: Unknown model_type: 'lstm'
```

**Solution:** Use exact model type name:
- ✅ `pso_lstm`
- ✅ `lstm_baseline`
- ✅ `xgboost`
- ❌ `lstm` (use `lstm_baseline`)
- ❌ `pso` (use `pso_lstm`)

---

## Advanced Usage

### Custom Configuration
```bash
python pipelines/run_backtest.py \
    --model_type pso_lstm \
    --model_path results/pso_lstm/best_model.pt \
    --ticker AAPL \
    --config config/custom_config.yaml
```

### Batch Processing (Shell Script)
```bash
#!/bin/bash
# backtest_all_tickers.sh

for ticker in AAPL MSFT GOOGL AMZN; do
    echo "Backtesting $ticker..."
    python pipelines/run_backtest.py \
        --model_type pso_lstm \
        --model_path results/pso_lstm/best_model.pt \
        --ticker $ticker
done
```

---

## Expected Runtime

| Data Size | Typical Runtime |
|-----------|----------------|
| ~500 days | 5-10 seconds |
| ~1000 days | 10-20 seconds |
| ~2000 days | 20-40 seconds |

*Note: Runtime varies based on model complexity and hardware.*

---

## Output Interpretation

### Reading the Equity Curve
- **Blue line:** Your strategy's portfolio value
- **Orange dashed line:** Buy-and-hold benchmark
- **Above benchmark:** Strategy outperforms
- **Below benchmark:** Strategy underperforms

### Reading the Drawdown Chart
- **Red area:** Underwater equity (drawdown)
- **Depth:** How much you're down from peak
- **Duration:** How long drawdown lasts
- **Recovery:** Return to previous peak

### Reading Signal Analysis
- **Left plot:** Distribution of signals (Long/Neutral/Short)
- **Right plot:** Trading activity over time
- **High bars:** Frequent position changes (high turnover)

---

## Best Practices

### 1. Always Use Test Set
```bash
# ✅ CORRECT: Backtest on unseen test data
python pipelines/run_backtest.py ... --split test

# ❌ WRONG: Backtesting on training data
python pipelines/run_backtest.py ... --split train
```

### 2. Compare Models Fairly
Use the same ticker and split for all models:
```bash
# PSO-LSTM
python pipelines/run_backtest.py --model_type pso_lstm ... --ticker AAPL

# Baseline LSTM
python pipelines/run_backtest.py --model_type lstm_baseline ... --ticker AAPL

# XGBoost
python pipelines/run_backtest.py --model_type xgboost ... --ticker AAPL
```

### 3. Check Multiple Metrics
Don't rely on a single metric. Look at:
- **Sharpe Ratio** (risk-adjusted return)
- **Max Drawdown** (downside risk)
- **Win Rate** (consistency)
- **Directional Accuracy** (prediction quality)

### 4. Validate Visualizations
Always check the plots to verify:
- Equity curve looks reasonable (no sudden jumps)
- Drawdowns are acceptable
- Returns distribution is not extreme

---

## Common Questions

### Q: Can I backtest multiple tickers at once?
**A:** Currently, the pipeline processes one ticker at a time. Use a shell script for batch processing (see Advanced Usage above).

### Q: Can I change transaction costs?
**A:** Yes, modify `config/default_config.yaml`:
```yaml
backtesting:
  transaction_cost: 0.0015  # Change this (0.15% default)
```

### Q: What if my model outputs probabilities instead of returns?
**A:** The current system expects return predictions. You'll need to adapt the model output to return format.

### Q: Can I use this for intraday data?
**A:** The system is designed for daily data. Intraday support would require modifications to the backtesting engine.

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review `BACKTEST_DESIGN.md` for architecture details
3. Review `BACKTEST_IMPLEMENTATION_COMPLETE.md` for technical details

---

**Last Updated:** 2026-04-21  
**Version:** PRODUCTION_2.1
