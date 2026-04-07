# LSTM SPY Analysis Guide
## Complete Workflow for Signal Generation and Visualization

**Last Updated**: April 7, 2026  
**Status**: Production Ready

---

## Overview

This guide explains how to train an LSTM model on SPY (or any ticker), generate trading signals, and create comprehensive visualizations with OHLCV charts.

### What's New

The updated pipeline now includes:
- ✅ **Signal generation** with configurable thresholds (±0.1% default)
- ✅ **OHLCV chart integration** with candlesticks and volume
- ✅ **Trading signal visualization** (buy/sell markers on price chart)
- ✅ **Signal statistics** (distribution, counts, percentages)
- ✅ **Comprehensive plots** (3-panel: price + volume + predictions)
- ✅ **Text summaries** with signal and prediction statistics

---

## Quick Start

### Option 1: Automated Script (Recommended)

Run the complete pipeline for SPY:

```bash
./scripts/run_spy_analysis.sh
```

This will:
1. Train LSTM model on SPY
2. Run walk-forward validation
3. Test on hold-out set
4. Generate trading signals
5. Create visualizations

### Option 2: Manual Step-by-Step

```bash
# Step 1: Train
python scripts/run_lstm_baseline.py --ticker SPY --mode train --seed 42

# Step 2: Validate
python scripts/run_lstm_baseline.py --ticker SPY --mode val --seed 42

# Step 3: Test (generates signals)
python scripts/run_lstm_baseline.py --ticker SPY --mode test --seed 42

# Step 4: Visualize
python plots/plot_lstm.py --ticker SPY --seed 42
```

---

## Detailed Usage

### 1. Training

```bash
python scripts/run_lstm_baseline.py \
    --ticker SPY \
    --mode train \
    --seed 42 \
    --num-layers 2 \
    --hidden-units 128 \
    --dropout 0.2 \
    --learning-rate 0.001 \
    --lookback 30 \
    --max-epochs 100 \
    --patience 10 \
    --batch-size 256
```

**Outputs**:
- `results/lstm_baseline_model_SPY_train_seed42.pth` - Trained model weights
- `results/lstm_baseline_params_SPY_train_seed42.json` - Hyperparameters
- `results/lstm_baseline_history_SPY_train_seed42.json` - Training history
- `results/plots/lstm_baseline_train_SPY_train_seed42.png` - Training plot

### 2. Validation (Walk-Forward)

```bash
python scripts/run_lstm_baseline.py \
    --ticker SPY \
    --mode val \
    --seed 42 \
    --wfv-fold-size 252 \
    --wfv-folds 10
```

**Outputs**:
- `results/lstm_baseline_wfv_SPY_val_seed42.json` - Walk-forward metrics

### 3. Testing (Signal Generation)

```bash
python scripts/run_lstm_baseline.py \
    --ticker SPY \
    --mode test \
    --seed 42
```

**Outputs**:
- `results/lstm_aligned_SPY_test_seed42.csv` - **Main output**: Aligned OHLCV + predictions + signals
- `results/lstm_baseline_test_SPY_test_seed42.json` - Test metrics + signal statistics
- `results/lstm_baseline_test_predictions_SPY_test_seed42.npy` - Raw predictions
- `results/lstm_baseline_test_signals_SPY_test_seed42.npy` - Raw signals
- `results/plots/lstm_baseline_test_SPY_test_seed42.png` - Prediction plot
- `results/plots/lstm_signals_SPY_test_seed42.png` - Signal distribution plot

### 4. Visualization

```bash
python plots/plot_lstm.py \
    --ticker SPY \
    --seed 42 \
    --results-dir results \
    --output results/plots/spy_signals.png
```

**Outputs**:
- `results/plots/lstm_ohlcv_SPY_seed42.png` - **Main visualization**: 3-panel OHLCV + signals
- `results/plots/lstm_ohlcv_SPY_seed42.txt` - Text summary of signals

---

## Output Files Explained

### CSV: `lstm_aligned_SPY_test_seed42.csv`

Contains aligned OHLCV data with predictions and signals:

| Column | Description |
|--------|-------------|
| `date` | Timestamp (if available) |
| `open` | Opening price |
| `high` | High price |
| `low` | Low price |
| `close` | Closing price |
| `volume` | Trading volume |
| `actual_return` | Actual log return |
| `predicted_return` | LSTM predicted log return |
| `signal` | Trading signal: `1` = Buy, `-1` = Sell, `0` = Hold |

**Example**:
```csv
date,open,high,low,close,volume,actual_return,predicted_return,signal
2025-01-02,475.23,476.89,474.12,476.45,85234567,0.0025,0.0018,1
2025-01-03,476.50,477.12,475.89,476.23,-0.0005,-0.0003,0
2025-01-06,476.00,478.45,475.67,478.12,92145678,0.0040,0.0035,1
```

### JSON: `lstm_baseline_test_SPY_test_seed42.json`

Contains test metrics and signal statistics:

```json
{
  "ticker": "SPY",
  "statistical_metrics": {
    "rmse": 0.012345,
    "mae": 0.009876,
    "r2": 0.234567,
    "directional_accuracy": 0.567890,
    "f1_ternary": 0.456789
  },
  "signal_statistics": {
    "total_predictions": 37656,
    "buy_signals": 8234,
    "sell_signals": 7891,
    "hold_signals": 21531,
    "buy_pct": 0.2187,
    "sell_pct": 0.2096,
    "hold_pct": 0.5717
  },
  "signal_threshold": 0.001
}
```

### Plot: `lstm_ohlcv_SPY_seed42.png`

**3-panel chart**:

1. **Top Panel**: Candlestick chart with:
   - Green/red candles (bullish/bearish)
   - Buy signals (green triangles ▲)
   - Sell signals (red triangles ▼)
   - Prediction strength (background shading)

2. **Middle Panel**: Volume bars
   - Green = up day
   - Red = down day

3. **Bottom Panel**: Predicted returns
   - Blue line = predictions
   - Green/red shading = positive/negative
   - Dashed lines = signal thresholds (±0.1%)
   - Markers = actual signals

---

## Signal Generation Logic

### Thresholds

```python
signal_threshold = 0.001  # 0.1% return threshold

if predicted_return > signal_threshold:
    signal = 1   # BUY
elif predicted_return < -signal_threshold:
    signal = -1  # SELL
else:
    signal = 0   # HOLD
```

### Why 0.1%?

- **Transaction costs**: ~10 bps (0.1%) per trade
- **Slippage**: ~5 bps (0.05%)
- **Total cost**: ~15 bps (0.15%) round-trip
- **Threshold**: 10 bps (0.1%) ensures signal > noise

**To change threshold**: Edit `signal_threshold` in `run_lstm_baseline.py` line 471.

---

## Interpreting Results

### Good Signal Distribution

```
Buy signals:  8,234 (21.9%)
Sell signals: 7,891 (21.0%)
Hold signals: 21,531 (57.2%)
```

**Interpretation**:
- ✅ Balanced buy/sell (similar counts)
- ✅ Majority hold (model is selective)
- ✅ ~40% actionable signals

### Bad Signal Distribution

```
Buy signals:  35,000 (93.0%)
Sell signals: 500 (1.3%)
Hold signals: 2,156 (5.7%)
```

**Interpretation**:
- ❌ Heavily skewed (model is biased)
- ❌ Too many signals (overtrading)
- ❌ Likely to lose money on transaction costs

### Prediction Quality Metrics

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| **RMSE** | < 0.01 | 0.01 - 0.02 | > 0.02 |
| **Directional Accuracy** | > 0.55 | 0.52 - 0.55 | < 0.52 |
| **F1 (Ternary)** | > 0.50 | 0.40 - 0.50 | < 0.40 |
| **R²** | > 0.20 | 0.10 - 0.20 | < 0.10 |

---

## Running for Other Tickers

The same workflow works for any ticker:

```bash
# AAPL
./scripts/run_spy_analysis.sh  # Edit ticker in script
# OR
python scripts/run_lstm_baseline.py --ticker AAPL --mode train --seed 42
python scripts/run_lstm_baseline.py --ticker AAPL --mode test --seed 42
python plots/plot_lstm.py --ticker AAPL --seed 42

# MSFT
python scripts/run_lstm_baseline.py --ticker MSFT --mode train --seed 42
python scripts/run_lstm_baseline.py --ticker MSFT --mode test --seed 42
python plots/plot_lstm.py --ticker MSFT --seed 42
```

---

## Troubleshooting

### Error: "Model not found"

```
FileNotFoundError: Model not found: results/lstm_baseline_model_SPY_train_seed42.pth
```

**Solution**: Run training first:
```bash
python scripts/run_lstm_baseline.py --ticker SPY --mode train --seed 42
```

### Error: "Aligned predictions not found"

```
FileNotFoundError: Aligned predictions not found: results/lstm_aligned_SPY_test_seed42.csv
```

**Solution**: Run testing first:
```bash
python scripts/run_lstm_baseline.py --ticker SPY --mode test --seed 42
```

### Error: "SPY features not found"

```
Error: SPY features not found in data/features/SPY
```

**Solution**: Run feature engineering:
```bash
python scripts/build_features.py --config config/default_config.yaml
```

### Warning: "No OHLCV data found"

```
Warning: No OHLCV data found for SPY, creating minimal DataFrame
```

**Impact**: Plot will not show candlesticks, only predictions.

**Solution**: Check if `data/processed/SPY.parquet` or `data/raw/SPY.parquet` exists. If not, re-run data ingestion:
```bash
python scripts/ingest_data.py --config config/default_config.yaml
```

---

## Advanced Usage

### Batch Processing Multiple Tickers

```bash
#!/bin/bash
TICKERS="SPY AAPL MSFT GOOGL NVDA"

for ticker in $TICKERS; do
    echo "Processing $ticker..."
    python scripts/run_lstm_baseline.py --ticker $ticker --mode train --seed 42
    python scripts/run_lstm_baseline.py --ticker $ticker --mode test --seed 42
    python plots/plot_lstm.py --ticker $ticker --seed 42
done
```

### Custom Hyperparameters

```bash
python scripts/run_lstm_baseline.py \
    --ticker SPY \
    --mode train \
    --num-layers 3 \
    --hidden-units 256 \
    --dropout 0.3 \
    --learning-rate 0.0005 \
    --lookback 60 \
    --max-epochs 200 \
    --batch-size 128
```

### Different Random Seeds

```bash
for seed in 42 123 456 789 1024; do
    python scripts/run_lstm_baseline.py --ticker SPY --mode train --seed $seed
    python scripts/run_lstm_baseline.py --ticker SPY --mode test --seed $seed
    python plots/plot_lstm.py --ticker SPY --seed $seed
done
```

---

## Performance Benchmarks

### Expected Runtime (SPY, 1-minute data, 5 years)

| Step | CPU (8 cores) | GPU (RTX 3090) | Memory |
|------|---------------|----------------|--------|
| Training (100 epochs) | ~30 min | ~8 min | ~8 GB |
| Validation (10 folds) | ~5 min | ~2 min | ~4 GB |
| Testing | ~2 min | ~1 min | ~4 GB |
| Plotting | ~30 sec | N/A | ~2 GB |
| **Total** | **~38 min** | **~12 min** | **~8 GB** |

### Expected Metrics (SPY, 1-minute data)

| Metric | Typical Range |
|--------|---------------|
| RMSE | 0.012 - 0.018 |
| Directional Accuracy | 0.52 - 0.56 |
| F1 (Ternary) | 0.42 - 0.48 |
| Buy Signal % | 18% - 25% |
| Sell Signal % | 18% - 25% |

---

## Next Steps

1. **Backtesting**: Use signals for backtesting with `scripts/backtest.py`
2. **Walk-Forward Validation**: Ensure signals are robust across time
3. **Portfolio Construction**: Combine signals from multiple tickers
4. **Risk Management**: Add position sizing and stop-loss logic
5. **Live Trading**: Deploy to paper trading account (Alpaca)

---

## References

- **Main Script**: `scripts/run_lstm_baseline.py`
- **Plotting Script**: `plots/plot_lstm.py`
- **Automation Script**: `scripts/run_spy_analysis.sh`
- **Feature Engineering**: `scripts/build_features.py`
- **Configuration**: `config/default_config.yaml`

---

## Contact

For issues or questions, check:
- Project README: `README.md`
- Technical docs: `docs/overviews/`
- Feature spec: `docs/overviews/feature_engineering_spec.md`
