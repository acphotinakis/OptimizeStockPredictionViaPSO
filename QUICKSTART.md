# Quick Start Guide
## Get Running in 5 Minutes

This guide will get you up and running with a single-ticker test as quickly as possible.

---

## 1. Install Dependencies (2 minutes)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install packages
pip install -r requirements.txt
```

---

## 2. Setup API Credentials (1 minute)

```bash
# Copy template
cp .env.example .env

# Edit .env and add your Alpaca API keys
# Get free keys at: https://alpaca.markets/
```

Your `.env` should look like:
```
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxx
ALPACA_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

---

## 3. Create Directories (10 seconds)

```bash
mkdir -p data/raw data/processed data/features results logs
```

---

## 4. Run Quick Test on AAPL (2 minutes setup + 3 hours compute)

### Option A: Full Pipeline (Recommended)

```bash
# Download data (~5 min)
python scripts/01_ingest_data.py

# Build features (~10 min)
python scripts/02_build_features.py

# Run PSO optimization (~3 hours on GPU, ~30 hours on CPU)
python scripts/03_run_pso.py --ticker AAPL

# Evaluate model (~5 min)
python scripts/04_evaluate.py --ticker AAPL

# Backtest (~2 min)
python scripts/05_backtest.py --ticker AAPL
```

### Option B: Fast Test (Skip PSO)

If you want to test the pipeline without waiting for PSO:

```bash
# 1. Download data
python scripts/01_ingest_data.py

# 2. Build features
python scripts/02_build_features.py

# 3. Skip PSO, just evaluate baselines
python scripts/04_evaluate.py --ticker AAPL

# This will run:
# - Persistence model (instant)
# - Vanilla LSTM with default params (~5 min)
# - XGBoost (~2 min)
```

---

## 5. Check Results

After running, check:

```bash
# PSO results
cat results/pso_results_AAPL.json

# Model comparison
cat results/evaluation/evaluation_AAPL.json

# Backtest metrics
cat results/backtest/backtest_AAPL.json

# Equity curve plot
open results/backtest/equity_curve_AAPL.png
```

---

## 6. Run Tests (Optional)

Verify everything works:

```bash
pytest tests/ -v
```

---

## Expected Output

After running the full pipeline, you should see:

### PSO Results
```json
{
  "ticker": "AAPL",
  "best_params": {
    "num_layers": 2,
    "hidden_units": 256,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "lookback": 60
  },
  "best_fitness": 0.234
}
```

### Model Comparison
```
Model                RMSE       DA         F1         R²
--------------------------------------------------------------
IPSO-LSTM            0.0012     0.5823     0.4567     0.3421
Vanilla-LSTM         0.0015     0.5512     0.4234     0.2987
XGBoost              0.0014     0.5634     0.4389     0.3156
Persistence          0.0018     0.5001     0.3876     0.0234
```

### Backtest Results
```
Sharpe Ratio:     1.45
Max Drawdown:     0.12
CAGR:             0.23
Win Rate:         0.54
Number of Trades: 1247
```

---

## Troubleshooting

### "No module named 'alpaca_trade_api'"
```bash
pip install alpaca-trade-api
```

### "CUDA out of memory"
Edit `config/default_config.yaml`:
```yaml
lstm:
  batch_size: 128  # Reduce from 256
```

### "Alpaca API authentication failed"
- Check your `.env` file has correct credentials
- Verify keys are active in Alpaca dashboard
- Ensure you're using paper trading URL

### "No data returned for ticker"
- Check ticker symbol is correct
- Verify date range has market data
- Try a different ticker (e.g., SPY)

---

## Next Steps

1. **Run on more tickers**: Edit `config/tickers.txt` to add more symbols
2. **Adjust PSO settings**: Edit `config/default_config.yaml`
3. **Analyze results**: Create notebooks in `notebooks/`
4. **Compare models**: Run walk-forward validation
5. **Optimize further**: Try different fitness weights

---

## Performance Tips

### Speed Up PSO (3x faster)
```yaml
pso:
  n_particles: 20      # Reduce from 30
  n_iterations: 30     # Reduce from 50
```

### Reduce Memory Usage
```yaml
lstm:
  batch_size: 64       # Reduce from 256
  
data:
  train_years: 2       # Reduce from 3
```

### Use CPU Only
```bash
export CUDA_VISIBLE_DEVICES=""
python scripts/03_run_pso.py --ticker AAPL
```

---

## Common Commands

```bash
# Run PSO with custom config
python scripts/03_run_pso.py --ticker AAPL --config my_config.yaml

# Run with different seed
python scripts/03_run_pso.py --ticker AAPL --seed 123

# Evaluate specific PSO results
python scripts/04_evaluate.py --ticker AAPL --pso-results results/

# Backtest with custom costs
python scripts/05_backtest.py --ticker AAPL  # Edit config for costs
```

---

## Getting Help

1. Check logs: `cat logs/03_run_pso_AAPL.log`
2. Read full setup: `SETUP.md`
3. Review documentation: `docs/`
4. Run tests: `pytest tests/ -v`

---

**Ready to go!** 🚀

Start with: `python scripts/01_ingest_data.py`
