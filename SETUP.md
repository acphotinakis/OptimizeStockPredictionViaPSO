agent --resume=abad2e15-8963-4d3e-8b04-04f3ee1a25df

# Setup Guide
## PSO-LSTM Stock Price Prediction System

This guide will help you set up and run the complete PSO-LSTM stock prediction pipeline.

---

## Prerequisites

- **Python 3.10+** (tested on 3.10 and 3.11)
- **CUDA-capable GPU** (recommended; CPU-only is ~10× slower)
- **8GB+ RAM** (16GB+ recommended for full universe)
- **Alpaca Markets account** (free paper-trading tier is sufficient)

---

## Step 1: Clone and Setup Environment

```bash
# Clone the repository
cd /path/to/ClaudePaper

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Configure API Credentials

1. Sign up for a free Alpaca Markets account at https://alpaca.markets/
2. Get your API keys from the dashboard
3. Copy the example environment file:

```bash
cp .env.example .env
```

4. Edit `.env` and add your credentials:

```bash
ALPACA_API_KEY=your_actual_api_key_here
ALPACA_API_SECRET=your_actual_api_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

---

## Step 3: Create Required Directories

```bash
mkdir -p data/raw data/processed data/features
mkdir -p results/checkpoints logs
mkdir -p notebooks
```

---

## Step 4: Run the Pipeline

### Quick Start (Single Ticker)

For a quick test run on a single ticker (AAPL):

```bash
# Step 1: Download and clean data
python scripts/01_ingest_data.py --config config/default_config.yaml

# Step 2: Build features
python scripts/02_build_features.py --config config/default_config.yaml

# Step 3: Run PSO optimization (this takes ~3-4 hours on GPU)
python scripts/03_run_pso.py --ticker AAPL --config config/default_config.yaml

# Step 4: Evaluate model
python scripts/04_evaluate.py --ticker AAPL --config config/default_config.yaml

# Step 5: Run backtest
python scripts/05_backtest.py --ticker AAPL --config config/default_config.yaml
```

### Full Pipeline (All Tickers)

To run the complete experiment on all 51 tickers:

```bash
# 1. Ingest all data (takes ~30 minutes due to API rate limits)
python scripts/01_ingest_data.py

# 2. Build features for all tickers (takes ~1 hour)
python scripts/02_build_features.py

# 3. Run PSO for each ticker (run in parallel if you have multiple GPUs)
for ticker in AAPL JPM JNJ AMZN BA; do
    python scripts/03_run_pso.py --ticker $ticker &
done
wait

# 4. Evaluate all models
for ticker in AAPL JPM JNJ AMZN BA; do
    python scripts/04_evaluate.py --ticker $ticker
done

# 5. Run backtests
for ticker in AAPL JPM JNJ AMZN BA; do
    python scripts/05_backtest.py --ticker $ticker
done
```

---

## Step 5: Run Tests

Verify the installation by running the test suite:

```bash
pytest tests/ -v
```

---

## Configuration

### Adjusting PSO Parameters

Edit `config/default_config.yaml` to customize:

- **PSO settings**: `n_particles`, `n_iterations`, `w_min`, `w_max`
- **LSTM search space**: `num_layers`, `hidden_units`, `dropout`, etc.
- **Fitness weights**: `rmse_weight`, `sharpe_weight`, `drawdown_weight`
- **Data split**: `train_years`, `val_years`, `test_years`

### Using a Subset of Tickers

To test on a smaller set of tickers, edit `config/tickers.txt` and comment out (with `#`) the tickers you don't want to use.

---

## Expected Outputs

After running the pipeline, you should have:

### Data Files
- `data/processed/aligned_universe.parquet` - Aligned OHLCV data
- `data/features/{ticker}/` - Feature matrices per ticker

### Results
- `results/pso_results_{ticker}.json` - Best hyperparameters and fitness history
- `results/evaluation/{ticker}/` - Model comparison metrics
- `results/backtest/{ticker}/` - Backtest results and equity curves

### Logs
- `logs/01_ingest_data.log` - Data ingestion logs
- `logs/03_run_pso_{ticker}.log` - PSO optimization logs
- etc.

---

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors during PSO:

1. Reduce `batch_size` in config (default: 256 → try 128 or 64)
2. Reduce `n_particles` (default: 30 → try 20 or 15)
3. Use CPU instead of GPU (slower but uses system RAM)

### Alpaca API Rate Limits

If data download is interrupted:

1. The script automatically resumes (skips already downloaded tickers)
2. Wait a few minutes and re-run `01_ingest_data.py`
3. Use `--skip-existing` flag to avoid re-downloading

### CUDA Out of Memory

If you get CUDA OOM during training:

1. Reduce `batch_size` in the config
2. Reduce `max_lookback` (default: 120 → try 60)
3. Set `CUDA_VISIBLE_DEVICES=""` to force CPU mode

---

## Performance Tips

### Speed Up PSO

1. **Use GPU**: Ensure PyTorch detects your GPU (`torch.cuda.is_available()`)
2. **Parallel particles**: Set `n_workers > 1` in config (experimental)
3. **Reduce iterations**: Try `n_iterations=30` for faster (but less optimal) results

### Reduce Data Size

1. **Fewer tickers**: Use only 5-10 representative tickers
2. **Shorter time range**: Reduce `train_years` in config
3. **Lower frequency**: Use 5-minute bars instead of 1-minute (requires code changes)

---

## Next Steps

1. **Analyze results**: See `notebooks/Results_Analysis.ipynb` (create if needed)
2. **Compare models**: Check `results/evaluation/` for performance comparisons
3. **Visualize convergence**: Plot fitness history from PSO results
4. **Walk-forward validation**: Implement using `src/evaluation/walk_forward.py`

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{photinakis2026pso,
  title={LSTM Hyperparameter Tuning with Improved Particle Swarm Optimization for Stock Price Prediction},
  author={Photinakis, Andrew and VanKlootwyk-Ford, Dory and Ukwuaba, Osita},
  year={2026},
  institution={Rochester Institute of Technology, CSCI 633},
  note={Course project, Biologically-Inspired Intelligent Systems}
}
```

---

## Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review the documentation in `docs/`
3. Run tests to verify installation: `pytest tests/ -v`
