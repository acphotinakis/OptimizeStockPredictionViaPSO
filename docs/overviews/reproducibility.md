# Reproducibility Guide
## PSO-LSTM Stock Price Prediction System

**Document Version:** 1.0 | March 2026

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment Setup](#2-environment-setup)
3. [requirements.txt Contents](#3-requirementstxt-contents)
4. [Seed Handling](#4-seed-handling)
5. [Run Commands](#5-run-commands)
6. [Expected Outputs and Checksums](#6-expected-outputs-and-checksums)
7. [Docker Setup (Optional)](#7-docker-setup-optional)
8. [Known Environment Issues](#8-known-environment-issues)

---

## 1. Project Structure

```
pso_lstm_stock/
│
├── README.md
├── requirements.txt
├── .env.example                  ← Copy to .env and add credentials
├── .gitignore
│
├── config/
│   ├── default_config.yaml       ← Main configuration file
│   └── tickers.txt               ← 51-ticker universe (1 per line)
│
├── data/                         ← Auto-created by scripts
│   ├── raw/                      ← Downloaded parquet files (gitignored)
│   ├── processed/                ← Cleaned and aligned data (gitignored)
│   └── features/                 ← Engineered feature matrices (gitignored)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── alpaca_ingestor.py
│   │   ├── cleaner.py
│   │   ├── aligner.py
│   │   └── splitter.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical.py
│   │   ├── statistical.py
│   │   ├── volume.py
│   │   ├── cross_ticker.py
│   │   ├── selector.py
│   │   └── pipeline.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py
│   │   ├── baselines.py
│   │   └── trainer.py
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── pso_core.py
│   │   ├── ipso.py
│   │   ├── particle.py
│   │   └── fitness.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── walk_forward.py
│   │   └── backtester.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── seed.py
│       └── config_loader.py
│
├── scripts/
│   ├── 01_ingest_data.py         ← Step 1: Download data from Alpaca
│   ├── 02_build_features.py      ← Step 2: Engineer feature matrices
│   ├── 03_run_pso.py             ← Step 3: IPSO hyperparameter search
│   ├── 04_evaluate.py            ← Step 4: Final model evaluation
│   └── 05_backtest.py            ← Step 5: Backtesting and reporting
│
├── results/                      ← Auto-created; outputs land here
│   ├── pso_log_<ticker>.json
│   ├── best_params_<ticker>.json
│   ├── metrics_<ticker>.json
│   └── backtest_<ticker>.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Results_Analysis.ipynb
│
├── ai_outputs/                   ← Design documents (this folder)
│
└── tests/
    ├── test_pso.py
    ├── test_lstm.py
    ├── test_features.py
    └── conftest.py
```

---

## 2. Environment Setup

### 2.1 Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 recommended |
| CUDA | 11.8+ | Optional but strongly recommended |
| Git | Any | For version control |
| Alpaca API credentials | — | Free paper-trading account |

### 2.2 Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/<team>/pso_lstm_stock.git
cd pso_lstm_stock

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate.bat       # Windows Command Prompt
# venv\Scripts\Activate.ps1       # Windows PowerShell

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set up environment variables
cp .env.example .env
# Edit .env with your Alpaca API credentials:
#   ALPACA_API_KEY=PKXXXXXXXXXXXXXXXXXXXXXXXX
#   ALPACA_API_SECRET=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#   ALPACA_BASE_URL=https://paper-api.alpaca.markets

# 6. Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import alpaca_trade_api; print('Alpaca OK')"

# 7. Run tests
pytest tests/ -v
```

### 2.3 Verify GPU Setup (if applicable)

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

---

## 3. requirements.txt Contents

```
# Core numerical computing
numpy==1.26.4
pandas==2.2.1
scipy==1.12.0

# Deep learning
torch==2.2.1
torchvision==0.17.1  # Required for some torch utilities

# Machine learning
scikit-learn==1.4.1
xgboost==2.0.3

# Technical analysis
pandas-ta==0.3.14b0

# Data ingestion
alpaca-trade-api==3.3.2

# Financial calendar
pandas-market-calendars==4.3.2

# Data storage
pyarrow==15.0.0
fastparquet==2024.2.0

# Visualization
matplotlib==3.8.3
seaborn==0.13.2
plotly==5.20.0

# Utilities
python-dotenv==1.0.1
pyyaml==6.0.1
tqdm==4.66.2
loguru==0.7.2

# Type hints and dataclasses
typing-extensions==4.10.0

# Testing
pytest==8.1.1
pytest-cov==5.0.0

# Notebook support (optional)
jupyter==1.0.0
ipykernel==6.29.3
```

**Installation command:**
```bash
pip install -r requirements.txt
```

**For CUDA 11.8 (if pip default installs CPU-only PyTorch):**
```bash
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
```

---

## 4. Seed Handling

### 4.1 Seed Management Module (`src/utils/seed.py`)

All stochastic components of the pipeline use a global seed for full reproducibility.

```python
"""
src/utils/seed.py
Global seed management for reproducibility.
"""
import random
import numpy as np
import torch
import os

def set_all_seeds(seed: int = 42) -> None:
    """
    Set random seeds for all relevant libraries to ensure reproducibility.
    
    Args:
        seed: Integer seed value. Default 42.
    
    Note:
        Must be called BEFORE any model instantiation or data generation.
        For PSO, the seed controls:
          - Particle position initialization (np.random)
          - Velocity initialization (np.random)
          - Mutation random draws (np.random)
          - r1, r2 uniform draws (np.random)
        For LSTM training, the seed controls:
          - Weight initialization (torch)
          - DataLoader shuffling (torch)
          - Dropout mask generation (torch)
    """
    # Python built-in random
    random.seed(seed)
    
    # NumPy (used by PSO and feature engineering)
    np.random.seed(seed)
    
    # PyTorch (used by LSTM training)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic CuDNN operations (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Python hash seed (affects dict ordering in some contexts)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_rng(seed: int) -> np.random.Generator:
    """
    Get a seeded NumPy random generator for a specific PSO particle.
    Used to ensure per-particle determinism.
    
    Args:
        seed: Unique seed per particle (e.g., global_seed * 1000 + particle_idx)
    Returns:
        np.random.Generator instance
    """
    return np.random.default_rng(seed)
```

### 4.2 Where Seeds Are Applied

| Component | Seed Usage |
|---|---|
| PSO initialization | `np.random.seed(42)` before swarm initialization |
| PSO velocity updates | NumPy `r1, r2 = rng.uniform(0, 1, size=2)` |
| PSO mutation | `xi = rng.uniform()` |
| LSTM weight init | `torch.manual_seed(42)` before model instantiation |
| LSTM DataLoader | `DataLoader(..., generator=torch.Generator().manual_seed(42))` |
| XGBoost | `random_state=42` in constructor |
| Train/val/test split | Chronological (no randomness; seed not needed) |

### 4.3 Multi-Seed Experiments

For statistical significance, 5 different seeds are used:

```python
SEEDS = [42, 123, 456, 789, 1024]

for seed in SEEDS:
    set_all_seeds(seed)
    optimizer = IPSO(..., seed=seed)
    best_params, fitness = optimizer.run(X_train, y_train, X_val, y_val)
    results[seed] = evaluate(best_params, X_test, y_test)
```

---

## 5. Run Commands

### 5.1 Full Pipeline (5 Primary Tickers)

```bash
# Step 1: Download data (≈ 20 min with rate limiting)
python scripts/01_ingest_data.py \
    --start 2019-01-02 \
    --end 2024-01-01 \
    --tickers AAPL JPM JNJ AMZN BA SPY \
    --output data/raw/

# Step 2: Build features (≈ 5 min per ticker)
python scripts/02_build_features.py \
    --tickers AAPL JPM JNJ AMZN BA \
    --target AAPL \
    --config config/default_config.yaml

# Step 3: Run IPSO (≈ 3–4 hours per ticker on GPU with 8-way parallelism)
python scripts/03_run_pso.py \
    --ticker AAPL \
    --particles 30 \
    --iterations 50 \
    --seed 42 \
    --n_workers 8

# Step 4: Evaluate final model
python scripts/04_evaluate.py \
    --ticker AAPL \
    --params results/best_params_AAPL.json \
    --seed 42

# Step 5: Backtest
python scripts/05_backtest.py \
    --ticker AAPL \
    --params results/best_params_AAPL.json \
    --transaction_cost 0.001 \
    --slippage 0.0005 \
    --stop_loss 0.02
```

### 5.2 Quick Test Run (Small Dataset)

For debugging and testing without a full 5-year dataset:

```bash
python scripts/01_ingest_data.py \
    --start 2023-01-02 \
    --end 2024-01-01 \
    --tickers AAPL SPY \
    --output data/raw_test/

python scripts/03_run_pso.py \
    --ticker AAPL \
    --particles 5 \
    --iterations 5 \
    --seed 42 \
    --data_dir data/raw_test/
```

Expected runtime: ~5 minutes total.

### 5.3 Ablation Study: IPSO vs. Standard PSO

```bash
# Run standard PSO (linear inertia, no mutation)
python scripts/03_run_pso.py \
    --ticker AAPL \
    --optimizer standard_pso \
    --particles 30 \
    --iterations 50 \
    --seed 42 \
    --output results/pso_standard/

# Run IPSO (default)
python scripts/03_run_pso.py \
    --ticker AAPL \
    --optimizer ipso \
    --particles 30 \
    --iterations 50 \
    --seed 42 \
    --output results/pso_improved/
```

### 5.4 Multi-Seed Reproducibility Test

```bash
for seed in 42 123 456 789 1024; do
    python scripts/03_run_pso.py \
        --ticker AAPL \
        --particles 30 \
        --iterations 50 \
        --seed $seed \
        --output results/seed_study/seed_$seed/
done

# Aggregate results
python notebooks/aggregate_seeds.py
```

### 5.5 Run Unit Tests

```bash
# All tests
pytest tests/ -v

# Specific test modules
pytest tests/test_pso.py -v
pytest tests/test_lstm.py -v
pytest tests/test_features.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## 6. Expected Outputs and Checksums

### 6.1 Key Output Files

After running all scripts for `AAPL`:

| File | Description | Approximate Size |
|---|---|---|
| `data/raw/AAPL.parquet` | Raw 1-min OHLCV | ~15 MB |
| `data/features/AAPL_features.npy` | Feature matrix | ~80 MB |
| `results/pso_log_AAPL.json` | PSO iteration log | ~500 KB |
| `results/best_params_AAPL.json` | gbest hyperparameters | < 1 KB |
| `results/metrics_AAPL.json` | Test-set evaluation metrics | < 10 KB |
| `results/backtest_AAPL.csv` | Trade-by-trade log | ~2 MB |

### 6.2 Expected `best_params_AAPL.json` Format

```json
{
  "ticker": "AAPL",
  "seed": 42,
  "pso_iterations": 50,
  "best_params": {
    "num_layers": 2,
    "hidden_units": 256,
    "dropout": 0.23,
    "learning_rate": 0.00087,
    "lookback": 30
  },
  "best_fitness": 0.3421,
  "gbest_rmse": 0.00142,
  "gbest_sharpe": 1.23,
  "gbest_mdd": 0.08,
  "runtime_seconds": 10824
}
```

### 6.3 Reproducibility Verification

To verify identical results across runs:

```bash
python scripts/03_run_pso.py --ticker AAPL --seed 42 --output results/run_a/
python scripts/03_run_pso.py --ticker AAPL --seed 42 --output results/run_b/

python -c "
import json
with open('results/run_a/best_params_AAPL.json') as f: a = json.load(f)
with open('results/run_b/best_params_AAPL.json') as f: b = json.load(f)
assert a['best_params'] == b['best_params'], 'Reproducibility FAILED'
print('Reproducibility check PASSED')
"
```

---

## 7. Docker Setup (Optional)

For maximum reproducibility across machines:

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default command
CMD ["bash"]
```

```bash
# Build
docker build -t pso-lstm .

# Run with GPU support and mounted data
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    --env-file .env \
    pso-lstm \
    python scripts/03_run_pso.py --ticker AAPL --seed 42
```

---

## 8. Known Environment Issues

### Issue 1: Alpaca API Rate Limiting

**Symptom:** `HTTPError: 429 Too Many Requests` during data download.

**Fix:** The `AlpacaIngestor` includes a 0.35s sleep between calls (≈ 171 req/min, below the 200 req/min limit). If still hitting limits, increase sleep to 0.5s:

```python
# In alpaca_ingestor.py
time.sleep(0.5)
```

### Issue 2: CUDA Out of Memory

**Symptom:** `torch.cuda.OutOfMemoryError` during PSO fitness evaluation.

**Fix:** Reduce batch size:
```yaml
# In config/default_config.yaml
lstm:
  batch_size: 128   # Reduce from 256
```

Or run on CPU (slower but reliable):
```bash
python scripts/03_run_pso.py --ticker AAPL --device cpu
```

### Issue 3: Non-Deterministic Results on GPU

**Symptom:** Slightly different results between runs even with the same seed.

**Cause:** Some CUDA operations are non-deterministic by default.

**Fix:** Already handled in `set_all_seeds()` via:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Note: This may reduce GPU throughput by ~5–10%.

### Issue 4: pandas-ta Installation

**Symptom:** `ModuleNotFoundError: No module named 'pandas_ta'`

**Fix:**
```bash
pip install pandas-ta==0.3.14b0
```

If that fails (common on Windows):
```bash
pip install git+https://github.com/twopirllc/pandas-ta.git@development
```

### Issue 5: Windows Path Separator

**Symptom:** File paths fail on Windows.

**Fix:** All paths in the codebase use `pathlib.Path` (OS-agnostic). Ensure scripts are called from the project root directory.