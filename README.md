# LSTM Hyperparameter Tuning with Improved PSO for Stock Price Prediction

> **CSCI 633 — Biologically-Inspired Intelligent Systems | Application Study**
> Authors: Andrew Photinakis · Dory VanKlootwyk-Ford · Osita Ukwuaba | March 2026

---

## Project Overview

This project implements and evaluates a hybrid **Improved Particle Swarm Optimization (IPSO) + Long Short-Term Memory (LSTM)** system for intraday stock price prediction using high-frequency (1-minute) OHLCV data. The IPSO algorithm automatically tunes LSTM hyperparameters (layers, hidden units, dropout, learning rate, lookback window) to maximize a composite fitness function combining prediction accuracy and trading performance.

### Key Features

- **51-ticker universe** (50 equities + SPY benchmark) via Alpaca Markets API
- **100+ engineered features** per ticker per timestep
- **Many-to-One LSTM** predicting next-period log returns / mid-price movement
- **Improved PSO** with non-linear inertia weight (tanh) and adaptive mutation factor
- **Walk-forward backtesting** with transaction costs, slippage, Sharpe, drawdown, CAGR
- **Baselines**: Persistence model, standard LSTM, XGBoost

---

## Repository Structure

```
pso_lstm_stock/
│
├── README.md
├── requirements.txt
├── config/
│   ├── default_config.yaml          # Hyperparameter ranges, PSO settings
│   └── tickers.txt                  # List of 51 tickers
│
├── data/
│   ├── raw/                         # Raw Alpaca API downloads (gitignored)
│   ├── processed/                   # Cleaned & aligned OHLCV
│   └── features/                    # Engineered feature matrices
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── alpaca_ingestor.py       # Alpaca API integration
│   │   ├── cleaner.py               # Data cleaning pipeline
│   │   ├── aligner.py               # Multi-ticker alignment
│   │   └── splitter.py              # Train/val/test split
│   │
│   ├── features/
│   │   ├── technical.py             # RSI, MACD, Bollinger, etc.
│   │   ├── statistical.py           # Rolling stats, skew, kurtosis
│   │   ├── volume.py                # VWAP, OBV, liquidity features
│   │   ├── cross_ticker.py          # SPY correlation, relative strength
│   │   ├── selector.py              # XGBoost importance + PSO mask
│   │   └── pipeline.py              # Unified feature pipeline
│   │
│   ├── models/
│   │   ├── lstm_model.py            # LSTM class (configurable)
│   │   ├── baselines.py             # Persistence, XGBoost
│   │   └── trainer.py               # Training loop with early stopping
│   │
│   ├── optimizer/
│   │   ├── pso_core.py              # Standard PSO equations
│   │   ├── ipso.py                  # Improved PSO (tanh inertia + mutation)
│   │   ├── particle.py              # Particle encoding / decoding
│   │   └── fitness.py               # Composite fitness function
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # RMSE, Sharpe, Drawdown, CAGR, F1
│   │   ├── walk_forward.py          # Walk-forward validation
│   │   └── backtester.py            # Signal generation + backtest engine
│   │
│   └── utils/
│       ├── logger.py
│       ├── seed.py                  # Global seed management
│       └── config_loader.py
│
├── scripts/
│   ├── 01_ingest_data.py
│   ├── 02_build_features.py
│   ├── 03_run_pso.py
│   ├── 04_evaluate.py
│   └── 05_backtest.py
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Results_Analysis.ipynb
│
├── ai_outputs/                      # All design documents (this folder)
│   ├── README.md
│   ├── technical_design_document.md
│   ├── feature_engineering_spec.md
│   ├── pso_mathematical_spec.md
│   ├── model_architecture.md
│   ├── data_pipeline.md
│   ├── experiment_plan.md
│   ├── backtesting_framework.md
│   ├── reproducibility.md
│   ├── literature_review.md
│   └── application_study.md
│
└── tests/
    ├── test_pso.py
    ├── test_lstm.py
    └── test_features.py
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended; CPU-only is ~10× slower for PSO evaluation)
- Alpaca Markets account (free paper-trading tier is sufficient for data access)

### 2. Clone and install

```bash
git clone https://github.com/<team>/pso_lstm_stock.git
cd pso_lstm_stock

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment variables

```bash
cp .env.example .env
# Edit .env with your credentials:
# ALPACA_API_KEY=your_key
# ALPACA_API_SECRET=your_secret
# ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 4. Download data

```bash
python scripts/01_ingest_data.py --start 2019-01-01 --end 2024-01-01 \
    --tickers config/tickers.txt --output data/raw/
```

---

## Pipeline Execution

### Step 1 — Data Ingestion & Cleaning

```bash
python scripts/01_ingest_data.py
```

Downloads 5 years of 1-minute OHLCV data for all 51 tickers from the Alpaca Markets API. Outputs aligned parquet files to `data/processed/`.

### Step 2 — Feature Engineering

```bash
python scripts/02_build_features.py --config config/default_config.yaml
```

Computes 100+ features per ticker, performs XGBoost importance filtering, and saves feature matrices to `data/features/`.

### Step 3 — PSO Hyperparameter Search

```bash
python scripts/03_run_pso.py \
    --ticker AAPL \
    --particles 30 \
    --iterations 50 \
    --seed 42
```

Runs the IPSO algorithm. Logs fitness per iteration to `logs/pso_run_<ticker>.json`. Outputs best hyperparameter set to `results/best_params_<ticker>.json`.

### Step 4 — Final Model Evaluation

```bash
python scripts/04_evaluate.py \
    --ticker AAPL \
    --params results/best_params_AAPL.json
```

Trains the LSTM with IPSO-optimal hyperparameters on the full training set and evaluates on the held-out test set. Prints RMSE, Directional Accuracy, Sharpe, Drawdown.

### Step 5 — Backtest

```bash
python scripts/05_backtest.py \
    --ticker AAPL \
    --transaction_cost 0.001 \
    --slippage 0.0005
```

Runs the walk-forward backtest and outputs performance report to `results/backtest_<ticker>.csv`.

---

## Example Usage (Python API)

```python
from src.optimizer.ipso import IPSO
from src.models.lstm_model import LSTMModel
from src.optimizer.fitness import composite_fitness

# Define search space
search_space = {
    "num_layers":    (1, 4),
    "hidden_units":  (32, 512),
    "dropout":       (0.0, 0.5),
    "learning_rate": (1e-5, 1e-1),
    "lookback":      [10, 30, 60, 120],
}

# Initialize improved PSO
optimizer = IPSO(
    n_particles=30,
    n_iterations=50,
    search_space=search_space,
    fitness_fn=composite_fitness,
    w_min=0.4, w_max=0.9,
    c1=1.5, c2=1.5,
    seed=42
)

best_params, best_fitness = optimizer.run(X_train, y_train, X_val, y_val)
print(f"Best params: {best_params}")
print(f"Best fitness: {best_fitness:.4f}")

# Build final model
model = LSTMModel(**best_params)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## Configuration File (`config/default_config.yaml`)

```yaml
pso:
  n_particles: 30
  n_iterations: 50
  w_min: 0.4
  w_max: 0.9
  c1: 1.5
  c2: 1.5
  seed: 42

lstm:
  num_layers: [1, 2, 3, 4]
  hidden_units: [32, 512]
  dropout: [0.0, 0.5]
  learning_rate: [1e-5, 1e-1]
  lookback: [10, 30, 60, 120]
  max_epochs: 100
  batch_size: 256
  early_stopping_patience: 10

fitness:
  rmse_weight: 0.4
  sharpe_weight: 0.4
  drawdown_weight: 0.2

data:
  train_years: 3
  val_years: 1
  test_years: 1
  freq: "1Min"
```

---

## Reproducing Results

See `ai_outputs/reproducibility.md` for full instructions including Docker setup, seed handling, and expected runtimes.

---

## Documentation Index

| Document | Description |
|---|---|
| `technical_design_document.md` | Full system architecture & module breakdown |
| `feature_engineering_spec.md` | 100+ feature dictionary with math definitions |
| `pso_mathematical_spec.md` | IPSO equations, particle encoding, convergence |
| `model_architecture.md` | LSTM structure, training loop, regularization |
| `data_pipeline.md` | Alpaca API integration, cleaning, alignment |
| `experiment_plan.md` | PSO config, baselines, evaluation budget |
| `backtesting_framework.md` | Signal generation, risk management, metrics |
| `reproducibility.md` | Environment setup, seeds, run commands |
| `literature_review.md` | Synthesis of all provided research papers |
| `application_study.md` | Full research paper (Abstract → Future Work) |

---

## Citation

```bibtex
@misc{photinakis2026pso,
  title={LSTM Hyperparameter Tuning with Improved Particle Swarm Optimization for Stock Price Prediction},
  author={Photinakis, Andrew and VanKlootwyk-Ford, Dory and Ukwuaba, Osita},
  year={2026},
  institution={Rochester Institute of Technology, CSCI 633},
  note={Course project, Biologically-Inspired Intelligent Systems}
}
```