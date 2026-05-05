# PSO-LSTM Stock Price Prediction System

**A production-grade financial machine learning system for stock return prediction using LSTM neural networks optimized via Particle Swarm Optimization (PSO) and improved PSO.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Data Ingestion](#1-data-ingestion)
  - [Feature Engineering](#2-feature-engineering)
  - [Model Training](#3-model-training)
  - [Evaluation](#4-evaluation)
- [Project Structure](#project-structure)
- [Technical Requirements Documents (TRDs)](#technical-requirements-documents-trds)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [References](#references)

---

## Overview

This system implements a **hybrid financial forecasting architecture** that combines:

- **Deep Learning**: 2-layer LSTM networks for temporal pattern recognition in stock price time series
- **Hyperparameter Optimization**: Improved Particle Swarm Optimization (IPSO) for automatic LSTM tuning
- **Gradient Boosting**: XGBoost as a comparative baseline
- **Rigorous Evaluation**: Walk-forward validation, backtesting with transaction costs, and production-grade metrics

### What Problem Does This Solve?

Stock price prediction is notoriously difficult due to:
- High noise-to-signal ratios
- Non-stationary market dynamics
- Complex temporal dependencies
- Overfitting risks with hyperparameter tuning

This system addresses these challenges through:
1. **Temporal Causality Enforcement**: Strict split-first architecture prevents data leakage
2. **Fit-Once-Freeze-Forever**: Feature pipelines trained once to ensure reproducibility
3. **PSO Hyperparameter Search**: Automatic LSTM optimization without manual tuning
4. **Comprehensive Backtesting**: Realistic trading simulation with costs and slippage

---

## Key Features

### ✅ **Production-Grade ML Pipeline**
- YAML-based configuration management
- Modular architecture with clear separation of concerns
- Extensive logging and error handling
- Reproducible experiments with seed control

### ✅ **Advanced Feature Engineering**
- **45+ Technical Indicators**: EMA, MACD, Bollinger Bands, ATR, RSI, Stochastic, ADX, etc.
- **Cross-Ticker Features**: Market context (SPY/QQQ), peer correlation, sector rotation
- **Wavelet Denoising**: Haar wavelet (3-level) with soft thresholding
- **4-Stage Feature Selection**: Variance threshold → Correlation → VIF → Mutual Information
- **MinMax Scaling**: Separate scalers for features and targets (prevents leakage)

### ✅ **Rigorous Data Leakage Prevention**
- **Split-First Architecture**: Chronological 70/10/20 train/val/test split before any fitting
- **Frozen Pipeline State**: Scalers, selectors, wavelet thresholds fit on train only
- **SPY-Aligned Timestamps**: All tickers synchronized to S&P 500 trading calendar
- **No Future Data Access**: Strict backward-looking features and causal forward-fill (≤5 bars)

### ✅ **IPSO-LSTM Hyperparameter Optimization**
- **6D Search Space**: Units (layer 1 & 2), dropout, learning rate, batch size, epochs
- **Fitness Function**: `f(x) = 0.9 × MSE + 0.1 × MSW` (accuracy + model complexity)
- **20 Particles × 50 Iterations**: Adaptive inertia weight (IPSO)
- **Two-Phase Protocol**: Phase 1 (PSO search) → Phase 2 (final training with best params)

### ✅ **Comprehensive Evaluation**
- **Walk-Forward Validation**: Expanding-window, per-fold retraining, per-fold PSO
- **Backtesting**: Transaction costs (0.15% one-way), slippage, stop-loss, daily limits
- **15+ Metrics**: RMSE, R², Directional Accuracy, Sharpe, Sortino, CAGR, Max Drawdown, Calmar, Profit Factor, Win Rate

---

## System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                               │
│  Alpaca API → Raw OHLCV → Cleaning → SPY Alignment → Synchronized  │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                             │
│  Technical Indicators → Cross-Ticker Features → Wavelet Denoising   │
│  → Feature Selection → MinMax Scaling → Temporal Windowing          │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│               CHRONOLOGICAL SPLIT (70/10/20)                        │
│  Train (fit all transformations) | Val (monitor) | Test (frozen)   │
└───────┬─────────────────┬─────────────────────────┬─────────────────┘
        │                 │                         │
        ▼                 ▼                         ▼
┌───────────────┐  ┌──────────────┐        ┌──────────────┐
│ BASELINE LSTM │  │  PSO-LSTM    │        │   XGBoost    │
│ Fixed Params  │  │ IPSO Search  │        │  Baseline    │
└───────┬───────┘  └──────┬───────┘        └──────┬───────┘
        │                 │                        │
        └─────────────────┴────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  EVALUATION & BACKTESTING                           │
│  Walk-Forward Validation → Metrics → Trading Simulation → Plots    │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | Purpose | Key Components |
|--------|---------|----------------|
| **`src/data/`** | Data pipeline | `alpaca_ingestor.py`, `cleaner.py`, `windowing.py` |
| **`src/features/`** | Feature engineering | `feature_generators.py`, `scaler.py`, `selector.py`, `wavelet.py` |
| **`src/models/`** | Model implementations | `lstm_model.py`, `lstm_trainer.py`, `xgboost_model.py` |
| **`src/optimizer/`** | PSO/IPSO algorithms | `pso_core.py`, `ipso.py`, `particle.py`, `fitness.py` |
| **`src/evaluation/`** | Evaluation & backtesting | `metrics.py`, `backtest.py`, `walk_forward_pso.py` |
| **`pipelines/`** | CLI entry points | `run_lstm.py`, `train_pso_lstm.py`, `run_backtest.py` |
| **`config/`** | Configuration | `default_config.yaml`, `symbol_universe.yaml` |

---

## Technology Stack

### Core Dependencies

- **Python**: 3.10+
- **Deep Learning**: PyTorch 2.2+ (CUDA support optional)
- **Machine Learning**: scikit-learn 1.4+, XGBoost 2.0+
- **Data Processing**: NumPy 1.26+, Pandas 2.1+
- **Signal Processing**: PyWavelets (via `pywt`)
- **Data Acquisition**: Alpaca Trade API 3.0+
- **Configuration**: PyYAML 6.0+
- **Visualization**: Matplotlib 3.8+, Seaborn 0.13+

### Full Dependency List

See [`requirements.txt`](requirements.txt) for complete list.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (optional, for GPU acceleration)
- Alpaca API key (for data ingestion)

### Step 1: Clone Repository

```bash
git clone https://github.com/acphotinakis/OptimizeStockPredictionViaPSO.git
cd OptimizeStockPredictionViaPSO
```

### Step 2: Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Alpaca API Credentials (for data ingestion)
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or live URL

# Optional: CUDA Configuration
CUDA_VISIBLE_DEVICES=0
```

### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
```

---

## Configuration

All system parameters are controlled via **YAML configuration files** in the `config/` directory.

### Main Configuration: `config/default_config.yaml`

Key sections:

#### Data Settings
```yaml
data:
  start_date: "2020-01-01"
  end_date: "2026-01-01"
  freq: "1Day"  # Options: 1Min, 5Min, 15Min, 1Hour, 1Day
  benchmark_ticker: "SPY"
  max_missing_fraction: 0.05
  max_ffill_bars: 5
```

#### LSTM Baseline Settings
```yaml
lstm_baseline:
  lookback: 20  # Temporal window size
  lstm_units_1: 128  # First LSTM layer units
  lstm_units_2: 64   # Second LSTM layer units
  dropout_rate: 0.2
  learning_rate: 0.001
  batch_size: 64
  epochs: 100
  early_stopping:
    enabled: true
    patience: 10
```

#### PSO Settings
```yaml
pso:
  n_particles: 20
  n_iterations: 50
  w_inertia: 0.7
  c1_cognitive: 1.5
  c2_social: 1.5
  search_space:
    units_1: [64, 256]
    units_2: [32, 128]
    dropout: [0.1, 0.5]
    learning_rate: [0.0001, 0.01]
    batch_size: [32, 128]
    epochs: [50, 200]
```

### Universe Configuration: `config/symbol_universe.yaml`

Define ticker universe and peer selection:

```yaml
tickers:
  - AAPL
  - MSFT
  - GOOGL
  - NVDA
  - TSLA
  # ... more tickers

peer_selection:
  mode: "dynamic"  # Options: dynamic, manual, hybrid
  max_peers: 3
  min_correlation: 0.3
  sector_constrained: true
```

---

## Usage

Run commands from the project root after activating the virtual environment.

### Yahoo Finance

Yahoo Finance does not require API credentials.

```powershell
python main.py --provider yfinance --run-index-suite --indices DJIA,SP500,NIFTY50 --device auto --plot
```

For a single ticker or index:

```powershell
python main.py --provider yfinance --symbol AAPL --method ipso --device auto --plot
```

You can also use the provider-specific entry point:

```powershell
python pipelines/run_yfinance.py --run-index-suite --indices DJIA,SP500,NIFTY50 --device auto --plot
```

### Alpaca

Set your Alpaca credentials locally before running. Do not commit `.env` or API keys.

```powershell
$env:ALPACA_API_KEY="your_key"
$env:ALPACA_API_SECRET="your_secret"
```

Run the Alpaca pipeline on selected traded companies:

```powershell
python main.py --provider alpaca --symbols AAPL,GOOGL,NVDA,META --timeframe 1Day --feed iex --adjustment raw --device auto --amp --plot
```

For the built-in index-proxy suite:

```powershell
python main.py --provider alpaca --run-index-suite --indices DJIA,SP500,NIFTY50 --timeframe 1Day --feed iex --adjustment raw --device auto --amp --plot
```

You can also use the provider-specific entry point:

```powershell
python pipelines/run_alpaca.py --symbols AAPL,GOOGL,NVDA,META --timeframe 1Day --feed iex --adjustment raw --device auto --amp --plot
```

### Outputs

By default, reports are written to `results_index_suite/` for Yahoo Finance and `results_alpaca_index_suite/` for Alpaca. Generated image artifacts are written to `img/`. Override these locations with `--output-dir` and `--img-dir` when needed.

---

## Project Structure

```
OptimizeStockPredictionViaPSO/
├── config/                          # Configuration files
│   ├── default_config.yaml         # Main system configuration
│   ├── symbol_universe.yaml        # Ticker universe and peer selection
│   └── tickers.txt                 # Ticker list
│
├── data/                           # Data storage (Git-ignored)
│   ├── raw/                        # Raw OHLCV from Alpaca
│   ├── cleaned/                    # Cleaned data (gap handling)
│   ├── aligned/                    # SPY-aligned synchronized data
│   └── features_v2/                # Feature-engineered datasets
│
├── docs/                           # Technical documentation
│   ├── TRD1.md                     # Feature Engineering TRD
│   ├── TRD2.md                     # Model Architecture TRD
│   └── TRD3.md                     # Evaluation TRD
│
├── logs/                           # Execution logs
│
├── pipelines/                      # CLI entry points
│   ├── data_ingest_align_clean.py  # Data acquisition pipeline
│   ├── run_build_features.py       # Feature engineering pipeline
│   ├── run_lstm.py                 # Baseline LSTM training
│   ├── train_pso_lstm.py           # PSO-LSTM training
│   ├── run_xgboost.py              # XGBoost training
│   ├── run_backtest.py             # Unified backtesting
│   └── lstm_walk_forward_evaluation.py  # Walk-forward validation
│
├── plans/                          # Design specifications and audits
│   ├── design_specs/               # TRD documents
│   ├── audits_and_analysis/        # Compliance audits
│   └── completion_summaries/       # Implementation summaries
│
├── results/                        # Experiment outputs
│   ├── experiments/                # Training results
│   ├── pso_lstm/                   # PSO-LSTM outputs
│   └── backtest/                   # Backtesting results
│
├── src/                            # Core library modules
│   ├── data/                       # Data ingestion and preprocessing
│   │   ├── alpaca_ingestor.py     # Alpaca API client
│   │   ├── cleaner.py             # Data cleaning (gap handling, validation)
│   │   └── windowing.py           # Temporal window construction
│   │
│   ├── features/                   # Feature engineering
│   │   ├── feature_generators.py  # Technical indicators + cross-ticker
│   │   ├── scaler.py              # MinMax scaling (frozen)
│   │   ├── selector.py            # 4-stage feature selection
│   │   └── wavelet.py             # Wavelet denoising (Haar, 3-level)
│   │
│   ├── models/                     # Model implementations
│   │   ├── lstm_model.py          # PyTorch LSTM wrapper
│   │   ├── lstm_trainer.py        # LSTM training loop (early stopping)
│   │   ├── xgboost_model.py       # XGBoost wrapper
│   │   └── utils.py               # Model utilities (seed control, etc.)
│   │
│   ├── optimizer/                  # PSO/IPSO algorithms
│   │   ├── pso_core.py            # PSO core logic
│   │   ├── ipso.py                # Improved PSO (adaptive inertia)
│   │   ├── particle.py            # Particle representation
│   │   └── fitness.py             # Fitness function (0.9×MSE + 0.1×MSW)
│   │
│   ├── evaluation/                 # Evaluation and backtesting
│   │   ├── metrics.py             # Statistical + trading metrics
│   │   ├── backtest.py            # Trading simulation
│   │   ├── walk_forward_pso.py    # Walk-forward with PSO
│   │   ├── model_loader.py        # Unified model loading (adapter pattern)
│   │   └── plotting.py            # Visualization utilities
│   │
│   └── utils/                      # Shared utilities
│       ├── config_loader.py       # YAML configuration loader
│       ├── logger.py              # Logging setup
│       └── seed.py                # Reproducibility utilities
│
├── .gemini/                        # Gemini CLI skills (custom agents)
│   ├── GEMINI.md                  # Global agent rules
│   └── skills/                    # Custom skill definitions
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .env                           # Environment variables (not in Git)
```

---

## Technical Requirements Documents (TRDs)

The system is governed by three authoritative Technical Requirements Documents:

### TRD1: Feature Engineering Pipeline
**File**: `docs/TRD1.md`

Specifies:
- 45+ technical indicators (formulas, parameters, edge cases)
- Cross-ticker feature computation (market context, peers, sector)
- Wavelet denoising protocol (Haar, 3-level, soft thresholding)
- 4-stage feature selection (variance → correlation → VIF → MI)
- Scaling protocol (separate feature/target scalers)
- Target definition: `r_t = (Close_{t+1} - Close_t) / Close_t`

### TRD2: Model Architecture
**File**: `docs/TRD2.md`

Specifies:
- LSTM architecture (2-layer, ReLU activation, dropout)
- Training protocol (Adam optimizer, MSE loss, early stopping)
- PSO/IPSO hyperparameter search (6D space, fitness function)
- XGBoost configuration (tree-based gradient boosting)
- Reproducibility requirements (seed control, deterministic=True)

### TRD3: Evaluation & Backtesting
**File**: `docs/TRD3.md`

Specifies:
- Data splitting (70/10/20, chronological)
- Walk-forward validation (expanding window, per-fold retraining)
- Backtesting protocol (transaction costs, slippage, stop-loss)
- Metrics definitions (statistical + trading)
- Visualization requirements

### Canonical Specification: FINAL_PLAN.md
**File**: `plans/design_specs/FINAL_PLAN.md`

Consolidates all TRDs and resolves conflicts. This is the **single source of truth** for system design.

---

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_features.py

# Run with coverage
pytest --cov=src --cov-report=html tests/
```

### Integration Tests

```bash
# Test full pipeline (small dataset)
python scripts/test_pipeline.py --ticker AAPL --n-samples 1000
```

### Validation Tests

```bash
# Verify data alignment
python scripts/verify_alignment.py --data-dir data/aligned/1Day

# Verify split correctness
python scripts/explore_splits.py --data-path data/features_v2/AAPL
```

---

## Troubleshooting

### Common Issues

#### 1. `RuntimeError: Model not built. Call build_model() first.`

**Solution**: LSTM models require explicit architecture construction before loading weights.

```python
# Incorrect
model = LSTMModel(seed=42)
model.load("model.pt")  # ERROR

# Correct
model = LSTMModel(seed=42)
model.build_model(model_config)  # Build architecture first
model.load("model.pt")
```

#### 2. `ValueError: Cannot align tickers`

**Cause**: Mismatched timestamp indices across tickers.

**Solution**: Ensure all tickers are SPY-aligned:

```bash
python pipelines/data_ingest_align_clean.py \
  --tickers AAPL MSFT SPY \
  --force-realign
```

#### 3. `ValueError: All arrays must be of the same length`

**Cause**: LSTM windowing reduces sample count (N - lookback + 1).

**Solution**: Ensure all arrays are windowed consistently or truncated to match.

#### 4. Directional Accuracy ~99.96% (Suspiciously High)

**Cause**: Bug in `np.sign()` usage where `sign(0) = 0` creates false positives.

**Status**: FIXED in commit `[2026-04-28]` via corrected `directional_accuracy()` in `src/evaluation/metrics.py`.

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/your-feature`)
3. Follow **TRD specifications** for any changes
4. Add **tests** for new functionality
5. Update **documentation** (README, docstrings)
6. Submit a **pull request**

### Code Quality Standards

- PEP 8 compliance (max line length: 120)
- Type hints for all function signatures
- Google-style docstrings
- Logging for key operations
- No data leakage (verify split-first architecture)

---

## References

### Academic Papers

1. **Ji et al. (2021)**: "Application of LSTM Model based on Particle Swarm Optimization Algorithm in Stock Market Trend Prediction"
2. **Zeng et al. (2025)**: "Enhancing stock index prediction: A hybrid LSTM-PSO model for improved forecasting accuracy"
3. **Deng & Peng (2025)**: "A Novel Improved Particle Swarm Optimization for LSTM"

### Related Documentation

- [Alpaca API Documentation](https://alpaca.markets/docs/)
- [PyTorch LSTM Tutorial](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Alpaca Markets** for data API access
- **PyTorch Team** for the deep learning framework
- **XGBoost Contributors** for the gradient boosting library
- Research papers cited above for algorithmic foundations

---

**⚠️ Disclaimer**: This system is for **educational and research purposes only**. It is NOT financial advice. Do NOT use for live trading without extensive backtesting and risk management. Past performance does not guarantee future results.
