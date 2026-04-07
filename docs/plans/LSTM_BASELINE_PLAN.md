# LSTM Baseline Implementation Plan

**Project:** PSO-LSTM Stock Price Prediction System  
**Document Version:** 1.0  
**Date:** April 6, 2026  
**Purpose:** Implement standalone LSTM baseline training script following project architecture patterns

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Implementation Plan](#3-implementation-plan)
4. [LSTM Baseline Configuration](#4-lstm-baseline-configuration)
5. [Script Structure](#5-script-structure)
6. [Testing Strategy](#6-testing-strategy)
7. [Success Criteria](#7-success-criteria)

---

## 1. Executive Summary

### Objective

Implement a standalone training script `run_lstm_baseline.py` that trains the vanilla LSTM model (without PSO optimization) following the same architectural patterns as `run_xgboost.py` and `run_pso.py`. This provides a direct baseline comparison for the PSO-optimized LSTM model.

### Key Requirements

1. **Consistent Architecture**: Follow the same structure as existing scripts (`run_xgboost.py`, `run_pso.py`)
2. **Model Integration**: Use existing `VanillaLSTM` class from `src/models/baselines.py`
3. **Mode Support**: Support train, validation, and test modes
4. **Metrics Tracking**: Track and save comprehensive metrics
5. **Result Storage**: Save models, predictions, and metrics in organized format

### Success Criteria

- [ ] Script trains LSTM baseline successfully with fixed hyperparameters
- [ ] Supports train/val/test modes like `run_xgboost.py`
- [ ] Saves model checkpoints and predictions
- [ ] Generates training history and metrics
- [ ] Creates visualization plots
- [ ] Follows project code style and conventions

---

## 2. Architecture Overview

### 2.1 Existing Components

**Available:**
- ✅ `src/models/baselines.py` - Contains `VanillaLSTM` class
- ✅ `src/models/lstm_model.py` - Core `LSTMModel` and `LSTMTrainer` classes
- ✅ `src/data/splitter.py` - `build_windows()` for sequence creation
- ✅ `src/evaluation/metrics.py` - Statistical and trading metrics
- ✅ `scripts/run_xgboost.py` - Reference script structure

**Missing:**
- ❌ `scripts/run_lstm_baseline.py` - Main training script (to be created)

### 2.2 Script Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    run_lstm_baseline.py                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Argument       │      │  Data Loading    │      │  Model Training  │
│   Parsing        │─────►│  & Windowing     │─────►│  (VanillaLSTM)   │
└──────────────────┘      └──────────────────┘      └──────────────────┘
                                                              │
                                                              ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Visualization  │◄─────│  Metrics         │◄─────│  Evaluation      │
│   & Plotting     │      │  Computation     │      │  & Prediction    │
└──────────────────┘      └──────────────────┘      └──────────────────┘
                                                              │
                                                              ▼
                                                     ┌──────────────────┐
                                                     │  Save Results    │
                                                     │  - Model (.pth)  │
                                                     │  - Metrics (.json)│
                                                     │  - Plots (.png)  │
                                                     └──────────────────┘
```

### 2.3 Data Flow

```
Raw Features [N, F]
    ↓
Build Windows → [N, T, F] where T = lookback (default 30)
    ↓
Train/Val/Test Split
    ↓
VanillaLSTM Training
    ↓
Predictions [N]
    ↓
Metrics Computation
    ↓
Results Storage
```

---

## 3. Implementation Plan

### Phase 1: Core Script Implementation

#### Task 1.1: Create `run_lstm_baseline.py` Script Structure

**File:** `scripts/run_lstm_baseline.py`

**Components:**

1. **Imports and Setup**
   - Standard library imports
   - Project imports (models, utils, evaluation)
   - Logger configuration

2. **Argument Parser**
   - Common arguments (ticker, seed, directories)
   - Mode selection (train, val, test)
   - Model-specific arguments (lookback, batch_size, etc.)
   - Optional override arguments for hyperparameters

3. **Helper Functions**
   - `load_data()` - Load feature arrays
   - `load_feature_names()` - Load metadata
   - `log_memory_usage()` - Memory profiling
   - `_print_info_metrics()` - Pretty print metrics

4. **Mode Handlers**
   - `run_train()` - Training mode
   - `run_val()` - Validation mode (walk-forward)
   - `run_test()` - Testing mode (backtesting)

5. **Main Function**
   - Setup (logging, seeds, directories)
   - Mode dispatch
   - Error handling

---

#### Task 1.2: Implement Training Mode (`run_train`)

**Purpose:** Train LSTM baseline with fixed hyperparameters and save model.

**Steps:**

1. **Load Data**
   ```python
   X_train_flat = np.load(ticker_dir / "X_train.npy")
   y_train = np.load(ticker_dir / "y_train.npy")
   X_val_flat = np.load(ticker_dir / "X_val.npy")
   y_val = np.load(ticker_dir / "y_val.npy")
   ```

2. **Build Windows**
   ```python
   max_lookback = args.lookback  # Default 30
   X_train_windows, y_train_windows = build_windows(
       X_train_flat, y_train, session_starts_train, max_lookback
   )
   X_val_windows, y_val_windows = build_windows(
       X_val_flat, y_val, session_starts_val, max_lookback
   )
   ```

3. **Initialize Model**
   ```python
   from src.models.baselines import VanillaLSTM
   
   model = VanillaLSTM(
       input_size=X_train_flat.shape[1],
       num_layers=args.num_layers,
       hidden_units=args.hidden_units,
       dropout=args.dropout,
       learning_rate=args.learning_rate,
       lookback=args.lookback,
       max_epochs=args.max_epochs,
       patience=args.patience,
       batch_size=args.batch_size,
   )
   ```

4. **Train Model**
   ```python
   history = model.fit(
       X_train_windows, y_train_windows,
       X_val_windows, y_val_windows
   )
   ```

5. **Evaluate on Validation Set**
   ```python
   y_pred_val = model.predict(X_val_windows)
   val_metrics = all_statistical_metrics(y_val_windows, y_pred_val)
   ```

6. **Save Results**
   ```python
   # Save model
   torch.save(model._trainer.model.state_dict(), 
              results_dir / f"lstm_baseline_model_{tag}.pth")
   
   # Save hyperparameters
   with open(results_dir / f"lstm_baseline_params_{tag}.json", "w") as f:
       json.dump({
           "ticker": ticker,
           "mode": "train",
           "seed": args.seed,
           "hyperparameters": {
               "num_layers": args.num_layers,
               "hidden_units": args.hidden_units,
               "dropout": args.dropout,
               "learning_rate": args.learning_rate,
               "lookback": args.lookback,
               "batch_size": args.batch_size,
           },
           "val_metrics": val_metrics,
           "runtime_seconds": elapsed,
       }, f, indent=2)
   
   # Save training history
   with open(results_dir / f"lstm_baseline_history_{tag}.json", "w") as f:
       json.dump(history, f, indent=2)
   
   # Save predictions
   np.save(results_dir / f"lstm_baseline_predictions_{tag}.npy", y_pred_val)
   ```

7. **Generate Plots**
   ```python
   # Training loss curves
   plt.figure(figsize=(12, 5))
   plt.subplot(1, 2, 1)
   plt.plot(history['train_loss'], label='Train Loss')
   plt.plot(history['val_loss'], label='Val Loss')
   plt.title(f'{ticker} LSTM Baseline Training History')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   
   # Predictions vs True
   plt.subplot(1, 2, 2)
   plt.plot(y_val_windows[:500], label='True', alpha=0.7)
   plt.plot(y_pred_val[:500], label='Predicted', alpha=0.7)
   plt.title(f'{ticker} LSTM Baseline Predictions')
   plt.xlabel('Sample')
   plt.ylabel('Return')
   plt.legend()
   
   plt.tight_layout()
   plt.savefig(plots_dir / f"lstm_baseline_train_{tag}.png", dpi=150)
   ```

---

#### Task 1.3: Implement Validation Mode (`run_val`)

**Purpose:** Walk-forward validation to assess model stability over time.

**Steps:**

1. **Load Trained Model**
   ```python
   model = VanillaLSTM(input_size=X_val_flat.shape[1])
   model._trainer.model.load_state_dict(
       torch.load(results_dir / f"lstm_baseline_model_{ticker}_train_seed{args.seed}.pth")
   )
   ```

2. **Walk-Forward Validation**
   ```python
   fold_size = args.wfv_fold_size  # Default 252 (trading days)
   max_folds = args.wfv_folds  # Default 10
   
   fold_metrics = []
   for fold_idx in range(max_folds):
       start_idx = fold_idx * fold_size
       end_idx = start_idx + fold_size
       
       if end_idx > len(X_val_windows):
           break
       
       X_fold = X_val_windows[start_idx:end_idx]
       y_fold = y_val_windows[start_idx:end_idx]
       
       y_pred_fold = model.predict(X_fold)
       metrics = all_statistical_metrics(y_fold, y_pred_fold)
       
       fold_metrics.append({
           "fold": fold_idx,
           "start_idx": start_idx,
           "end_idx": end_idx,
           "metrics": metrics,
       })
   ```

3. **Aggregate Metrics**
   ```python
   avg_metrics = {}
   for metric_name in fold_metrics[0]["metrics"].keys():
       values = [f["metrics"][metric_name] for f in fold_metrics]
       avg_metrics[metric_name] = {
           "mean": np.mean(values),
           "std": np.std(values),
           "min": np.min(values),
           "max": np.max(values),
       }
   ```

4. **Save Validation Results**
   ```python
   with open(results_dir / f"lstm_baseline_wfv_{tag}.json", "w") as f:
       json.dump({
           "ticker": ticker,
           "fold_metrics": fold_metrics,
           "aggregate_metrics": avg_metrics,
       }, f, indent=2)
   ```

---

#### Task 1.4: Implement Test Mode (`run_test`)

**Purpose:** Final evaluation on held-out test set with backtesting.

**Steps:**

1. **Load Trained Model**
   ```python
   model = VanillaLSTM(input_size=X_test_flat.shape[1])
   model._trainer.model.load_state_dict(
       torch.load(results_dir / f"lstm_baseline_model_{ticker}_train_seed{args.seed}.pth")
   )
   ```

2. **Generate Test Predictions**
   ```python
   X_test_flat = np.load(ticker_dir / "X_test.npy")
   y_test = np.load(ticker_dir / "y_test.npy")
   
   X_test_windows, y_test_windows = build_windows(
       X_test_flat, y_test, session_starts_test, args.lookback
   )
   
   y_pred_test = model.predict(X_test_windows)
   ```

3. **Compute Statistical Metrics**
   ```python
   test_metrics = all_statistical_metrics(y_test_windows, y_pred_test)
   ```

4. **Run Backtesting**
   ```python
   from src.evaluation.backtester import Backtester
   
   backtester = Backtester(
       initial_capital=args.initial_capital,
       position_fraction=args.position_fraction,
       transaction_cost=args.transaction_cost,
       slippage=args.slippage,
       stop_loss=args.stop_loss,
       daily_loss_limit=args.daily_loss_limit,
   )
   
   backtest_results = backtester.run(
       predictions=y_pred_test,
       true_returns=y_test_windows,
       prices=test_prices,  # Load from metadata
   )
   ```

5. **Save Test Results**
   ```python
   with open(results_dir / f"lstm_baseline_test_{tag}.json", "w") as f:
       json.dump({
           "ticker": ticker,
           "statistical_metrics": test_metrics,
           "backtest_results": backtest_results,
       }, f, indent=2)
   
   np.save(results_dir / f"lstm_baseline_test_predictions_{tag}.npy", y_pred_test)
   ```

6. **Generate Test Plots**
   ```python
   # Equity curve
   plt.figure(figsize=(12, 5))
   plt.plot(backtest_results['equity_curve'])
   plt.title(f'{ticker} LSTM Baseline Equity Curve')
   plt.xlabel('Time')
   plt.ylabel('Portfolio Value ($)')
   plt.savefig(plots_dir / f"lstm_baseline_equity_{tag}.png", dpi=150)
   ```

---

### Phase 2: Configuration and Integration

#### Task 2.1: Add LSTM Baseline Config Section

**File:** `config/default_config.yaml`

**Add:**

```yaml
lstm_baseline:
  # Fixed hyperparameters (no PSO optimization)
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
  use_amp: true  # Mixed precision training
  accumulation_steps: 1
  
  # Walk-forward validation
  wfv_fold_size: 252  # ~1 trading year
  wfv_folds: 10
  
  # Backtesting
  initial_capital: 100000.0
  position_fraction: 0.02
  transaction_cost: 0.001
  slippage: 0.0005
  stop_loss: 0.02
  daily_loss_limit: 0.05
```

---

#### Task 2.2: Update Model Registry

**File:** `src/models/registry.py` (if exists)

**Add:**

```python
from .baselines import VanillaLSTM

def build_vanilla_lstm(params, X_train, y_train, X_val, y_val):
    """Vanilla LSTM builder function."""
    model = VanillaLSTM(
        input_size=X_train.shape[2],
        num_layers=params.get("num_layers", 2),
        hidden_units=params.get("hidden_units", 128),
        dropout=params.get("dropout", 0.2),
        learning_rate=params.get("learning_rate", 0.001),
        lookback=params.get("lookback", 30),
        max_epochs=params.get("max_epochs", 100),
        patience=params.get("patience", 10),
        batch_size=params.get("batch_size", 256),
    )
    model.fit(X_train, y_train, X_val, y_val)
    return model.predict(X_val)

# Register baseline LSTM
ModelRegistry.register("lstm_baseline", VanillaLSTM, build_vanilla_lstm, None)
```

---

### Phase 3: Testing and Documentation

#### Task 3.1: Create Unit Tests

**File:** `tests/test_lstm_baseline.py`

**Tests:**

```python
import pytest
import numpy as np
from pathlib import Path
from src.models.baselines import VanillaLSTM

def test_vanilla_lstm_initialization():
    """Test VanillaLSTM can be initialized."""
    model = VanillaLSTM(input_size=10)
    assert model is not None
    assert model.lookback == 30

def test_vanilla_lstm_training():
    """Test VanillaLSTM can train on dummy data."""
    X_train = np.random.randn(100, 30, 10)
    y_train = np.random.randn(100)
    X_val = np.random.randn(50, 30, 10)
    y_val = np.random.randn(50)
    
    model = VanillaLSTM(input_size=10, max_epochs=2)
    history = model.fit(X_train, y_train, X_val, y_val)
    
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) > 0

def test_vanilla_lstm_prediction():
    """Test VanillaLSTM can generate predictions."""
    X_train = np.random.randn(100, 30, 10)
    y_train = np.random.randn(100)
    X_val = np.random.randn(50, 30, 10)
    y_val = np.random.randn(50)
    
    model = VanillaLSTM(input_size=10, max_epochs=2)
    model.fit(X_train, y_train, X_val, y_val)
    
    y_pred = model.predict(X_val)
    assert y_pred.shape == (50,)
    assert not np.isnan(y_pred).any()

def test_vanilla_lstm_custom_params():
    """Test VanillaLSTM accepts custom hyperparameters."""
    model = VanillaLSTM(
        input_size=10,
        num_layers=3,
        hidden_units=64,
        dropout=0.3,
        learning_rate=0.01,
    )
    assert model._trainer.model.num_layers == 3
    assert model._trainer.model.hidden_units == 64
```

---

#### Task 3.2: Create Integration Test

**File:** `tests/test_run_lstm_baseline.py`

**Test:**

```python
import pytest
import subprocess
from pathlib import Path

def test_run_lstm_baseline_script():
    """Test run_lstm_baseline.py executes without errors."""
    # Assumes test data exists
    result = subprocess.run([
        "python", "scripts/run_lstm_baseline.py",
        "--ticker", "TEST",
        "--mode", "train",
        "--seed", "42",
        "--max-epochs", "2",  # Quick test
        "--features-dir", "tests/fixtures/features",
        "--results-dir", "tests/fixtures/results",
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Training complete" in result.stdout
```

---

#### Task 3.3: Update Documentation

**Files to Update:**

1. **README.md**
   - Add LSTM baseline to model list
   - Add usage example

2. **docs/overviews/model_architecture.md**
   - Document LSTM baseline architecture
   - Compare with PSO-optimized LSTM

3. **Create Usage Guide**
   - `docs/guides/LSTM_BASELINE_USAGE.md`

---

## 4. LSTM Baseline Configuration

### 4.1 Default Hyperparameters

Based on `VanillaLSTM.DEFAULT_PARAMS` in `src/models/baselines.py`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_layers` | 2 | Ji et al. default; Zeng et al. best result |
| `hidden_units` | 128 | Common choice in literature |
| `dropout` | 0.2 | Standard regularization |
| `learning_rate` | 0.001 | Adam optimizer default |
| `lookback` | 30 | Lanbouri & Achchab optimal short-term window |
| `max_epochs` | 100 | Sufficient for convergence |
| `patience` | 10 | Early stopping patience |
| `batch_size` | 256 | Balance speed and memory |

### 4.2 Optional Overrides

Users can override defaults via command-line arguments:

```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --num-layers 3 \
    --hidden-units 256 \
    --dropout 0.3 \
    --learning-rate 0.0005 \
    --lookback 60 \
    --batch-size 128
```

### 4.3 Memory Optimization Settings

For low-memory environments:

```yaml
lstm_baseline:
  use_amp: true              # Mixed precision (FP16)
  accumulation_steps: 4      # Gradient accumulation
  batch_size: 64             # Smaller batches
  use_checkpointing: true    # Gradient checkpointing
```

---

## 5. Script Structure

### 5.1 Complete File Structure

```python
#!/usr/bin/env python3
"""
scripts/run_lstm_baseline.py

Train vanilla LSTM baseline (no PSO optimization) for stock return prediction.

Usage:
    # Training
    python scripts/run_lstm_baseline.py --ticker AAPL --mode train
    
    # Validation
    python scripts/run_lstm_baseline.py --ticker AAPL --mode val
    
    # Testing
    python scripts/run_lstm_baseline.py --ticker AAPL --mode test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baselines import VanillaLSTM
from src.data.splitter import build_windows
from src.evaluation.metrics import all_statistical_metrics
from src.utils import set_all_seeds, setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LSTM Baseline Training, Validation, Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Common arguments
    parser.add_argument("--ticker", required=True, help="Target ticker symbol")
    parser.add_argument(
        "--mode",
        choices=["train", "val", "test"],
        required=True,
        help="Execution mode: train, val, or test",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument(
        "--features-dir", default="data/features", help="Features directory"
    )
    parser.add_argument("--results-dir", default="results", help="Output directory")
    parser.add_argument(
        "--log-file", default="logs/run_lstm_baseline.log", help="Log file path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Config file path",
    )
    
    # Model hyperparameters (overrides)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--hidden-units", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    
    # Validation-specific arguments
    parser.add_argument("--wfv-fold-size", type=int, default=252)
    parser.add_argument("--wfv-folds", type=int, default=10)
    
    # Testing-specific arguments
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--position-fraction", type=float, default=0.02)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--stop-loss", type=float, default=0.02)
    parser.add_argument("--daily-loss-limit", type=float, default=0.05)
    
    args = parser.parse_args()
    
    # Filter arguments based on mode
    if args.mode == "train":
        for attr in [
            "wfv_fold_size", "wfv_folds",
            "initial_capital", "position_fraction",
            "transaction_cost", "slippage",
            "stop_loss", "daily_loss_limit",
        ]:
            if hasattr(args, attr):
                delattr(args, attr)
    elif args.mode == "val":
        for attr in [
            "initial_capital", "position_fraction",
            "transaction_cost", "slippage",
            "stop_loss", "daily_loss_limit",
        ]:
            if hasattr(args, attr):
                delattr(args, attr)
    elif args.mode == "test":
        for attr in ["wfv_fold_size", "wfv_folds"]:
            if hasattr(args, attr):
                delattr(args, attr)
    
    return args


# ======================================================================
# Helper functions
# ======================================================================

def load_data(ticker_dir: Path, split: str):
    """Load feature arrays for given split."""
    X = np.load(ticker_dir / f"X_{split}.npy")
    y = np.load(ticker_dir / f"y_{split}.npy")
    return X, y


def log_memory_usage(label: str):
    """Log current memory usage."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logger.info(f"[{label}] Memory: {mem_info.rss / 1024**3:.2f} GB")
    except ImportError:
        pass


def _print_info_metrics(label: str, m: dict) -> None:
    """Pretty-print metrics dict."""
    logger.info(
        f"  {label} — "
        f"RMSE={m['rmse']:.6f}  "
        f"DA={m['directional_accuracy']:.4f}  "
        f"F1={m['f1_ternary']:.4f}  "
        f"R²={m['r2']:.4f}"
    )


# ======================================================================
# Mode handlers
# ======================================================================

def run_train(args, features_dir, results_dir, ticker, tag):
    """Training mode: train LSTM baseline and save model."""
    cfg = load_config(args.config)
    
    logger.info("\n[TRAIN] Loading data...")
    ticker_dir = features_dir / ticker
    
    X_train_flat, y_train = load_data(ticker_dir, "train")
    X_val_flat, y_val = load_data(ticker_dir, "val")
    
    log_memory_usage("After loading data")
    
    # Build windows
    lookback = args.lookback or cfg.lstm_baseline.get("lookback", 30)
    session_starts_train = np.zeros(len(X_train_flat), dtype=bool)
    session_starts_val = np.zeros(len(X_val_flat), dtype=bool)
    
    X_train_windows, y_train_windows = build_windows(
        X_train_flat, y_train, session_starts_train, lookback
    )
    X_val_windows, y_val_windows = build_windows(
        X_val_flat, y_val, session_starts_val, lookback
    )
    
    logger.info("Windowed Shapes:")
    logger.info("  X_train: %s", X_train_windows.shape)
    logger.info("  X_val: %s", X_val_windows.shape)
    
    log_memory_usage("After windowing")
    
    # Build hyperparameters
    hyperparams = {
        "num_layers": args.num_layers or cfg.lstm_baseline.get("num_layers", 2),
        "hidden_units": args.hidden_units or cfg.lstm_baseline.get("hidden_units", 128),
        "dropout": args.dropout or cfg.lstm_baseline.get("dropout", 0.2),
        "learning_rate": args.learning_rate or cfg.lstm_baseline.get("learning_rate", 0.001),
        "lookback": lookback,
        "max_epochs": args.max_epochs or cfg.lstm_baseline.get("max_epochs", 100),
        "patience": args.patience or cfg.lstm_baseline.get("patience", 10),
        "batch_size": args.batch_size or cfg.lstm_baseline.get("batch_size", 256),
    }
    
    logger.info("Hyperparameters: %s", hyperparams)
    
    # Initialize model
    model = VanillaLSTM(
        input_size=X_train_flat.shape[1],
        **hyperparams
    )
    
    # Train
    logger.info("\nTraining model...")
    t0 = time.time()
    history = model.fit(X_train_windows, y_train_windows, X_val_windows, y_val_windows)
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s")
    
    log_memory_usage("After training")
    
    # Evaluate
    y_pred_val = model.predict(X_val_windows)
    val_metrics = all_statistical_metrics(y_val_windows, y_pred_val)
    _print_info_metrics("Validation", val_metrics)
    
    # Save model
    model_path = results_dir / f"lstm_baseline_model_{tag}.pth"
    torch.save(model._trainer.model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save params
    with open(results_dir / f"lstm_baseline_params_{tag}.json", "w") as f:
        json.dump({
            "ticker": ticker,
            "mode": "train",
            "seed": args.seed,
            "hyperparameters": hyperparams,
            "val_metrics": val_metrics,
            "runtime_seconds": elapsed,
        }, f, indent=2)
    
    # Save history
    with open(results_dir / f"lstm_baseline_history_{tag}.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Save predictions
    np.save(results_dir / f"lstm_baseline_predictions_{tag}.npy", y_pred_val)
    
    # Plot
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(16, 6))
    
    # Training history
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', alpha=0.7)
    plt.plot(history['val_loss'], label='Val Loss', alpha=0.7)
    plt.title(f'{ticker} LSTM Baseline Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Predictions
    plt.subplot(1, 2, 2)
    n_samples = min(500, len(y_val_windows))
    plt.plot(y_val_windows[:n_samples], label='True', alpha=0.7)
    plt.plot(y_pred_val[:n_samples], label='Predicted', alpha=0.7)
    plt.title(f'{ticker} LSTM Baseline Predictions')
    plt.xlabel('Sample')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = plots_dir / f"lstm_baseline_train_{tag}.png"
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Plot saved to {plot_path}")
    plt.close()
    
    logger.info("Training complete.")


def run_val(args, features_dir, results_dir, ticker, tag):
    """Validation mode: walk-forward validation."""
    logger.info("\n[VAL] Walk-forward validation...")
    
    # Load model
    model_path = results_dir / f"lstm_baseline_model_{ticker}_train_seed{args.seed}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")
    
    # Load data
    ticker_dir = features_dir / ticker
    X_val_flat, y_val = load_data(ticker_dir, "val")
    
    # Load params to get lookback
    params_path = results_dir / f"lstm_baseline_params_{ticker}_train_seed{args.seed}.json"
    with open(params_path) as f:
        params = json.load(f)
    lookback = params["hyperparameters"]["lookback"]
    
    # Build windows
    session_starts_val = np.zeros(len(X_val_flat), dtype=bool)
    X_val_windows, y_val_windows = build_windows(
        X_val_flat, y_val, session_starts_val, lookback
    )
    
    # Initialize model and load weights
    model = VanillaLSTM(input_size=X_val_flat.shape[1])
    model._trainer.model.load_state_dict(torch.load(model_path))
    
    # Walk-forward validation
    fold_size = args.wfv_fold_size
    max_folds = args.wfv_folds
    
    fold_metrics = []
    for fold_idx in range(max_folds):
        start_idx = fold_idx * fold_size
        end_idx = start_idx + fold_size
        
        if end_idx > len(X_val_windows):
            break
        
        X_fold = X_val_windows[start_idx:end_idx]
        y_fold = y_val_windows[start_idx:end_idx]
        
        y_pred_fold = model.predict(X_fold)
        metrics = all_statistical_metrics(y_fold, y_pred_fold)
        
        fold_metrics.append({
            "fold": fold_idx,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "metrics": metrics,
        })
        
        logger.info(f"Fold {fold_idx}: RMSE={metrics['rmse']:.6f}")
    
    # Aggregate
    avg_metrics = {}
    for metric_name in fold_metrics[0]["metrics"].keys():
        values = [f["metrics"][metric_name] for f in fold_metrics]
        avg_metrics[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    
    logger.info("\nAggregate Metrics:")
    logger.info(f"  RMSE: {avg_metrics['rmse']['mean']:.6f} ± {avg_metrics['rmse']['std']:.6f}")
    
    # Save
    with open(results_dir / f"lstm_baseline_wfv_{tag}.json", "w") as f:
        json.dump({
            "ticker": ticker,
            "fold_metrics": fold_metrics,
            "aggregate_metrics": avg_metrics,
        }, f, indent=2)
    
    logger.info("Validation complete.")


def run_test(args, features_dir, results_dir, ticker, tag):
    """Test mode: final evaluation on test set."""
    logger.info("\n[TEST] Final evaluation...")
    
    # Load model
    model_path = results_dir / f"lstm_baseline_model_{ticker}_train_seed{args.seed}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")
    
    # Load data
    ticker_dir = features_dir / ticker
    X_test_flat, y_test = load_data(ticker_dir, "test")
    
    # Load params
    params_path = results_dir / f"lstm_baseline_params_{ticker}_train_seed{args.seed}.json"
    with open(params_path) as f:
        params = json.load(f)
    lookback = params["hyperparameters"]["lookback"]
    
    # Build windows
    session_starts_test = np.zeros(len(X_test_flat), dtype=bool)
    X_test_windows, y_test_windows = build_windows(
        X_test_flat, y_test, session_starts_test, lookback
    )
    
    # Initialize model and load weights
    model = VanillaLSTM(input_size=X_test_flat.shape[1])
    model._trainer.model.load_state_dict(torch.load(model_path))
    
    # Predict
    y_pred_test = model.predict(X_test_windows)
    
    # Metrics
    test_metrics = all_statistical_metrics(y_test_windows, y_pred_test)
    _print_info_metrics("Test", test_metrics)
    
    # Save
    with open(results_dir / f"lstm_baseline_test_{tag}.json", "w") as f:
        json.dump({
            "ticker": ticker,
            "statistical_metrics": test_metrics,
        }, f, indent=2)
    
    np.save(results_dir / f"lstm_baseline_test_predictions_{tag}.npy", y_pred_test)
    
    # Plot
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(16, 6))
    n_samples = min(1000, len(y_test_windows))
    plt.plot(y_test_windows[:n_samples], label='True', alpha=0.7)
    plt.plot(y_pred_test[:n_samples], label='Predicted', alpha=0.7)
    plt.title(f'{ticker} LSTM Baseline Test Predictions')
    plt.xlabel('Sample')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = plots_dir / f"lstm_baseline_test_{tag}.png"
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Plot saved to {plot_path}")
    plt.close()
    
    logger.info("Testing complete.")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()
    
    # Setup
    setup_logger(args.log_file)
    set_all_seeds(args.seed)
    
    features_dir = Path(args.features_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    ticker = args.ticker
    tag = f"{ticker}_{args.mode}_seed{args.seed}"
    
    logger.info(f"{'='*60}")
    logger.info(
        f"LSTM Baseline Pipeline | mode={args.mode} | ticker={ticker} | seed={args.seed}"
    )
    logger.info(f"{'='*60}")
    
    # Mode dispatch
    if args.mode == "train":
        run_train(args, features_dir, results_dir, ticker, tag)
    elif args.mode == "val":
        run_val(args, features_dir, results_dir, ticker, tag)
    elif args.mode == "test":
        run_test(args, features_dir, results_dir, ticker, tag)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Coverage Goals:**
- VanillaLSTM class: 90%
- Helper functions: 80%

**Test Files:**
```
tests/
├── test_lstm_baseline.py       # VanillaLSTM tests
└── test_run_lstm_baseline.py   # Script integration tests
```

### 6.2 Integration Tests

**Scenarios:**
1. **Full Training Pipeline**
   - Load data → Train → Save → Evaluate
   - Expected time: < 10 minutes (small dataset)

2. **Model Persistence**
   - Train → Save → Load → Predict
   - Verify predictions match

3. **Mode Transitions**
   - Train → Val → Test
   - Verify each mode works correctly

### 6.3 Validation Tests

**Checks:**
- [ ] Model trains without errors
- [ ] Predictions are in reasonable range
- [ ] No NaN/Inf in outputs
- [ ] Model files are saved correctly
- [ ] Metrics are computed correctly

---

## 7. Success Criteria

### 7.1 Implementation Success

- [ ] Script executes without errors in all modes (train/val/test)
- [ ] Model trains and converges properly
- [ ] Results are saved in correct format
- [ ] Plots are generated successfully
- [ ] Code follows project style conventions

### 7.2 Performance Success

- [ ] Training completes in reasonable time (< 30 min for typical dataset)
- [ ] Validation metrics are stable across folds
- [ ] Test metrics are comparable to literature baselines
- [ ] No memory leaks or crashes

### 7.3 Usability Success

- [ ] Clear command-line interface
- [ ] Helpful error messages
- [ ] Comprehensive logging
- [ ] Easy to compare with PSO-optimized LSTM

---

## 8. Usage Examples

### 8.1 Basic Training

```bash
# Train LSTM baseline with default hyperparameters
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --seed 42
```

### 8.2 Training with Custom Hyperparameters

```bash
# Train with custom settings
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

### 8.3 Walk-Forward Validation

```bash
# Validate trained model
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode val \
    --seed 42 \
    --wfv-fold-size 252 \
    --wfv-folds 10
```

### 8.4 Final Testing

```bash
# Test on held-out set
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode test \
    --seed 42 \
    --initial-capital 100000 \
    --position-fraction 0.02
```

---

## 9. Comparison with PSO-Optimized LSTM

### 9.1 Key Differences

| Aspect | LSTM Baseline | PSO-Optimized LSTM |
|--------|---------------|-------------------|
| Hyperparameters | Fixed (manually chosen) | Optimized via IPSO |
| Training Time | Fast (~10-30 min) | Slow (~6-12 hours) |
| Performance | Good baseline | Better (optimized) |
| Use Case | Quick baseline | Production model |
| Script | `run_lstm_baseline.py` | `run_pso.py` |

### 9.2 When to Use Each

**Use LSTM Baseline:**
- Quick experiments
- Baseline comparisons
- Resource-constrained environments
- Initial model validation

**Use PSO-Optimized LSTM:**
- Production deployment
- Maximum performance needed
- Sufficient compute resources
- Final model selection

---

## 10. File Checklist

### 10.1 Files to Create

- [ ] `scripts/run_lstm_baseline.py` - Main training script
- [ ] `tests/test_lstm_baseline.py` - Unit tests
- [ ] `tests/test_run_lstm_baseline.py` - Integration tests
- [ ] `docs/guides/LSTM_BASELINE_USAGE.md` - Usage guide

### 10.2 Files to Modify

- [ ] `config/default_config.yaml` - Add lstm_baseline section
- [ ] `README.md` - Add LSTM baseline documentation
- [ ] `docs/overviews/model_architecture.md` - Document baseline

### 10.3 Files to Reference

- ✅ `src/models/baselines.py` - VanillaLSTM class (already exists)
- ✅ `src/models/lstm_model.py` - Core LSTM classes (already exists)
- ✅ `scripts/run_xgboost.py` - Reference script structure (already exists)

---

## Document Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-04-06 | System | Initial document creation |

---

**End of Document**
