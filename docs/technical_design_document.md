# Technical Design Document
## PSO-LSTM Stock Price Prediction System

**Project:** LSTM Hyperparameter Tuning with Improved PSO for Stock Price Prediction
**Course:** CSCI 633 — Biologically-Inspired Intelligent Systems
**Version:** 1.0 | March 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Data Flow](#3-data-flow)
4. [Module Breakdown](#4-module-breakdown)
5. [Class and Interface Design](#5-class-and-interface-design)
6. [Pipeline Orchestration](#6-pipeline-orchestration)
7. [Technology Stack](#7-technology-stack)
8. [Design Decisions and Trade-offs](#8-design-decisions-and-trade-offs)

---

## 1. System Overview

The system is a five-stage machine learning pipeline that ingests high-frequency equity data, engineers a rich feature set, tunes an LSTM model via Improved Particle Swarm Optimization, and evaluates predictive and trading performance against multiple baselines.

### Goals

| Goal | Metric | Target |
|---|---|---|
| Predict next-minute log return direction | Directional Accuracy | > 55% |
| Risk-adjusted portfolio performance | Sharpe Ratio | > 1.0 (annualized) |
| Hyperparameter optimization | Composite fitness | Minimize RMSE, maximize Sharpe |
| Benchmark PSO vs. baselines | RMSE, Sharpe vs. persistence/LSTM/XGB | IPSO-LSTM wins ≥ 2 of 3 metrics |

### Scope

- **Universe:** 51 tickers (50 equities + SPY ETF as benchmark)
- **Data frequency:** 1-minute OHLCV bars
- **Horizon:** Predict 1-step (1-minute) ahead
- **Time range:** 5 years (2019-01-01 to 2024-01-01)
- **Prediction target:** Log return of mid-price at t+1

---

## 2. Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         PSO-LSTM SYSTEM ARCHITECTURE                       │
└────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────┐
   │   ALPACA MARKETS API │
   │  (1-min OHLCV bars)  │
   └──────────┬───────────┘
              │  REST / WebSocket
              ▼
   ┌──────────────────────┐
   │   DATA INGESTION     │  AlpacaIngestor
   │  - Rate-limited DL   │  TickerAligner
   │  - 51 tickers × 5yr  │  DataCleaner
   │  - Parquet storage   │
   └──────────┬───────────┘
              │  Aligned OHLCV DataFrames
              ▼
   ┌──────────────────────┐
   │  FEATURE ENGINEERING │  FeaturePipeline
   │  - Technical (40+)   │   ├─ TechnicalFeatures
   │  - Statistical (25+) │   ├─ StatisticalFeatures
   │  - Volume/Liq (20+)  │   ├─ VolumeFeatures
   │  - Cross-ticker (15+)│   └─ CrossTickerFeatures
   │  - Feature selection │  FeatureSelector (XGB + PSO mask)
   └──────────┬───────────┘
              │  Selected feature tensors [N, T, F]
              ▼
   ┌──────────────────────┐       ┌────────────────────────────┐
   │  DATA SPLITTER       │       │   BASELINE MODELS          │
   │  Train: 3yr          │──────►│   - PersistenceModel       │
   │  Val:   1yr          │       │   - XGBoostRegressor       │
   │  Test:  1yr          │       │   - VanillaLSTM (manual)   │
   └──────────┬───────────┘       └────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                    IPSO OPTIMIZER                            │
   │  ┌─────────────────────────────────────────────────────────┐ │
   │  │  Swarm: 30 particles × 5-dimensional search space      │ │
   │  │  Particle: [n_layers, hidden, dropout, lr, lookback]   │ │
   │  │                                                         │ │
   │  │  For each particle at each iteration:                  │ │
   │  │    1. Decode particle → LSTM hyperparameters           │ │
   │  │    2. Train LSTM on Train set                          │ │
   │  │    3. Evaluate on Val set → RMSE, Sharpe, Drawdown     │ │
   │  │    4. Compute composite fitness F(x)                   │ │
   │  │    5. Update pbest, gbest                              │ │
   │  │    6. Apply tanh inertia weight update                 │ │
   │  │    7. Apply adaptive mutation (if ξ > μ_mf)           │ │
   │  │    8. Update velocity + position                       │ │
   │  └─────────────────────────────────────────────────────────┘ │
   └──────────────────────────┬───────────────────────────────────┘
                              │  gbest hyperparameters
                              ▼
   ┌──────────────────────┐
   │  LSTM MODEL          │  LSTMModel(gbest_params)
   │  (Final training on  │  - Retrain on Train+Val
   │   Train+Val)         │  - Inference on Test set
   └──────────┬───────────┘
              │  Predictions ŷ
              ▼
   ┌──────────────────────┐
   │  EVALUATION ENGINE   │  MetricsCalculator
   │  Statistical:        │  WalkForwardValidator
   │   RMSE, DA, F1, AUC  │  Backtester
   │  Trading:            │
   │   Sharpe, MDD, CAGR  │
   └──────────────────────┘
```

---

## 3. Data Flow

### 3.1 Raw Data to Features

```
Alpaca API
    │
    ├─[GET /v2/stocks/{ticker}/bars?timeframe=1Min]
    │
    ▼
Raw OHLCV DataFrame (per ticker)
    │  Columns: [timestamp, open, high, low, close, volume]
    │  Missing value imputation (forward fill ≤ 5 bars)
    │  Session filtering (09:30–16:00 ET)
    │
    ▼
Cleaned OHLCV (per ticker)
    │  51 DataFrames aligned to common timestamp index
    │
    ▼
Feature Matrix [N_timestamps × N_tickers × N_features]
    │  Per-ticker features stacked horizontally
    │  Normalized per-feature using RobustScaler (fit on train only)
    │
    ▼
Selected Features [N_timestamps × F_selected]
    │  XGBoost importance threshold (top-70% cumulative)
    │  Optional PSO binary mask
    │
    ▼
Sliding Window Tensors [N_samples × T_lookback × F_selected]
    │  Target: log return at t+1
    │
    ▼
Train / Val / Test splits
```

### 3.2 PSO Fitness Evaluation Loop

```
Particle x_i = [n_layers, hidden, dropout, lr, lookback]
    │
    ▼
Decode → LSTM Hyperparameters
    │
    ▼
Build LSTMModel(n_layers, hidden, dropout, lr)
    │
    ▼
Train on (X_train, y_train) with early stopping (patience=10)
    │
    ▼
Predict on (X_val)
    │
    ├── Compute RMSE(y_val, ŷ_val)
    ├── Simulate signals → compute Sharpe ratio
    └── Compute Max Drawdown
    │
    ▼
F(x_i) = 0.4 × norm_RMSE + 0.4 × (1 - norm_Sharpe) + 0.2 × norm_MDD
    │
    ▼
Return F(x_i) to PSO core (minimize)
```

---

## 4. Module Breakdown

### 4.1 Data Ingestion (`src/data/`)

#### `alpaca_ingestor.py`

Responsible for downloading and caching historical bar data via the Alpaca Markets REST API.

**Key functions:**
```python
class AlpacaIngestor:
    def download_bars(ticker: str, start: date, end: date,
                      timeframe: str = "1Min") -> pd.DataFrame
    def download_universe(tickers: List[str], start: date,
                          end: date, output_dir: Path) -> None
    def _rate_limit_sleep() -> None  # 200 req/min limit
```

**Output schema:**
```
timestamp (UTC)  | open   | high   | low    | close  | volume
2019-01-02 14:30 | 249.54 | 249.80 | 249.40 | 249.75 | 1452300
```

#### `cleaner.py`

Handles missing values, outlier detection, and session filtering.

**Steps:**
1. Filter to NYSE trading hours (09:30–16:00 ET)
2. Forward-fill gaps ≤ 5 bars
3. Drop bars with volume = 0 (pre/post market artifacts)
4. Flag and cap price returns > 5 σ (fat-tail outliers)

#### `aligner.py`

Aligns 51 independent DataFrames to a single master timestamp index.

```python
def align_universe(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Inner join on timestamp index.
    Returns MultiIndex columns: (ticker, field)
    """
```

#### `splitter.py`

Performs chronological train/val/test split.

```python
def temporal_split(df: pd.DataFrame,
                   train_years=3, val_years=1, test_years=1
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
```

---

### 4.2 Feature Engineering (`src/features/`)

See `feature_engineering_spec.md` for the complete 100+ feature dictionary.

#### `pipeline.py`

```python
class FeaturePipeline:
    def __init__(self, target_ticker: str, universe: List[str])
    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray
    def transform(self, df: pd.DataFrame) -> np.ndarray
    # Normalizer fit on train only; applied to val/test
```

---

### 4.3 PSO Optimizer (`src/optimizer/`)

See `pso_mathematical_spec.md` for full mathematical specification.

#### `particle.py`

```python
@dataclass
class Particle:
    position: np.ndarray        # [n_layers, hidden, dropout, lr, lookback_idx]
    velocity: np.ndarray
    pbest_position: np.ndarray
    pbest_fitness: float
    fitness: float

    def decode(self) -> Dict[str, Any]:
        """Map continuous position to discrete/bounded hyperparameters."""
```

**Encoding scheme:**

| Dimension | Variable | Type | Range |
|---|---|---|---|
| 0 | `num_layers` | Integer | [1, 4] |
| 1 | `hidden_units` | Integer | [32, 512] |
| 2 | `dropout` | Float | [0.0, 0.5] |
| 3 | `learning_rate` | Float (log scale) | [1e-5, 1e-1] |
| 4 | `lookback_idx` | Integer index | {0,1,2,3} → {10,30,60,120} |

#### `ipso.py`

```python
class IPSO:
    def __init__(self, n_particles, n_iterations, search_space,
                 fitness_fn, w_min=0.4, w_max=0.9, c1=1.5, c2=1.5, seed=42)
    def run(self, X_train, y_train, X_val, y_val
            ) -> Tuple[Dict, float]  # (best_params, best_fitness)
    def _update_inertia(self, t: int) -> float
    def _adaptive_mutation(self, particle: Particle, t: int) -> Particle
    def _update_velocity(self, particle: Particle, t: int) -> Particle
```

#### `fitness.py`

```python
def composite_fitness(y_true: np.ndarray, y_pred: np.ndarray,
                      prices: np.ndarray,
                      rmse_weight=0.4, sharpe_weight=0.4,
                      mdd_weight=0.2) -> float:
    """
    Lower is better.
    Normalizes each component to [0,1] using running min/max across the swarm.
    """
```

---

### 4.4 LSTM Model (`src/models/`)

See `model_architecture.md` for full architectural specification.

#### `lstm_model.py`

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, num_layers: int,
                 hidden_units: int, dropout: float,
                 output_size: int = 1)
    def forward(self, x: Tensor) -> Tensor  # [batch, seq, features] → [batch, 1]

class LSTMTrainer:
    def __init__(self, model: LSTMModel, lr: float,
                 patience: int = 10, max_epochs: int = 100)
    def fit(self, X_train, y_train, X_val, y_val) -> history: Dict
    def predict(self, X: np.ndarray) -> np.ndarray
```

---

### 4.5 Evaluation Engine (`src/evaluation/`)

#### `metrics.py`

```python
def rmse(y_true, y_pred) -> float
def directional_accuracy(y_true, y_pred) -> float
def sharpe_ratio(returns: np.ndarray, freq: int = 390) -> float
    # freq=390 for 1-minute bars × 390 bars/trading day
def max_drawdown(equity_curve: np.ndarray) -> float
def cagr(equity_curve: np.ndarray, n_years: float) -> float
def f1_ternary(y_true, y_pred, threshold=0.0001) -> float
    # Classes: up / flat / down
```

#### `walk_forward.py`

```python
class WalkForwardValidator:
    """
    Expanding window: retrain on all data up to fold boundary.
    Window size: 3 months; step size: 1 month.
    Reports mean ± std of each metric across folds.
    """
    def validate(self, model_cls, params, X, y) -> Dict[str, float]
```

#### `backtester.py`

See `backtesting_framework.md` for full backtesting specification.

```python
class Backtester:
    def __init__(self, transaction_cost=0.001, slippage=0.0005,
                 position_size=1.0, stop_loss=0.02)
    def run(self, signals: np.ndarray, prices: np.ndarray
            ) -> pd.DataFrame  # equity curve + trade log
```

---

## 5. Class and Interface Design

### 5.1 Core Abstract Interfaces

```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> None: ...
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def get_params(self) -> Dict[str, Any]: ...

class BaseOptimizer(ABC):
    @abstractmethod
    def run(self, X_train, y_train, X_val, y_val
            ) -> Tuple[Dict[str, Any], float]: ...

class BaseFeatureExtractor(ABC):
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

### 5.2 Data Schemas

```python
# Raw bar schema (per ticker)
RawBar = TypedDict('RawBar', {
    'timestamp': pd.Timestamp,
    'open': float, 'high': float, 'low': float,
    'close': float, 'volume': int
})

# Feature output schema
FeatureMatrix = np.ndarray  # shape: [n_samples, n_features]

# Sliding window tensors
WindowTensor = np.ndarray   # shape: [n_samples, lookback, n_features]

# Particle encoding
ParticleVector = np.ndarray  # shape: [5] — continuous encoding
HyperParams = TypedDict('HyperParams', {
    'num_layers': int,       # 1–4
    'hidden_units': int,     # 32–512
    'dropout': float,        # 0.0–0.5
    'learning_rate': float,  # 1e-5 to 1e-1
    'lookback': int          # 10, 30, 60, or 120
})
```

### 5.3 Configuration Schema (`config/default_config.yaml`)

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
  num_layers:    {type: int,   min: 1,    max: 4}
  hidden_units:  {type: int,   min: 32,   max: 512}
  dropout:       {type: float, min: 0.0,  max: 0.5}
  learning_rate: {type: float, min: 1e-5, max: 1e-1, scale: log}
  lookback:      {type: categorical, choices: [10, 30, 60, 120]}
  max_epochs:    100
  batch_size:    256
  patience:      10

fitness:
  rmse_weight:     0.4
  sharpe_weight:   0.4
  drawdown_weight: 0.2

data:
  train_years: 3
  val_years:   1
  test_years:  1
  freq:        "1Min"
  session_start: "09:30"
  session_end:   "16:00"
```

---

## 6. Pipeline Orchestration

### 6.1 Sequential Script Execution

The pipeline is designed to be run sequentially via numbered scripts. Each script reads from and writes to clearly-defined intermediate artifacts (Parquet files, JSON configs), enabling restartability at any stage.

```
scripts/01_ingest_data.py
    ↓ data/raw/<ticker>.parquet (51 files)

scripts/02_build_features.py
    ↓ data/features/feature_matrix_<ticker>.npy
    ↓ data/features/scaler_<ticker>.pkl
    ↓ data/features/selected_features_<ticker>.json

scripts/03_run_pso.py
    ↓ results/pso_log_<ticker>.json
    ↓ results/best_params_<ticker>.json

scripts/04_evaluate.py
    ↓ results/metrics_<ticker>.json
    ↓ results/predictions_<ticker>.npy

scripts/05_backtest.py
    ↓ results/backtest_<ticker>.csv
    ↓ results/equity_curve_<ticker>.png
```

### 6.2 Parallelization Opportunities

The PSO fitness evaluation is the computational bottleneck. Two parallelization strategies are available:

**Strategy A — Particle-level parallelism (recommended for single-machine)**
```python
# In IPSO.run():
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(evaluate_particle, p, X_train, y_train, X_val, y_val)
               for p in self.swarm]
    fitnesses = [f.result() for f in futures]
```

**Strategy B — Ticker-level parallelism (multi-node cluster)**
```bash
# Run PSO independently per ticker
for ticker in $(cat config/tickers.txt); do
    python scripts/03_run_pso.py --ticker $ticker &
done
wait
```

### 6.3 Checkpointing

The IPSO optimizer saves state every 10 iterations:

```python
# Checkpoint format: results/pso_checkpoint_<ticker>_iter_<n>.pkl
{
  "iteration": 40,
  "swarm": [particle_1, ..., particle_30],
  "gbest_position": np.ndarray,
  "gbest_fitness": float,
  "fitness_history": List[float]
}
```

To resume a run:
```bash
python scripts/03_run_pso.py --ticker AAPL --resume
```

---

## 7. Technology Stack

| Component | Library / Tool | Version |
|---|---|---|
| Data download | `alpaca-trade-api` | 3.x |
| Data manipulation | `pandas`, `numpy` | 2.x, 1.26 |
| Feature computation | `pandas-ta`, `scipy` | 0.3.14b, 1.12 |
| ML model | `torch` (PyTorch) | 2.2 |
| Baseline model | `xgboost` | 2.0 |
| Optimization | Custom PSO (pure Python + NumPy) | — |
| Serialization | `parquet` (via `pyarrow`) | 15.x |
| Experiment tracking | `json` logs + `matplotlib` plots | — |
| Testing | `pytest` | 8.x |
| Environment | `venv` + `requirements.txt` | — |

---

## 8. Design Decisions and Trade-offs

### 8.1 PyTorch vs. TensorFlow for LSTM

**Decision:** PyTorch

**Rationale:** PyTorch's eager execution makes it easier to integrate LSTM training within the PSO fitness evaluation loop (no graph recompilation per particle). The dynamic computation graph also simplifies varying `num_layers` across particles.

### 8.2 Many-to-One vs. Many-to-Many LSTM

**Decision:** Many-to-One (predict one step ahead only)

**Rationale:** The project goal is next-period return prediction. Many-to-One is computationally cheaper (one output per sequence), and the error signal is concentrated on the single prediction target. Many-to-many would require auxiliary loss weighting, complicating the PSO fitness landscape.

### 8.3 Particle Encoding: Continuous vs. Discrete

**Decision:** Continuous encoding with decoding at fitness evaluation

**Rationale:** Standard PSO operates in continuous space. Discrete hyperparameters (num_layers, lookback) are decoded via rounding/indexing after velocity/position updates. This avoids the overhead of discrete PSO variants while preserving PSO's natural gradient-free search.

### 8.4 Fitness Function Normalization

**Decision:** Online normalization using running min/max across all particle evaluations

**Rationale:** RMSE, Sharpe, and MDD have very different scales and units. Normalizing to [0,1] using empirical bounds observed during the PSO run ensures equal weighting without requiring prior domain knowledge of typical metric ranges.

### 8.5 Feature Selection: XGBoost Importance vs. PSO Mask

**Decision:** Two-stage: XGBoost first (coarse), PSO mask optional (fine)

**Rationale:** Running PSO over a 100+ dimensional feature mask simultaneously with LSTM hyperparameters would create a very high-dimensional search space. Pre-filtering with XGBoost to the top 70% of features reduces dimensionality before PSO, making the search tractable within the evaluation budget (30 particles × 50 iterations = 1,500 LSTM training runs).