# Software Design and Modularity Audit

**Project:** ClaudePaper (PSO-LSTM Stock Prediction System)  
**Date:** April 7, 2026  
**Auditor:** AI Architecture Review Agent  
**Total Files Analyzed:** 50+ Python modules  
**Lines of Code:** ~10,000 LOC  

---

## Executive Summary

This audit evaluates the ClaudePaper codebase for **software engineering principles** including Separation of Concerns (SoC), cohesion, coupling, code duplication, and architectural layering. The analysis identifies **68 structural issues** affecting maintainability, testability, and extensibility.

### Severity Distribution

| Severity | Count | % of Total |
|----------|-------|------------|
| **CRITICAL** | 12 | 18% |
| **HIGH** | 28 | 41% |
| **MEDIUM** | 20 | 29% |
| **LOW** | 8 | 12% |
| **TOTAL** | **68** | **100%** |

### Key Findings

1. **Inverted Dependencies:** `pipelines/` imports from `scripts/` (wrong direction)
2. **God Objects:** 684-line scripts mixing concerns; 500-line training classes
3. **Massive Duplication:** Feature loading, windowing, price alignment repeated 5+ times
4. **Missing Abstractions:** No shared interfaces for models, no feature block protocol
5. **Business Logic in Scripts:** Training, evaluation, plotting embedded in entry points
6. **Inconsistent Patterns:** LSTM has Model+Trainer split, XGBoost doesn't, RL is different again
7. **Dead Code:** 1000+ lines of commented-out code across multiple files

---

## Table of Contents

1. [Critical Issues](#1-critical-issues)
2. [High Priority Issues](#2-high-priority-issues)
3. [Medium Priority Issues](#3-medium-priority-issues)
4. [Low Priority Issues](#4-low-priority-issues)
5. [Architectural Recommendations](#5-architectural-recommendations)
6. [Refactoring Roadmap](#6-refactoring-roadmap)

---

## 1. Critical Issues

### 1.1 Inverted Dependency: Pipelines Import Scripts

**Files:** `pipelines/run_lstm_baseline.py` line 625-640  
**Principle Violated:** Dependency Inversion, Layering  
**Severity:** CRITICAL  

**Issue:**
```python
from scripts.plots_lstm import plot_lstm_pnl
```

Pipeline code (library layer) depends on `scripts/` (application layer). This inverts the normal dependency flow where scripts should orchestrate library code, not vice versa.

**Impact:**
- Circular dependency risk
- Scripts become part of the library API
- Cannot use pipelines without scripts
- Violates clean architecture principles

**Refactoring:**
```
1. Move `scripts/plots_lstm.py` → `src/visualization/lstm_plots.py`
2. Update imports:
   - pipelines/run_lstm_baseline.py: from src.visualization.lstm_plots import plot_lstm_pnl
   - scripts/: from src.visualization.lstm_plots import plot_lstm_pnl
3. Keep scripts/ as thin CLI wrappers only
```

---

### 1.2 God Script: 684-Line run_lstm_baseline.py

**File:** `pipelines/run_lstm_baseline.py`  
**Principle Violated:** Single Responsibility Principle  
**Severity:** CRITICAL  

**Issue:**
One file contains:
- CLI argument parsing (lines 51-140)
- Training logic (lines 148-352)
- Walk-forward validation (lines 353-441)
- Test-time threshold sweeping (lines 488-534)
- Backtesting simulation (lines 535-600)
- JSON serialization (lines 604-620)
- Plotting integration (lines 625-640)

**Impact:**
- Impossible to unit test individual components
- Cannot reuse training without CLI
- Difficult to understand control flow
- High risk of merge conflicts

**Refactoring:**
```
Create modular structure mirroring XGBoost:

src/models/lstm/
├── train.py         # run_train() logic
├── validate.py      # run_validation() logic  
├── test.py          # run_test() logic
└── factory.py       # Model construction

pipelines/run_lstm_baseline.py:
- Parse args
- Dispatch to src/models/lstm/{train,validate,test}.py
- ~100 lines total
```

---

### 1.3 Massive Code Duplication: Feature Loading

**Files:** 
- `scripts/evaluate.py` lines 138-153
- `scripts/run_pso.py` lines 172-207
- `scripts/backtest.py` lines 97-112
- `scripts/train_rl_agent.py` lines 92-104
- `scripts/evaluate_rl_agent.py` lines 82-98

**Principle Violated:** DRY (Don't Repeat Yourself)  
**Severity:** CRITICAL  

**Issue:**
Same pattern repeated 5+ times:
```python
# Load features
X_train = np.load(ticker_dir / "X_train.npy")
y_train = np.load(ticker_dir / "y_train.npy")
X_val = np.load(ticker_dir / "X_val.npy")
y_val = np.load(ticker_dir / "y_val.npy")
X_test = np.load(ticker_dir / "X_test.npy")
y_test = np.load(ticker_dir / "y_test.npy")

# Build windows
session_starts = np.zeros(len(X_train), dtype=bool)
X_train_w, y_train_w = build_windows(X_train, y_train, session_starts, lookback)
# ... repeat for val and test
```

**Impact:**
- 6 files to update when changing data format
- Inconsistent error handling
- Difficult to add validation or caching

**Refactoring:**
```python
# src/data/loaders.py
@dataclass
class FeatureSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    
    @classmethod
    def load(cls, ticker: str, features_dir: Path) -> 'FeatureSplits':
        """Load all splits for a ticker."""
        ticker_dir = features_dir / ticker
        return cls(
            X_train=np.load(ticker_dir / "X_train.npy"),
            y_train=np.load(ticker_dir / "y_train.npy"),
            # ...
        )

def build_windowed_splits(
    splits: FeatureSplits, 
    lookback: int,
    session_starts: Optional[np.ndarray] = None
) -> tuple:
    """Apply windowing to all splits."""
    if session_starts is None:
        session_starts = np.zeros(len(splits.X_train), dtype=bool)
    # ...

# Usage in all scripts:
splits = FeatureSplits.load(ticker, features_dir)
X_train_w, y_train_w, X_val_w, y_val_w, X_test_w, y_test_w = build_windowed_splits(
    splits, lookback
)
```

---

### 1.4 Business Logic in Scripts: RLTrainer Class

**File:** `scripts/train_rl_agent.py` lines 39-499  
**Principle Violated:** Separation of Concerns, Layering  
**Severity:** CRITICAL  

**Issue:**
A 460-line `RLTrainer` class lives in `scripts/`, containing:
- Training loop (lines 154-222)
- Episode rollout (lines 154-200)
- History tracking (lines 72-88)
- Checkpointing (lines 244-252)
- Two large plotting methods (lines 338-493)

This is **application logic** in what should be a thin CLI wrapper.

**Impact:**
- Cannot reuse training without the script
- No unit tests possible
- Violates "scripts are thin" principle
- Duplicates concerns with other trainers

**Refactoring:**
```
Move to src/models/rl/trainer.py:

src/models/rl/
├── trading_env.py      # Existing
├── ppo_agent.py        # Existing
├── trainer.py          # NEW: RLTrainer class
└── plotting.py         # NEW: plot_training_curves, plot_final_evaluation

scripts/train_rl_agent.py:
- Parse args (~50 lines)
- Create RLTrainer(ticker, config)
- Call trainer.train()
- ~80 lines total
```

---

### 1.5 Inconsistent Model Interfaces

**Files:** `src/models/lstm_model.py`, `src/models/xgboost/xgboost_model.py`, `src/models/rl/ppo_agent.py`  
**Principle Violated:** Liskov Substitution, Interface Segregation  
**Severity:** CRITICAL  

**Issue:**
Three model types with completely different APIs:

| Model | Training Method | Prediction | Parameters | Save/Load |
|-------|----------------|------------|------------|-----------|
| LSTM | `LSTMTrainer.fit()` | `trainer.predict()` | No `get_params()` | `state_dict()` |
| XGBoost | `XGBoostModel.fit()` | `model.predict()` | `get_params()` | `save()`/`load()` |
| RL | `PPOAgent.update()` | `select_action()` | No params API | `save()`/`load()` |

**Impact:**
- Cannot write generic evaluation code
- Difficult to compare models
- Each model needs custom pipeline code
- No polymorphism possible

**Refactoring:**
```python
# src/models/base.py
from abc import ABC, abstractmethod
from typing import Protocol

class Predictor(Protocol):
    """Common interface for all models."""
    
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        """Train model. Returns training history."""
        ...
    
    def predict(self, X) -> np.ndarray:
        """Make predictions."""
        ...
    
    def get_params(self) -> dict:
        """Get model hyperparameters."""
        ...
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        ...
    
    @classmethod
    def load(cls, path: str) -> 'Predictor':
        """Load model from disk."""
        ...

# Adapt existing models to implement this protocol
class LSTMTrainerAdapter(Predictor):
    def __init__(self, trainer: LSTMTrainer):
        self.trainer = trainer
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        return self.trainer.fit(X_train, y_train, X_val, y_val)
    
    # ... implement other methods

# Now evaluation code can be generic:
def evaluate_model(model: Predictor, X_test, y_test):
    predictions = model.predict(X_test)
    return compute_metrics(y_test, predictions)
```

---

### 1.6 XGBoost Tuner Exposes Private API

**File:** `src/models/xgboost/xgboost_train.py` lines 81-84  
**Principle Violated:** Encapsulation, Law of Demeter  
**Severity:** CRITICAL  

**Issue:**
```python
# In xgboost_train.py
tuner._run_single_trial(...)  # Calling PRIVATE method
```

Training script reaches into private `XGBoostTuner` internals and reimplements trial loop logic that already exists in `XGBoostTuner.fit()`.

**Impact:**
- Breaks encapsulation
- Duplicates trial logic
- Fragile to tuner refactoring
- Violates "don't touch privates"

**Refactoring:**
```python
# src/models/xgboost/xgboost_model.py
class XGBoostTuner:
    def fit(self, ..., progress_callback=None):
        """Public API with optional callback."""
        for trial_idx in range(self.n_trials):
            result = self._run_single_trial(trial_idx, ...)
            if progress_callback:
                progress_callback(trial_idx, result)
            # ... rest of logic
    
    # OR provide iterator:
    def trials(self):
        """Yield trial results one at a time."""
        for trial_idx in range(self.n_trials):
            yield self._run_single_trial(trial_idx, ...)

# In xgboost_train.py:
for trial_idx, result in enumerate(tuner.trials()):
    logger.info(f"Trial {trial_idx}: {result}")
```

---

### 1.7 Duplicate True Range Calculation

**Files:**
- `src/features/technical.py` lines 76-80
- `src/features/pipeline.py` lines 150-164
- `src/features/ttm_squeeze.py` lines 44-50

**Principle Violated:** DRY  
**Severity:** CRITICAL  

**Issue:**
True Range formula implemented three times with slight variations:
```python
# technical.py
prev_close = C.shift(1)
tr = pd.concat([H - L, (H - prev_close).abs(), (L - prev_close).abs()], axis=1).max(axis=1)

# pipeline.py  
prev_close = df["close"].shift(1)
tr = pd.concat([...], axis=1).max(axis=1)

# ttm_squeeze.py
true_range = pd.concat([...], axis=1).max(axis=1)
```

**Impact:**
- Bug fixes need 3 updates
- Inconsistent behavior possible
- Violates single source of truth

**Refactoring:**
```python
# src/features/indicators_common.py
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Calculate True Range.
    
    TR = max(H - L, |H - C_prev|, |L - C_prev|)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        
    Returns:
        True Range series
    """
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

# Usage in all three modules:
from src.features.indicators_common import true_range
tr = true_range(H, L, C)
```

---

### 1.8 Flattening Contract Mismatch

**Files:**
- `src/models/baselines.py` lines 120-122
- `src/models/xgboost/xgboost_model.py` lines 425-442

**Principle Violated:** Liskov Substitution, Consistent Abstraction  
**Severity:** CRITICAL  

**Issue:**
Two "XGBoost" implementations with incompatible flattening:

```python
# baselines.py - XGBoostBaseline
def _flatten(self, X):
    N, T, F = X.shape
    return X.reshape(N, T * F)  # Uses ALL timesteps

# xgboost_model.py - XGBoostModel
def _flatten(self, X):
    N, T, F = X.shape
    return X[:, -self.lookback:, :].reshape(N, self.lookback * F)  # Uses LAST lookback only
```

**Impact:**
- Same input produces different feature spaces
- "Baseline" vs "production" XGBoost not comparable
- Silent bugs when swapping implementations
- Violates principle of least surprise

**Refactoring:**
```python
# src/data/sequence_utils.py
from enum import Enum

class FlattenMode(Enum):
    FULL_SEQUENCE = "full"
    TAIL_LOOKBACK = "tail"

def flatten_sequences(
    X: np.ndarray, 
    mode: FlattenMode = FlattenMode.TAIL_LOOKBACK,
    lookback: Optional[int] = None
) -> np.ndarray:
    """
    Flatten 3D sequences to 2D.
    
    Args:
        X: [N, T, F] array
        mode: FULL_SEQUENCE uses all T, TAIL_LOOKBACK uses last `lookback` steps
        lookback: Required for TAIL_LOOKBACK mode
        
    Returns:
        [N, T*F] or [N, lookback*F] array
    """
    N, T, F = X.shape
    
    if mode == FlattenMode.FULL_SEQUENCE:
        return X.reshape(N, T * F)
    elif mode == FlattenMode.TAIL_LOOKBACK:
        if lookback is None:
            raise ValueError("lookback required for TAIL_LOOKBACK mode")
        return X[:, -lookback:, :].reshape(N, lookback * F)
    else:
        raise ValueError(f"Unknown mode: {mode}")

# Usage:
# Baseline
X_flat = flatten_sequences(X, mode=FlattenMode.FULL_SEQUENCE)

# Production
X_flat = flatten_sequences(X, mode=FlattenMode.TAIL_LOOKBACK, lookback=30)
```

---

### 1.9 Invalid Python File

**File:** `src/models/xgboost/helpers.py` lines 184-188  
**Principle Violated:** Syntax, Code Integrity  
**Severity:** CRITICAL  

**Issue:**
```python
def create_experiement_setup():
    # NO BODY - file is syntactically invalid
```

Function has no implementation, making the entire module invalid Python.

**Impact:**
- Import fails if function is called
- Indicates incomplete refactoring
- Typo in name ("experiement")

**Refactoring:**
```python
# Option 1: Remove if unused
# Delete the function

# Option 2: Implement if needed
def create_experiment_setup(config, ticker):
    """
    Create experiment configuration.
    
    Args:
        config: Configuration object
        ticker: Ticker symbol
        
    Returns:
        Experiment setup dictionary
    """
    return {
        'ticker': ticker,
        'timestamp': datetime.now(),
        'config': config,
        # ...
    }
```

---

### 1.10 Duplicate Constants with Conflicting Values

**File:** `src/models/xgboost/consts.py` lines 8-9 vs 16-17  
**Principle Violated:** DRY, Single Source of Truth  
**Severity:** CRITICAL  

**Issue:**
```python
# Lines 8-9
PSEUDO_VAL_MIN_SIZE = 100
PSEUDO_VAL_FRACTION = 0.01

# Lines 16-17 (OVERWRITES ABOVE)
PSEUDO_VAL_MIN_SIZE = 100
PSEUDO_VAL_FRACTION = 0.05  # Different value!
```

Second definition silently overwrites first, causing confusion.

**Impact:**
- Unclear which value is used
- Different modules may expect different values
- Silent bugs from overwriting

**Refactoring:**
```python
# Single definition per constant
PSEUDO_VAL_MIN_SIZE = 100
PSEUDO_VAL_FRACTION = 0.05

# If different contexts need different values:
PSEUDO_VAL_FRACTION_TRAIN = 0.01
PSEUDO_VAL_FRACTION_TUNE = 0.05

# Or use config:
# config/default_config.yaml
xgboost:
  pseudo_val:
    min_size: 100
    fraction: 0.05
```

---

### 1.11 Tight Coupling: Pipeline Hard-Wired to Feature Blocks

**File:** `src/features/pipeline.py` lines 15-19, 112-141  
**Principle Violated:** Open/Closed Principle, Dependency Inversion  
**Severity:** CRITICAL  

**Issue:**
```python
from src.features.technical import compute_technical_features
from src.features.statistical import compute_statistical_features
from src.features.volume import compute_volume_features
from src.features.cross_ticker import compute_cross_ticker_features

# In FeaturePipeline._compute_features:
out = compute_technical_features(df_target)
out2 = compute_statistical_features(df_target)
out3 = compute_volume_features(df_target)
out4 = compute_cross_ticker_features(...)
```

Pipeline directly imports and sequences four specific functions. Adding a feature block requires editing the pipeline class.

**Impact:**
- Cannot add features without modifying pipeline
- Difficult to test individual blocks
- Cannot configure which blocks to use
- Violates open/closed principle

**Refactoring:**
```python
# src/features/blocks.py
from abc import ABC, abstractmethod
from typing import Protocol

class FeatureBlock(Protocol):
    """Protocol for feature computation blocks."""
    
    @property
    def name(self) -> str:
        """Block identifier."""
        ...
    
    def compute(self, df: pd.DataFrame, context: dict) -> pd.DataFrame:
        """
        Compute features.
        
        Args:
            df: Input DataFrame with OHLCV
            context: Shared context (e.g., universe data)
            
        Returns:
            DataFrame with computed features
        """
        ...

# src/features/technical_block.py
class TechnicalBlock:
    name = "technical"
    
    def compute(self, df, context):
        return compute_technical_features(df)

# src/features/pipeline.py
class FeaturePipeline:
    def __init__(
        self,
        target_ticker: str,
        universe_tickers: List[str],
        blocks: Optional[List[FeatureBlock]] = None
    ):
        self.target_ticker = target_ticker
        self.universe_tickers = universe_tickers
        
        # Default blocks if none provided
        if blocks is None:
            from src.features.technical_block import TechnicalBlock
            from src.features.statistical_block import StatisticalBlock
            # ...
            blocks = [TechnicalBlock(), StatisticalBlock(), ...]
        
        self.blocks = blocks
    
    def _compute_features(self, dfs, fit):
        df_target = dfs[self.target_ticker]
        context = {'dfs': dfs, 'fit': fit}
        
        # Run all blocks
        feature_dfs = []
        for block in self.blocks:
            logger.info(f"Computing {block.name} features")
            features = block.compute(df_target, context)
            feature_dfs.append(features)
        
        # Combine
        combined = pd.concat(feature_dfs, axis=1)
        # ... rest of pipeline
```

---

### 1.12 Missing Abstraction: No Shared Data Loader

**Files:** Multiple scripts and pipelines  
**Principle Violated:** DRY, Abstraction  
**Severity:** CRITICAL  

**Issue:**
Every script manually constructs paths and loads data:
```python
# Repeated everywhere
ticker_dir = features_dir / ticker
X_train = np.load(ticker_dir / "X_train.npy")
# ...
```

No abstraction for "load feature splits" or "load OHLCV prices".

**Impact:**
- Cannot change storage format easily
- No validation or error handling
- Difficult to add caching
- Repeated code across 10+ files

**Refactoring:**
See issue 1.3 above for detailed `FeatureSplits` solution.

---

## 2. High Priority Issues

### 2.1 sys.path Injection in Every Script

**Files:** All scripts and pipelines (10+ files)  
**Principle Violated:** Packaging, Dependency Management  
**Severity:** HIGH  

**Issue:**
```python
# Repeated in every file
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

**Refactoring:**
```bash
# Create pyproject.toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "claude-paper"
version = "0.1.0"
dependencies = [
    "numpy>=1.20",
    "pandas>=1.3",
    # ...
]

[project.scripts]
ingest-data = "src.pipelines.ingest_data:main"
build-features = "src.pipelines.build_features:main"
train-lstm = "src.pipelines.run_lstm_baseline:main"
# ...

# Install in development mode
pip install -e .

# Remove all sys.path hacks
```

---

### 2.2 Long Functions: 200+ Line main()

**Files:**
- `scripts/evaluate.py` main() ~207 lines
- `scripts/run_pso.py` main() ~270 lines
- `scripts/backtest.py` main() ~207 lines

**Principle Violated:** Single Responsibility, Readability  
**Severity:** HIGH  

**Refactoring:**
```python
# Before
def main():
    # 200+ lines of logic
    ...

# After
def main():
    args = parse_args()
    config = load_configuration(args)
    data = load_data(config)
    model = train_model(data, config)
    results = evaluate_model(model, data)
    save_results(results, config)

# Each helper is 20-50 lines, testable independently
```

---

### 2.3 Duplicate CLI Argument Parsing

**Files:**
- `pipelines/run_lstm_baseline.py` lines 51-140
- `pipelines/run_xgboost.py` lines 35-164

**Principle Violated:** DRY  
**Severity:** HIGH  

**Issue:**
Both pipelines have nearly identical argparse setup with mode-based attribute deletion.

**Refactoring:**
```python
# src/cli/common_args.py
def add_model_args(parser):
    """Add common model training arguments."""
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--mode", choices=["train", "validate", "test"])
    parser.add_argument("--config", default="config/default_config.yaml")
    # ...
    return parser

def filter_args_for_mode(args, mode):
    """Remove mode-specific arguments."""
    if mode == "train":
        for attr in ["pso_results", "model_path"]:
            delattr(args, attr)
    # ...
    return args

# Usage in both pipelines:
from src.cli.common_args import add_model_args, filter_args_for_mode

parser = argparse.ArgumentParser()
add_model_args(parser)
args = parser.parse_args()
args = filter_args_for_mode(args, args.mode)
```

---

### 2.4 Duplicate Ticker List Loading

**Files:**
- `pipelines/ingest_data.py` lines 60-63
- `pipelines/build_features.py` lines 108-111

**Principle Violated:** DRY  
**Severity:** HIGH  

**Refactoring:**
```python
# src/utils/tickers.py
def load_ticker_list(path: Path) -> List[str]:
    """
    Load ticker symbols from file.
    
    Format: One ticker per line, # for comments.
    
    Args:
        path: Path to tickers file
        
    Returns:
        List of ticker symbols
    """
    with open(path) as f:
        return [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]

# Usage:
from src.utils.tickers import load_ticker_list
tickers = load_ticker_list(Path("config/tickers.txt"))
```

---

### 2.5 Hard-Coded Trading Parameters

**Files:** Multiple scripts with duplicate values  
**Principle Violated:** Configuration Management  
**Severity:** HIGH  

**Issue:**
```python
# Repeated in multiple files
initial_capital = 100_000.0
transaction_cost = 0.001
max_position = 1.0
stop_loss = 0.02
```

**Refactoring:**
```yaml
# config/default_config.yaml
trading:
  initial_capital: 100000.0
  transaction_cost: 0.001
  max_position: 1.0
  stop_loss: 0.02
  slippage: 0.0001

# Usage:
from src.utils.config_loader import load_config
config = load_config("config/default_config.yaml")
capital = config.trading.initial_capital
```

---

### 2.6 FeaturePipeline Stores Unused Parameter

**File:** `src/features/pipeline.py` lines 55-59  
**Principle Violated:** YAGNI, Clear Intent  
**Severity:** HIGH  

**Issue:**
```python
def __init__(self, target_ticker, universe_tickers, ...):
    self.universe_tickers = universe_tickers  # Stored but never used
```

**Refactoring:**
```python
# Option 1: Remove if truly unused
def __init__(self, target_ticker, ...):
    # Don't store universe_tickers

# Option 2: Use for validation
def __init__(self, target_ticker, universe_tickers, ...):
    self.universe_tickers = universe_tickers
    
def _compute_features(self, dfs, fit):
    # Validate universe
    missing = set(self.universe_tickers) - set(dfs.keys())
    if missing:
        logger.warning(f"Missing tickers: {missing}")
```

---

### 2.7 Inconsistent Return Types: cross_ticker

**File:** `src/features/cross_ticker.py` lines 19-25, 171-175  
**Principle Violated:** Interface Segregation, Least Surprise  
**Severity:** HIGH  

**Issue:**
```python
def compute_cross_ticker_features(..., return_peer_tickers=False):
    # ...
    if return_peer_tickers:
        return result, peer_tickers  # Tuple
    else:
        return result  # DataFrame
```

Callers must know the flag value to unpack correctly.

**Refactoring:**
```python
# Option 1: Two functions
def compute_cross_ticker_features(...) -> pd.DataFrame:
    """Compute cross-ticker features."""
    return result

def compute_cross_ticker_features_with_peers(...) -> tuple[pd.DataFrame, List[str]]:
    """Compute cross-ticker features and return selected peers."""
    return result, peer_tickers

# Option 2: Always return NamedTuple
from typing import NamedTuple

class CrossTickerResult(NamedTuple):
    features: pd.DataFrame
    peer_tickers: Optional[List[str]] = None

def compute_cross_ticker_features(..., return_peer_tickers=False) -> CrossTickerResult:
    # ...
    if return_peer_tickers:
        return CrossTickerResult(result, peer_tickers)
    else:
        return CrossTickerResult(result, None)

# Callers always unpack the same way:
result = compute_cross_ticker_features(...)
features = result.features
peers = result.peer_tickers  # None if not requested
```

---

### 2.8 Unstable Column Names: peer_corr

**File:** `src/features/cross_ticker.py` lines 158-167  
**Principle Violated:** Stable Contracts, Predictability  
**Severity:** HIGH  

**Issue:**
```python
# If peer exists in dfs:
out[f"peer_corr_{peer}"] = ...  # Column name is ticker symbol

# If peer missing:
out[f"peer_corr_{rank}"] = 0.0  # Column name is rank number
```

Column schema changes based on data availability.

**Refactoring:**
```python
# Always use fixed schema
for rank in range(1, 4):  # peer_corr_1, peer_corr_2, peer_corr_3
    if rank <= len(peer_tickers):
        peer = peer_tickers[rank-1]
        if peer in dfs:
            r_peer = dfs[peer]["log_return"].reindex(df_target.index).fillna(0.0)
            out[f"peer_corr_{rank}"] = r_target.rolling(60, min_periods=10).corr(r_peer)
        else:
            out[f"peer_corr_{rank}"] = 0.0
    else:
        out[f"peer_corr_{rank}"] = 0.0

# Optional: Store metadata separately
metadata = {
    'peer_1': peer_tickers[0] if len(peer_tickers) > 0 else None,
    'peer_2': peer_tickers[1] if len(peer_tickers) > 1 else None,
    'peer_3': peer_tickers[2] if len(peer_tickers) > 2 else None,
}
```

---

### 2.9 Duplicate Logger Declarations

**File:** `src/models/lstm_model.py` lines 20-21, 123-124  
**Principle Violated:** DRY  
**Severity:** HIGH  

**Refactoring:**
```python
# Keep only one at module level
logger = logging.getLogger(__name__)

# Remove duplicate
```

---

### 2.10 Duplicate Imports

**File:** `pipelines/run_lstm_baseline.py` lines 36-39  
**Principle Violated:** DRY, Clean Code  
**Severity:** HIGH  

**Issue:**
```python
from src.evaluation.metrics import all_statistical_metrics
from src.evaluation import all_statistical_metrics  # Duplicate
```

**Refactoring:**
```python
# Keep one import
from src.evaluation.metrics import all_statistical_metrics
```

---

### 2.11 Leaky Abstraction: Reaching Through Private Trainer

**File:** `pipelines/run_lstm_baseline.py` lines 291-292, 358-359  
**Principle Violated:** Encapsulation, Law of Demeter  
**Severity:** HIGH  

**Issue:**
```python
model._trainer.model.state_dict()  # Reaching through private attribute
```

**Refactoring:**
```python
# src/models/baselines.py
class VanillaLSTM:
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(self._trainer.model.state_dict(), path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        self._trainer.model.load_state_dict(
            torch.load(path, map_location="cpu")
        )

# Usage:
model.save_checkpoint(model_path)
model.load_checkpoint(model_path)
```

---

### 2.12 Business Logic in Pipeline: Threshold Sweeping

**File:** `pipelines/run_lstm_baseline.py` lines 488-534  
**Principle Violated:** Separation of Concerns  
**Severity:** HIGH  

**Issue:**
Threshold grid search, signal generation, transaction costs, and equity calculation embedded in pipeline script.

**Refactoring:**
```python
# src/evaluation/threshold_optimization.py
def optimize_signal_threshold(
    predictions: np.ndarray,
    returns: np.ndarray,
    closes: np.ndarray,
    threshold_grid: np.ndarray,
    initial_capital: float = 100_000,
    transaction_cost: float = 0.001,
    stop_loss: Optional[float] = None
) -> dict:
    """
    Find optimal signal threshold by grid search.
    
    Returns:
        dict with best_threshold, best_sharpe, equity_curve, etc.
    """
    # Implementation moved from pipeline
    ...

# Usage in pipeline:
from src.evaluation.threshold_optimization import optimize_signal_threshold

threshold_grid = np.arange(0.0001, 0.0021, 0.0001)
result = optimize_signal_threshold(
    predictions, y_test, closes, threshold_grid
)
best_threshold = result['best_threshold']
```

---

### 2.13 Plotting in Training Loop

**File:** `pipelines/run_lstm_baseline.py` lines 321-347  
**Principle Violated:** Separation of Concerns  
**Severity:** HIGH  

**Issue:**
Matplotlib plotting code embedded in training function.

**Refactoring:**
```python
# src/visualization/training_curves.py
def plot_training_history(
    history: dict,
    save_path: Optional[Path] = None
) -> None:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    # ... plotting logic
    if save_path:
        fig.savefig(save_path)
    plt.close()

# Usage in pipeline:
from src.visualization.training_curves import plot_training_history

history = trainer.fit(X_train, y_train, X_val, y_val)
plot_training_history(history, plots_dir / "training_curves.png")
```

---

### 2.14 XGBoost I/O Helpers in Model Module

**File:** `src/models/xgboost/xgboost_model.py` lines 72-126  
**Principle Violated:** Single Responsibility, Layering  
**Severity:** HIGH  

**Issue:**
```python
def load_windows(features_dir, ticker, lookback):
    """Load numpy arrays from disk."""
    # File I/O logic in model module

def load_feature_names(features_dir, ticker):
    """Load pickle from disk."""
    # More I/O logic
```

**Refactoring:**
```python
# src/data/feature_io.py
def load_feature_windows(
    ticker: str,
    features_dir: Path,
    lookback: int
) -> tuple:
    """
    Load windowed features for a ticker.
    
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    ticker_dir = features_dir / ticker
    # ... loading logic

def load_feature_names(ticker: str, features_dir: Path) -> List[str]:
    """Load feature names from metadata."""
    # ... loading logic

# Usage in model:
from src.data.feature_io import load_feature_windows
X_train, y_train, ... = load_feature_windows(ticker, features_dir, lookback)
```

---

### 2.15 GPU Memory Setup in Model Class

**File:** `src/models/xgboost/xgboost_model.py` lines 160-181  
**Principle Violated:** Single Responsibility  
**Severity:** HIGH  

**Issue:**
`XGBoostModel` contains GPU memory probing and CuPy configuration.

**Refactoring:**
```python
# src/utils/gpu_utils.py
def configure_xgboost_device(tree_method: str = "hist") -> tuple[str, str]:
    """
    Configure XGBoost device and tree method.
    
    Returns:
        (tree_method, device) tuple
    """
    if tree_method == "gpu_hist":
        try:
            import cupy as cp
            # ... GPU setup
            return "gpu_hist", "cuda"
        except Exception as e:
            logger.warning(f"GPU unavailable: {e}, falling back to CPU")
            return "hist", "cpu"
    else:
        return tree_method, "cpu"

# Usage in model:
from src.utils.gpu_utils import configure_xgboost_device
tree_method, device = configure_xgboost_device(self.tree_method)
```

---

### 2.16 Duplicate Validation Logic

**File:** `src/models/xgboost/xgboost_model.py` lines 244-256, 519-529  
**Principle Violated:** DRY  
**Severity:** HIGH  

**Issue:**
Same X/y shape validation in `XGBoostModel.fit()` and `XGBoostTuner.fit()`.

**Refactoring:**
```python
def _validate_supervised_3d(X, y, name="data"):
    """Validate 3D supervised learning inputs."""
    if X.ndim != 3:
        raise ValueError(f"{name} X must be 3D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"{name} y must be 1D, got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"{name} length mismatch: X={len(X)}, y={len(y)}")

# Usage in both classes:
_validate_supervised_3d(X_train, y_train, "train")
_validate_supervised_3d(X_val, y_val, "val")
```

---

### 2.17 Duplicate RMSE Extraction

**File:** `src/models/xgboost/xgboost_model.py` lines 547-552, 608-614  
**Principle Violated:** DRY  
**Severity:** HIGH  

**Refactoring:**
```python
def _extract_val_rmse(model, y_val, X_val):
    """Extract validation RMSE from model or compute it."""
    if hasattr(model, 'evals_result_'):
        history = model.evals_result_['validation_0']
        if 'rmse' in history:
            return min(history['rmse'])
    
    # Fallback: compute
    y_pred = model.predict(X_val)
    return rmse_fn(y_val, y_pred)
```

---

### 2.18 LSTMTrainer Monolithic

**File:** `src/models/lstm_model.py` lines 126-299  
**Principle Violated:** Single Responsibility  
**Severity:** HIGH  

**Issue:**
`LSTMTrainer` combines DataLoader creation, AMP, gradient accumulation, clipping, early stopping, and prediction.

**Refactoring:**
```python
# Option 1: Extract components
class EarlyStoppingTracker:
    def __init__(self, patience):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
    
    def should_stop(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class TorchTrainingLoop:
    def train_epoch(self, model, dataloader, optimizer, ...):
        # Extract epoch logic
        ...

# Option 2: Keep monolithic but extract methods
class LSTMTrainer:
    def fit(self, ...):
        train_loader, val_loader = self._create_dataloaders(...)
        early_stopping = self._create_early_stopping()
        
        for epoch in range(self.max_epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)
            
            if early_stopping.should_stop(val_loss):
                break
```

---

### 2.19 PPOAgent Mixes Algorithm and I/O

**File:** `src/models/rl/ppo_agent.py` lines 159-404  
**Principle Violated:** Single Responsibility  
**Severity:** HIGH  

**Issue:**
`PPOAgent` contains policy network, GAE computation, PPO update logic, AND checkpoint save/load.

**Refactoring:**
```python
# Option 1: Split into learner and agent
class PPOLearner:
    """Handles PPO update logic only."""
    def __init__(self, policy, optimizer, ...):
        self.policy = policy
        self.optimizer = optimizer
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        # PPO update logic only
        ...

class PPOAgent:
    """Agent for action selection and I/O."""
    def __init__(self, state_dim, action_dim, ...):
        self.policy = ActorCritic(...)
        self.learner = PPOLearner(self.policy, ...)
    
    def select_action(self, state):
        # Action selection
        ...
    
    def update(self, *args):
        return self.learner.update(*args)
    
    def save(self, path):
        # I/O only
        ...
```

---

### 2.20 TradingEnv Contains Portfolio Logic

**File:** `src/models/rl/trading_env.py` lines 168-258  
**Principle Violated:** Separation of Concerns  
**Severity:** HIGH  

**Issue:**
`TradingEnv.step()` implements position changes, transaction costs, equity tracking, drawdown penalties, and reward calculation.

**Refactoring:**
```python
# src/models/rl/portfolio_simulator.py
class PortfolioSimulator:
    """Simulates portfolio dynamics."""
    def __init__(self, initial_capital, transaction_cost, ...):
        self.capital = initial_capital
        self.tc = transaction_cost
        # ...
    
    def execute_trade(self, position_change, current_return):
        """Execute trade and update equity."""
        tc_cost = abs(position_change) * self.tc * self.equity
        position_pnl = self.position * current_return * self.equity
        self.equity += position_pnl - tc_cost
        return position_pnl, tc_cost

class RewardModel:
    """Computes RL rewards."""
    def compute_reward(self, pnl, tc_cost, drawdown, position):
        return (
            pnl 
            - tc_cost 
            - self.drawdown_penalty * (drawdown ** 2) 
            - self.hold_penalty * abs(position)
        )

# TradingEnv uses these:
class TradingEnv(gym.Env):
    def __init__(self, ...):
        self.simulator = PortfolioSimulator(...)
        self.reward_model = RewardModel(...)
    
    def step(self, action):
        pnl, tc_cost = self.simulator.execute_trade(...)
        reward = self.reward_model.compute_reward(...)
        # ...
```

---

### 2.21 Duplicate SMA/EMA Implementations

**Files:**
- `src/features/technical.py` lines 14-19
- `src/features/ttm_squeeze.py` lines 36-41

**Principle Violated:** DRY  
**Severity:** HIGH  

**Refactoring:**
```python
# src/features/indicators_common.py
def sma(series: pd.Series, period: int, min_periods: int = 1) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(period, min_periods=min_periods).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()

# Usage in both modules:
from src.features.indicators_common import sma, ema
```

---

### 2.22 Duplicate Typical Price

**Files:**
- `src/features/volume.py` line 33
- `src/features/technical.py` lines 84-85

**Principle Violated:** DRY  
**Severity:** HIGH  

**Refactoring:**
```python
# src/features/indicators_common.py
def typical_price(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Typical Price = (H + L + C) / 3"""
    return (high + low + close) / 3

# Usage:
from src.features.indicators_common import typical_price
tp = typical_price(H, L, C)
```

---

### 2.23 Hard-Coded Magic Numbers in Features

**Files:** Multiple feature modules  
**Principle Violated:** Configurability, Magic Numbers  
**Severity:** HIGH  

**Issue:**
Dozens of hard-coded periods: 5, 14, 20, 26, 60, etc.

**Refactoring:**
```python
# src/features/config.py
from dataclasses import dataclass

@dataclass
class TechnicalPeriods:
    """Configurable periods for technical indicators."""
    rsi: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    # ...

@dataclass
class CrossTickerWindows:
    """Configurable windows for cross-ticker features."""
    beta_window: int = 60
    corr_short: int = 20
    corr_long: int = 60
    spy_atr: int = 14
    vix_window: int = 30
    bars_per_year: int = 252 * 390

# Usage:
def compute_technical_features(
    df: pd.DataFrame,
    periods: TechnicalPeriods = TechnicalPeriods()
) -> pd.DataFrame:
    out["rsi"] = _rsi(C, periods.rsi)
    # ...
```

---

### 2.24 Duplicate Artifact Loading

**File:** `src/models/xgboost/helpers.py` lines 24-45, 55-86  
**Principle Violated:** DRY  
**Severity:** HIGH  

**Issue:**
`load_artefacts()` and `load_model()` have nearly identical logic.

**Refactoring:**
```python
def _load_xgb_from_disk(
    results_dir: Path,
    ticker: str,
    tag: str,
    return_tag: bool = False
) -> tuple:
    """Internal loader."""
    # Unified loading logic
    ...
    if return_tag:
        return model, meta, tag
    else:
        return model, meta

def load_artefacts(results_dir, ticker, tag):
    return _load_xgb_from_disk(results_dir, ticker, tag, return_tag=True)

def load_model(results_dir, ticker, tag):
    return _load_xgb_from_disk(results_dir, ticker, tag, return_tag=False)
```

---

### 2.25 Import Clutter in xgboost_train.py

**File:** `src/models/xgboost/xgboost_train.py` lines 23-31  
**Principle Violated:** Clean Code, Explicit Dependencies  
**Severity:** HIGH  

**Issue:**
```python
from src.models.xgboost.consts import *  # Wildcard import
from src.utils.config_loader import Config, load_config  # Duplicate imports
```

**Refactoring:**
```python
# Explicit imports
from src.models.xgboost.consts import (
    FINAL_ESTIMATOR_BUFFER,
    PSEUDO_VAL_FRACTION,
    # ... only what's needed
)
from src.utils.config_loader import load_config  # Remove duplicate
```

---

### 2.26 Base Features in Pipeline Class

**File:** `src/features/pipeline.py` lines 141-196  
**Principle Violated:** Single Responsibility, Testability  
**Severity:** HIGH  

**Issue:**
Base feature computation (returns, ratios, session cumulative return) embedded in pipeline instead of separate module.

**Refactoring:**
```python
# src/features/base_features.py
def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute base price-derived features.
    
    Features:
    - log_return
    - close_to_open_ratio
    - high_low_ratio
    - cum_return_session
    - intrabar_volatility
    """
    out = pd.DataFrame(index=df.index)
    
    # Log return
    out["log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    # Ratios
    out["close_to_open_ratio"] = df["close"] / (df["open"] + 1e-10)
    # ...
    
    return out

# Usage in pipeline:
from src.features.base_features import compute_base_features
base_features = compute_base_features(df_target)
```

---

### 2.27 volume.py Mutates Input

**File:** `src/features/volume.py` lines 93-99  
**Principle Violated:** Referential Transparency, Immutability  
**Severity:** HIGH  

**Issue:**
```python
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")  # MUTATES CALLER'S DATA
```

**Refactoring:**
```python
def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume features without mutating input."""
    # Work on index copy
    index = df.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    
    et_index = index.tz_convert("America/New_York")
    
    # Build features using et_index
    out = pd.DataFrame(index=df.index)  # Original index
    out["dow_sin"] = np.sin(2 * np.pi * et_index.dayofweek / 5)
    # ...
    
    return out
```

---

### 2.28 ttm_squeeze Mixes Concerns

**File:** `src/features/ttm_squeeze.py` entire file  
**Principle Violated:** Separation of Concerns, Module Cohesion  
**Severity:** HIGH  

**Issue:**
One file contains:
- Indicator math (lines 1-180)
- Matplotlib plotting (lines 182-379)
- yfinance normalization (lines 380-450)
- `if __name__` demo (lines 452-498)

**Refactoring:**
```
Split into:

src/features/indicators/ttm_squeeze_core.py
- Pure indicator math
- No plotting dependencies

src/visualization/ttm_squeeze_plot.py
- Plotting functions
- Imports from core

examples/ttm_squeeze_demo.py
- Demo script
- Imports from both
```

---

## 3. Medium Priority Issues

### 3.1 Unused LAG_SOURCES Configuration

**File:** `src/features/pipeline.py` lines 23-35  
**Principle Violated:** YAGNI, Dead Code  
**Severity:** MEDIUM  

**Issue:**
`LAG_SOURCES` is defined but never used; only `LAG_DEPTHS` is referenced.

**Refactoring:**
```python
# Option 1: Remove if unused
# Delete LAG_SOURCES

# Option 2: Use for validation
LAG_SOURCES = ["log_return", "close_to_open_ratio", ...]
LAG_DEPTHS = {1, 2, 3, 5}

def _build_lag_features_np(self, X, feature_names):
    # Validate that lagged features are in LAG_SOURCES
    for name in feature_names:
        if any(name.endswith(f"_lag{d}") for d in LAG_DEPTHS):
            base_name = name.rsplit('_lag', 1)[0]
            if base_name not in LAG_SOURCES:
                logger.warning(f"Unexpected lag feature: {name}")
```

---

### 3.2 Repeated Epsilon Values

**Files:** Multiple feature modules  
**Principle Violated:** DRY, Magic Numbers  
**Severity:** MEDIUM  

**Issue:**
`1e-10` repeated dozens of times for division safety.

**Refactoring:**
```python
# src/features/constants.py
EPSILON = 1e-10

# Usage everywhere:
from src.features.constants import EPSILON
ratio = numerator / (denominator + EPSILON)
```

---

### 3.3 Hard-Coded Market Calendar

**File:** `src/features/volume.py` lines 13-14, 89-104  
**Principle Violated:** Configurability  
**Severity:** MEDIUM  

**Issue:**
```python
TRADING_MINUTES = 390  # US market only
TRADING_DAYS = 5  # Mon-Fri only
# Hard-coded America/New_York
```

**Refactoring:**
```python
# src/features/config.py
@dataclass
class MarketCalendar:
    trading_minutes: int = 390
    trading_days: int = 5
    timezone: str = "America/New_York"
    session_start: time = time(9, 30)
    session_end: time = time(16, 0)

# Usage:
def compute_volume_features(
    df: pd.DataFrame,
    calendar: MarketCalendar = MarketCalendar()
) -> pd.DataFrame:
    tod = (df.index.hour * 60 + df.index.minute) / calendar.trading_minutes
    # ...
```

---

### 3.4 Hard-Coded SPY Benchmark

**File:** `src/features/cross_ticker.py` throughout  
**Principle Violated:** Configurability  
**Severity:** MEDIUM  

**Issue:**
`"SPY"` hard-coded as benchmark ticker.

**Refactoring:**
```python
def compute_cross_ticker_features(
    target_ticker: str,
    dfs: Dict[str, pd.DataFrame],
    benchmark_ticker: str = "SPY",
    rolling_window: int = 60,
    peer_tickers: Optional[List[str]] = None,
    ...
) -> pd.DataFrame:
    # Use benchmark_ticker instead of hard-coded "SPY"
    if benchmark_ticker in dfs and target_ticker != benchmark_ticker:
        # ...
```

---

### 3.5 XGBoost Selector Hard-Coded Params

**File:** `src/features/selector.py` lines 187-197  
**Principle Violated:** Configurability  
**Severity:** MEDIUM  

**Issue:**
XGBoost hyperparameters hard-coded in selector.

**Refactoring:**
```python
class FeatureSelector:
    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        importance_cumulative: float = 0.95,
        xgb_params: Optional[dict] = None
    ):
        self.xgb_params = xgb_params or {
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
```

---

### 3.6 Correlation Dedup Drops Wrong Feature

**File:** `src/features/selector.py` lines 110-127  
**Principle Violated:** Correctness, Documented Behavior  
**Severity:** MEDIUM  

**Issue:**
Comment says "assumed higher importance" but code drops by index order, not importance.

**Refactoring:**
```python
# After Stage 3, reorder by importance before dedup
if self.importance_cumulative is not None:
    # Reorder columns by importance descending
    importance_order = sorted(
        enumerate(feature_importances),
        key=lambda x: x[1],
        reverse=True
    )
    X_selected = X_selected[:, [i for i, _ in importance_order]]
    selected_features = [selected_features[i] for i, _ in importance_order]

# Now correlation dedup drops less important features
corr_matrix = np.corrcoef(X_selected.T)
# ...
```

---

### 3.7 Imputation Strategy Not Explicit

**File:** `src/features/pipeline.py` lines 224-225  
**Principle Violated:** Explicitness, Configurability  
**Severity:** MEDIUM  

**Issue:**
```python
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
```

Strong modeling assumption (impute to zero) is buried in pipeline.

**Refactoring:**
```python
@dataclass
class ImputationStrategy:
    nan_value: float = 0.0
    posinf_value: float = 0.0
    neginf_value: float = 0.0

class FeaturePipeline:
    def __init__(self, ..., imputation: ImputationStrategy = ImputationStrategy()):
        self.imputation = imputation
    
    def _compute_features(self, ...):
        # ...
        X = np.nan_to_num(
            X,
            nan=self.imputation.nan_value,
            posinf=self.imputation.posinf_value,
            neginf=self.imputation.neginf_value
        )
```

---

### 3.8 Peer Selection Default is Unsafe

**File:** `src/features/cross_ticker.py` lines 135-144  
**Principle Violated:** Fail-Safe Defaults  
**Severity:** MEDIUM  

**Issue:**
When `peer_tickers=None`, computes on full series (look-ahead bias) with only a warning.

**Refactoring:**
```python
def compute_cross_ticker_features(
    target_ticker: str,
    dfs: Dict[str, pd.DataFrame],
    peer_tickers: Optional[List[str]] = None,
    allow_auto_peers: bool = False,  # NEW: explicit opt-in
    ...
):
    if peer_tickers is None:
        if not allow_auto_peers:
            raise ValueError(
                "peer_tickers is None and allow_auto_peers=False. "
                "Auto peer selection creates look-ahead bias. "
                "Either pass pre-selected peers or set allow_auto_peers=True."
            )
        # Auto-select with warning
        logger.warning("Auto-selecting peers on full data (look-ahead bias)")
        # ...
```

---

### 3.9 Missing Type Hints in Key Functions

**Files:** Multiple modules  
**Principle Violated:** Type Safety, Documentation  
**Severity:** MEDIUM  

**Refactoring:**
```python
# Add type hints throughout
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

def compute_technical_features(
    df: pd.DataFrame,
    periods: Optional[TechnicalPeriods] = None
) -> pd.DataFrame:
    """Compute technical indicators."""
    ...
```

---

### 3.10 No Validation in Data Loaders

**Files:** Feature loading code in multiple scripts  
**Principle Violated:** Robustness, Fail-Fast  
**Severity:** MEDIUM  

**Refactoring:**
```python
# src/data/loaders.py
def load_feature_splits(ticker: str, features_dir: Path) -> FeatureSplits:
    """Load and validate feature splits."""
    ticker_dir = features_dir / ticker
    
    # Validate directory exists
    if not ticker_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {ticker_dir}")
    
    # Load arrays
    try:
        X_train = np.load(ticker_dir / "X_train.npy")
        y_train = np.load(ticker_dir / "y_train.npy")
        # ...
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing feature file for {ticker}: {e}")
    
    # Validate shapes
    if len(X_train) != len(y_train):
        raise ValueError(f"Train length mismatch: X={len(X_train)}, y={len(y_train)}")
    
    # Validate no NaN/Inf
    if not np.isfinite(X_train).all():
        raise ValueError(f"X_train contains NaN/Inf for {ticker}")
    
    return FeatureSplits(X_train, y_train, ...)
```

---

### 3.11 No Caching for Expensive Operations

**Files:** Feature computation, data loading  
**Principle Violated:** Performance, DRY  
**Severity:** MEDIUM  

**Refactoring:**
```python
# src/utils/cache.py
from functools import lru_cache
import hashlib
import pickle

def cache_to_disk(cache_dir: Path):
    """Decorator for disk-based caching."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key from args
            key = hashlib.md5(
                pickle.dumps((func.__name__, args, kwargs))
            ).hexdigest()
            cache_file = cache_dir / f"{key}.pkl"
            
            if cache_file.exists():
                return pickle.load(open(cache_file, 'rb'))
            
            result = func(*args, **kwargs)
            pickle.dump(result, open(cache_file, 'wb'))
            return result
        return wrapper
    return decorator

# Usage:
@cache_to_disk(Path("cache/features"))
def compute_expensive_features(df):
    # ...
```

---

### 3.12 No Progress Reporting

**Files:** Long-running operations in pipelines  
**Principle Violated:** User Experience  
**Severity:** MEDIUM  

**Refactoring:**
```python
# Use tqdm for progress bars
from tqdm import tqdm

# In build_features.py:
for ticker in tqdm(tickers, desc="Building features"):
    process_ticker(ticker, ...)

# In parallel jobs:
from joblib import Parallel, delayed
results = Parallel(n_jobs=n_jobs)(
    delayed(process_ticker)(ticker, ...)
    for ticker in tqdm(tickers, desc="Building features")
)
```

---

### 3.13 Inconsistent Logging Levels

**Files:** All modules  
**Principle Violated:** Observability  
**Severity:** MEDIUM  

**Issue:**
Mix of `print()`, `logger.info()`, `logger.debug()`, `logger.warning()` without clear convention.

**Refactoring:**
```
Establish logging convention:

- DEBUG: Detailed diagnostic info (variable values, intermediate steps)
- INFO: High-level progress (starting/finishing operations)
- WARNING: Unexpected but recoverable situations
- ERROR: Errors that prevent operation completion

Remove all print() statements, use logger instead.
```

---

### 3.14 No Configuration Validation

**Files:** Config loading throughout  
**Principle Violated:** Fail-Fast, Robustness  
**Severity:** MEDIUM  

**Refactoring:**
```python
# src/utils/config_validator.py
from pydantic import BaseModel, validator

class LSTMConfig(BaseModel):
    hidden_size: int
    num_layers: int
    dropout: float
    
    @validator('dropout')
    def dropout_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('dropout must be in [0, 1]')
        return v

class Config(BaseModel):
    lstm: LSTMConfig
    xgboost: XGBoostConfig
    # ...

# Usage:
config = Config.parse_file("config/default_config.yaml")
# Automatically validated
```

---

### 3.15 Tight Coupling to File Paths

**Files:** Multiple modules with hard-coded paths  
**Principle Violated:** Dependency Inversion  
**Severity:** MEDIUM  

**Refactoring:**
```python
# src/data/storage.py
from abc import ABC, abstractmethod

class FeatureStorage(ABC):
    """Abstract storage interface."""
    
    @abstractmethod
    def load_splits(self, ticker: str) -> FeatureSplits:
        pass
    
    @abstractmethod
    def save_splits(self, ticker: str, splits: FeatureSplits):
        pass

class FileSystemStorage(FeatureStorage):
    """File-based storage."""
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
    
    def load_splits(self, ticker: str):
        ticker_dir = self.base_dir / ticker
        # ... load from disk

# Now code depends on interface, not implementation:
def train_model(ticker: str, storage: FeatureStorage):
    splits = storage.load_splits(ticker)
    # ...

# Can easily swap to database, S3, etc.
```

---

### 3.16 No Metrics Aggregation Utility

**Files:** Multiple evaluation scripts  
**Principle Violated:** DRY  
**Severity:** MEDIUM  

**Refactoring:**
```python
# src/evaluation/aggregation.py
def aggregate_metrics(
    metrics_list: List[dict],
    aggregations: dict = None
) -> dict:
    """
    Aggregate metrics across multiple runs.
    
    Args:
        metrics_list: List of metric dictionaries
        aggregations: Dict mapping metric name to aggregation function
                     (default: mean for all)
    
    Returns:
        Aggregated metrics with mean, std, min, max
    """
    if aggregations is None:
        aggregations = {k: np.mean for k in metrics_list[0].keys()}
    
    result = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        result[f"{key}_mean"] = np.mean(values)
        result[f"{key}_std"] = np.std(values)
        result[f"{key}_min"] = np.min(values)
        result[f"{key}_max"] = np.max(values)
    
    return result
```

---

### 3.17 No Model Registry

**Files:** Model loading scattered across scripts  
**Principle Violated:** Centralization, Discoverability  
**Severity:** MEDIUM  

**Refactoring:**
```python
# src/models/registry.py
from typing import Dict, Type
from src.models.base import Predictor

class ModelRegistry:
    """Central registry for all model types."""
    
    _models: Dict[str, Type[Predictor]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a model."""
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[Predictor]:
        """Get model class by name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}")
        return cls._models[name]
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models."""
        return list(cls._models.keys())

# Usage:
@ModelRegistry.register("lstm")
class LSTMModel(Predictor):
    ...

@ModelRegistry.register("xgboost")
class XGBoostModel(Predictor):
    ...

# In scripts:
model_class = ModelRegistry.get(args.model_type)
model = model_class(**params)
```

---

### 3.18 No Experiment Tracking Integration

**Files:** All training scripts  
**Principle Violated:** Observability, Reproducibility  
**Severity:** MEDIUM  

**Refactoring:**
```python
# src/utils/experiment.py
import mlflow

class ExperimentTracker:
    """Wrapper for experiment tracking."""
    
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)
    
    def log_params(self, params: dict):
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, path: str):
        mlflow.log_artifact(path)

# Usage in training:
tracker = ExperimentTracker("lstm-baseline")
tracker.log_params(config.to_dict())
tracker.log_metrics({"train_loss": loss}, step=epoch)
tracker.log_artifact("model.pt")
```

---

### 3.19 No Data Versioning

**Files:** Data loading throughout  
**Principle Violated:** Reproducibility  
**Severity:** MEDIUM  

**Refactoring:**
```python
# Add metadata to feature artifacts
# src/data/feature_io.py
def save_feature_splits(
    ticker: str,
    splits: FeatureSplits,
    features_dir: Path,
    metadata: dict = None
):
    """Save features with metadata."""
    ticker_dir = features_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.save(ticker_dir / "X_train.npy", splits.X_train)
    # ...
    
    # Save metadata
    meta = {
        'ticker': ticker,
        'created_at': datetime.now().isoformat(),
        'data_version': '1.0',
        'feature_version': '1.0',
        'config_hash': hash_config(config),
        **(metadata or {})
    }
    with open(ticker_dir / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
```

---

### 3.20 No Health Checks

**Files:** Pipeline scripts  
**Principle Violated:** Robustness  
**Severity:** MEDIUM  

**Refactoring:**
```python
# src/utils/health.py
def check_environment():
    """Verify environment is properly configured."""
    checks = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        checks.append("Python 3.8+ required")
    
    # Check required packages
    try:
        import torch
        import pandas
        import numpy
    except ImportError as e:
        checks.append(f"Missing package: {e}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU available, using CPU")
    
    # Check data directories
    for dir_path in [Path("data/raw"), Path("data/processed")]:
        if not dir_path.exists():
            checks.append(f"Missing directory: {dir_path}")
    
    if checks:
        raise EnvironmentError(f"Environment checks failed:\n" + "\n".join(checks))
    
    logger.info("Environment checks passed")

# Usage at start of pipelines:
check_environment()
```

---

## 4. Low Priority Issues

### 4.1 Inconsistent Naming Conventions

**Files:** Throughout codebase  
**Principle Violated:** Consistency  
**Severity:** LOW  

**Issue:**
Mix of `snake_case`, `camelCase`, and abbreviations.

**Refactoring:**
```
Establish naming convention:
- Functions/variables: snake_case
- Classes: PascalCase
- Constants: UPPER_SNAKE_CASE
- Private: _leading_underscore

Examples:
- compute_features() ✓
- computeFeatures() ✗
- ComputeFeatures() ✗ (unless class)
```

---

### 4.2 Inconsistent Docstring Styles

**Files:** Throughout codebase  
**Principle Violated:** Documentation Consistency  
**Severity:** LOW  

**Refactoring:**
```python
# Adopt consistent style (Google, NumPy, or Sphinx)
# Example: Google style

def compute_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute technical features from OHLCV data.
    
    Args:
        df: DataFrame with OHLCV columns
        config: Configuration dictionary with periods
        
    Returns:
        DataFrame with computed features
        
    Raises:
        ValueError: If required columns missing
        
    Example:
        >>> df = pd.DataFrame(...)
        >>> features = compute_features(df, config)
    """
    ...
```

---

### 4.3 Missing __all__ Exports

**Files:** Most `__init__.py` files  
**Principle Violated:** Explicit Public API  
**Severity:** LOW  

**Refactoring:**
```python
# src/features/__init__.py
__all__ = [
    'FeaturePipeline',
    'FeatureSelector',
    'compute_technical_features',
    'compute_statistical_features',
    'compute_volume_features',
    'compute_cross_ticker_features',
]

from .pipeline import FeaturePipeline
from .selector import FeatureSelector
# ...
```

---

### 4.4 No Module-Level Documentation

**Files:** Many modules  
**Principle Violated:** Documentation  
**Severity:** LOW  

**Refactoring:**
```python
"""
src/features/technical.py

Technical indicators for OHLCV data.

This module computes common technical analysis indicators including:
- Trend: SMA, EMA, MACD
- Momentum: RSI, Stochastic, ROC
- Volatility: Bollinger Bands, ATR
- Volume: OBV, MFI

All functions accept pandas Series/DataFrame and return the same index.
"""

import pandas as pd
import numpy as np
# ...
```

---

### 4.5 Inconsistent Error Messages

**Files:** Throughout codebase  
**Principle Violated:** User Experience  
**Severity:** LOW  

**Refactoring:**
```python
# Establish error message format:
# "[Component] Error description. Suggestion."

# Bad:
raise ValueError("Invalid input")

# Good:
raise ValueError(
    "[FeaturePipeline] X_train must be 3D array, got shape {X_train.shape}. "
    "Ensure data is windowed before passing to pipeline."
)
```

---

### 4.6 No Code Style Enforcement

**Files:** Project configuration  
**Principle Violated:** Consistency  
**Severity:** LOW  

**Refactoring:**
```toml
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py38']

[tool.isort]
profile = "black"
line_length = 100

[tool.pylint]
max-line-length = 100
disable = ["C0111"]  # missing-docstring

# Add pre-commit hooks
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
```

---

### 4.7 No Unit Tests

**Files:** `tests/` directory mostly empty  
**Principle Violated:** Quality Assurance  
**Severity:** LOW (but should be HIGH)  

**Refactoring:**
```python
# tests/test_features.py
import pytest
import pandas as pd
import numpy as np
from src.features.technical import compute_technical_features

def test_compute_technical_features():
    """Test technical feature computation."""
    # Create synthetic OHLCV
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [99, 100, 101],
        'close': [103, 104, 105],
        'volume': [1000, 1100, 1200]
    })
    
    features = compute_technical_features(df)
    
    # Verify output shape
    assert len(features) == len(df)
    
    # Verify expected columns
    assert 'rsi_14' in features.columns
    assert 'sma_20' in features.columns
    
    # Verify no NaN in final rows
    assert features.iloc[-1].notna().all()
```

---

### 4.8 No Integration Tests

**Files:** Missing  
**Principle Violated:** Quality Assurance  
**Severity:** LOW (but should be HIGH)  

**Refactoring:**
```python
# tests/integration/test_pipeline.py
def test_full_feature_pipeline():
    """Test complete feature engineering pipeline."""
    # Load sample data
    dfs = load_sample_universe()
    
    # Create pipeline
    pipeline = FeaturePipeline(
        target_ticker="AAPL",
        universe_tickers=list(dfs.keys())
    )
    
    # Fit and transform
    X_train, y_train, feature_names = pipeline.fit_transform(
        dfs_train, fit=True
    )
    
    # Verify outputs
    assert X_train.shape[0] > 0
    assert X_train.shape[1] == len(feature_names)
    assert len(y_train) == X_train.shape[0]
    assert not np.isnan(X_train).any()
```

---

## 5. Architectural Recommendations

### 5.1 Adopt Layered Architecture

**Current State:** Mixed concerns across all layers  
**Recommended:**

```
Presentation Layer (scripts/, CLI)
├── Thin argument parsing
├── Orchestration only
└── Delegates to Application Layer

Application Layer (src/pipelines/, src/training/)
├── High-level workflows
├── Coordinates Domain Layer
└── Handles I/O and configuration

Domain Layer (src/models/, src/features/, src/evaluation/)
├── Core business logic
├── Pure functions where possible
└── No I/O or presentation concerns

Infrastructure Layer (src/data/, src/utils/, src/database/)
├── Data access
├── External services
└── Cross-cutting concerns
```

**Benefits:**
- Clear separation of concerns
- Testable business logic
- Flexible presentation layer
- Reusable components

---

### 5.2 Implement Dependency Injection

**Current State:** Hard-coded dependencies throughout  
**Recommended:**

```python
# src/core/container.py
class Container:
    """Dependency injection container."""
    
    def __init__(self, config):
        self.config = config
        self._storage = None
        self._model_factory = None
    
    @property
    def storage(self) -> FeatureStorage:
        if self._storage is None:
            self._storage = FileSystemStorage(
                Path(self.config.data.features_dir)
            )
        return self._storage
    
    @property
    def model_factory(self) -> ModelFactory:
        if self._model_factory is None:
            self._model_factory = ModelFactory(self.config)
        return self._model_factory

# Usage:
container = Container(config)
storage = container.storage
model = container.model_factory.create("lstm")
```

---

### 5.3 Create Unified Model Interface

**Current State:** Three incompatible model APIs  
**Recommended:** See issue 1.5 for detailed `Predictor` protocol

---

### 5.4 Establish Feature Block Protocol

**Current State:** Hard-wired feature pipeline  
**Recommended:** See issue 1.11 for detailed `FeatureBlock` protocol

---

### 5.5 Implement Repository Pattern

**Current State:** Direct file I/O scattered everywhere  
**Recommended:**

```python
# src/data/repositories.py
class FeatureRepository:
    """Repository for feature data."""
    
    def __init__(self, storage: FeatureStorage):
        self.storage = storage
    
    def get_splits(self, ticker: str) -> FeatureSplits:
        """Get feature splits for ticker."""
        return self.storage.load_splits(ticker)
    
    def save_splits(self, ticker: str, splits: FeatureSplits):
        """Save feature splits for ticker."""
        self.storage.save_splits(ticker, splits)
    
    def exists(self, ticker: str) -> bool:
        """Check if features exist for ticker."""
        return self.storage.exists(ticker)

# Usage:
repo = FeatureRepository(FileSystemStorage(features_dir))
splits = repo.get_splits("AAPL")
```

---

### 5.6 Add Configuration Management Layer

**Current State:** Config loading duplicated, no validation  
**Recommended:**

```python
# src/config/manager.py
from pydantic import BaseModel

class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._config = None
    
    @property
    def config(self) -> Config:
        if self._config is None:
            self._config = self._load_and_validate()
        return self._config
    
    def _load_and_validate(self) -> Config:
        """Load and validate configuration."""
        raw_config = load_yaml(self.config_path)
        return Config.parse_obj(raw_config)  # Pydantic validation
    
    def get(self, key: str, default=None):
        """Get nested config value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = getattr(value, k, default)
            if value is default:
                return default
        return value
```

---

### 5.7 Introduce Service Layer

**Current State:** Business logic in scripts and pipelines  
**Recommended:**

```python
# src/services/training_service.py
class TrainingService:
    """Service for model training workflows."""
    
    def __init__(
        self,
        feature_repo: FeatureRepository,
        model_factory: ModelFactory,
        experiment_tracker: ExperimentTracker
    ):
        self.feature_repo = feature_repo
        self.model_factory = model_factory
        self.tracker = experiment_tracker
    
    def train_model(
        self,
        ticker: str,
        model_type: str,
        config: dict
    ) -> TrainingResult:
        """Complete training workflow."""
        # Load data
        splits = self.feature_repo.get_splits(ticker)
        
        # Create model
        model = self.model_factory.create(model_type, config)
        
        # Train
        self.tracker.log_params(config)
        history = model.fit(splits.X_train, splits.y_train)
        self.tracker.log_metrics(history)
        
        # Evaluate
        predictions = model.predict(splits.X_test)
        metrics = compute_metrics(splits.y_test, predictions)
        self.tracker.log_metrics(metrics)
        
        return TrainingResult(model, history, metrics)

# Usage in scripts:
service = TrainingService(repo, factory, tracker)
result = service.train_model("AAPL", "lstm", config)
```

---

### 5.8 Adopt Factory Pattern for Models

**Current State:** Direct instantiation with complex logic  
**Recommended:**

```python
# src/models/factory.py
class ModelFactory:
    """Factory for creating models."""
    
    def __init__(self, config):
        self.config = config
    
    def create(self, model_type: str, **overrides) -> Predictor:
        """Create model instance."""
        if model_type == "lstm":
            return self._create_lstm(**overrides)
        elif model_type == "xgboost":
            return self._create_xgboost(**overrides)
        elif model_type == "rl":
            return self._create_rl(**overrides)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_lstm(self, **overrides):
        params = {**self.config.lstm.dict(), **overrides}
        model = LSTMModel(**params)
        trainer = LSTMTrainer(model, **params)
        return LSTMTrainerAdapter(trainer)
    
    # ... other creation methods
```

---

## 6. Refactoring Roadmap

### Phase 1: Critical Fixes (Week 1)

**Priority:** Fix execution-blocking and security issues

1. **Move plots_lstm.py** from scripts/ to src/visualization/
2. **Extract RLTrainer** to src/models/rl/trainer.py
3. **Create FeatureSplits loader** in src/data/loaders.py
4. **Fix XGBoost tuner** public API
5. **Consolidate True Range** calculation
6. **Fix flattening contract** mismatch
7. **Fix invalid helpers.py** function
8. **Fix duplicate constants** in consts.py

**Estimated Effort:** 2-3 days  
**Risk:** Low (mostly moving code)

---

### Phase 2: High Priority Refactoring (Week 2-3)

**Priority:** Reduce duplication and improve modularity

1. **Refactor run_lstm_baseline.py** to match XGBoost structure
2. **Create common CLI utilities** for argparse
3. **Extract ticker list loading** to utils
4. **Consolidate trading parameters** in config
5. **Create shared indicator functions** (SMA, EMA, typical price)
6. **Extract base features** to separate module
7. **Fix volume.py** input mutation
8. **Split ttm_squeeze.py** into core and plotting

**Estimated Effort:** 5-7 days  
**Risk:** Medium (requires testing)

---

### Phase 3: Medium Priority Improvements (Week 4-5)

**Priority:** Improve configurability and robustness

1. **Add configuration validation** with Pydantic
2. **Implement feature block protocol**
3. **Create model interface** (Predictor protocol)
4. **Add data validation** to loaders
5. **Implement caching** for expensive operations
6. **Add progress reporting** throughout
7. **Standardize logging** levels
8. **Create experiment tracking** integration

**Estimated Effort:** 7-10 days  
**Risk:** Medium (new abstractions)

---

### Phase 4: Architectural Improvements (Week 6-8)

**Priority:** Long-term maintainability

1. **Implement layered architecture**
2. **Add dependency injection** container
3. **Create repository pattern** for data access
4. **Implement service layer** for workflows
5. **Add model factory** pattern
6. **Create configuration manager**
7. **Add model registry**
8. **Implement storage abstraction**

**Estimated Effort:** 10-15 days  
**Risk:** High (major restructuring)

---

### Phase 5: Quality Assurance (Week 9-10)

**Priority:** Ensure reliability

1. **Write unit tests** for all modules (target 80% coverage)
2. **Add integration tests** for pipelines
3. **Implement health checks**
4. **Add data versioning**
5. **Set up code style** enforcement
6. **Add pre-commit hooks**
7. **Create CI/CD pipeline**
8. **Write comprehensive documentation**

**Estimated Effort:** 10-12 days  
**Risk:** Low (additive)

---

## Summary Statistics

### Issues by Category

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Scripts | 4 | 8 | 3 | 1 | 16 |
| Pipelines | 2 | 6 | 2 | 0 | 10 |
| Models | 3 | 9 | 4 | 1 | 17 |
| Features | 3 | 8 | 7 | 2 | 20 |
| Infrastructure | 0 | 2 | 4 | 4 | 10 |
| **TOTAL** | **12** | **28** | **20** | **8** | **68** |

### Issues by Principle

| Principle Violated | Count | % |
|-------------------|-------|---|
| DRY (Don't Repeat Yourself) | 18 | 26% |
| Single Responsibility | 12 | 18% |
| Separation of Concerns | 10 | 15% |
| Dependency Inversion | 6 | 9% |
| Encapsulation | 5 | 7% |
| Interface Segregation | 4 | 6% |
| Open/Closed | 3 | 4% |
| Other | 10 | 15% |

### Estimated Refactoring Effort

| Phase | Duration | Risk | Priority |
|-------|----------|------|----------|
| Phase 1: Critical Fixes | 2-3 days | Low | URGENT |
| Phase 2: High Priority | 5-7 days | Medium | High |
| Phase 3: Medium Priority | 7-10 days | Medium | Medium |
| Phase 4: Architecture | 10-15 days | High | Low |
| Phase 5: Quality Assurance | 10-12 days | Low | Medium |
| **TOTAL** | **34-47 days** | | |

---

## Conclusion

The ClaudePaper codebase demonstrates solid algorithmic foundations but suffers from **architectural debt** accumulated through rapid development. The primary issues are:

1. **Inverted dependencies** (pipelines importing scripts)
2. **Massive code duplication** (feature loading, windowing, indicators)
3. **Missing abstractions** (no model interface, no feature blocks)
4. **Business logic in scripts** (should be in src/)
5. **Inconsistent patterns** (LSTM vs XGBoost vs RL)

**Immediate Actions Required:**
- Fix critical issues (Phase 1) within 1 week
- Extract business logic from scripts
- Consolidate duplicate code
- Establish consistent interfaces

**Long-Term Goals:**
- Implement layered architecture
- Add comprehensive testing
- Improve configurability
- Enhance observability

The refactoring roadmap provides a structured path from the current state to a maintainable, extensible architecture suitable for production deployment.

---

**End of Modular Audit Report**
