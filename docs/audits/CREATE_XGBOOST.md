# XGBoost Model Implementation Plan with IPSO Optimization

**Project:** PSO-LSTM Stock Price Prediction System  
**Document Version:** 1.0  
**Date:** April 6, 2026  
**Purpose:** Implement XGBoost model with IPSO hyperparameter optimization pipeline similar to LSTM-IPSO

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Implementation Plan](#3-implementation-plan)
4. [XGBoost Hyperparameter Search Space](#4-xgboost-hyperparameter-search-space)
5. [Pipeline Integration](#5-pipeline-integration)
6. [Model Tracking & Metrics System](#6-model-tracking--metrics-system)
7. [Code Cleanup & Bug Analysis](#7-code-cleanup--bug-analysis)
8. [Testing Strategy](#8-testing-strategy)
9. [Timeline & Dependencies](#9-timeline--dependencies)

---

## 1. Executive Summary

### Objective

Implement a complete XGBoost model training pipeline with IPSO hyperparameter optimization that mirrors the existing LSTM-IPSO architecture. The system must support multiple model types (LSTM, XGBoost, future models) with unified metrics tracking and clean, maintainable architecture.

### Key Requirements

1. **XGBoost Model Wrapper**: Production-grade wrapper matching `LSTMModel` + `LSTMTrainer` interface
2. **IPSO Integration**: Adapt PSO optimizer to tune XGBoost hyperparameters
3. **Unified Metrics System**: Track and compare metrics across all model types
4. **Clean Architecture**: Modular, extensible design for easy addition of new models
5. **Bug Fixes**: Address existing issues in codebase (detailed in Section 7)

### Success Criteria

- [ ] XGBoost model trains successfully with IPSO optimization
- [ ] Metrics are tracked consistently across LSTM and XGBoost models
- [ ] Results are stored in organized, queryable format
- [ ] Code passes all tests and linting checks
- [ ] Documentation is complete and accurate

---

## 2. Architecture Overview

### 2.1 Current State Analysis

**Existing Components:**
- ✅ `src/models/lstm_model.py` - LSTM model with trainer
- ✅ `src/models/xgboost_model.py` - Basic XGBoost wrapper (needs enhancement)
- ✅ `src/optimizer/ipso.py` - IPSO optimizer (LSTM-specific)
- ✅ `src/optimizer/fitness.py` - Composite fitness function
- ✅ `src/evaluation/metrics.py` - Statistical and trading metrics
- ✅ `scripts/run_model_train.py` - LSTM training script
- ⚠️ `scripts/run_xgboost.py` - Incomplete XGBoost script

**Issues Identified:**
1. `run_xgboost.py` references LSTM model instead of XGBoost (lines 38-80)
2. No XGBoost-specific particle encoding/decoding
3. No unified model registry or tracking system
4. Results storage is model-specific, not centralized
5. Missing XGBoost fitness evaluation wrapper

### 2.2 Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED MODEL PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Base Model     │      │  LSTM Model      │      │  XGBoost Model   │
│   Interface      │◄─────┤  + Trainer       │      │  + Trainer       │
│                  │      └──────────────────┘      └──────────────────┘
│  - fit()         │
│  - predict()     │      ┌──────────────────────────────────────────┐
│  - get_params()  │      │     Model Registry & Factory             │
└──────────────────┘      │  - register_model(name, class)           │
                          │  - create_model(name, params)            │
                          │  - list_models()                         │
                          └──────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    IPSO Optimizer (Generic)                     │
│  - Accepts model_builder callable                               │
│  - Particle encoding/decoding per model type                    │
│  - Unified fitness evaluation                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  Unified Metrics Tracker                        │
│  - Experiment metadata (model, params, timestamp)               │
│  - Training metrics (loss curves, convergence)                  │
│  - Evaluation metrics (RMSE, Sharpe, MDD, etc.)                 │
│  - Storage: JSON + SQLite for queryability                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Results Storage                            │
│  results/                                                       │
│    ├── experiments.db          (SQLite database)                │
│    ├── models/                                                  │
│    │   ├── lstm/                                                │
│    │   │   ├── {ticker}_{run_id}/                              │
│    │   │   │   ├── model.pth                                   │
│    │   │   │   ├── params.json                                 │
│    │   │   │   ├── metrics.json                                │
│    │   │   │   └── predictions.npy                             │
│    │   └── xgboost/                                            │
│    │       └── {ticker}_{run_id}/                              │
│    │           ├── model.pkl                                   │
│    │           ├── params.json                                 │
│    │           ├── metrics.json                                │
│    │           └── predictions.npy                             │
│    └── comparisons/                                            │
│        └── {ticker}_model_comparison.json                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Plan

### Phase 1: Core XGBoost Infrastructure (Priority: HIGH)

#### Task 1.1: Enhance XGBoost Model Wrapper
**File:** `src/models/xgboost_model.py`

**Current Issues:**
- Model is hardcoded for classification (`multi:softprob`)
- Should support regression for log return prediction
- Missing validation metrics extraction
- No model serialization support

**Changes Required:**

```python
class XGBoostModel:
    """Configurable XGBoost model for regression/classification.
    
    Supports both regression (log returns) and classification (ternary signals).
    """
    
    def __init__(self, params: Dict, task: str = "regression") -> None:
        self.params = params.copy()
        self.task = task
        
        if task == "regression":
            self.model = xgb.XGBRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.7),
                colsample_bytree=params.get("colsample_bytree", 0.6),
                min_child_weight=params.get("min_child_weight", 5),
                gamma=params.get("gamma", 0.1),
                reg_alpha=params.get("reg_alpha", 0.0),
                reg_lambda=params.get("reg_lambda", 1.0),
                tree_method=params.get("tree_method", "hist"),
                max_bin=params.get("max_bin", 128),
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        elif task == "classification":
            # Keep existing classification setup
            pass
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
        """Load model from disk."""
        import pickle
        with open(path, "rb") as f:
            model = pickle.load(f)
        instance = cls.__new__(cls)
        instance.model = model
        instance._is_fitted = True
        return instance
```

**Action Items:**
- [ ] Add regression mode support
- [ ] Implement model serialization (save/load)
- [ ] Add validation metrics extraction
- [ ] Add early stopping callback tracking
- [ ] Update docstrings with examples

---

#### Task 1.2: Create XGBoost Particle Encoding
**New File:** `src/optimizer/xgboost_particle.py`

**Purpose:** Define XGBoost-specific hyperparameter search space and encoding/decoding.

```python
"""
src/optimizer/xgboost_particle.py

Particle encoding/decoding for XGBoost hyperparameters.
"""

from typing import Dict, Any
import numpy as np

# XGBoost hyperparameter bounds
XGBOOST_BOUNDS = {
    "n_estimators": (50, 500),          # Number of trees
    "max_depth": (3, 10),                # Tree depth
    "learning_rate": (0.001, 0.3),       # Boosting rate (log scale)
    "subsample": (0.5, 1.0),             # Row sampling
    "colsample_bytree": (0.5, 1.0),      # Column sampling
    "min_child_weight": (1, 10),         # Minimum leaf weight
    "gamma": (0.0, 1.0),                 # Min split loss
    "reg_alpha": (0.0, 1.0),             # L1 regularization
    "reg_lambda": (0.0, 2.0),            # L2 regularization
}

# Dimension: 9 (vs LSTM's 5)
DIM = 9

# Bounds for PSO position vector
LB = np.array([0.0] * DIM)
UB = np.array([1.0] * DIM)


def encode(params: Dict[str, Any]) -> np.ndarray:
    """Encode XGBoost hyperparameters to [0,1]^9 position vector."""
    bounds = XGBOOST_BOUNDS
    position = np.zeros(DIM)
    
    # Linear scaling for most params
    position[0] = (params["n_estimators"] - bounds["n_estimators"][0]) / \
                  (bounds["n_estimators"][1] - bounds["n_estimators"][0])
    position[1] = (params["max_depth"] - bounds["max_depth"][0]) / \
                  (bounds["max_depth"][1] - bounds["max_depth"][0])
    
    # Log scaling for learning rate
    lr_min, lr_max = bounds["learning_rate"]
    position[2] = (np.log10(params["learning_rate"]) - np.log10(lr_min)) / \
                  (np.log10(lr_max) - np.log10(lr_min))
    
    position[3] = (params["subsample"] - bounds["subsample"][0]) / \
                  (bounds["subsample"][1] - bounds["subsample"][0])
    position[4] = (params["colsample_bytree"] - bounds["colsample_bytree"][0]) / \
                  (bounds["colsample_bytree"][1] - bounds["colsample_bytree"][0])
    position[5] = (params["min_child_weight"] - bounds["min_child_weight"][0]) / \
                  (bounds["min_child_weight"][1] - bounds["min_child_weight"][0])
    position[6] = (params["gamma"] - bounds["gamma"][0]) / \
                  (bounds["gamma"][1] - bounds["gamma"][0])
    position[7] = (params["reg_alpha"] - bounds["reg_alpha"][0]) / \
                  (bounds["reg_alpha"][1] - bounds["reg_alpha"][0])
    position[8] = (params["reg_lambda"] - bounds["reg_lambda"][0]) / \
                  (bounds["reg_lambda"][1] - bounds["reg_lambda"][0])
    
    return np.clip(position, LB, UB)


def decode(position: np.ndarray) -> Dict[str, Any]:
    """Decode [0,1]^9 position vector to XGBoost hyperparameters."""
    position = np.clip(position, LB, UB)
    bounds = XGBOOST_BOUNDS
    
    params = {}
    
    # Integer parameters
    params["n_estimators"] = int(np.round(
        position[0] * (bounds["n_estimators"][1] - bounds["n_estimators"][0]) + 
        bounds["n_estimators"][0]
    ))
    params["max_depth"] = int(np.round(
        position[1] * (bounds["max_depth"][1] - bounds["max_depth"][0]) + 
        bounds["max_depth"][0]
    ))
    
    # Log-scale learning rate
    lr_min, lr_max = bounds["learning_rate"]
    log_lr = position[2] * (np.log10(lr_max) - np.log10(lr_min)) + np.log10(lr_min)
    params["learning_rate"] = float(10 ** log_lr)
    
    # Float parameters
    params["subsample"] = float(
        position[3] * (bounds["subsample"][1] - bounds["subsample"][0]) + 
        bounds["subsample"][0]
    )
    params["colsample_bytree"] = float(
        position[4] * (bounds["colsample_bytree"][1] - bounds["colsample_bytree"][0]) + 
        bounds["colsample_bytree"][0]
    )
    params["min_child_weight"] = float(
        position[5] * (bounds["min_child_weight"][1] - bounds["min_child_weight"][0]) + 
        bounds["min_child_weight"][0]
    )
    params["gamma"] = float(
        position[6] * (bounds["gamma"][1] - bounds["gamma"][0]) + 
        bounds["gamma"][0]
    )
    params["reg_alpha"] = float(
        position[7] * (bounds["reg_alpha"][1] - bounds["reg_alpha"][0]) + 
        bounds["reg_alpha"][0]
    )
    params["reg_lambda"] = float(
        position[8] * (bounds["reg_lambda"][1] - bounds["reg_lambda"][0]) + 
        bounds["reg_lambda"][0]
    )
    
    return params


def random_position(rng: np.random.Generator) -> np.ndarray:
    """Generate random position in [0,1]^9."""
    return rng.uniform(LB, UB, size=DIM)


def random_velocity(rng: np.random.Generator, v_clamp: float = 0.2) -> np.ndarray:
    """Generate random velocity."""
    return rng.uniform(-v_clamp, v_clamp, size=DIM)
```

**Action Items:**
- [ ] Create `xgboost_particle.py` with encoding/decoding
- [ ] Add unit tests for encode/decode round-trip
- [ ] Validate bounds match config file
- [ ] Document hyperparameter choices with references

---

#### Task 1.3: Create XGBoost Model Builder
**New File:** `src/models/xgboost_builder.py`

**Purpose:** Wrapper function for IPSO fitness evaluation.

```python
"""
src/models/xgboost_builder.py

Model builder for XGBoost IPSO optimization.
"""

from typing import Dict
import numpy as np
from .xgboost_model import XGBoostModel, XGBoostTrainer


def build_and_train_xgboost(
    params: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "regression",
) -> np.ndarray:
    """Build and train XGBoost model, return validation predictions.
    
    Args:
        params: Decoded hyperparameters from particle
        X_train: [N_train, F] feature matrix (flattened)
        y_train: [N_train] targets
        X_val: [N_val, F] validation features
        y_val: [N_val] validation targets
        task: "regression" or "classification"
    
    Returns:
        y_pred_val: [N_val] validation predictions
    """
    # Add fixed params not tuned by PSO
    full_params = {
        **params,
        "tree_method": "hist",
        "max_bin": 128,
        "random_state": 42,
        "early_stopping_rounds": 50,
    }
    
    model = XGBoostModel(full_params, task=task)
    trainer = XGBoostTrainer(model)
    
    # Flatten if needed (XGBoost expects 2D)
    if X_train.ndim == 3:
        N_train, T, F = X_train.shape
        X_train = X_train.reshape(N_train, T * F)
    if X_val.ndim == 3:
        N_val, T, F = X_val.shape
        X_val = X_val.reshape(N_val, T * F)
    
    trainer.fit(X_train, y_train, X_val, y_val)
    y_pred_val = trainer.predict(X_val)
    
    return y_pred_val
```

**Action Items:**
- [ ] Create `xgboost_builder.py`
- [ ] Add input validation and error handling
- [ ] Support both 2D and 3D input (auto-flatten)
- [ ] Add logging for training progress

---

### Phase 2: IPSO Integration (Priority: HIGH)

#### Task 2.1: Create Generic IPSO Wrapper
**New File:** `src/optimizer/generic_ipso.py`

**Purpose:** Model-agnostic IPSO that accepts any model builder function.

```python
"""
src/optimizer/generic_ipso.py

Generic IPSO optimizer that works with any model type.
"""

from typing import Callable, Dict, Any, Tuple
import numpy as np
from .ipso import IPSO


class GenericIPSO(IPSO):
    """IPSO optimizer for any model type.
    
    Accepts a model_builder callable and particle encode/decode functions.
    """
    
    def __init__(
        self,
        model_builder: Callable,
        encode_fn: Callable,
        decode_fn: Callable,
        random_position_fn: Callable,
        random_velocity_fn: Callable,
        dimension: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        fitness_fn: Callable,
        n_particles: int = 30,
        n_iterations: int = 50,
        **kwargs
    ):
        """
        Args:
            model_builder: Function(params, X_train, y_train, X_val, y_val) -> y_pred
            encode_fn: Function(params_dict) -> position_vector
            decode_fn: Function(position_vector) -> params_dict
            random_position_fn: Function(rng) -> position_vector
            random_velocity_fn: Function(rng) -> velocity_vector
            dimension: Search space dimensionality
            bounds: (lower_bounds, upper_bounds) arrays
            fitness_fn: CompositeFitness instance
            n_particles: Swarm size
            n_iterations: PSO iterations
        """
        self.model_builder = model_builder
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.random_position_fn = random_position_fn
        self.random_velocity_fn = random_velocity_fn
        self.dimension = dimension
        self.lb, self.ub = bounds
        
        # Initialize parent IPSO
        super().__init__(
            n_particles=n_particles,
            n_iterations=n_iterations,
            fitness_fn=fitness_fn,
            **kwargs
        )
    
    def _evaluate_particle(
        self,
        particle,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Evaluate fitness for one particle."""
        # Decode position to hyperparameters
        params = self.decode_fn(particle.position)
        
        # Build and train model
        y_pred_val = self.model_builder(params, X_train, y_train, X_val, y_val)
        
        # Compute fitness
        fitness = self.fitness_fn(y_val, y_pred_val)
        
        return fitness
```

**Action Items:**
- [ ] Create `generic_ipso.py`
- [ ] Test with both LSTM and XGBoost builders
- [ ] Add progress logging and checkpointing
- [ ] Handle exceptions in model training gracefully

---

#### Task 2.2: Fix `run_xgboost.py` Script
**File:** `scripts/run_xgboost.py`

**Current Issues:**
- Lines 38-80: Uses LSTM model instead of XGBoost
- Missing XGBoost particle import
- Incorrect fitness function setup

**Required Changes:**

```python
# Replace lines 38-80 with:

from src.optimizer.xgboost_particle import (
    encode, decode, random_position, random_velocity,
    XGBOOST_BOUNDS, DIM, LB, UB
)
from src.models.xgboost_builder import build_and_train_xgboost
from src.optimizer.generic_ipso import GenericIPSO


def main():
    # ... (argument parsing remains same) ...
    
    # Load and prepare data (flatten for XGBoost)
    X_train_flat = np.load(ticker_dir / "X_train.npy")
    y_train = np.load(ticker_dir / "y_train.npy")
    X_val_flat = np.load(ticker_dir / "X_val.npy")
    y_val = np.load(ticker_dir / "y_val.npy")
    
    # XGBoost doesn't need windowing - works on flattened features
    logger.info("Train: %s, Val: %s", X_train_flat.shape, X_val_flat.shape)
    
    # Initialize fitness function
    fitness_fn = CompositeFitness(
        weights={
            "rmse": cfg.fitness.rmse_weight,
            "sharpe": cfg.fitness.sharpe_weight,
            "mdd": cfg.fitness.drawdown_weight,
        },
        signal_threshold=cfg.fitness.signal_threshold,
        transaction_cost=cfg.fitness.transaction_cost,
    )
    
    # Initialize Generic IPSO for XGBoost
    optimizer = GenericIPSO(
        model_builder=build_and_train_xgboost,
        encode_fn=encode,
        decode_fn=decode,
        random_position_fn=random_position,
        random_velocity_fn=random_velocity,
        dimension=DIM,
        bounds=(LB, UB),
        fitness_fn=fitness_fn,
        n_particles=cfg.pso.n_particles,
        n_iterations=cfg.pso.n_iterations,
        w_min=cfg.pso.w_min,
        w_max=cfg.pso.w_max,
        c1=cfg.pso.c1,
        c2=cfg.pso.c2,
        seed=seed,
        checkpoint_dir=cfg.pso.checkpoint_dir,
    )
    
    logger.info("Starting XGBoost IPSO optimization...")
    best_params, best_fitness = optimizer.run(
        X_train_flat, y_train, X_val_flat, y_val
    )
    
    # ... (rest of saving logic) ...
```

**Action Items:**
- [ ] Replace LSTM references with XGBoost
- [ ] Update imports
- [ ] Remove windowing logic (XGBoost uses flat features)
- [ ] Update logging messages
- [ ] Test end-to-end execution

---

### Phase 3: Unified Metrics & Tracking System (Priority: MEDIUM)

#### Task 3.1: Create Experiment Tracker
**New File:** `src/utils/experiment_tracker.py`

**Purpose:** Centralized tracking for all model experiments.

```python
"""
src/utils/experiment_tracker.py

Unified experiment tracking across all model types.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np


class ExperimentTracker:
    """Track experiments across all model types with SQLite backend."""
    
    def __init__(self, db_path: Path = Path("results/experiments.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                run_id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                config_hash TEXT,
                status TEXT,
                params_json TEXT,
                metrics_json TEXT,
                artifacts_path TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                run_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                metric_type TEXT,
                FOREIGN KEY (run_id) REFERENCES experiments(run_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pso_iterations (
                run_id TEXT,
                iteration INTEGER,
                gbest_fitness REAL,
                swarm_diversity REAL,
                FOREIGN KEY (run_id) REFERENCES experiments(run_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def start_experiment(
        self,
        model_type: str,
        ticker: str,
        params: Dict[str, Any],
        config: Optional[Dict] = None,
    ) -> str:
        """Start new experiment, return run_id."""
        run_id = f"{model_type}_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO experiments 
            (run_id, model_type, ticker, timestamp, status, params_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            model_type,
            ticker,
            datetime.now().isoformat(),
            "running",
            json.dumps(params),
        ))
        
        conn.commit()
        conn.close()
        
        return run_id
    
    def log_metrics(self, run_id: str, metrics: Dict[str, float], metric_type: str = "eval"):
        """Log metrics for an experiment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for name, value in metrics.items():
            cursor.execute("""
                INSERT INTO metrics (run_id, metric_name, metric_value, metric_type)
                VALUES (?, ?, ?, ?)
            """, (run_id, name, float(value), metric_type))
        
        # Update experiments table
        cursor.execute("""
            UPDATE experiments
            SET metrics_json = ?
            WHERE run_id = ?
        """, (json.dumps(metrics), run_id))
        
        conn.commit()
        conn.close()
    
    def log_pso_iteration(self, run_id: str, iteration: int, gbest_fitness: float, diversity: float):
        """Log PSO iteration metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO pso_iterations (run_id, iteration, gbest_fitness, swarm_diversity)
            VALUES (?, ?, ?, ?)
        """, (run_id, iteration, gbest_fitness, diversity))
        
        conn.commit()
        conn.close()
    
    def finish_experiment(self, run_id: str, status: str = "completed"):
        """Mark experiment as finished."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE experiments
            SET status = ?
            WHERE run_id = ?
        """, (status, run_id))
        
        conn.commit()
        conn.close()
    
    def get_best_run(self, ticker: str, model_type: str, metric: str = "sharpe") -> Optional[Dict]:
        """Get best run for a ticker and model type by metric."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT e.run_id, e.params_json, e.metrics_json, m.metric_value
            FROM experiments e
            JOIN metrics m ON e.run_id = m.run_id
            WHERE e.ticker = ? AND e.model_type = ? AND m.metric_name = ?
            ORDER BY m.metric_value DESC
            LIMIT 1
        """, (ticker, model_type, metric))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "run_id": row[0],
                "params": json.loads(row[1]),
                "metrics": json.loads(row[2]),
                "best_metric_value": row[3],
            }
        return None
    
    def compare_models(self, ticker: str) -> Dict[str, Dict]:
        """Compare all models for a ticker."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT model_type, metrics_json
            FROM experiments
            WHERE ticker = ? AND status = 'completed'
            ORDER BY timestamp DESC
        """, (ticker,))
        
        results = {}
        for row in cursor.fetchall():
            model_type = row[0]
            metrics = json.loads(row[1])
            if model_type not in results:
                results[model_type] = metrics
        
        conn.close()
        return results
```

**Action Items:**
- [ ] Create `experiment_tracker.py`
- [ ] Add unit tests for database operations
- [ ] Create CLI tool for querying experiments
- [ ] Add export functionality (CSV, JSON)

---

#### Task 3.2: Create Model Registry
**New File:** `src/models/registry.py`

**Purpose:** Central registry for all model types.

```python
"""
src/models/registry.py

Model registry for dynamic model creation.
"""

from typing import Dict, Type, Callable, Any
from pathlib import Path


class ModelRegistry:
    """Registry for all model types in the system."""
    
    _models: Dict[str, Type] = {}
    _builders: Dict[str, Callable] = {}
    _particle_modules: Dict[str, Any] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type,
        builder_fn: Callable,
        particle_module: Any,
    ):
        """Register a new model type.
        
        Args:
            name: Model identifier (e.g., "lstm", "xgboost")
            model_class: Model class
            builder_fn: Function to build and train model
            particle_module: Module with encode/decode/random_* functions
        """
        cls._models[name] = model_class
        cls._builders[name] = builder_fn
        cls._particle_modules[name] = particle_module
    
    @classmethod
    def get_model_class(cls, name: str) -> Type:
        """Get model class by name."""
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not registered. Available: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def get_builder(cls, name: str) -> Callable:
        """Get builder function by name."""
        if name not in cls._builders:
            raise ValueError(f"Builder for '{name}' not registered.")
        return cls._builders[name]
    
    @classmethod
    def get_particle_module(cls, name: str) -> Any:
        """Get particle encoding module by name."""
        if name not in cls._particle_modules:
            raise ValueError(f"Particle module for '{name}' not registered.")
        return cls._particle_modules[name]
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered models."""
        return list(cls._models.keys())


# Register models at import time
from .lstm_model import LSTMModel
from .xgboost_model import XGBoostModel
from ..optimizer import particle as lstm_particle
# from ..optimizer import xgboost_particle  # Will add after creation

def build_lstm(params, X_train, y_train, X_val, y_val):
    """LSTM builder function."""
    from .lstm_model import LSTMTrainer
    model = LSTMModel(
        input_size=X_train.shape[2],
        num_layers=params["num_layers"],
        hidden_units=params["hidden_units"],
        dropout=params["dropout"],
    )
    trainer = LSTMTrainer(
        model=model,
        lr=params["learning_rate"],
        max_epochs=100,
        patience=10,
        batch_size=256,
    )
    trainer.fit(X_train, y_train, X_val, y_val)
    return trainer.predict(X_val)


ModelRegistry.register("lstm", LSTMModel, build_lstm, lstm_particle)
# ModelRegistry.register("xgboost", XGBoostModel, build_and_train_xgboost, xgboost_particle)
```

**Action Items:**
- [ ] Create `registry.py`
- [ ] Register LSTM and XGBoost models
- [ ] Add validation for registration
- [ ] Create helper functions for model creation

---

#### Task 3.3: Create Unified Training Script
**New File:** `scripts/train_model.py`

**Purpose:** Single script to train any model type with IPSO.

```python
"""
scripts/train_model.py

Unified training script for all model types.
"""

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.registry import ModelRegistry
from src.optimizer.generic_ipso import GenericIPSO
from src.optimizer.fitness import CompositeFitness
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.config_loader import load_config
from src.utils.seed import set_all_seeds
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Train any model with IPSO")
    parser.add_argument("--model", type=str, required=True, 
                       choices=ModelRegistry.list_models(),
                       help="Model type to train")
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Setup
    cfg = load_config(args.config)
    set_all_seeds(args.seed)
    setup_logger(log_file=f"logs/train_{args.model}_{args.ticker}.log")
    
    # Initialize tracker
    tracker = ExperimentTracker()
    
    # Load data
    ticker_dir = Path(args.features_dir) / args.ticker
    X_train = np.load(ticker_dir / "X_train.npy")
    y_train = np.load(ticker_dir / "y_train.npy")
    X_val = np.load(ticker_dir / "X_val.npy")
    y_val = np.load(ticker_dir / "y_val.npy")
    
    # Get model-specific components
    builder = ModelRegistry.get_builder(args.model)
    particle_module = ModelRegistry.get_particle_module(args.model)
    
    # Start experiment
    run_id = tracker.start_experiment(
        model_type=args.model,
        ticker=args.ticker,
        params={},  # Will be updated after PSO
        config=cfg.__dict__,
    )
    
    # Initialize fitness
    fitness_fn = CompositeFitness(
        weights={
            "rmse": cfg.fitness.rmse_weight,
            "sharpe": cfg.fitness.sharpe_weight,
            "mdd": cfg.fitness.drawdown_weight,
        },
        signal_threshold=cfg.fitness.signal_threshold,
        transaction_cost=cfg.fitness.transaction_cost,
    )
    
    # Initialize IPSO
    optimizer = GenericIPSO(
        model_builder=builder,
        encode_fn=particle_module.encode,
        decode_fn=particle_module.decode,
        random_position_fn=particle_module.random_position,
        random_velocity_fn=particle_module.random_velocity,
        dimension=particle_module.DIM,
        bounds=(particle_module.LB, particle_module.UB),
        fitness_fn=fitness_fn,
        n_particles=cfg.pso.n_particles,
        n_iterations=cfg.pso.n_iterations,
        w_min=cfg.pso.w_min,
        w_max=cfg.pso.w_max,
        c1=cfg.pso.c1,
        c2=cfg.pso.c2,
        seed=args.seed,
    )
    
    # Run optimization
    best_params, best_fitness = optimizer.run(X_train, y_train, X_val, y_val)
    
    # Log results
    tracker.log_metrics(run_id, {"fitness": best_fitness}, metric_type="pso")
    tracker.finish_experiment(run_id, status="completed")
    
    print(f"✓ Training complete. Run ID: {run_id}")
    print(f"✓ Best fitness: {best_fitness:.6f}")
    print(f"✓ Best params: {best_params}")


if __name__ == "__main__":
    main()
```

**Action Items:**
- [ ] Create `train_model.py`
- [ ] Add progress bars for PSO iterations
- [ ] Add model saving after training
- [ ] Add resume capability from checkpoint

---

## 4. XGBoost Hyperparameter Search Space

### 4.1 Hyperparameter Definitions

| Parameter | Type | Range | Scale | Description |
|-----------|------|-------|-------|-------------|
| `n_estimators` | int | [50, 500] | linear | Number of boosting rounds |
| `max_depth` | int | [3, 10] | linear | Maximum tree depth |
| `learning_rate` | float | [0.001, 0.3] | log | Boosting learning rate (eta) |
| `subsample` | float | [0.5, 1.0] | linear | Row sampling ratio per tree |
| `colsample_bytree` | float | [0.5, 1.0] | linear | Column sampling ratio per tree |
| `min_child_weight` | float | [1, 10] | linear | Minimum sum of instance weight in child |
| `gamma` | float | [0.0, 1.0] | linear | Minimum loss reduction for split |
| `reg_alpha` | float | [0.0, 1.0] | linear | L1 regularization term |
| `reg_lambda` | float | [0.0, 2.0] | linear | L2 regularization term |

### 4.2 Fixed Parameters (Not Tuned)

```yaml
tree_method: "hist"           # Histogram-based algorithm (fast)
max_bin: 128                  # Number of bins for histogram
random_state: 42              # Reproducibility
n_jobs: -1                    # Use all CPU cores
verbosity: 0                  # Silent mode
early_stopping_rounds: 50     # Stop if no improvement
```

### 4.3 Rationale

**Dimension Comparison:**
- LSTM: 5 dimensions (layers, hidden, dropout, lr, lookback)
- XGBoost: 9 dimensions (n_estimators, depth, lr, subsample, colsample, min_child, gamma, alpha, lambda)

**Why more dimensions?**
- XGBoost has more regularization knobs (alpha, lambda, gamma)
- Tree structure parameters (depth, min_child_weight)
- Sampling parameters (subsample, colsample_bytree)

**PSO Considerations:**
- Higher dimensionality → may need more particles (suggest 40-50 vs 30 for LSTM)
- May need more iterations (60-80 vs 50 for LSTM)
- Expected runtime: ~2-3x longer than LSTM due to dimensionality

---

## 5. Pipeline Integration

### 5.1 Data Flow for XGBoost

```
Raw Features [N, F]
    ↓
No windowing needed (XGBoost handles temporal features directly)
    ↓
Optional: Add lag features explicitly if not already in feature set
    ↓
Train/Val/Test Split
    ↓
XGBoost Training
    ↓
Predictions [N]
```

**Key Difference from LSTM:**
- LSTM requires windowing: `[N, T, F]` where T = lookback
- XGBoost uses flat features: `[N, F]`
- If temporal context needed, add lag features explicitly

### 5.2 Feature Engineering Considerations

**Current Feature Set:**
- Technical indicators (RSI, MACD, etc.)
- Statistical features (rolling mean, std, etc.)
- Volume features
- Cross-ticker features

**Recommendations for XGBoost:**
1. **Add explicit lag features** if not present:
   ```python
   for lag in [1, 5, 10, 30, 60]:
       df[f'close_lag_{lag}'] = df['close'].shift(lag)
       df[f'return_lag_{lag}'] = df['return'].shift(lag)
   ```

2. **Add interaction features**:
   ```python
   df['volume_price_interaction'] = df['volume'] * df['close']
   df['rsi_macd_interaction'] = df['rsi'] * df['macd']
   ```

3. **Add time-based features**:
   ```python
   df['hour'] = df.index.hour
   df['minute'] = df.index.minute
   df['day_of_week'] = df.index.dayofweek
   ```

### 5.3 Updated Config File

**File:** `config/default_config.yaml`

Add XGBoost section:

```yaml
xgboost:
  # IPSO search space
  n_estimators:
    min: 50
    max: 500
  max_depth:
    min: 3
    max: 10
  learning_rate:
    min: 0.001
    max: 0.3
    scale: "log"
  subsample:
    min: 0.5
    max: 1.0
  colsample_bytree:
    min: 0.5
    max: 1.0
  min_child_weight:
    min: 1
    max: 10
  gamma:
    min: 0.0
    max: 1.0
  reg_alpha:
    min: 0.0
    max: 1.0
  reg_lambda:
    min: 0.0
    max: 2.0
  
  # Fixed parameters
  tree_method: "hist"
  max_bin: 128
  early_stopping_rounds: 50
  
  # Task type
  task: "regression"  # or "classification"
```

---

## 6. Model Tracking & Metrics System

### 6.1 Metrics to Track

#### Training Metrics
- PSO iteration number
- Global best fitness per iteration
- Swarm diversity per iteration
- Particle positions (optional, for analysis)
- Training time per iteration

#### Evaluation Metrics

**Statistical:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Directional Accuracy
- F1 Score (ternary classification)
- AUC (ternary classification)

**Trading/Financial:**
- Sharpe Ratio (annualized)
- Sortino Ratio
- Maximum Drawdown
- CAGR (Compound Annual Growth Rate)
- Calmar Ratio
- Profit Factor
- Win Rate
- Information Ratio (vs benchmark)

### 6.2 Storage Schema

#### SQLite Schema (Detailed)

```sql
-- Main experiments table
CREATE TABLE experiments (
    run_id TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    ticker TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    config_hash TEXT,
    status TEXT CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
    params_json TEXT,
    metrics_json TEXT,
    artifacts_path TEXT,
    duration_seconds REAL,
    git_commit TEXT,
    notes TEXT
);

-- Detailed metrics table
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    metric_name TEXT,
    metric_value REAL,
    metric_type TEXT CHECK(metric_type IN ('train', 'val', 'test', 'pso')),
    timestamp TEXT,
    FOREIGN KEY (run_id) REFERENCES experiments(run_id)
);

-- PSO iteration tracking
CREATE TABLE pso_iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INTEGER,
    gbest_fitness REAL,
    gbest_params_json TEXT,
    swarm_diversity REAL,
    avg_fitness REAL,
    timestamp TEXT,
    FOREIGN KEY (run_id) REFERENCES experiments(run_id)
);

-- Model comparisons
CREATE TABLE model_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    comparison_date TEXT,
    models_json TEXT,
    winner TEXT,
    criteria TEXT
);

-- Indices for fast queries
CREATE INDEX idx_experiments_ticker ON experiments(ticker);
CREATE INDEX idx_experiments_model_type ON experiments(model_type);
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_metrics_run_id ON metrics(run_id);
CREATE INDEX idx_metrics_name ON metrics(metric_name);
```

### 6.3 Artifacts Storage Structure

```
results/
├── experiments.db                    # SQLite database
├── models/
│   ├── lstm/
│   │   └── {ticker}_{run_id}/
│   │       ├── model.pth             # PyTorch model weights
│   │       ├── params.json           # Hyperparameters
│   │       ├── metrics.json          # All metrics
│   │       ├── predictions.npy       # Test set predictions
│   │       ├── training_history.json # Loss curves
│   │       └── pso_history.json      # PSO convergence data
│   └── xgboost/
│       └── {ticker}_{run_id}/
│           ├── model.pkl             # XGBoost model
│           ├── params.json
│           ├── metrics.json
│           ├── predictions.npy
│           ├── feature_importance.json
│           └── pso_history.json
├── comparisons/
│   └── {ticker}_comparison_{date}.json
└── plots/
    ├── {run_id}_pso_convergence.png
    ├── {run_id}_predictions.png
    └── {run_id}_equity_curve.png
```

### 6.4 Query Examples

**Get best model for ticker:**
```python
tracker = ExperimentTracker()
best_lstm = tracker.get_best_run("AAPL", "lstm", metric="sharpe")
best_xgb = tracker.get_best_run("AAPL", "xgboost", metric="sharpe")
```

**Compare all models:**
```python
comparison = tracker.compare_models("AAPL")
# Returns: {"lstm": {...metrics...}, "xgboost": {...metrics...}}
```

**Get PSO convergence data:**
```python
conn = sqlite3.connect("results/experiments.db")
df = pd.read_sql("""
    SELECT iteration, gbest_fitness, swarm_diversity
    FROM pso_iterations
    WHERE run_id = ?
    ORDER BY iteration
""", conn, params=(run_id,))
```

---

## 7. Code Cleanup & Bug Analysis

### 7.1 Critical Bugs

#### Bug 1: `run_xgboost.py` Uses LSTM Model
**File:** `scripts/run_xgboost.py`  
**Lines:** 38-80  
**Severity:** HIGH  
**Impact:** Script cannot run XGBoost optimization

**Issue:**
```python
# Current (WRONG):
model = LSTMModel(
    input_size=input_size,
    num_layers=params["num_layers"],
    ...
)
```

**Fix:**
```python
# Corrected:
from src.models.xgboost_builder import build_and_train_xgboost
# Use build_and_train_xgboost instead of LSTM model
```

**Status:** ❌ Not Fixed  
**Priority:** P0 - Blocks XGBoost implementation

---

#### Bug 2: Missing XGBoost Particle Encoding
**File:** `src/optimizer/` (missing file)  
**Severity:** HIGH  
**Impact:** Cannot encode XGBoost hyperparameters for PSO

**Issue:**
- No `xgboost_particle.py` file exists
- LSTM particle encoding is hardcoded for 5 dimensions
- XGBoost needs 9 dimensions

**Fix:**
- Create `src/optimizer/xgboost_particle.py` (see Task 1.2)
- Implement encode/decode for 9-dimensional space

**Status:** ❌ Not Implemented  
**Priority:** P0 - Blocks XGBoost implementation

---

#### Bug 3: XGBoost Model Hardcoded for Classification
**File:** `src/models/xgboost_model.py`  
**Lines:** 35-52  
**Severity:** MEDIUM  
**Impact:** Cannot use XGBoost for regression (log return prediction)

**Issue:**
```python
self.model = xgb.XGBClassifier(
    objective="multi:softprob",  # Classification only
    num_class=3,
    ...
)
```

**Fix:**
```python
if task == "regression":
    self.model = xgb.XGBRegressor(
        objective="reg:squarederror",
        ...
    )
elif task == "classification":
    self.model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        ...
    )
```

**Status:** ❌ Not Fixed  
**Priority:** P1 - Required for proper XGBoost usage

---

#### Bug 4: No Model Serialization
**File:** `src/models/xgboost_model.py`  
**Severity:** MEDIUM  
**Impact:** Cannot save/load trained models

**Issue:**
- No `save()` method
- No `load()` class method
- Models are lost after training

**Fix:**
```python
def save(self, path: Path) -> None:
    import pickle
    with open(path, "wb") as f:
        pickle.dump(self.model, f)

@classmethod
def load(cls, path: Path) -> "XGBoostModel":
    import pickle
    with open(path, "rb") as f:
        model = pickle.load(f)
    instance = cls.__new__(cls)
    instance.model = model
    instance._is_fitted = True
    return instance
```

**Status:** ❌ Not Implemented  
**Priority:** P1 - Required for model persistence

---

#### Bug 5: Inconsistent Metrics Extraction
**File:** `src/models/xgboost_model.py`  
**Lines:** 128-138  
**Severity:** LOW  
**Impact:** Training metrics not properly tracked

**Issue:**
```python
evals_result = self.model.model.evals_result()
# Inconsistent dataset naming ("validation_0" vs "validation_1")
# No error handling if eval_set not provided
```

**Fix:**
```python
def fit(self, X_train, y_train, X_val=None, y_val=None):
    eval_set = [(X_train, y_train)]
    eval_names = ["train"]
    
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))
        eval_names.append("val")
    
    self.model.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_names=eval_names,  # Explicit naming
        verbose=False,
    )
    
    evals_result = self.model.evals_result()
    if evals_result:
        self.history["train_metric"] = evals_result.get("train", {}).get("rmse", [])
        self.history["val_metric"] = evals_result.get("val", {}).get("rmse", [])
```

**Status:** ❌ Not Fixed  
**Priority:** P2 - Improves tracking quality

---

### 7.2 Code Quality Issues

#### Issue 1: Duplicate Code in Training Scripts
**Files:** `scripts/run_model_train.py`, `scripts/run_xgboost.py`  
**Severity:** LOW  
**Impact:** Maintenance burden, inconsistency risk

**Issue:**
- Both scripts have nearly identical argument parsing
- Both scripts have identical data loading logic
- Both scripts have identical config loading

**Fix:**
- Create unified `scripts/train_model.py` (see Task 3.3)
- Use `--model` flag to select model type
- Deprecate separate scripts

**Status:** ❌ Not Refactored  
**Priority:** P2 - Improves maintainability

---

#### Issue 2: No Input Validation
**Files:** Multiple model files  
**Severity:** MEDIUM  
**Impact:** Cryptic errors when invalid inputs provided

**Issue:**
- No validation of hyperparameter ranges
- No validation of input array shapes
- No validation of config file structure

**Fix:**
```python
def validate_params(params: Dict, bounds: Dict) -> None:
    """Validate hyperparameters are within bounds."""
    for key, value in params.items():
        if key in bounds:
            min_val, max_val = bounds[key]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Parameter '{key}' = {value} out of bounds [{min_val}, {max_val}]"
                )

def validate_input_shape(X: np.ndarray, expected_dims: int, name: str) -> None:
    """Validate input array shape."""
    if X.ndim != expected_dims:
        raise ValueError(
            f"{name} expected {expected_dims}D array, got {X.ndim}D"
        )
```

**Status:** ❌ Not Implemented  
**Priority:** P2 - Improves user experience

---

#### Issue 3: Inconsistent Logging
**Files:** Multiple  
**Severity:** LOW  
**Impact:** Difficult to debug issues

**Issue:**
- Some files use `print()`, others use `logger`
- No consistent log levels
- No structured logging (JSON logs)

**Fix:**
```python
# Standardize on logger everywhere
import logging
logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed info for debugging")
logger.info("General information")
logger.warning("Warning messages")
logger.error("Error messages")

# Add structured logging
logger.info("Training complete", extra={
    "model": "xgboost",
    "ticker": "AAPL",
    "rmse": 0.0123,
    "sharpe": 1.45,
})
```

**Status:** ⚠️ Partially Implemented  
**Priority:** P3 - Nice to have

---

#### Issue 4: No Type Hints in Some Files
**Files:** `src/optimizer/fitness.py`, `src/evaluation/backtester.py`  
**Severity:** LOW  
**Impact:** Reduced code clarity, no static type checking

**Issue:**
- Some functions lack type hints
- Makes IDE autocomplete less effective
- Harder to catch type errors

**Fix:**
```python
# Before:
def sharpe_from_signals(signals, y_true, transaction_cost=0.001):
    ...

# After:
def sharpe_from_signals(
    signals: np.ndarray,
    y_true: np.ndarray,
    transaction_cost: float = 0.001,
) -> float:
    ...
```

**Status:** ⚠️ Partially Implemented  
**Priority:** P3 - Code quality improvement

---

#### Issue 5: Missing Docstrings
**Files:** Multiple  
**Severity:** LOW  
**Impact:** Reduced code documentation

**Issue:**
- Some functions lack docstrings
- Some docstrings don't follow NumPy/Google style
- No examples in docstrings

**Fix:**
```python
def build_and_train_xgboost(
    params: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> np.ndarray:
    """Build and train XGBoost model, return validation predictions.
    
    Args:
        params: Hyperparameters decoded from PSO particle.
            Must include: n_estimators, max_depth, learning_rate, etc.
        X_train: Training features, shape [N_train, F]
        y_train: Training targets, shape [N_train]
        X_val: Validation features, shape [N_val, F]
        y_val: Validation targets, shape [N_val]
    
    Returns:
        Validation predictions, shape [N_val]
    
    Raises:
        ValueError: If params are invalid or out of bounds
        RuntimeError: If training fails
    
    Example:
        >>> params = {"n_estimators": 200, "max_depth": 5, ...}
        >>> y_pred = build_and_train_xgboost(params, X_train, y_train, X_val, y_val)
        >>> rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
    """
    ...
```

**Status:** ⚠️ Partially Implemented  
**Priority:** P3 - Documentation improvement

---

### 7.3 Testing Gaps

#### Gap 1: No XGBoost Tests
**File:** `tests/` (missing)  
**Severity:** HIGH  
**Impact:** No confidence in XGBoost implementation

**Required Tests:**
```python
# tests/test_xgboost.py

def test_xgboost_model_regression():
    """Test XGBoost in regression mode."""
    params = {"n_estimators": 10, "max_depth": 3, ...}
    model = XGBoostModel(params, task="regression")
    X_train = np.random.randn(100, 20)
    y_train = np.random.randn(100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    assert y_pred.shape == (100,)

def test_xgboost_particle_encoding():
    """Test encode/decode round-trip."""
    from src.optimizer.xgboost_particle import encode, decode
    params = {"n_estimators": 200, "max_depth": 5, ...}
    position = encode(params)
    params_decoded = decode(position)
    assert params_decoded["n_estimators"] == params["n_estimators"]

def test_xgboost_builder():
    """Test XGBoost builder function."""
    from src.models.xgboost_builder import build_and_train_xgboost
    params = {"n_estimators": 10, ...}
    X_train = np.random.randn(100, 20)
    y_train = np.random.randn(100)
    X_val = np.random.randn(50, 20)
    y_val = np.random.randn(50)
    y_pred = build_and_train_xgboost(params, X_train, y_train, X_val, y_val)
    assert y_pred.shape == (50,)
```

**Status:** ❌ Not Implemented  
**Priority:** P1 - Required for confidence

---

#### Gap 2: No Integration Tests
**File:** `tests/` (missing)  
**Severity:** MEDIUM  
**Impact:** No end-to-end validation

**Required Tests:**
```python
# tests/test_integration.py

def test_full_xgboost_pipeline():
    """Test complete XGBoost IPSO pipeline."""
    # Load small test dataset
    # Run 2 particles, 2 iterations
    # Verify results are saved correctly
    # Verify metrics are tracked
    pass

def test_model_comparison():
    """Test comparing LSTM vs XGBoost."""
    # Train both models on same data
    # Compare metrics
    # Verify comparison is saved
    pass
```

**Status:** ❌ Not Implemented  
**Priority:** P2 - Improves confidence

---

#### Gap 3: No Performance Tests
**File:** `tests/` (missing)  
**Severity:** LOW  
**Impact:** No performance regression detection

**Required Tests:**
```python
# tests/test_performance.py

def test_xgboost_training_speed():
    """Ensure XGBoost training completes in reasonable time."""
    import time
    start = time.time()
    # Train model
    elapsed = time.time() - start
    assert elapsed < 60, f"Training took {elapsed:.1f}s, expected < 60s"

def test_pso_iteration_speed():
    """Ensure PSO iteration completes in reasonable time."""
    # Run 1 iteration with 5 particles
    # Should complete in < 5 minutes
    pass
```

**Status:** ❌ Not Implemented  
**Priority:** P3 - Nice to have

---

### 7.4 Bug Fix Checklist

**Critical (P0) - Must fix before XGBoost works:**
- [ ] Fix `run_xgboost.py` to use XGBoost model (Bug 1)
- [ ] Create `xgboost_particle.py` with encoding/decoding (Bug 2)
- [ ] Add regression mode to XGBoost model (Bug 3)

**High Priority (P1) - Should fix soon:**
- [ ] Add model serialization (Bug 4)
- [ ] Fix metrics extraction (Bug 5)
- [ ] Add XGBoost unit tests (Gap 1)

**Medium Priority (P2) - Fix when time permits:**
- [ ] Refactor duplicate code (Issue 1)
- [ ] Add input validation (Issue 2)
- [ ] Add integration tests (Gap 2)

**Low Priority (P3) - Nice to have:**
- [ ] Standardize logging (Issue 3)
- [ ] Add type hints everywhere (Issue 4)
- [ ] Improve docstrings (Issue 5)
- [ ] Add performance tests (Gap 3)

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Coverage Goals:**
- Model classes: 90%
- Particle encoding: 100%
- Fitness functions: 90%
- Utility functions: 80%

**Test Files:**
```
tests/
├── test_xgboost_model.py       # XGBoost model tests
├── test_xgboost_particle.py    # Particle encoding tests
├── test_xgboost_builder.py     # Builder function tests
├── test_registry.py            # Model registry tests
├── test_experiment_tracker.py  # Tracker tests
├── test_generic_ipso.py        # Generic IPSO tests
└── conftest.py                 # Pytest fixtures
```

### 8.2 Integration Tests

**Scenarios:**
1. **Full XGBoost Pipeline**
   - Load data → Train with IPSO → Save model → Evaluate
   - Expected time: < 5 minutes (small dataset, 5 particles, 5 iterations)

2. **Model Comparison**
   - Train LSTM and XGBoost on same data
   - Compare metrics
   - Verify winner selection

3. **Resume from Checkpoint**
   - Start training → Stop mid-way → Resume
   - Verify results are identical to uninterrupted run

### 8.3 Validation Tests

**Data Validation:**
- [ ] Verify feature matrices have correct shape
- [ ] Verify no NaN/Inf values
- [ ] Verify train/val/test splits don't overlap
- [ ] Verify temporal ordering is preserved

**Model Validation:**
- [ ] Verify predictions are in reasonable range
- [ ] Verify no data leakage (future → past)
- [ ] Verify metrics are computed correctly
- [ ] Verify backtesting respects trading constraints

### 8.4 Performance Benchmarks

**Targets:**
- XGBoost training (1 particle): < 30 seconds
- IPSO iteration (30 particles): < 15 minutes
- Full optimization (30 particles, 50 iterations): < 12 hours
- Metrics computation: < 1 second

---

## 9. Timeline & Dependencies

### 9.1 Implementation Phases

#### Phase 1: Core Infrastructure (Week 1)
**Duration:** 3-5 days  
**Dependencies:** None

Tasks:
- [ ] Task 1.1: Enhance XGBoost model wrapper
- [ ] Task 1.2: Create XGBoost particle encoding
- [ ] Task 1.3: Create XGBoost model builder
- [ ] Unit tests for above

**Deliverables:**
- Working XGBoost model with regression support
- Particle encoding/decoding for 9D space
- Builder function for IPSO integration

---

#### Phase 2: IPSO Integration (Week 1-2)
**Duration:** 2-3 days  
**Dependencies:** Phase 1 complete

Tasks:
- [ ] Task 2.1: Create generic IPSO wrapper
- [ ] Task 2.2: Fix `run_xgboost.py` script
- [ ] Integration tests

**Deliverables:**
- Generic IPSO that works with any model
- Working XGBoost optimization script
- End-to-end test passing

---

#### Phase 3: Metrics & Tracking (Week 2)
**Duration:** 3-4 days  
**Dependencies:** Phase 2 complete

Tasks:
- [ ] Task 3.1: Create experiment tracker
- [ ] Task 3.2: Create model registry
- [ ] Task 3.3: Create unified training script
- [ ] Database schema and queries

**Deliverables:**
- SQLite database with all experiments
- Model registry with LSTM and XGBoost
- Single `train_model.py` script
- Query tools for analysis

---

#### Phase 4: Code Cleanup (Week 2-3)
**Duration:** 2-3 days  
**Dependencies:** Phase 3 complete

Tasks:
- [ ] Fix all P0 and P1 bugs
- [ ] Refactor duplicate code
- [ ] Add input validation
- [ ] Standardize logging
- [ ] Add type hints
- [ ] Improve docstrings

**Deliverables:**
- Clean, maintainable codebase
- All critical bugs fixed
- Consistent code style

---

#### Phase 5: Testing & Documentation (Week 3)
**Duration:** 2-3 days  
**Dependencies:** Phase 4 complete

Tasks:
- [ ] Complete unit test suite
- [ ] Complete integration tests
- [ ] Add performance benchmarks
- [ ] Update all documentation
- [ ] Create usage examples

**Deliverables:**
- 90%+ test coverage
- Comprehensive documentation
- Working examples

---

### 9.2 Dependency Graph

```
Phase 1 (Core Infrastructure)
    ↓
Phase 2 (IPSO Integration)
    ↓
Phase 3 (Metrics & Tracking)
    ↓
Phase 4 (Code Cleanup)
    ↓
Phase 5 (Testing & Documentation)
```

**Critical Path:**
1. XGBoost particle encoding (Task 1.2)
2. Generic IPSO (Task 2.1)
3. Experiment tracker (Task 3.1)
4. Bug fixes (Section 7)
5. Tests (Section 8)

---

### 9.3 Risk Assessment

#### High Risk
**Risk:** XGBoost performance is significantly worse than LSTM  
**Mitigation:** 
- Ensure feature engineering is appropriate for XGBoost
- Add lag features explicitly
- Tune PSO parameters (more particles/iterations)
- Consider ensemble approach

**Risk:** IPSO doesn't converge for 9D space  
**Mitigation:**
- Increase swarm size to 40-50 particles
- Increase iterations to 60-80
- Add adaptive swarm size based on dimensionality
- Consider dimension reduction techniques

#### Medium Risk
**Risk:** Database becomes bottleneck for large-scale experiments  
**Mitigation:**
- Use batch inserts
- Add database indices
- Consider PostgreSQL for production
- Implement caching layer

**Risk:** Results storage grows too large  
**Mitigation:**
- Compress model files
- Implement retention policy
- Store only best N models per ticker
- Use cloud storage for archival

#### Low Risk
**Risk:** Code refactoring introduces bugs  
**Mitigation:**
- Comprehensive test suite
- Gradual refactoring
- Code review process
- Git branching strategy

---

## 10. Success Metrics

### 10.1 Implementation Success

- [ ] XGBoost model trains successfully with IPSO
- [ ] All P0 and P1 bugs fixed
- [ ] Test coverage > 90% for new code
- [ ] Documentation complete and accurate
- [ ] Code passes linting (flake8, mypy)

### 10.2 Performance Success

- [ ] XGBoost achieves comparable metrics to LSTM
- [ ] IPSO converges within 50 iterations
- [ ] Full optimization completes in < 12 hours
- [ ] No memory leaks or crashes

### 10.3 Usability Success

- [ ] Single command to train any model
- [ ] Easy to query and compare results
- [ ] Clear error messages
- [ ] Comprehensive examples and tutorials

---

## 11. Appendix

### 11.1 File Checklist

**New Files to Create:**
- [ ] `src/optimizer/xgboost_particle.py`
- [ ] `src/models/xgboost_builder.py`
- [ ] `src/optimizer/generic_ipso.py`
- [ ] `src/utils/experiment_tracker.py`
- [ ] `src/models/registry.py`
- [ ] `scripts/train_model.py`
- [ ] `tests/test_xgboost_model.py`
- [ ] `tests/test_xgboost_particle.py`
- [ ] `tests/test_xgboost_builder.py`
- [ ] `tests/test_registry.py`
- [ ] `tests/test_experiment_tracker.py`
- [ ] `tests/test_generic_ipso.py`
- [ ] `tests/test_integration.py`

**Files to Modify:**
- [ ] `src/models/xgboost_model.py`
- [ ] `scripts/run_xgboost.py`
- [ ] `config/default_config.yaml`
- [ ] `requirements.txt` (if needed)
- [ ] `README.md` (update with XGBoost info)

**Files to Deprecate:**
- [ ] `scripts/run_model_train.py` (replaced by `train_model.py`)
- [ ] `scripts/run_xgboost.py` (replaced by `train_model.py`)

---

### 11.2 Configuration Reference

**Complete XGBoost Config:**

```yaml
xgboost:
  # IPSO search space
  n_estimators:
    min: 50
    max: 500
  max_depth:
    min: 3
    max: 10
  learning_rate:
    min: 0.001
    max: 0.3
    scale: "log"
  subsample:
    min: 0.5
    max: 1.0
  colsample_bytree:
    min: 0.5
    max: 1.0
  min_child_weight:
    min: 1
    max: 10
  gamma:
    min: 0.0
    max: 1.0
  reg_alpha:
    min: 0.0
    max: 1.0
  reg_lambda:
    min: 0.0
    max: 2.0
  
  # Fixed parameters
  tree_method: "hist"
  max_bin: 128
  early_stopping_rounds: 50
  task: "regression"
  
  # PSO settings (may need adjustment for 9D space)
  pso_particles: 40  # Increased from 30 for higher dimensionality
  pso_iterations: 60  # Increased from 50 for better convergence
```

---

### 11.3 Example Usage

**Train XGBoost with IPSO:**
```bash
python scripts/train_model.py \
    --model xgboost \
    --ticker AAPL \
    --config config/default_config.yaml \
    --features-dir data/features \
    --output-dir results \
    --seed 42
```

**Train LSTM with IPSO:**
```bash
python scripts/train_model.py \
    --model lstm \
    --ticker AAPL \
    --config config/default_config.yaml \
    --features-dir data/features \
    --output-dir results \
    --seed 42
```

**Query best models:**
```python
from src.utils.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()

# Get best XGBoost model for AAPL
best_xgb = tracker.get_best_run("AAPL", "xgboost", metric="sharpe")
print(f"Best XGBoost Sharpe: {best_xgb['best_metric_value']:.4f}")
print(f"Params: {best_xgb['params']}")

# Compare all models
comparison = tracker.compare_models("AAPL")
for model_type, metrics in comparison.items():
    print(f"{model_type}: Sharpe={metrics['sharpe']:.4f}, RMSE={metrics['rmse']:.4f}")
```

---

## Document Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-04-06 | System | Initial document creation |

---

**End of Document**
