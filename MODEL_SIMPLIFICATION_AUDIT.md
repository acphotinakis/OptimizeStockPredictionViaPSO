# Model Simplification Audit Report
## `src/models/` Layer — Code Clarity, Maintainability & Simplification

**Date:** 2026-04-23  
**Scope:** `baseline_lstm_model.py`, `baseline_lstm_trainer.py`, `pso_lstm_model.py`, `pso_lstm_trainer.py`, `xgboost_model.py`, `xgboost_trainer.py`, `utils.py`  
**Objective:** Identify unnecessary complexity, duplication, and architectural confusion.  

---

## 1. Executive Summary

The `src/models/` layer is ~60–70% redundant. The core problems are:

1. **Phantom PSO/Baseline dichotomy.** Four LSTM classes exist (`LSTMNetwork`, `PSOLSTMNetwork`, `LSTMModel`, `PSOLSTMModel`) where two would suffice. The "PSO" variants are copy-pasted clones with minor, broken differences.
2. **Training logic lives in two places.** Both `*Model` wrappers and `*Trainer` classes contain complete training loops, making it ambiguous which entry point is canonical.
3. **Broken early stopping.** `PSOLSTMTrainer.train()` declares early stopping but never updates the improvement counter, rendering it non-functional.
4. **No shared abstractions.** There are no base classes or protocols, so every model/trainer invents its own API surface.
5. **XGBoost structural isolation.** XGBoost follows a different pattern than PyTorch models, and its trainer is a pass-through wrapper that adds no value.

**Estimated code reduction:** 40–50% of model-layer LOC without losing functionality.

---

## 2. Duplication Map

| Duplicated Component | Primary Location | Duplicated In | Extent | Notes |
|----------------------|------------------|---------------|--------|-------|
| **LSTM Network (PyTorch `nn.Module`)** | `baseline_lstm_model.py` — `LSTMNetwork` (lines 29–116) | `pso_lstm_model.py` — `PSOLSTMNetwork` (lines 29–116) | **100%** | Identical `__init__`, `forward`, layer definitions, and `save()`. Same docstrings. |
| **Model Wrapper** | `baseline_lstm_model.py` — `LSTMModel` (lines 119–365) | `pso_lstm_model.py` — `PSOLSTMModel` (lines 119–365) | **~95%** | Same state (`history`, `best_weights`, `best_epoch`), same `_validate_inputs`, same `build_model`, same `predict`, same `evaluate`, same `save_weights`/`load_weights`. Only difference: PSO version adds `tqdm` to the training loop. |
| **Training Loop** | `baseline_lstm_trainer.py` — `LSTMTrainer.train()` (lines 56–181) | `pso_lstm_trainer.py` — `PSOLSTMTrainer.train()` (lines 56–148) | **~80%** | Same optimizer (Adam), same loss (MSELoss), same DataLoader setup (`shuffle=False`), same device handling. PSO version strips out the **entire validation phase** and breaks early stopping. |
| **Evaluation Logic** | `baseline_lstm_trainer.py` — `evaluate()` (lines 183–246) | `pso_lstm_trainer.py` — `evaluate()` (lines 150–213) | **~95%** | Identical metric computation: MSE, MAE, RMSE, R², directional accuracy, inverse-transform logic. |
| **Input Validation** | `baseline_lstm_model.py` — `_validate_inputs()` (lines 256–300) | `pso_lstm_model.py` — `_validate_inputs()` (lines 256–300) | **~100%** | Same 3D tensor checks, dtype checks, NaN checks, shape assertions. |
| **XGBoost Training** | `xgboost_model.py` — `train()` (lines 120–220) | `xgboost_trainer.py` — `train()` (lines 56–89) | **~100%** | Trainer instantiates `XGBoostModel` and immediately delegates to `model.train()`. No additional behavior. |

---

## 3. Architecture Problems

### 3.1 Training Logic Lives in Two Places (Critical Design Flaw)

Both `*Model` wrappers and `*Trainer` classes contain complete, independent training loops. This violates the single-responsibility principle and creates maintenance drift.

- **`LSTMModel.train()`** (`baseline_lstm_model.py:183`) builds DataLoaders, runs the epoch loop, implements early stopping, and restores best weights.
- **`LSTMTrainer.train()`** (`baseline_lstm_trainer.py:56`) does the exact same thing.
- The canonical training script (`train_baseline_lstm.py`) uses `LSTMTrainer`, making `LSTMModel.train()` **dead code** that is nonetheless fully maintained in parallel.

**Same issue for XGBoost:**
- `XGBoostModel.train()` (`xgboost_model.py:120`) owns the full `XGBRegressor.fit()` call, eval-set tracking, and feature-importance extraction.
- `XGBoostTrainer.train()` (`xgboost_trainer.py:56`) is a thin wrapper that instantiates `XGBoostModel` and delegates.

### 3.2 False PSO / Baseline Dichotomy

The codebase treats "PSO LSTM" and "Baseline LSTM" as different architectures. They are not.

- The **only** semantic difference is the *source of hyperparameters* (PSO search vs. fixed config). The network graph, loss function, optimizer, and evaluation protocol are identical.
- Despite this, there are **4 separate classes** and **2 separate trainers**.
- `PSOLSTMTrainer` is an incomplete copy of `LSTMTrainer`: validation logic was stripped out, but early-stopping variables were left behind in a broken state.

### 3.3 Broken Early Stopping in `PSOLSTMTrainer` (Critical Bug)

`PSOLSTMTrainer.train()` (`pso_lstm_trainer.py:56`) declares early stopping in its docstring but the implementation is non-functional:

- `best_weights` is initialized to `None` and **never updated**.
- `epochs_no_improve` is initialized to `0` and **never incremented**.
- The condition `if epochs_no_improve >= patience` (line ~137) will **never** evaluate to `True` for any `patience > 0`, so the loop always runs to `epochs`.
- There is **no validation set evaluation** inside the loop, so even if the counter were fixed, there would be no metric to judge improvement against.

This appears to be a rushed copy-paste where validation code was deleted but the early-stopping skeleton was left intact.

### 3.4 Inconsistent Trainer Interfaces

There is no common contract. Each trainer invents its own signature:

| Trainer | `train()` Signature | Validation Required? | Returns |
|---------|---------------------|----------------------|---------|
| `LSTMTrainer` | `(X_train, y_train, X_val, y_val, lstm_units_1, lstm_units_2, dropout_rate, learning_rate, epochs, batch_size, patience, shuffle=False)` | Yes | `(model, history)` |
| `PSOLSTMTrainer` | `(X_train, y_train, lstm_units_1, lstm_units_2, dropout_rate, learning_rate, epochs, batch_size, patience)` | **No** | `(model, history)` |
| `XGBoostTrainer` | `(X_train, y_train, X_val, y_val, feature_names=None)` | Yes | `(model, history)` |

Hyperparameters are passed as **individual kwargs** in LSTM trainers but as a **config dict** in XGBoost. This inconsistency forces callers to know implementation details.

### 3.5 XGBoost Structural Isolation

XGBoost uses a completely different organizational pattern than PyTorch models:
- No `Network` / `Model` split (unnecessary for sklearn-compatible estimators).
- `XGBoostTrainer` is redundant; it adds no behavior beyond logging and delegation.
- `build_xgboost_lag_features` is a pure data-transformation function stranded inside the trainer file.

### 3.6 Misplaced & Redundant Utilities

`utils.py` contains logic that does not belong in the model layer:
- `build_lstm_windows` — Data preprocessing / temporal windowing. Belongs in `src/data/` or the feature pipeline.
- `build_xgboost_lag_features` — Already in `xgboost_trainer.py`, but also belongs in the data layer.
- `save_model_weights` / `load_model_weights` — PyTorch-specific I/O with generic names. Belongs inside `LSTMModel` or a dedicated PyTorch mixin.

---

## 4. Simplification Strategy

### 4.1 Merge PSO and Baseline LSTM → Delete 2 Files
- **Delete** `pso_lstm_model.py` and `pso_lstm_trainer.py`.
- **Keep** `baseline_lstm_model.py` and `baseline_lstm_trainer.py` as the single canonical LSTM implementation.
- **Rename** `baseline_lstm_*` → `lstm_*` to remove the misleading "baseline" prefix.
- The PSO orchestration layer should import the **same** `LSTMTrainer` and `LSTMModel`; it does not need its own model variants.

### 4.2 Remove Training Logic from Model Wrappers
- **Strip** `LSTMModel.train()` entirely. The model wrapper should only:
  - Build / hold the `nn.Module`.
  - Provide `predict()`, `evaluate()`, `save()`, `load()`.
- **Strip** `XGBoostModel.train()` similarly, or eliminate `XGBoostTrainer` and let `XGBoostModel` be the self-contained owner of training (sklearn-style). **Do not keep both.**

### 4.3 Unify LSTM Trainer into a Single Correct Implementation
Create one `LSTMTrainer` that:
- Accepts hyperparameters via a `config: Dict` (like XGBoost) instead of 8 individual kwargs.
- **Always** requires validation data and performs validation-based early stopping.
- Is usable by both single-fit scripts and PSO loops without code duplication.

### 4.4 Introduce Minimal Base Abstractions
Create lightweight protocols (or ABCs) in a new `base.py`:
- `BaseModel`: enforces `predict()`, `evaluate()`, `save()`, `load()`.
- `BaseTrainer`: enforces `train(X_train, y_train, X_val, y_val, **kwargs) -> (model, history)`.

This forces XGBoost and LSTM to expose the same public API even though their internals differ.

### 4.5 Relocate Utilities
- Move `build_lstm_windows` → `src/data/temporal.py`.
- Move `build_xgboost_lag_features` → `src/data/tabular.py`.
- Keep `set_seeds` in a general `src/utils/random.py`.
- Delete `save_model_weights` / `load_model_weights`; use `torch.save` / `torch.load` directly inside `LSTMModel`.

### 4.6 Simplify XGBoost Layer
**Recommendation:** Keep `XGBoostTrainer` as the canonical training entry point and strip `XGBoostModel.train()` down to a thin `fit()` wrapper or remove it entirely. Trainers should own training; models should own architecture and inference.

---

## 5. Proposed Clean Architecture

```
src/models/
├── base.py                 # BaseModel & BaseTrainer protocols
├── lstm/
│   ├── network.py          # LSTMNetwork (nn.Module only)
│   ├── model.py            # LSTMModel (wrapper: build, predict, evaluate, save/load)
│   └── trainer.py          # LSTMTrainer (single canonical training loop)
├── xgboost/
│   ├── model.py            # XGBoostModel (thin wrapper around xgb.XGBRegressor)
│   └── trainer.py          # XGBoostTrainer (owns training + lag feature building)
└── utils.py                # set_seeds ONLY (or delete and use src/utils/)
```

### Standardized Interface

```python
# src/models/base.py
from typing import Protocol, Dict, Tuple, Any
import numpy as np

class BaseModel(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

class BaseTrainer(Protocol):
    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        **kwargs: Any
    ) -> Tuple[BaseModel, Dict]: ...
```

### Unified LSTM Trainer

```python
# src/models/lstm/trainer.py
class LSTMTrainer:
    def train(
        self,
        X_train, y_train, X_val, y_val,
        config: Dict,          # {lstm_units_1, lstm_units_2, dropout_rate, ...}
        seed: int = 42
    ) -> Tuple[LSTMModel, Dict[str, List[float]]]:
        # Single, correct implementation of:
        # - DataLoader creation (shuffle=False)
        # - Training + validation loop
        # - Early stopping on validation loss
        # - Best weight restoration
```

### What PSO Scripts Should Look Like

```python
# In PSO orchestration layer (NOT in src/models/)
from src.models.lstm.trainer import LSTMTrainer

trainer = LSTMTrainer(seed=seed)
for particle in swarm:
    config = decode_particle_to_dict(particle)
    model, history = trainer.train(
        X_train, y_train, X_val, y_val, config=config
    )
    fitness = min(history["val_loss"])
```

---

## 6. Priority Fix List

### CRITICAL — Must Fix Before Production

| # | Issue | Action | Files Affected |
|---|-------|--------|----------------|
| 1 | **Broken early stopping in PSOLSTMTrainer** | Delete `PSOLSTMTrainer`; use unified `LSTMTrainer` which has working validation-based early stopping. | `pso_lstm_trainer.py` |
| 2 | **Phantom PSO model classes** | Delete `PSOLSTMModel` and `PSOLSTMNetwork`. PSO should use canonical `LSTMModel` / `LSTMNetwork`. | `pso_lstm_model.py` |
| 3 | **Dual ownership of training logic** | Remove `train()` from `LSTMModel` and `XGBoostModel`. Trainers are the single owner of training loops. | `baseline_lstm_model.py`, `xgboost_model.py` |

### HIGH — Should Fix in Next Sprint

| # | Issue | Action | Files Affected |
|---|-------|--------|----------------|
| 4 | **Unify LSTM trainers** | Merge `LSTMTrainer` and `PSOLSTMTrainer` into one class accepting a `config` dict. | `baseline_lstm_trainer.py`, `pso_lstm_trainer.py` |
| 5 | **Standardize trainer signatures** | All trainers should accept `(X_train, y_train, X_val, y_val, config, seed)` and return `(model, history)`. | All `*_trainer.py` |
| 6 | **Relocate data utilities** | Move `build_lstm_windows` and `build_xgboost_lag_features` to `src/data/`. | `utils.py`, `xgboost_trainer.py` |
| 7 | **Introduce base protocols** | Add `BaseModel` and `BaseTrainer` to enforce consistent APIs across LSTM and XGBoost. | New `base.py` |
| 8 | **Simplify XGBoost wrapper** | Remove either `XGBoostTrainer` or `XGBoostModel.train()` to eliminate the redundant delegation layer. | `xgboost_model.py`, `xgboost_trainer.py` |

### OPTIONAL — Nice to Have

| # | Issue | Action | Files Affected |
|---|-------|--------|----------------|
| 9 | **Rename baseline files** | Rename `baseline_lstm_model.py` → `lstm_model.py` and `baseline_lstm_trainer.py` → `lstm_trainer.py`. | `baseline_lstm_*.py` |
| 10 | **Remove generic weight I/O utils** | Delete `save_model_weights` / `load_model_weights`; use native `torch.save` inside `LSTMModel.save()`. | `utils.py` |
| 11 | **Standardize history schema** | Ensure all trainers return `Dict[str, List[float]]` with identical keys (`train_loss`, `val_loss`). | All trainers |
| 12 | **Remove dead code** | `LSTMModel.build_model()` and `PSOLSTMModel.build_model()` are public but only used internally; consider making them private (`_build_model`). | `baseline_lstm_model.py`, `pso_lstm_model.py` |

---

## Appendix: Line-Level Evidence

### A.1 Identical Networks
`baseline_lstm_model.py:29-116` and `pso_lstm_model.py:29-116` are character-for-character identical (except class name).

### A.2 Broken Early Stopping
`pso_lstm_trainer.py:56-148`:
```python
best_weights = None       # Never updated
best_epoch = 0            # Never updated
epochs_no_improve = 0     # Never incremented

# ... training loop ...

if epochs_no_improve >= patience:   # Always False for patience > 0
    logger.info(f"Early stopping triggered at epoch {epoch+1}")
    break
```

### A.3 Delegation-Only XGBoost Trainer
`xgboost_trainer.py:56-89`:
```python
def train(self, X_train, y_train, X_val, y_val, feature_names=None):
    model = XGBoostModel(self.config, seed=self.seed)
    model.train(X_train, y_train, X_val, y_val, feature_names)  # All work done here
    history = model.evals_result if model.evals_result else {}
    return model, history
```

### A.4 Misplaced Lag Feature Builder
`xgboost_trainer.py:92-155` (`build_xgboost_lag_features`) is a pure NumPy transformation with no dependency on XGBoost. It belongs in the data pipeline.

---

*End of Audit*
