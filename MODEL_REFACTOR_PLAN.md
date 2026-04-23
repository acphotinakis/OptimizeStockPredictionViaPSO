# Model Layer Refactor - Implementation Plan

**Date:** 2026-04-22  
**Status:** PLANNING PHASE  
**Authority:** MODEL_SIMPLIFICATION_AUDIT.md

---

## Dependencies Analysis

### Current Imports That Must Be Preserved (or Updated):

1. **From `pipelines/train_baseline_lstm.py`:**
   ```python
   from src.models import LSTMModel, LSTMTrainer, build_lstm_windows, set_seeds
   ```

2. **From `pipelines/train_pso_lstm.py`:**
   ```python
   from src.models import PSOLSTMModel, PSOLSTMTrainer
   ```

3. **From `pipelines/train_xgboost.py`:**
   ```python
   from src.models import XGBoostModel, XGBoostTrainer, build_xgboost_lag_features
   ```

4. **From `src/evaluation/model_loader.py`:**
   ```python
   from src.models import PSOLSTMModel, LSTMModel, XGBoostModel, build_lstm_windows
   ```

5. **From `src/evaluation/walk_forward_pso.py`:**
   ```python
   from src.models import build_lstm_windows
   from src.models.pso_lstm_model import PSOLSTMModel
   from src.models.pso_lstm_trainer import PSOLSTMTrainer
   ```

---

## Refactor Strategy

### Phase 1: Create New Unified Structure ✓

**Status:** COMPLETE

- [x] Create `src/models/base.py` with BaseModel and BaseTrainer protocols

### Phase 2: Move Data Utilities

**Target Location:** `src/data/`

1. **Create `src/data/windowing.py`:**
   - Move `build_lstm_windows()` from `src/models/utils.py`
   - Move `build_xgboost_lag_features()` from `src/models/xgboost_trainer.py`

2. **Update all imports:**
   - `pipelines/train_baseline_lstm.py`
   - `pipelines/train_xgboost.py`
   - `src/evaluation/model_loader.py`
   - `src/evaluation/walk_forward_pso.py`

### Phase 3: Create Unified LSTM Implementation

**New Structure:**
```
src/models/
├── base.py (✓ done)
├── lstm_network.py (new - PyTorch nn.Module)
├── lstm_model.py (new - Model wrapper)
├── lstm_trainer.py (new - Unified trainer)
```

**Source Files to Consolidate:**
- `baseline_lstm_model.py` → Extract `LSTMNetwork` → `lstm_network.py`
- `baseline_lstm_model.py` → Extract `LSTMModel` → `lstm_model.py`
- `baseline_lstm_trainer.py` + `pso_lstm_trainer.py` → Merge → `lstm_trainer.py`

**Key Changes:**
1. **Remove `train()` method from LSTMModel** - Trainers own training
2. **Fix early stopping** - Use validation loss consistently
3. **Config-driven interface** - Accept `config: Dict` not 8 kwargs
4. **Remove duplication** - One trainer serves both baseline and PSO use cases

### Phase 4: Refactor XGBoost

**New Structure:**
```
src/models/
├── xgboost_model.py (simplified)
├── xgboost_trainer.py (simplified)
```

**Changes:**
1. **Remove `train()` from XGBoostModel** - Trainer owns training
2. **Move lag feature builder** to `src/data/windowing.py`
3. **Standardize trainer interface** - Match LSTM trainer pattern

### Phase 5: Update src/models/__init__.py

**New Exports:**
```python
# Unified LSTM
from .lstm_model import LSTMModel
from .lstm_network import LSTMNetwork
from .lstm_trainer import LSTMTrainer

# XGBoost
from .xgboost_model import XGBoostModel
from .xgboost_trainer import XGBoostTrainer

# Base abstractions
from .base import BaseModel, BaseTrainer

# Utilities (backward compat - will point to src.data)
from src.data.windowing import build_lstm_windows, build_xgboost_lag_features
from src.utils.random import set_seeds

__all__ = [
    # LSTM
    "LSTMModel",
    "LSTMNetwork",
    "LSTMTrainer",
    # XGBoost
    "XGBoostModel",
    "XGBoostTrainer",
    # Base
    "BaseModel",
    "BaseTrainer",
    # Utils (backward compat)
    "build_lstm_windows",
    "build_xgboost_lag_features",
    "set_seeds",
]
```

### Phase 6: Update Pipeline Scripts

**Files to Update:**

1. **`pipelines/train_baseline_lstm.py`:**
   ```python
   # OLD
   from src.models import LSTMModel, LSTMTrainer, build_lstm_windows, set_seeds
   
   # NEW (no changes needed - imports remain compatible!)
   from src.models import LSTMModel, LSTMTrainer, build_lstm_windows, set_seeds
   ```

2. **`pipelines/train_pso_lstm.py`:**
   ```python
   # OLD
   from src.models import PSOLSTMModel, PSOLSTMTrainer
   
   # NEW (uses unified LSTM)
   from src.models import LSTMModel, LSTMTrainer
   ```
   - Update code to use `LSTMModel` instead of `PSOLSTMModel`
   - Update code to use `LSTMTrainer` instead of `PSOLSTMTrainer`

3. **`pipelines/train_xgboost.py`:**
   ```python
   # OLD
   from src.models import XGBoostModel, XGBoostTrainer, build_xgboost_lag_features
   
   # NEW (no changes - backward compat maintained)
   from src.models import XGBoostModel, XGBoostTrainer, build_xgboost_lag_features
   ```

### Phase 7: Update Evaluation Scripts

1. **`src/evaluation/model_loader.py`:**
   - Update `pso_lstm` case to use `LSTMModel` instead of `PSOLSTMModel`
   - No changes needed for `lstm_baseline` case
   - Update import from `src.models import build_lstm_windows` (backward compat)

2. **`src/evaluation/walk_forward_pso.py`:**
   ```python
   # OLD
   from src.models.pso_lstm_model import PSOLSTMModel
   from src.models.pso_lstm_trainer import PSOLSTMTrainer
   
   # NEW
   from src.models import LSTMModel, LSTMTrainer
   ```

### Phase 8: Delete Obsolete Files

**Files to Delete:**
- `src/models/pso_lstm_model.py`
- `src/models/pso_lstm_trainer.py`
- `src/models/baseline_lstm_model.py`
- `src/models/baseline_lstm_trainer.py`
- `src/models/utils.py` (content moved to src/data/ and src/utils/)

**Do NOT delete:**
- `src/models/__init__.py` (updated, not deleted)
- `src/models/xgboost_model.py` (refactored, not deleted)
- `src/models/xgboost_trainer.py` (refactored, not deleted)

---

## Critical Compatibility Requirements

### 1. Model Save/Load Format

**MUST remain compatible with existing saved models:**
- LSTM models use PyTorch `.pt` or `.h5` format
- XGBoost models use pickle `.pkl` format
- Model config JSON files must remain readable

### 2. Trainer Interface Changes

**Old Interface (baseline_lstm_trainer.py):**
```python
def train(
    self,
    X_train, y_train, X_val, y_val,
    lstm_units_1, lstm_units_2, dropout_rate,
    learning_rate, epochs, batch_size, patience,
    shuffle=False
)
```

**New Interface (unified lstm_trainer.py):**
```python
def train(
    self,
    X_train, y_train, X_val, y_val,
    config: Dict,  # Contains all hyperparameters
    seed: int = 42
)
```

**Migration Strategy:**
- Keep old interface as **deprecated** method for one version
- Add new config-based interface as primary
- Update all callers to use new interface

### 3. History Schema

**Must remain consistent:**
```python
{
    "train_loss": List[float],
    "val_loss": List[float]
}
```

---

## Testing Strategy

### Unit Tests
1. Test LSTM model can load old saved models
2. Test XGBoost model can load old saved models
3. Test trainer produces same results as before
4. Test windowing functions produce same output

### Integration Tests
1. Run `train_baseline_lstm.py` end-to-end
2. Run `train_pso_lstm.py` end-to-end (with updated imports)
3. Run `train_xgboost.py` end-to-end
4. Run `run_backtest.py` with all model types
5. Run `walk_forward_evaluation.py`

### Regression Tests
1. Compare metrics before/after refactor on same data
2. Verify saved models are still loadable
3. Verify no broken imports in any pipeline

---

## Risk Mitigation

### High Risk
- **Model loading compatibility** - Existing saved models must load
  - Mitigation: Test with real saved models before deployment

### Medium Risk
- **PSO pipeline updates** - `train_pso_lstm.py` needs significant changes
  - Mitigation: Update carefully, test thoroughly

### Low Risk
- **Import updates** - Some scripts need import changes
  - Mitigation: Backward-compatible exports in `__init__.py`

---

## Code Reduction Estimate

| Component | Before (LOC) | After (LOC) | Reduction |
|-----------|--------------|-------------|-----------|
| LSTM (4 files) | ~1,800 | ~1,000 | **44%** |
| XGBoost (2 files) | ~450 | ~350 | **22%** |
| Utils | ~150 | ~50 | **67%** |
| Base abstractions | 0 | ~120 | +120 |
| **Total** | **~2,400** | **~1,520** | **37%** |

**Target: 40-50% reduction ✓ Achievable**

---

## Implementation Order

1. ✓ Create `src/models/base.py`
2. ⏳ Create `src/data/windowing.py` and move utilities
3. ⏳ Create unified LSTM files (network, model, trainer)
4. ⏳ Update `src/models/__init__.py`
5. ⏳ Update pipeline scripts
6. ⏳ Update evaluation scripts
7. ⏳ Test everything
8. ⏳ Delete obsolete files

---

**Status:** Ready to implement Phase 2
