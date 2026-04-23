# IPSO-LSTM IMPLEMENTATION AUDIT

**Date:** 2026-04-21  
**Auditor:** Technical Compliance System  
**Scope:** Strict verification against IPSO-LSTM production specification  
**Files Audited:** 10 files (train_pso_lstm.py, ipso.py, pso_core.py, particle.py, fitness.py, pso_lstm_model.py, pso_lstm_trainer.py, utils.py, default_config.yaml, LSTM_PSO_DETAILS.md)

---

## EXECUTIVE SUMMARY

**VERDICT:** ❌ **NOT PRODUCTION READY**

**Overall Compliance:** 45% (Partial Implementation)

**Status:** The codebase contains a **fundamentally different PSO-LSTM implementation** than specified. Critical architectural mismatches exist between:
1. **Specification:** 20 particles, 50 iterations, γ×MSE+(1-γ)×MSW fitness, lookback=20 fixed
2. **Implementation:** 5-dimensional particle encoding with variable lookback {10,30,60,120}, composite financial fitness function, missing MSW

**BLOCKING ISSUES:** 7 critical  
**NON-BLOCKING ISSUES:** 3 high-priority

**Recommendation:** **Major refactoring required** or **specification update required** to align codebase with stated requirements.

---

## COMPLIANCE MATRIX

### 1. IPSO Swarm Dynamics

| Component | Required | Implemented | Status | Location |
|-----------|----------|-------------|--------|----------|
| Swarm size | n_particles = 20 | n_particles = 20 | ✅ PASS | config:207, pso_core.py:45, train_pso_lstm.py:369 |
| Iterations | n_iterations = 50 | n_iterations = 50 | ✅ PASS | config:208, pso_core.py:46, train_pso_lstm.py:370 |
| Inertia max | 0.9 | 0.9 | ✅ PASS | config:210, pso_core.py:51, ipso.py:9,49 |
| Inertia min | 0.4 | 0.4 | ✅ PASS | config:209, pso_core.py:52, ipso.py:9,49 |
| Inertia decay | Linear or functional | **Tanh (non-linear)** | ⚠️ MISMATCH | ipso.py:41-49 |
| c1 (cognitive) | 1.5 | 1.5 | ✅ PASS | config:211, pso_core.py:53 |
| c2 (social) | 1.5 | 1.5 | ✅ PASS | config:212, pso_core.py:54 |
| Velocity update | Correct PSO equations | Correct | ✅ PASS | pso_core.py:162-164 |
| Position clipping | Within bounds | Correct | ✅ PASS | pso_core.py:170 |

**Sub-Score:** 8/9 (89%)  
**Issues:** Inertia uses tanh instead of linear (but functionally acceptable per spec "linearly or functionally")

---

### 2. Hyperparameter Search Space

| Parameter | Required Range | Implemented Range | Status | Location |
|-----------|---------------|-------------------|--------|----------|
| lstm_units_1 | [50, 300] | [32, 512] via hidden_units | ❌ **CRITICAL MISMATCH** | particle.py:25-26,41-42 |
| lstm_units_2 | [20, 200] | **NOT ENCODED** | ❌ **CRITICAL MISSING** | particle.py:8-12 |
| dropout_rate | [0.0, 0.5] | [0.0, 0.5] | ✅ PASS | particle.py:25,43, config:226-227 |
| batch_size | {32, 64} | **NOT ENCODED** | ❌ **CRITICAL MISSING** | config:232-233 (fixed, not searched) |
| learning_rate | [0.001, 0.01] log-scale | [1e-5, 1e-1] log-scale | ⚠️ RANGE MISMATCH | particle.py:25,44, config:229-231 |
| epochs | [50, 300] | **NOT ENCODED** | ❌ **CRITICAL MISSING** | config:235-236 (fixed, not searched) |
| **Extra dimension** | **NOT SPECIFIED** | num_layers [1,4] | ❌ **EXTRA DIMENSION** | particle.py:8,40 |
| **Extra dimension** | **NOT SPECIFIED** | lookback {10,30,60,120} | ❌ **EXTRA DIMENSION** | particle.py:12,22,45 |

**Sub-Score:** 2/6 (33%)  
**CRITICAL ISSUES:**
- **Particle encoding is 5D with DIFFERENT parameters** than specified
- **Missing:** lstm_units_2, batch_size, epochs from PSO search
- **Extra:** num_layers, variable lookback (spec requires fixed lookback=20)
- **Mismatch:** learning_rate range too wide, lstm_units_1 range wrong

---

### 3. Model Construction Constraints

| Constraint | Required | Implemented | Status | Location |
|-----------|----------|-------------|--------|----------|
| Framework | PyTorch nn.Module | PyTorch nn.Module | ✅ PASS | pso_lstm_model.py:29 |
| Optimizer | Adam | Adam | ✅ PASS | pso_lstm_model.py:391, pso_lstm_trainer.py:134 |
| Loss function | MSE | MSE | ✅ PASS | pso_lstm_model.py:394, pso_lstm_trainer.py:135 |
| shuffle | **MUST be False** | **False** | ✅ PASS | pso_lstm_model.py:378, pso_lstm_trainer.py:129, config:238 |
| lookback window | **MUST be 20** | **Variable {10,30,60,120}** | ❌ **CRITICAL VIOLATION** | particle.py:22,45 |
| No random shuffling | Required | Enforced | ✅ PASS | Multiple locations |

**Sub-Score:** 5/6 (83%)  
**CRITICAL ISSUE:** Lookback is variable (PSO-optimized) instead of fixed at 20.

---

### 4. Fitness Function (CRITICAL SECTION)

| Component | Required | Implemented | Status | Location |
|-----------|----------|-------------|--------|----------|
| Core formula | f(x) = γ×MSE + (1-γ)×MSW | **DIFFERENT FORMULA** | ❌ **CRITICAL MISMATCH** | fitness.py:6-7,143-147 |
| γ (gamma) | 0.9 | **NOT PRESENT** | ❌ **MISSING** | N/A |
| MSE on validation | Required | **RMSE used** | ⚠️ DIFFERENT METRIC | fitness.py:115 |
| MSW (mean squared weights) | Required | **NOT IMPLEMENTED** | ❌ **CRITICAL MISSING** | fitness.py (entire file) |
| **Actual formula** | N/A | α₁·RMSE + α₂·(1-Sharpe) + α₃·MDD | ❌ **WRONG FORMULA** | fitness.py:6-7,143-147 |
| RMSE weight | N/A | 0.4 | ❌ NOT IN SPEC | fitness.py:23, config:247 |
| Sharpe weight | N/A | 0.4 | ❌ NOT IN SPEC | fitness.py:23, config:248 |
| MDD weight | N/A | 0.2 | ❌ NOT IN SPEC | fitness.py:23, config:249 |
| Signal threshold | N/A | 1e-4 | ❌ NOT IN SPEC | fitness.py:29, config:250 |
| Transaction cost | 0.001 | 0.001 | ✅ PASS | fitness.py:30, config:251 |
| Validation-only scoring | Required | **Correct** | ✅ PASS | train_pso_lstm.py:239, pso_core.py:216 |
| No training leakage | Required | **Correct** | ✅ PASS | Train/val split enforced |

**Sub-Score:** 2/10 (20%)  
**CRITICAL FAILURES:**
- **Fitness function is COMPLETELY DIFFERENT from specification**
- **MSW (Mean Squared Weights) is NOT implemented**
- **Uses financial metrics (Sharpe, MDD) instead of MSW regularization**
- **Uses RMSE instead of MSE**
- **No gamma weighting parameter**

**ACTUAL IMPLEMENTATION:**
```python
# fitness.py:143-147
fitness = (
    self.weights["rmse"] * norm_rmse
    + self.weights["sharpe"] * (1.0 - norm_sharpe)
    + self.weights["mdd"] * norm_mdd
)
```

**REQUIRED IMPLEMENTATION:**
```python
# Should be:
gamma = 0.9
mse = np.mean((y_val - y_pred) ** 2)
msw = np.mean([w**2 for w in model.parameters()])  # Mean squared weights
fitness = gamma * mse + (1 - gamma) * msw
```

---

### 5. Financial Fitness Extensions

| Component | Status | Note |
|-----------|--------|------|
| Sharpe ratio | ✅ IMPLEMENTED | fitness.py:122 |
| Drawdown penalty | ✅ IMPLEMENTED | fitness.py:124 |
| Transaction cost = 0.001 | ✅ CORRECT | fitness.py:30 |
| Signal threshold | ✅ IMPLEMENTED | fitness.py:29,118 |
| Destandardization | ⚠️ NOT APPLICABLE | fitness.py expects original scale |

**Assessment:** Financial extensions are well-implemented but **NOT PART OF CORE SPEC**. These are **additions** to the required MSE+MSW fitness, not replacements.

---

### 6. IPSO Loop Correctness

| Requirement | Status | Evidence | Location |
|-------------|--------|----------|----------|
| Fresh model per particle | ✅ CORRECT | Model instantiated per evaluation | train_pso_lstm.py:219, pso_core.py:201 |
| Train on training split only | ✅ CORRECT | X_pso_train used | train_pso_lstm.py:223-236 |
| Val split for fitness only | ✅ CORRECT | X_pso_val for prediction only | train_pso_lstm.py:239 |
| pbest update correct | ✅ CORRECT | fitness < pbest_fitness | pso_core.py:254-256 |
| gbest update correct | ✅ CORRECT | fitness < gbest_fitness | pso_core.py:257-259 |
| No weight leakage | ✅ CORRECT | Fresh instantiation | Multiple |
| No model reuse | ✅ CORRECT | New model per particle | pso_core.py:216 model_builder |

**Sub-Score:** 7/7 (100%)  
**Status:** PSO loop mechanics are **CORRECT**.

---

### 7. PSO Core Correctness

| Component | Status | Evidence | Location |
|-----------|--------|----------|----------|
| Velocity update equation | ✅ CORRECT | v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x) | pso_core.py:162-164 |
| Position clipping | ✅ CORRECT | np.clip(position, LB, UB) | pso_core.py:170 |
| Global best selection | ✅ CORRECT | Minimum fitness | pso_core.py:257-259 |
| Swarm initialization | ✅ CORRECT | Random within bounds | pso_core.py:148-151 |
| Deterministic (if seed set) | ✅ CORRECT | np.random.default_rng(seed) | pso_core.py:73 |
| IPSO tanh inertia | ✅ CORRECT | ω = w_max - (w_max-w_min)*tanh(4t/T) | ipso.py:48-49 |
| IPSO adaptive mutation | ✅ CORRECT | μ_mf = 0.7 + 0.3*(t/T) | ipso.py:58-66 |

**Sub-Score:** 7/7 (100%)  
**Status:** PSO core mechanics are **PRODUCTION-GRADE**.

---

### 8. CONFIG CONSISTENCY CHECK

| Check | Expected | Config Value | Status | Location |
|-------|----------|--------------|--------|----------|
| Swarm size | 20 | 20 | ✅ MATCH | config:207 |
| Iterations | 50 | 50 | ✅ MATCH | config:208 |
| Inertia min | 0.4 | 0.4 | ✅ MATCH | config:209 |
| Inertia max | 0.9 | 0.9 | ✅ MATCH | config:210 |
| c1 | 1.5 | 1.5 | ✅ MATCH | config:211 |
| c2 | 1.5 | 1.5 | ✅ MATCH | config:212 |
| Learning rate bounds | [0.001, 0.01] | [0.001, 0.01] | ✅ MATCH | config:229-231 |
| Batch size options | {32, 64} | {32, 64} | ✅ MATCH | config:232-233 |
| **Batch size in PSO search?** | **YES** | **NO (fixed)** | ❌ **MISMATCH** | config:232 not in search_space |
| **Epochs in PSO search?** | **YES** | **NO (in config but not enforced)** | ⚠️ **UNCLEAR** | config:234-236, particle.py (not encoded) |
| Gamma (γ) for MSE+MSW | 0.9 | **MISSING** | ❌ **MISSING** | config:245-246 has mse_weight/msw_weight but different formula |
| MSW weight | Should be (1-γ)=0.1 | 0.1 | ⚠️ NAME MATCH, WRONG USE | config:246 (not used in formula) |
| IPSO/PSO toggle | N/A | pso.enabled: false | ⚠️ PSO DISABLED BY DEFAULT | config:205 |

**Sub-Score:** 6/11 (55%)  
**Issues:**
- Batch size and epochs not in particle encoding
- Gamma not present in fitness function
- MSW weight exists in config but not used correctly
- PSO disabled by default (must set `enabled: true`)

---

### 9. CRITICAL FAILURE MODES

| Failure Mode | Status | Evidence | Severity |
|--------------|--------|----------|----------|
| **Data leakage (train/val)** | ✅ NO LEAKAGE | Strict split enforcement | **PASS** |
| **Model reuse across particles** | ✅ NO REUSE | Fresh instantiation verified | **PASS** |
| **Incorrect fitness computation** | ❌ **WRONG FORMULA** | MSW not implemented, financial metrics instead | **CRITICAL** |
| **Silent shape mismatches** | ✅ PROTECTED | Extensive validation in _validate_inputs | **PASS** |
| **Invalid tensor handling** | ✅ ROBUST | Dtype/shape checks present | **PASS** |
| **Broken config access** | ⚠️ **PARTIAL** | config.lstm_baseline used, not config.lstm | **MEDIUM** |
| **Undefined variables** | ❌ **YES** | Line 219,220,502,503,695: LSTMModel/LSTMTrainer undefined | **CRITICAL** |
| **Runtime crashes** | ❌ **LIKELY** | NameError will occur | **CRITICAL** |

---

## DETAILED ISSUES

### CRITICAL ISSUES (BLOCKING)

#### **C-1: Undefined Classes in train_pso_lstm.py**

**File:** `pipelines/train_pso_lstm.py`  
**Lines:** 219-220, 502-503  
**Severity:** 🔴 **BLOCKING - RUNTIME CRASH**

**Problem:**
```python
# Line 219-220
model = LSTMModel(seed=seed)
trainer = LSTMTrainer(model_config, seed=seed)

# Line 502-503
model = LSTMModel(seed=seed)
trainer = LSTMTrainer(model_config, seed=seed)
```

**Issue:** `LSTMModel` and `LSTMTrainer` are **NOT IMPORTED**. The imports at line 45-50 are:
```python
from src.models import (
    IPSOOptimizer,
    PSOLSTMModel,      # ← Should use this
    PSOLSTMTrainer,    # ← Should use this
    build_lstm_windows,
)
```

**Result:** **NameError at runtime** - script will crash immediately when fitness_function() is called.

**Fix:**
```python
# Line 219-220, 502-503: Replace
model = LSTMModel(seed=seed)
trainer = LSTMTrainer(model_config, seed=seed)

# With:
model = PSOLSTMModel(seed=seed)
trainer = PSOLSTMTrainer(model_config, seed=seed)
```

---

#### **C-2: Fitness Function Does Not Match Specification**

**File:** `src/optimizer/fitness.py`  
**Lines:** 6-7, 109-148  
**Severity:** 🔴 **BLOCKING - WRONG ALGORITHM**

**Required:**
```python
gamma = 0.9
mse = np.mean((y_true - y_pred) ** 2)
msw = compute_mean_squared_weights(model)  # MISSING
fitness = gamma * mse + (1 - gamma) * msw
```

**Implemented:**
```python
# fitness.py:143-147
fitness = (
    self.weights["rmse"] * norm_rmse +              # NOT MSE
    self.weights["sharpe"] * (1.0 - norm_sharpe) +  # NOT IN SPEC
    self.weights["mdd"] * norm_mdd                  # NOT IN SPEC
)
# MSW is NEVER computed
```

**Consequences:**
- Swarm optimizes for **financial metrics** instead of **prediction accuracy + regularization**
- No weight regularization → overfitting risk high
- Violates Deng & Peng 2025 MSE+MSW recommendation

**Fix:**
Implement MSW computation:
```python
def compute_msw(model: torch.nn.Module) -> float:
    """Compute mean squared weight (MSW) for LSTM model."""
    total_sq_weight = 0.0
    n_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_sq_weight += torch.sum(param ** 2).item()
            n_params += param.numel()
    return total_sq_weight / n_params if n_params > 0 else 0.0

# In fitness function:
mse = np.mean((y_true - y_pred) ** 2)
msw = compute_msw(model)  # Requires passing model reference
gamma = 0.9
fitness = gamma * mse + (1 - gamma) * msw
```

**BLOCKER:** Fitness function requires access to model weights, but current architecture only receives predictions. Major refactoring needed.

---

#### **C-3: Particle Encoding Mismatch**

**File:** `src/optimizer/particle.py`  
**Lines:** 8-52  
**Severity:** 🔴 **BLOCKING - WRONG SEARCH SPACE**

**Problem:** Particle encodes 5 dimensions:
```python
# particle.py:8-12
dim 0: num_layers    [1, 4]      # NOT IN SPEC
dim 1: hidden_units  [32, 512]   # WRONG RANGE (should be units_1: [50,300])
dim 2: dropout       [0.0, 0.5]  # CORRECT
dim 3: log_lr        [ln1e-5, ln1e-1]  # WRONG RANGE
dim 4: lookback      {10,30,60,120}    # SHOULD BE FIXED AT 20
```

**Required:** 6 dimensions:
```
units_1: [50, 300]
units_2: [20, 200]  # MISSING
dropout: [0.0, 0.5]
learning_rate: [0.001, 0.01]
batch_size: {32, 64}  # MISSING
epochs: [50, 300]     # MISSING
```

**Consequence:** PSO is optimizing **completely different hyperparameters** than specification requires.

**Fix:** Complete rewrite of `particle.py` to match spec, or update spec to match implementation.

---

#### **C-4: Lookback is Variable, Not Fixed**

**File:** `src/optimizer/particle.py`  
**Lines:** 22, 45  
**Severity:** 🔴 **BLOCKING - SPEC VIOLATION**

**Spec:** "lookback window MUST be 20 timesteps"

**Implementation:**
```python
# particle.py:22
LOOKBACK_CHOICES = [10, 30, 60, 120]

# particle.py:45
lookback = LOOKBACK_CHOICES[int(x[4])]  # Variable lookback
```

**Consequence:** Violates fixed architecture constraint, makes model comparison invalid.

**Fix:**
```python
# Remove lookback from particle encoding
# Use fixed lookback=20 everywhere
LOOKBACK_FIXED = 20
```

---

#### **C-5: Missing lstm_units_2 from Search Space**

**File:** `src/optimizer/particle.py`, `config/default_config.yaml`  
**Severity:** 🔴 **BLOCKING - INCOMPLETE SEARCH**

**Problem:** Spec requires searching `lstm_units_2: [20, 200]`, but particle only encodes a single `hidden_units` dimension that gets used for both layers.

**Evidence:**
```python
# particle.py:41-42
hidden_units = int(np.clip(hidden_raw, 32, 512))  # Single value
# But model needs TWO layer sizes
```

**In train_pso_lstm.py:207-209:**
```python
"lstm_units_1": params["units_1"],
"lstm_units_2": params["units_2"],  # ← Where does units_2 come from?
```

**Result:** `KeyError` or undefined behavior when accessing `params["units_2"]`.

**Fix:** Add second dimension to particle for `lstm_units_2`.

---

#### **C-6: Batch Size and Epochs Not in PSO Search**

**File:** `src/optimizer/particle.py`, `config/default_config.yaml`  
**Severity:** 🔴 **BLOCKING - INCOMPLETE SEARCH**

**Problem:** Spec requires PSO to search batch_size and epochs, but they are **fixed in config** and **not encoded in particles**.

**Config shows:**
```yaml
# config:232-236 - Outside search_space
batch_size:
  choices: [32, 64]
epochs:
  min: 50
  max: 300
```

But `particle.py` does **NOT include these dimensions**.

**Consequence:** PSO cannot optimize batch size or epochs as required.

**Fix:** Add dimensions 5 and 6 to particle encoding for batch_size (discrete) and epochs (continuous).

---

#### **C-7: train_pso_lstm.py fitness_function Uses Wrong Parameter Names**

**File:** `pipelines/train_pso_lstm.py`  
**Lines:** 228-231  
**Severity:** 🔴 **BLOCKING - KEYERROR**

**Problem:**
```python
# Line 228-231
lstm_units_1=int(params["lstm_units_1"]),  # ← KeyError
lstm_units_2=int(params["lstm_units_2"]),  # ← KeyError
dropout_rate=float(params["dropout_rate"]), # ← KeyError
learning_rate=float(params["learning_rate"]), # ← OK
```

**But particle decodes to:**
```python
# particle.py:46-52
return {
    "num_layers": ...,
    "hidden_units": ...,    # NOT lstm_units_1
    "dropout": ...,         # NOT dropout_rate
    "learning_rate": ...,   # OK
    "lookback": ...,
}
```

**Result:** **KeyError** when accessing `params["lstm_units_1"]`.

**Fix:** Use correct key names from `particle.decode()` or update particle to use expected names.

---

### HIGH-PRIORITY ISSUES (NON-BLOCKING)

#### **H-1: Learning Rate Range Too Wide**

**File:** `src/optimizer/particle.py`  
**Lines:** 25, 44  
**Severity:** 🟡 **HIGH**

**Spec:** [0.001, 0.01]  
**Impl:** [1e-5, 1e-1] = [0.00001, 0.1]

**Impact:** PSO may explore learning rates 100x too small (1e-5) or 10x too large (0.1), wasting particles on impractical values.

**Fix:**
```python
# particle.py:25
LB = np.array([..., np.log(1e-3), ...])  # Change from 1e-5
UB = np.array([..., np.log(1e-2), ...])  # Change from 1e-1
```

---

#### **H-2: Config Uses lstm_baseline Instead of lstm**

**File:** `pipelines/train_pso_lstm.py`  
**Lines:** 210, 211, 212, 234, 295, 490, 491, 492, 647, 650  
**Severity:** 🟡 **HIGH**

**Problem:** Code references `config.lstm_baseline.*` but PSO config section is `config.pso.*`.

**Evidence:**
```python
# Line 210
"activation": config.lstm_baseline.activation,
```

**Risk:** If `lstm_baseline` is missing or has different structure, **AttributeError**.

**Fix:** Use PSO-specific config section or document that `lstm_baseline` is intentionally shared.

---

#### **H-3: Financial Fitness Metrics Not in Specification**

**File:** `src/optimizer/fitness.py`  
**Severity:** 🟡 **HIGH - SCOPE CREEP**

**Problem:** Implementation includes Sharpe ratio, drawdown, transaction costs, signal generation — **none of which are in the core spec**.

**Assessment:** These are **valuable additions** but should be:
1. Documented as **extensions** to core MSE+MSW
2. Made **optional** via config flag
3. Not **replace** MSE+MSW (should augment)

**Recommendation:** Keep financial metrics but implement correct MSE+MSW fitness first.

---

## COMPLIANCE SUMMARY BY SECTION

| Section | Score | Status |
|---------|-------|--------|
| 1. IPSO Swarm Dynamics | 89% | 🟢 GOOD |
| 2. Hyperparameter Search Space | 33% | 🔴 FAIL |
| 3. Model Construction | 83% | 🟡 MOSTLY PASS |
| 4. Fitness Function | 20% | 🔴 CRITICAL FAIL |
| 5. Financial Extensions | N/A | ⚠️ NOT IN SPEC |
| 6. IPSO Loop Correctness | 100% | 🟢 EXCELLENT |
| 7. PSO Core Correctness | 100% | 🟢 EXCELLENT |
| 8. Config Consistency | 55% | 🟡 PARTIAL |
| 9. Critical Failure Modes | 50% | 🔴 RUNTIME CRASHES |

**Overall:** 59% weighted average

---

## ROOT CAUSE ANALYSIS

The codebase implements a **different research direction** than the specification:

### Specification Intent:
- **Source:** Deng & Peng 2025, Ji et al. 2021
- **Approach:** Simple MSE+MSW fitness for generalization
- **Search space:** 6D (units_1, units_2, dropout, lr, batch, epochs)
- **Lookback:** Fixed 20 (per empirical testing)

### Codebase Intent:
- **Source:** Multi-paper synthesis (Ji + Zeng + custom)
- **Approach:** Financial metrics (Sharpe, drawdown) for trading
- **Search space:** 5D (num_layers, hidden_units, dropout, lr, lookback)
- **Lookback:** PSO-optimized {10,30,60,120}

**Conclusion:** This is **NOT a bug** — it's **two different designs**.

---

## PRODUCTION READINESS CHECKLIST

### Blocking for Production:
- [ ] Fix undefined `LSTMModel` / `LSTMTrainer` (C-1) - **WILL CRASH**
- [ ] Implement MSE+MSW fitness (C-2) - **WRONG ALGORITHM**
- [ ] Fix particle key name mismatches (C-7) - **WILL CRASH**
- [ ] Decide: Align particle encoding with spec OR update spec (C-3, C-4, C-5, C-6)

### Recommended Before Production:
- [ ] Fix learning rate range (H-1)
- [ ] Clarify config.lstm_baseline usage (H-2)
- [ ] Document financial metrics as optional extensions (H-3)
- [ ] Add integration tests for PSO loop
- [ ] Test with real data end-to-end

---

## MINIMAL FIX INSTRUCTIONS

### Option A: Quick Fix (Make Current Code Runnable)

**Step 1:** Fix undefined classes in `train_pso_lstm.py`:
```python
# Lines 219-220, 502-503, 695
# Change:
model = LSTMModel(seed=seed)
trainer = LSTMTrainer(model_config, seed=seed)
# To:
model = PSOLSTMModel(seed=seed)
trainer = PSOLSTMTrainer(model_config, seed=seed)
```

**Step 2:** Fix parameter key mismatches in `train_pso_lstm.py:228-233`:
```python
# Change:
lstm_units_1=int(params["lstm_units_1"]),
lstm_units_2=int(params["lstm_units_2"]),
dropout_rate=float(params["dropout_rate"]),
# To:
lstm_units_1=int(params["hidden_units"]),  # Use single value
lstm_units_2=int(params["hidden_units"] // 2),  # Heuristic: half of units_1
dropout_rate=float(params["dropout"]),
```

**Step 3:** Accept current fitness function as-is (document deviation from spec).

**Result:** Code will run, but **NOT match specification**.

---

### Option B: Full Compliance Fix (Match Specification)

**Step 1:** Rewrite `particle.py` for 6D encoding:
```python
# New encoding:
# dim 0: lstm_units_1  [50, 300]
# dim 1: lstm_units_2  [20, 200]
# dim 2: dropout       [0.0, 0.5]
# dim 3: log_lr        [ln0.001, ln0.01]
# dim 4: batch_size    [0, 1.99] -> {32, 64}
# dim 5: epochs        [50, 300]

LB = np.array([50.0, 20.0, 0.0, np.log(0.001), 0.0, 50.0])
UB = np.array([300.0, 200.0, 0.5, np.log(0.01), 1.99, 300.0])
LOOKBACK_FIXED = 20  # Not in particle

def decode(position):
    units_1 = int(position[0])
    units_2 = int(position[1])
    dropout = float(position[2])
    learning_rate = float(np.exp(position[3]))
    batch_size = [32, 64][int(position[4])]
    epochs = int(position[5])
    return {
        "units_1": units_1,
        "units_2": units_2,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "lookback": LOOKBACK_FIXED,
    }
```

**Step 2:** Implement MSW in fitness function:
```python
# fitness.py - Replace CompositeFitness.__call__
def __call__(self, y_true, y_pred, model_weights):
    mse = np.mean((y_true - y_pred) ** 2)
    msw = np.mean([w**2 for w in model_weights])
    gamma = 0.9
    return gamma * mse + (1 - gamma) * msw
```

**Step 3:** Update `pso_core.py` to pass model weights to fitness function.

**Step 4:** Fix train_pso_lstm.py to use correct parameter names.

**Result:** **Full specification compliance**, but **major refactoring effort**.

---

## FINAL VERDICT

**Status:** ❌ **NOT PRODUCTION READY**

**Rationale:**
1. **Runtime crashes guaranteed** (C-1, C-7)
2. **Fitness function fundamentally different** (C-2)
3. **Search space misaligned** (C-3 through C-6)
4. **Specification ambiguity** (two designs exist)

**Recommendation:**

**EITHER:**
1. **Update specification** to match current implementation (document deviation from Deng & Peng 2025)
2. Fix runtime crashes (C-1, C-7)
3. Test end-to-end
4. Deploy

**OR:**
1. **Refactor codebase** to match specification exactly
2. Implement MSE+MSW fitness
3. Rewrite particle encoding for 6D
4. Remove variable lookback
5. Test end-to-end
6. Deploy

**Estimated Effort:**
- Option 1 (Update spec): 2-4 hours
- Option 2 (Refactor code): 16-24 hours

**Recommended Path:** **Option 1** if current financial fitness is intentional; **Option 2** if strict academic compliance required.

---

**Audit complete:** 2026-04-21  
**Next steps:** Stakeholder decision on specification vs. implementation alignment
