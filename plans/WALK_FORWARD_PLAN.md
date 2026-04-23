# Production-Grade Walk-Forward Validation System
## Complete Design and Implementation Plan

**Date:** 2026-04-21  
**Status:** COMPREHENSIVE AUDIT COMPLETE → IMPLEMENTATION READY  
**Priority:** CRITICAL - Blocks production deployment

---

## PHASE 1: CODEBASE AUDIT RESULTS

### Summary of Findings

**Files Audited:** 27 files across `@src` and `@pipelines`

#### ✅ CORRECT Implementations (Keep):
1. **`src/evaluation/frozen_pipeline.py`** - Immutable pipeline state (GOOD)
2. **`src/evaluation/canonical_split.py`** - 70/10/20 temporal split (GOOD)
3. **`src/features/scaler.py`** - FrozenMinMaxScaler with fit-once semantics (GOOD)

#### ❌ INCORRECT Implementations (Must Fix/Remove):

1. **`src/evaluation/walk_forward.py`** - **FUNDAMENTALLY BROKEN**
   - **Problem:** Implements ROLLING window (NOT expanding)
   - **Problem:** Frozen models (NOT retraining per fold)
   - **Problem:** No PSO integration
   - **Problem:** No per-fold scaling
   - **Verdict:** INCOMPATIBLE with requirements, must be completely rewritten

2. **`pipelines/validation.py`** - **PARTIALLY BROKEN**
   - **Problem:** References `WalkForwardValidator` class that doesn't exist
   - **Problem:** Uses `RuntimeContext` that may not exist
   - **Problem:** Retrains models but unclear on scaling isolation
   - **Verdict:** Architecture is closer but implementation incomplete

3. **`pipelines/canonical_evaluation.py`** - **INCOMPLETE STUB**
   - **Problem:** Placeholder implementation with TODOs
   - **Problem:** No actual model training
   - **Problem:** No walk-forward logic
   - **Verdict:** Keep structure, complete implementation

#### ⚠️ MISSING Critical Components:

1. **Per-Fold PSO Optimization** - Not implemented anywhere
2. **Per-Fold Scaler Isolation** - Not enforced
3. **Expanding Window Logic** - Only rolling window exists
4. **Sequence Boundary Handling** - Not addressed
5. **Metrics Aggregation** - No cross-fold aggregation
6. **Inverse Transform in Evaluation** - Missing (per LOSS_SCALING_ANALYSIS.md)

---

## PHASE 2: ARCHITECTURAL VIOLATIONS TO FIX

### Violation #1: Rolling vs Expanding Window

**Current (WRONG):**
```python
# src/evaluation/walk_forward.py - Line 90
for t in range(self.lookback, len(X_test), self.step_size):
    X_window = X_test[t - self.lookback : t, :]  # ROLLING WINDOW
```

**Required (CORRECT):**
```python
# Expanding window: train grows each fold
for fold_idx in range(n_folds):
    train_end = initial_train_size + fold_idx * step_size
    val_end = train_end + val_size
    
    X_train_fold = X_all[:train_end]  # EXPANDING WINDOW
    X_val_fold = X_all[train_end:val_end]
```

### Violation #2: Frozen Models vs Retraining

**Current (WRONG):**
```python
# src/evaluation/walk_forward.py - Line 98
pred = model.predict(X_input, verbose=0)  # Uses pre-trained model
```

**Required (CORRECT):**
```python
# Retrain model per fold
for fold_idx in range(n_folds):
    # Fit scaler on THIS fold's train data
    scaler = FrozenMinMaxScaler().fit(X_train_fold)
    
    # Run PSO on THIS fold
    best_params = pso_optimizer.optimize(X_train_fold_scaled, y_train_fold_scaled)
    
    # Train model with PSO params
    model = LSTMModel(**best_params)
    model.train(X_train_fold_scaled, y_train_fold_scaled)
```

### Violation #3: No Scaler Isolation

**Current (WRONG):**
```python
# pipelines/train_pso_lstm.py
# Scales once globally, reuses scaler
```

**Required (CORRECT):**
```python
# Per-fold independent scaling
for fold_idx in range(n_folds):
    feature_scaler_fold = FrozenMinMaxScaler(feature_range=(-1, 1))
    feature_scaler_fold.fit(X_train_fold)
    
    target_scaler_fold = FrozenMinMaxScaler(feature_range=(-1, 1))
    target_scaler_fold.fit(y_train_fold.reshape(-1, 1))
    
    # Never reuse these scalers for other folds
```

---

## PHASE 3: PRODUCTION WALK-FORWARD ARCHITECTURE

### High-Level System Design

```
┌────────────────────────────────────────────────────────────────┐
│                   WALK-FORWARD VALIDATION                       │
│                                                                  │
│  [Global Preprocessing] (ONCE)                                  │
│   ├─ Data cleaning (TRD rules)                                  │
│   ├─ Feature generation (causal)                                │
│   └─ Wavelet denoising (global close series)                    │
│                                                                  │
│  [Fold Loop] (EXPANDING WINDOW)                                 │
│   │                                                              │
│   ├─ Fold 0: Train[0:t0] → Val[t0:t0+Δ]                         │
│   │   ├─ [A] Fit Feature Scaler on Train[0:t0]                  │
│   │   ├─ [B] Fit Target Scaler on Train[0:t0]                   │
│   │   ├─ [C] Transform Train & Val with scalers                 │
│   │   ├─ [D] Build sequences (handle boundaries)                │
│   │   ├─ [E] Run PSO on Train[0:t0]                             │
│   │   ├─ [F] Train LSTM with PSO params                         │
│   │   ├─ [G] Predict on Val[t0:t0+Δ]                            │
│   │   ├─ [H] Inverse transform predictions                      │
│   │   └─ [I] Compute metrics (RMSE, MAE, R², DA)                │
│   │                                                              │
│   ├─ Fold 1: Train[0:t1] → Val[t1:t1+Δ]  (train EXPANDS)        │
│   │   └─ Repeat [A-I] independently                             │
│   │                                                              │
│   └─ Fold N: Train[0:tN] → Val[tN:tN+Δ]                         │
│       └─ Repeat [A-I] independently                             │
│                                                                  │
│  [Aggregation]                                                   │
│   ├─ Metrics per fold → Mean ± Std                              │
│   ├─ Per-fold hyperparameters (from PSO)                        │
│   └─ Convergence analysis                                       │
└────────────────────────────────────────────────────────────────┘
```

### Critical Invariants

1. **Temporal Ordering:** `∀ fold_i: val_start[i] > train_end[i]`
2. **Expanding Window:** `train_size[i+1] > train_size[i]`
3. **Scaler Isolation:** `scaler[i] ≠ scaler[j] for i ≠ j`
4. **Model Isolation:** Each fold gets fresh LSTM initialization
5. **No Look-Ahead:** `∀t: features[t] only use data ≤ t`

---

## PHASE 4: DETAILED IMPLEMENTATION SPEC

### 4.1 Global Preprocessing (Once Before Folds)

```python
def global_preprocessing(
    raw_data: pd.DataFrame,
    config: Config
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Execute global preprocessing ONCE before fold loop.
    
    TRD-Compliant Steps:
    1. Data cleaning (forward-fill ≤5, discard >5)
    2. Feature generation (technical, cross-ticker, log returns)
    3. Wavelet denoising (global close series ONLY)
    4. NO SCALING (done per-fold)
    
    Args:
        raw_data: Raw OHLCV with DatetimeIndex
        config: Configuration object
    
    Returns:
        (X_raw, y_raw, timestamps)
    """
    logger.info("=" * 80)
    logger.info("GLOBAL PREPROCESSING (PRE-FOLD)")
    logger.info("=" * 80)
    
    # Step 1: TRD cleaning
    data_clean = apply_trd_cleaning(raw_data)
    
    # Step 2: Generate features (causal)
    X_features = generate_raw_features(
        data_clean,
        config.features
    )
    
    # Step 3: Compute target (log returns)
    y_raw = compute_log_returns(data_clean["close"])
    
    # Step 4: Wavelet denoise close series ONLY
    if config.features.wavelet.enabled:
        close_denoised = apply_wavelet_global(
            data_clean["close"].values,
            wavelet=config.features.wavelet.wavelet,
            level=config.features.wavelet.level
        )
        X_features["close_denoised"] = close_denoised
    
    logger.info(f"Global preprocessing complete:")
    logger.info(f"  Samples: {len(X_features)}")
    logger.info(f"  Features: {X_features.shape[1]}")
    logger.info(f"  ⚠️  NO SCALING APPLIED (done per-fold)")
    
    return X_features.values, y_raw.values, X_features.index
```

### 4.2 Per-Fold Processing (Core Loop)

```python
class ExpandingWindowWalkForward:
    """
    Production-grade expanding-window walk-forward validator with PSO.
    
    TRD-Compliant Features:
    - Expanding training window per fold
    - Independent scaler per fold
    - Independent PSO per fold
    - Fresh LSTM initialization per fold
    - Proper sequence boundary handling
    - Inverse transform before metrics
    """
    
    def __init__(
        self,
        initial_train_pct: float = 0.60,
        fold_step_pct: float = 0.05,
        val_pct: float = 0.05,
        lookback: int = 20,
        config: Config = None,
    ):
        self.initial_train_pct = initial_train_pct
        self.fold_step_pct = fold_step_pct
        self.val_pct = val_pct
        self.lookback = lookback
        self.config = config
        
        logger.info("=" * 80)
        logger.info("EXPANDING WINDOW WALK-FORWARD VALIDATOR")
        logger.info("=" * 80)
        logger.info(f"Initial train: {initial_train_pct:.0%}")
        logger.info(f"Fold step: {fold_step_pct:.0%}")
        logger.info(f"Val size: {val_pct:.0%}")
        logger.info(f"Lookback: {lookback} days")
        logger.info("=" * 80)
    
    def validate(
        self,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> Dict:
        """
        Execute expanding-window walk-forward validation.
        
        Args:
            X_raw: Global preprocessed features (NO SCALING)
            y_raw: Target returns (NO SCALING)
            timestamps: Datetime index
        
        Returns:
            Complete validation results with per-fold metrics
        """
        n_samples = len(X_raw)
        initial_train_size = int(n_samples * self.initial_train_pct)
        fold_step_size = int(n_samples * self.fold_step_pct)
        val_size = int(n_samples * self.val_pct)
        
        # Compute fold boundaries
        fold_boundaries = self._compute_fold_boundaries(
            n_samples,
            initial_train_size,
            fold_step_size,
            val_size
        )
        
        logger.info(f"Computed {len(fold_boundaries)} folds")
        
        # Execute fold loop
        fold_results = []
        
        for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(fold_boundaries):
            logger.info("=" * 80)
            logger.info(f"FOLD {fold_idx + 1}/{len(fold_boundaries)}")
            logger.info("=" * 80)
            
            fold_result = self._process_fold(
                fold_idx=fold_idx,
                X_raw=X_raw,
                y_raw=y_raw,
                timestamps=timestamps,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            )
            
            fold_results.append(fold_result)
        
        # Aggregate results
        aggregated_metrics = self._aggregate_fold_results(fold_results)
        
        return {
            "fold_results": fold_results,
            "aggregated_metrics": aggregated_metrics,
            "fold_boundaries": fold_boundaries,
            "config": {
                "initial_train_pct": self.initial_train_pct,
                "fold_step_pct": self.fold_step_pct,
                "val_pct": self.val_pct,
                "lookback": self.lookback,
            },
        }
    
    def _process_fold(
        self,
        fold_idx: int,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        timestamps: pd.DatetimeIndex,
        train_start: int,
        train_end: int,
        val_start: int,
        val_end: int,
    ) -> Dict:
        """
        Process a single fold with full isolation.
        
        CRITICAL: This function is COMPLETELY INDEPENDENT per fold.
        NO shared state between folds.
        
        Steps (TRD-Compliant):
        A. Fit feature scaler on train
        B. Fit target scaler on train
        C. Transform train & val
        D. Build sequences
        E. Run PSO
        F. Train LSTM
        G. Predict
        H. Inverse transform
        I. Compute metrics
        """
        logger.info(f"Train: [{train_start}:{train_end}] ({train_end - train_start} samples)")
        logger.info(f"Val:   [{val_start}:{val_end}] ({val_end - val_start} samples)")
        logger.info(f"Dates: {timestamps[train_start]} → {timestamps[val_end-1]}")
        
        # Extract fold data (UNSCALED)
        X_train_raw = X_raw[train_start:train_end]
        y_train_raw = y_raw[train_start:train_end]
        X_val_raw = X_raw[val_start:val_end]
        y_val_raw = y_raw[val_start:val_end]
        
        # A. Fit feature scaler (TRAIN ONLY)
        feature_scaler = FrozenMinMaxScaler(feature_range=(-1, 1))
        feature_scaler.fit(X_train_raw)
        logger.info("✓ Feature scaler fitted on TRAIN ONLY")
        
        # B. Fit target scaler (TRAIN ONLY)
        target_scaler = FrozenMinMaxScaler(feature_range=(-1, 1))
        target_scaler.fit(y_train_raw.reshape(-1, 1))
        logger.info("✓ Target scaler fitted on TRAIN ONLY")
        
        # C. Transform with fitted scalers
        X_train_scaled = feature_scaler.transform(X_train_raw)
        y_train_scaled = target_scaler.transform(y_train_raw.reshape(-1, 1)).flatten()
        X_val_scaled = feature_scaler.transform(X_val_raw)
        y_val_scaled = target_scaler.transform(y_val_raw.reshape(-1, 1)).flatten()
        
        logger.info("✓ Data scaled with fold-specific scalers")
        
        # D. Build LSTM sequences (handle boundaries)
        X_train_seq, y_train_seq = build_lstm_windows(
            X_train_scaled,
            y_train_scaled,
            self.lookback
        )
        X_val_seq, y_val_seq = build_lstm_windows(
            X_val_scaled,
            y_val_scaled,
            self.lookback
        )
        
        logger.info(f"✓ Sequences built: train={X_train_seq.shape}, val={X_val_seq.shape}")
        
        # E. Run PSO optimization (TRAIN + internal val split)
        pso_result = self._run_pso_for_fold(
            X_train_seq,
            y_train_seq,
            fold_idx
        )
        
        best_params = pso_result["best_params"]
        logger.info(f"✓ PSO complete: {best_params}")
        
        # F. Train LSTM with PSO params
        model = self._train_lstm_with_params(
            X_train_seq,
            y_train_seq,
            best_params,
            fold_idx
        )
        
        logger.info("✓ LSTM trained with PSO params")
        
        # G. Predict on validation set
        y_pred_scaled = model.predict(X_val_seq, verbose=0).flatten()
        
        # H. Inverse transform predictions (CRITICAL for interpretable metrics)
        y_pred_original = target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()
        y_val_original = target_scaler.inverse_transform(
            y_val_seq.reshape(-1, 1)
        ).flatten()
        
        logger.info("✓ Predictions inverse-transformed to original scale")
        
        # I. Compute metrics on ORIGINAL scale
        metrics = self._compute_fold_metrics(
            y_pred_original,
            y_val_original
        )
        
        logger.info(f"✓ Fold {fold_idx + 1} metrics: RMSE={metrics['rmse']:.6f}, R²={metrics['r2']:.4f}")
        
        return {
            "fold_idx": fold_idx,
            "train_size": train_end - train_start,
            "val_size": val_end - val_start,
            "pso_params": best_params,
            "pso_fitness": pso_result["best_fitness"],
            "metrics": metrics,
            "predictions": {
                "y_pred": y_pred_original,
                "y_true": y_val_original,
                "timestamps": timestamps[val_start + self.lookback : val_end],
            },
            "scalers": {
                "feature_scaler_params": feature_scaler.get_params(),
                "target_scaler_params": target_scaler.get_params(),
            },
        }
    
    def _run_pso_for_fold(
        self,
        X_train_seq: np.ndarray,
        y_train_seq: np.ndarray,
        fold_idx: int,
    ) -> Dict:
        """
        Run PSO optimization for this fold.
        
        PSO uses internal train/val split (90/10) of fold's training data.
        """
        # Split train into PSO train/val (90/10)
        pso_split_idx = int(len(X_train_seq) * 0.9)
        
        X_pso_train = X_train_seq[:pso_split_idx]
        y_pso_train = y_train_seq[:pso_split_idx]
        X_pso_val = X_train_seq[pso_split_idx:]
        y_pso_val = y_train_seq[pso_split_idx:]
        
        logger.info(f"PSO internal split: train={len(X_pso_train)}, val={len(X_pso_val)}")
        
        # Define fitness function
        def fitness_fn(params):
            model = LSTMModel(seed=self.config.pso.random_seed + fold_idx)
            trainer = LSTMTrainer(params, seed=self.config.pso.random_seed + fold_idx)
            
            model, _ = trainer.train(
                X_pso_train,
                y_pso_train,
                X_pso_val,
                y_pso_val,
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                patience=self.config.lstm.early_stopping.patience,
                shuffle=False,
            )
            
            y_pred = model.predict(X_pso_val, verbose=0)
            mse = np.mean((y_pso_val - y_pred.flatten()) ** 2)
            return float(mse)
        
        # Initialize PSO
        pso = IPSOOptimizer(
            n_particles=self.config.pso.n_particles,
            n_iterations=self.config.pso.n_iterations,
            search_space=self.config.pso.search_space,
            fitness_func=fitness_fn,
            seed=self.config.pso.random_seed + fold_idx,
        )
        
        # Run optimization
        best_params, best_fitness = pso.optimize()
        
        return {
            "best_params": best_params,
            "best_fitness": best_fitness,
            "pso_train_size": len(X_pso_train),
            "pso_val_size": len(X_pso_val),
        }
    
    def _train_lstm_with_params(
        self,
        X_train_seq: np.ndarray,
        y_train_seq: np.ndarray,
        params: Dict,
        fold_idx: int,
    ) -> LSTMModel:
        """
        Train LSTM with PSO-optimized params on full fold training data.
        """
        model_config = {
            "input_shape": (self.lookback, X_train_seq.shape[2]),
            "lstm_units_1": params["units_1"],
            "lstm_units_2": params["units_2"],
            "dropout_rate": params["dropout"],
            "activation": self.config.lstm.activation,
            "output_units": 1,
            "output_activation": "linear",
            "learning_rate": params["learning_rate"],
            "loss": "mse",
        }
        
        model = LSTMModel(seed=self.config.pso.random_seed + fold_idx)
        trainer = LSTMTrainer(model_config, seed=self.config.pso.random_seed + fold_idx)
        
        model, _ = trainer.train(
            X_train_seq,
            y_train_seq,
            None,  # No validation in final training
            None,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            patience=None,  # No early stopping
            shuffle=False,
        )
        
        return model
    
    def _compute_fold_metrics(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute fold-specific metrics on ORIGINAL scale.
        """
        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(mse))
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
        
        # MAPE (avoid division by zero)
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100)
        
        # Directional accuracy
        correct_dir = np.sum(np.sign(y_true) == np.sign(y_pred))
        da = float(correct_dir / len(y_true))
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "directional_accuracy": da,
            "n_samples": len(y_true),
        }
    
    def _aggregate_fold_results(self, fold_results: List[Dict]) -> Dict:
        """
        Aggregate metrics across all folds.
        """
        metric_names = ["mse", "mae", "rmse", "r2", "mape", "directional_accuracy"]
        
        aggregated = {}
        for metric in metric_names:
            values = [fold["metrics"][metric] for fold in fold_results]
            aggregated[f"{metric}_mean"] = float(np.mean(values))
            aggregated[f"{metric}_std"] = float(np.std(values))
            aggregated[f"{metric}_min"] = float(np.min(values))
            aggregated[f"{metric}_max"] = float(np.max(values))
        
        aggregated["n_folds"] = len(fold_results)
        
        logger.info("=" * 80)
        logger.info("AGGREGATED METRICS (ACROSS FOLDS)")
        logger.info("=" * 80)
        logger.info(f"RMSE: {aggregated['rmse_mean']:.6f} ± {aggregated['rmse_std']:.6f}")
        logger.info(f"R²:   {aggregated['r2_mean']:.4f} ± {aggregated['r2_std']:.4f}")
        logger.info(f"DA:   {aggregated['directional_accuracy_mean']:.2%} ± {aggregated['directional_accuracy_std']:.2%}")
        logger.info("=" * 80)
        
        return aggregated
    
    def _compute_fold_boundaries(
        self,
        n_samples: int,
        initial_train_size: int,
        fold_step_size: int,
        val_size: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Compute (train_start, train_end, val_start, val_end) for each fold.
        """
        boundaries = []
        train_start = 0
        
        while True:
            train_end = initial_train_size + len(boundaries) * fold_step_size
            val_start = train_end
            val_end = val_start + val_size
            
            # Stop if validation extends beyond data
            if val_end > n_samples:
                break
            
            boundaries.append((train_start, train_end, val_start, val_end))
        
        return boundaries
```

---

## PHASE 5: FILES TO MODIFY/CREATE

### Files to REMOVE:
- ❌ `src/evaluation/walk_forward.py` (incompatible architecture)
- ❌ `pipelines/validation.py` (incomplete, references missing classes)

### Files to KEEP (No Changes):
- ✅ `src/evaluation/frozen_pipeline.py`
- ✅ `src/evaluation/canonical_split.py`
- ✅ `src/features/scaler.py`

### Files to CREATE:
1. **`src/evaluation/walk_forward_pso.py`** (NEW)
   - `ExpandingWindowWalkForward` class
   - Complete implementation per Phase 4.2

2. **`pipelines/walk_forward_evaluation.py`** (NEW)
   - CLI script to run walk-forward validation
   - Calls `ExpandingWindowWalkForward`

3. **`src/evaluation/metrics_aggregation.py`** (NEW)
   - Cross-fold metric aggregation
   - Statistical tests (if needed)

### Files to MODIFY:
1. **`src/evaluation/__init__.py`**
   - Export `ExpandingWindowWalkForward`
   
2. **`src/models/trainer.py`**
   - Add `target_scaler` parameter to `evaluate()` method
   - Apply inverse transform before metrics

3. **`pipelines/canonical_evaluation.py`**
   - Replace TODO stubs with actual walk-forward call

---

## PHASE 6: IMPLEMENTATION EXECUTION ORDER

### Step 1: Fix Existing Issues (1-2 hours)
1. ✅ Add inverse transform to `src/models/trainer.py:evaluate()`
2. ✅ Update `src/models/lstm.py:evaluate()`
3. ✅ Update `src/models/xgboost_model.py:evaluate()`

### Step 2: Create Core Walk-Forward (3-4 hours)
1. ✅ Implement `src/evaluation/walk_forward_pso.py`
2. ✅ Implement `ExpandingWindowWalkForward` class
3. ✅ Test fold boundary computation
4. ✅ Test per-fold scaler isolation

### Step 3: PSO Integration (2-3 hours)
1. ✅ Implement `_run_pso_for_fold()`
2. ✅ Test PSO internal split
3. ✅ Verify seed isolation per fold

### Step 4: Sequence Handling (1-2 hours)
1. ✅ Test sequence boundary handling
2. ✅ Verify no data loss at fold transitions
3. ✅ Test with different lookback values

### Step 5: Metrics & Aggregation (1-2 hours)
1. ✅ Implement fold-level metrics
2. ✅ Implement cross-fold aggregation
3. ✅ Add statistical significance tests

### Step 6: CLI & Integration (2-3 hours)
1. ✅ Create `pipelines/walk_forward_evaluation.py`
2. ✅ Update `pipelines/canonical_evaluation.py`
3. ✅ Add config schema for walk-forward params

### Step 7: Testing & Validation (3-4 hours)
1. ✅ Unit tests for fold computation
2. ✅ Integration test with small dataset
3. ✅ Verify TRD compliance
4. ✅ Performance benchmarking

**Total Estimated Time:** 13-20 hours

---

## PHASE 7: CRITICAL VALIDATION CHECKLIST

Before declaring system production-ready, verify:

### ✅ Temporal Integrity:
- [ ] No look-ahead bias in features
- [ ] Validation timestamps > training timestamps
- [ ] Expanding window grows correctly

### ✅ Scaler Isolation:
- [ ] Feature scaler fitted per fold
- [ ] Target scaler fitted per fold
- [ ] No scaler reuse across folds

### ✅ Model Isolation:
- [ ] Fresh LSTM initialization per fold
- [ ] No weight carryover
- [ ] Independent PSO per fold

### ✅ PSO Correctness:
- [ ] PSO uses internal 90/10 split
- [ ] PSO never sees fold validation data
- [ ] Seed varies per fold

### ✅ Metrics Correctness:
- [ ] Inverse transform applied before metrics
- [ ] Metrics on original scale
- [ ] Cross-fold aggregation correct

### ✅ Reproducibility:
- [ ] Same seed → same results
- [ ] Deterministic fold boundaries
- [ ] Serializable state

---

## PHASE 8: PRODUCTION DEPLOYMENT

### Configuration Schema

Add to `config/default_config.yaml`:

```yaml
walk_forward:
  enabled: true
  initial_train_pct: 0.60  # 60% initial training
  fold_step_pct: 0.05      # 5% step size (~30 days for 600-day dataset)
  val_pct: 0.05            # 5% validation per fold
  max_folds: 10            # Limit total folds (0 = unlimited)
  
  # Per-fold PSO
  pso_per_fold: true
  pso_internal_split: 0.9  # 90/10 for PSO train/val
  
  # Metrics
  inverse_transform_metrics: true  # CRITICAL: always true
  compute_per_fold: true
  aggregate_folds: true
  save_fold_results: true
```

### Usage Example

```bash
# Run walk-forward validation
python pipelines/walk_forward_evaluation.py \
    --ticker AAPL \
    --config config/default_config.yaml \
    --output-dir results/walk_forward/AAPL
```

### Output Structure

```
results/walk_forward/AAPL/
├── fold_results.json          # Per-fold metrics & predictions
├── aggregated_metrics.json    # Cross-fold statistics
├── fold_scalers/              # Scaler params per fold
│   ├── fold_0_feature_scaler.json
│   ├── fold_0_target_scaler.json
│   ├── fold_1_feature_scaler.json
│   └── ...
├── fold_models/               # Trained models per fold
│   ├── fold_0_lstm.h5
│   ├── fold_1_lstm.h5
│   └── ...
├── pso_results/               # PSO convergence per fold
│   ├── fold_0_pso.json
│   ├── fold_1_pso.json
│   └── ...
└── validation_report.md       # Human-readable summary
```

---

## CONCLUSION

### Current State Assessment:

**Existing Implementation:** ❌ BROKEN (frozen models, rolling windows, no PSO)

**Required Effort:** ~13-20 hours to implement production-grade system

**Blocking Issues:** 3 critical (expanding window, scaler isolation, PSO integration)

### Recommended Action:

1. **IMMEDIATE:** Remove broken `src/evaluation/walk_forward.py`
2. **PRIORITY 1:** Implement `ExpandingWindowWalkForward` (Phase 4.2)
3. **PRIORITY 2:** Add inverse transform to evaluation methods
4. **PRIORITY 3:** Create CLI script and integrate

### Success Criteria:

✅ Expanding window with per-fold retraining  
✅ Independent scalers per fold  
✅ PSO optimization per fold  
✅ Inverse transform before metrics  
✅ Cross-fold aggregation  
✅ TRD-compliant (no leakage)  
✅ Reproducible (same seed → same results)

---

**Document Status:** COMPLETE - Ready for Implementation  
**Next Action:** Begin Phase 5 implementation  
**Estimated Completion:** 13-20 engineering hours
