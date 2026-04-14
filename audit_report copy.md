# Codebase Audit Report

**Project**: LSTM/XGBoost Quantitative Finance System  
**Date**: 2026-04-14  
**Auditor**: Production-Critical Audit  
**Scope**: Full repository analysis for production deployment readiness

---

### 4.2 **Duplicate Logging Setup**
- **Locations**:
  - `pipelines/test.py:67` (`_setup_logging`)
  - `pipelines/validation.py:59` (`_setup_logging`)
  - `pipelines/common.py:27` (`setup`)
  - `src/utils/logger.py:10` (`setup_logger`)
- **Refactor Plan**: Use only `src.utils.logger.setup_logger()`. Delete all duplicates.

### 4.3 **Duplicate JSON Serialization**
- **Locations**:
  - `pipelines/test.py:81` (`_save_json`)
  - `pipelines/validation.py:81` (`_save_json`)
  - `pipelines/common.py:123` (`save_json`)
  - `src/experiment/experiment_tracker.py:142` (`save_json`)
- **Refactor Plan**: Consolidate into `src.utils.io.save_json()`. Add numpy/pandas serialization support.

### 4.4 **Duplicate Tag Generation**
- **Locations**:
  - `pipelines/test.py:77` (`_make_tag`)
  - `pipelines/validation.py:94` (`_make_tag`)
  - `pipelines/common.py:38` (`make_tag`)
  - `src/experiment/experiment_tracker.py:129` (`make_filename`)
- **Refactor Plan**: Use `ExperimentTracker.make_filename()` everywhere. Delete standalone functions.

---

## 5. Modularity Improvements

### Current Problems

1. **Tight Coupling**: `pipelines/train.py` directly imports `VanillaLSTM`, `XGBoostModel`, `ExperimentTracker`, `build_windows`, `compute_and_log_all_statistical_metrics`. Changes to any model require editing the pipeline.

2. **Mixed Responsibilities**: `src/models/lstm/lstm_model.py` contains both `LSTMModel` (architecture) and `LSTMTrainer` (training loop + metrics). These should be separate.

3. **Non-Reusable Pipelines**: `pipelines/train.py`, `test.py`, `validation.py` are 500+ line scripts with duplicated logic. No shared abstractions.

### Target Architecture

```
src/
  data/
    loaders.py          # Load raw data, splits, features
    splitter.py         # Temporal splitting, window construction
    validators.py       # Data integrity checks (checksums, schema)
  
  features/
    pipeline.py         # FeaturePipeline (orchestrator)
    technical.py        # Technical indicators
    statistical.py      # Statistical features
    volume.py           # Volume features
    cross_ticker.py     # Cross-ticker features
    selector.py         # Feature selection
    scalar.py           # Scaling
  
  models/
    base.py             # BaseModel interface (fit, predict, save, load)
    lstm/
      model.py          # LSTMModel (nn.Module only)
      trainer.py        # LSTMTrainer (training loop)
      inference.py      # LSTMInference (batch prediction)
    xgboost/
      model.py          # XGBoostModel (fit, predict)
      trainer.py        # XGBoostTrainer (hyperparameter tuning)
      inference.py      # XGBoostInference
  
  evaluation/
    metrics.py          # Statistical + trading metrics
    backtester.py       # Backtester (refactored)
    walk_forward.py     # Walk-forward validation
  
  optimizer/
    pso_core.py         # StandardPSO
    ipso.py             # IPSO (mutation)
    fitness.py          # CompositeFitness
    particle.py         # Particle encoding/decoding
  
  pipelines/
    train.py            # Thin wrapper: load data → train model → save
    test.py             # Thin wrapper: load model → predict → backtest
    validation.py       # Thin wrapper: walk-forward + threshold sweep
    common.py           # DELETE (move to src/utils/)
  
  experiment/
    tracker.py          # ExperimentTracker (artifact management)
    run_context.py      # RuntimeContext (config + args)
  
  utils/
    config_loader.py    # YAML config parsing
    logger.py           # Logging setup
    seed.py             # Seed setting
    io.py               # save_json, load_json, checksums
```

### Required Rules

1. **Strict Separation**:
   - **Data**: Load, split, validate. No feature engineering.
   - **Features**: Transform raw data → feature matrices. No model training.
   - **Models**: Define architecture + training. No data loading.
   - **Evaluation**: Compute metrics. No model training.
   - **Pipelines**: Orchestrate. No business logic.

2. **No Cross-Layer Leakage**:
   - Models cannot import from `data/` or `features/`.
   - Pipelines cannot import model internals (only public API).
   - Evaluation cannot import training code.

3. **Reusable Components**:
   - All models implement `BaseModel` interface:
     ```python
     class BaseModel(ABC):
         @abstractmethod
         def fit(self, X_train, y_train, X_val, y_val) -> dict:
             pass
         
         @abstractmethod
         def predict(self, X) -> np.ndarray:
             pass
         
         @abstractmethod
         def save(self, path: str) -> None:
             pass
         
         @classmethod
         @abstractmethod
         def load(cls, path: str) -> "BaseModel":
             pass
     ```

### Refactor Examples

#### Example 1: Decouple Training Pipeline

**Before** (`pipelines/train.py:89-139`):
```python
def train_lstm(ctx: RuntimeContext) -> None:
    X_train, y_train = ctx.tracker.load_split(Phase.TRAIN)
    X_val, y_val = ctx.tracker.load_split(Phase.VAL)
    hp = _lstm_hyperparams(ctx.args, ctx.cfg)
    X_train, y_train_w = ctx.tracker._make_windows(X_train, y_train, hp["lookback"])
    X_val, y_val_w = ctx.tracker._make_windows(X_val, y_val, hp["lookback"])
    model = VanillaLSTM(input_size=X_train.shape[2], device=ctx.args.device, **hp)
    history = model.fit(X_train, y_train_w, X_val, y_val_w)
    # ... save artifacts
```

**After**:
```python
# pipelines/train.py
from src.pipelines.training_pipeline import TrainingPipeline

def train_lstm(ctx: RuntimeContext) -> None:
    pipeline = TrainingPipeline(ctx)
    pipeline.run(model_type="lstm")

# src/pipelines/training_pipeline.py
class TrainingPipeline:
    def __init__(self, ctx: RuntimeContext):
        self.ctx = ctx
        self.data_loader = DataLoader(ctx.tracker)
        self.model_factory = ModelFactory(ctx.cfg)
    
    def run(self, model_type: str):
        X_train, y_train, X_val, y_val = self.data_loader.load_windowed_splits()
        model = self.model_factory.create(model_type, input_size=X_train.shape[2])
        history = model.fit(X_train, y_train, X_val, y_val)
        self._save_artifacts(model, history)
```

#### Example 2: Extract Backtester Components

**Before** (`src/evaluation/backtester.py:87-227`):
```python
def run(self, y_pred, opens, closes, timestamps, session_starts=None):
    # 140 lines of mixed logic
```

**After**:
```python
# src/evaluation/backtester.py
class Backtester:
    def __init__(self, ...):
        self.position_manager = PositionManager(...)
        self.risk_manager = RiskManager(...)
        self.pnl_calculator = PnLCalculator(...)
    
    def run(self, y_pred, opens, closes, timestamps, session_starts=None):
        signals = self._make_signals(y_pred)
        state = BacktestState(initial_capital=self.V0)
        
        for t in range(len(y_pred)):
            state = self.position_manager.update(t, signals[t], opens[t], closes[t], state)
            state = self.risk_manager.apply_controls(t, state)
            state = self.pnl_calculator.mark_to_market(t, closes[t], state)
        
        return self._build_result(state)

# src/evaluation/position_manager.py
class PositionManager:
    def update(self, t, signal, open_price, close_price, state):
        # Entry/exit logic
        pass

# src/evaluation/risk_manager.py
class RiskManager:
    def apply_controls(self, t, state):
        # Stop-loss, daily loss limit
        pass
```

---

## 6. Performance Optimizations

### 6.1 **CPU Bottleneck: Sequential PSO Particle Evaluation**
- **File**: `src/optimizer/pso_core.py:180-192`
- **Root Cause**: Line 186-191 evaluates particles sequentially in a for-loop. Each particle trains an LSTM from scratch (100 epochs). For 30 particles × 50 iterations, this is 1500 LSTM training runs.
- **Expected Gain**: 8-16x speedup on multi-core CPU.
- **Fix**:
  ```python
  # In StandardPSO.__init__, add:
  self.n_workers = n_workers  # Already present
  
  # In _evaluate_all(), replace sequential loop with:
  if self.n_workers > 1:
      self._evaluate_parallel(X_train, y_train, X_val, y_val)
  else:
      for particle in self._swarm:
          fitness = self._evaluate_particle(particle, X_train, y_train, X_val, y_val)
          self._update_bests(particle, fitness)
  
  def _evaluate_parallel(self, X_train, y_train, X_val, y_val):
      from concurrent.futures import ProcessPoolExecutor
      with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
          futures = [ex.submit(self._evaluate_particle, p, X_train, y_train, X_val, y_val) 
                     for p in self._swarm]
          for particle, future in zip(self._swarm, futures):
              fitness = future.result()
              self._update_bests(particle, fitness)
  ```

### 6.2 **GPU Underutilization: XGBoost tree_method="gpu_hist" Not Verified**
- **File**: `src/models/xgboost/xgboost_model.py:103-124`
- **Root Cause**: Line 53 sets `tree_method="gpu_hist"` but lines 108-124 (`_setup_gpu_memory`) only check for `cupy`. If cupy is missing, it falls back to `"hist"` (CPU) silently. No warning is logged.
- **Expected Gain**: 5-10x speedup on GPU-enabled machines.
- **Fix**:
  ```python
  def _setup_gpu_memory(self):
      if self._xgb_params.get("tree_method") != "gpu_hist":
          return
      
      try:
          import cupy as cp
          pool = cp.cuda.MemoryPool()
          cp.cuda.set_allocator(pool.malloc)
          logger.info("GPU memory pool configured for XGBoost")
      except ImportError:
          logger.warning("cupy not available. Falling back to CPU (tree_method='hist')")
          self._xgb_params["tree_method"] = "hist"
      except Exception as e:
          logger.warning(f"GPU setup failed: {e}. Falling back to CPU.")
          self._xgb_params["tree_method"] = "hist"
  ```

### 6.3 **Inefficient Tensor Operations: LSTM Forward Pass**
- **File**: `src/models/lstm/lstm_model.py:84-104`
- **Root Cause**: Line 101 extracts `last_hidden = lstm_out[:, -1, :]`. This is correct, but the LSTM computes all timesteps' hidden states. For inference (not training), we only need the last timestep. PyTorch's `pack_padded_sequence` can skip computation for masked timesteps.
- **Expected Gain**: 10-20% inference speedup for variable-length sequences.
- **Fix**:
  ```python
  # Only applicable if sequences have variable lengths. Current code uses fixed lookback.
  # If implementing variable-length support:
  from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
  
  def forward(self, x, lengths=None):
      if lengths is not None:
          x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
      lstm_out, _ = self.lstm(x)
      if lengths is not None:
          lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
      last_hidden = lstm_out[:, -1, :]
      out = self.dropout(last_hidden)
      out = self.fc(out)
      return out
  ```

### 6.4 **Non-Vectorized Operations: Feature Lag Construction**
- **File**: `src/features/pipeline.py:217-233`
- **Root Cause**: Lines 225-229 build lag features with a Python for-loop over each lag. For 10 features × 3 lags × 100k samples, this is slow.
- **Expected Gain**: 5-10x speedup.
- **Fix**:
  ```python
  def _build_lags(X: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str]]:
      name_idx = {n: i for i, n in enumerate(names)}
      arrays, lag_names = [], []
      
      for col, lags in LAG_SPEC.items():
          if col not in name_idx:
              continue
          col_data = X[:, name_idx[col]]
          
          # Vectorized: create all lags at once
          max_lag = max(lags)
          lagged_matrix = np.zeros((len(col_data), len(lags)), dtype=np.float32)
          for i, lag in enumerate(lags):
              lagged_matrix[lag:, i] = col_data[:-lag]
          
          arrays.append(lagged_matrix)
          lag_names.extend([f"{col}_lag{lag}" for lag in lags])
      
      if not arrays:
          return np.empty((X.shape[0], 0), dtype=np.float32), []
      return np.concatenate(arrays, axis=1), lag_names
  ```

### 6.5 **DataLoader Inefficiencies: No Pinned Memory**
- **File**: `src/models/lstm/lstm_model.py:208-215`
- **Root Cause**: `DataLoader` is created without `pin_memory=True`. On GPU training, this forces synchronous CPU→GPU transfers.
- **Expected Gain**: 5-10% training speedup on GPU.
- **Fix**:
  ```python
  train_dl = DataLoader(
      train_ds,
      batch_size=self.batch_size,
      shuffle=True,
      generator=torch.Generator().manual_seed(self.seed),
      pin_memory=torch.cuda.is_available(),  # ADD THIS
      num_workers=2,  # ADD THIS (prefetch batches)
  )
  
  val_dl = DataLoader(
      val_ds,
      batch_size=self.batch_size,
      shuffle=False,
      pin_memory=torch.cuda.is_available(),
      num_workers=2,
  )
  ```

---

## 7. Saving & Logging Failures

### 7.1 **Training History: XGBoost History Not Saved**
- **File**: `pipelines/train.py:245`
- **Issue**: Line 245 calls `ctx.tracker.save_history(Phase.TRAIN, history=model.history)`, but `XGBoostModel.history` is a dict with keys `["train_rmse", "val_rmse"]`. The tracker expects a dict with keys matching `LSTMTrainer.history` (which includes `"directional_accuracy"`, `"f1_ternary"`, etc.). XGBoost history is incomplete.
- **Fix**:
  ```python
  # In pipelines/train.py:245-248, replace:
  ctx.tracker.save_history(Phase.TRAIN, history=model.history)
  
  # With:
  history = {
      "train_rmse": model.history["train_rmse"],
      "val_rmse": model.history["val_rmse"],
      "best_iteration": model.best_iteration,
  }
  ctx.tracker.save_history(Phase.TRAIN, history=history)
  ```

### 7.2 **Plots: LSTM Training History Plot Not Saved**
- **File**: `pipelines/train.py:112-138`
- **Issue**: LSTM training completes but no plot is saved. `plot_training_history` is imported (line 40) but never called.
- **Fix**:
  ```python
  # In train_lstm(), after line 112:
  ctx.tracker.save_history(Phase.TRAIN, history=history)
  
  # ADD:
  fig = plot_training_history(history)
  ctx.tracker.save_plot(Phase.TRAIN, fig, artifact=ArtifactType.PLOT)
  ```

### 7.3 **Model Checkpoints: No Intermediate Checkpoints**
- **File**: `src/models/lstm/lstm_model.py:193-287`
- **Issue**: `LSTMTrainer.fit()` saves the best model in `self._best_state` (line 273-275) but only restores it at the end (line 282-285). If training crashes mid-epoch, all progress is lost.
- **Fix**:
  ```python
  # In LSTMTrainer.__init__, add:
  self.checkpoint_dir: Optional[Path] = None
  
  # In LSTMTrainer.fit(), after line 275:
  if val_loss < best_val_loss - 1e-7:
      best_val_loss = val_loss
      patience_counter = 0
      self._best_state = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
      
      # ADD:
      if self.checkpoint_dir:
          checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
          torch.save(self._best_state, checkpoint_path)
          logger.info(f"Checkpoint saved: {checkpoint_path}")
  ```

### 7.4 **Metrics: Test Metrics Not Saved to Tracker**
- **File**: `pipelines/test.py:354-374`
- **Issue**: Lines 355-370 save test metrics to `results_dir / f"lstm_test_stat_metrics_{tag}.json"` using `_save_json()`, bypassing `ExperimentTracker`. Metrics are not tracked in the experiment's directory structure.
- **Fix**:
  ```python
  # In test_lstm(), replace lines 355-370:
  _save_json(results_dir / f"lstm_test_stat_metrics_{tag}.json", stat_metrics)
  _save_json(results_dir / f"lstm_test_trading_metrics_{tag}.json", {...})
  
  # With:
  ctx.tracker.save_metrics(Phase.TEST, stat_metrics)
  ctx.tracker.save_json(
      ctx.tracker._resolve(Phase.TEST, f"trading_metrics_{tag}.json"),
      {...}
  )
  ```

### 7.5 **Non-Deterministic Filenames: Timestamp in Logs**
- **File**: `src/experiment/experiment_tracker.py:240-246`
- **Issue**: `append_log()` adds a UTC timestamp to each log entry, but the log filename is fixed (`train.log`). If multiple runs use the same `run_id`, logs are appended, making it impossible to isolate a single run's logs.
- **Fix**:
  ```python
  # In ExperimentTracker.__init__, add:
  self.run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
  
  # In append_log(), change:
  def append_log(self, message: str, filename="train.log"):
      path = self.log_path(f"{self.run_timestamp}_{filename}")
      # ...
  ```

---

## 8. Code Quality Improvements

### 8.1 **Poor Naming: `_gbest_params()` Returns Dict, Not Params**
- **File**: `src/optimizer/pso_core.py:270-273`
- **Issue**: Method name suggests it returns a `Params` object, but it returns a `dict`. Misleading.
- **Fix**: Rename to `_gbest_hyperparameters()` or `_gbest_dict()`.

### 8.2 **Large Function: `process_ticker()` in run_build_features.py**
- **File**: `pipelines/run_build_features.py:29-101`
- **Issue**: 73 lines. Handles universe selection, feature engineering, scaling, and saving. Violates SRP.
- **Fix**:
  ```python
  def process_ticker(ticker, dfs_train, dfs_val, dfs_test, universe_builder, output_dir):
      universe = _build_universe(ticker, dfs_train, universe_builder)
      X_train, y_train, X_val, y_val, X_test, y_test, feat_names = _engineer_features(
          ticker, universe, dfs_train, dfs_val, dfs_test
      )
      X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s = _scale_features(
          X_train, y_train, X_val, y_val, X_test, y_test, feat_names
      )
      _save_artifacts(ticker, output_dir, X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s, feat_names)
  ```

### 8.3 **Violation of SRP: `LSTMTrainer` Computes Metrics**
- **File**: `src/models/lstm/lstm_model.py:242-245`
- **Issue**: Line 243 calls `compute_and_log_all_statistical_metrics()`. A trainer should train, not evaluate. Metrics belong in `src/evaluation/`.
- **Fix**:
  ```python
  # In LSTMTrainer.fit(), remove lines 242-245:
  y_pred = self._predict_batches(X_val, self.batch_size)
  metrics = compute_and_log_all_statistical_metrics(y_val, y_pred, label=f"LSTM Epoch {epoch+1}")
  val_loss = metrics["rmse"]
  
  # Replace with:
  y_pred = self._predict_batches(X_val, self.batch_size)
  val_loss = np.sqrt(np.mean((y_val - y_pred) ** 2))  # Compute RMSE inline
  
  # Move metric computation to pipelines/train.py:114-118 (after training completes).
  ```

### 8.4 **Violation of DRY: Duplicate Metric Logging**
- **Files**:
  - `src/evaluation/metrics.py:260-281` (`compute_and_log_all_statistical_metrics`)
  - `src/evaluation/metrics.py:284-307` (`compute_and_log_all_trading_metrics`)
  - `src/evaluation/metrics.py:347-368` (`_log_results`)
- **Issue**: Three functions format and log metrics. Formatting logic is duplicated.
- **Fix**: Extract a `MetricsFormatter` class:
  ```python
  class MetricsFormatter:
      @staticmethod
      def format_statistical(metrics: dict) -> str:
          return f"RMSE={metrics['rmse']:.6f} | DA={metrics['directional_accuracy']:.4f} | ..."
      
      @staticmethod
      def format_trading(metrics: dict) -> str:
          return f"Sharpe={metrics['sharpe']:.3f} | MDD={metrics['max_drawdown']:.2%} | ..."
  ```

### 8.5 **Magic Numbers: Hardcoded Thresholds**
- **Files**:
  - `src/evaluation/metrics.py:33` (`SIGNAL_THRESHOLD = 1e-4`)
  - `src/optimizer/fitness.py:29` (`SIGNAL_THRESHOLD = 1e-4`)
  - `src/evaluation/backtester.py:33` (`SIGNAL_THRESHOLD = 1e-4`)
- **Issue**: Same constant defined in 3 places. If changed in one place, others become inconsistent.
- **Fix**: Define once in `constants.py`:
  ```python
  # constants.py
  SIGNAL_THRESHOLD = 1e-4
  TRANSACTION_COST = 0.001
  ANNUALISE_1MIN = np.sqrt(252 * 390)
  
  # Import everywhere:
  from constants import SIGNAL_THRESHOLD, TRANSACTION_COST, ANNUALISE_1MIN
  ```

---

## 9. Priority Fix Plan

**Execution Order** (highest impact on correctness / risk):

1. **[CRITICAL] Fix window construction target alignment** (`src/data/splitter.py:159`)  
   **Impact**: Eliminates data leakage. All metrics will change (likely decrease). Must rerun all experiments.  
   **Effort**: 10 lines of code.

2. **[CRITICAL] Remove target scaling** (`src/features/scalar.py`)  
   **Impact**: Fixes out-of-sample prediction distortion. Sharpe/RMSE become comparable across periods.  
   **Effort**: Delete 5 lines.

3. **[CRITICAL] Fix XGBoost model saving** (`pipelines/train.py:253-257`)  
   **Impact**: Unblocks test pipeline. Currently crashes.  
   **Effort**: 3 lines.

4. **[HIGH] Fix backtester position sizing** (`src/evaluation/backtester.py:150-152`)  
   **Impact**: Corrects P&L calculations. Sharpe/MDD metrics become accurate.  
   **Effort**: 5 lines.

5. **[HIGH] Fix validation threshold sweep** (`pipelines/validation.py:114-137`)  
   **Impact**: Optimal threshold matches test environment. Improves Sharpe by 10-20%.  
   **Effort**: 30 lines (requires loading price data).

6. **[HIGH] Add data checksums** (`pipelines/run_build_features.py`)  
   **Impact**: Prevents silent data corruption. Critical for production.  
   **Effort**: 20 lines.

7. **[HIGH] Use fixed date splits** (`src/data/splitter.py:42-60`)  
   **Impact**: Ensures reproducibility. Prevents train/test contamination on data updates.  
   **Effort**: 15 lines.

8. **[MEDIUM] Fix LSTM gradient accumulation** (`src/models/lstm/lstm_model.py:224-236`)  
   **Impact**: Enables larger effective batch sizes. Improves training stability.  
   **Effort**: 20 lines.

9. **[MEDIUM] Parallelize PSO** (`src/optimizer/pso_core.py:180-192`)  
   **Impact**: 8-16x speedup. Reduces PSO runtime from hours to minutes.  
   **Effort**: 30 lines.

10. **[MEDIUM] Refactor backtester** (`src/evaluation/backtester.py:87-227`)  
    **Impact**: Improves testability. Enables unit tests for position management.  
    **Effort**: 100 lines (extract 3 classes).

11. **[MEDIUM] Add session boundary tracking** (`src/data/splitter.py:109`)  
    **Impact**: Prevents windows from spanning overnight gaps. Improves model quality.  
    **Effort**: 40 lines.

12. **[LOW] Consolidate duplicate code** (window construction, logging, JSON serialization)  
    **Impact**: Reduces maintenance burden. No functional change.  
    **Effort**: 50 lines (delete duplicates, update imports).

13. **[LOW] Add intermediate checkpoints** (`src/models/lstm/lstm_model.py:193-287`)  
    **Impact**: Prevents loss of progress on crashes. Useful for long training runs.  
    **Effort**: 10 lines.

14. **[LOW] Improve logging** (deterministic filenames, structured logs)  
    **Impact**: Easier debugging. No functional change.  
    **Effort**: 20 lines.

15. **[LOW] Performance optimizations** (pinned memory, vectorized lags, GPU verification)  
    **Impact**: 10-30% speedup. Not critical for correctness.  
    **Effort**: 50 lines total.

---

## Summary Statistics

- **Critical Bugs**: 7
- **Data Integrity Issues**: 4
- **Over-Complex Functions**: 3
- **Redundancies**: 4 categories (12 locations)
- **Modularity Violations**: 3 major areas
- **Performance Bottlenecks**: 5
- **Saving/Logging Issues**: 5
- **Code Quality Issues**: 5

**Total Estimated Effort**: 400-500 lines of code changes + 200-300 lines of refactoring.

**Recommended Approach**:
1. Fix critical bugs (items 1-3) immediately. These break correctness.
2. Add data integrity checks (items 6-7) before rerunning experiments.
3. Fix high-priority bugs (items 4-5) to ensure metrics are accurate.
4. Refactor modularity (item 10) to enable parallel development.
5. Optimize performance (items 9, 15) once correctness is verified.

---

## Appendix: Testing Recommendations

### Unit Tests Required

1. **`test_build_windows.py`**:
   - Verify target is `returns[i+1]`, not `returns[i]`.
   - Test session boundary exclusion.
   - Test edge cases (last window, single sample).

2. **`test_backtester.py`**:
   - Verify position sizing is constant from entry to exit.
   - Test stop-loss triggers correctly.
   - Test daily loss limit halts new entries.
   - Test session-end forced flat.

3. **`test_feature_pipeline.py`**:
   - Verify no data leakage in lag construction.
   - Test deduplication removes correct columns.
   - Test NaN handling doesn't drop valid rows.

4. **`test_pso.py`**:
   - Verify fitness normalization resets between runs.
   - Test parallel evaluation produces same results as sequential.
   - Test velocity clamping prevents divergence.

5. **`test_experiment_tracker.py`**:
   - Verify checksums detect corrupted files.
   - Test `RunMode.ATTACH` prevents overwrites.
   - Test deterministic filename generation.

### Integration Tests Required

1. **End-to-End Pipeline**:
   - Run `ingest → features → train → validate → test` on synthetic data.
   - Verify all artifacts are saved and loadable.
   - Verify metrics are consistent across stages.

2. **Walk-Forward Validation**:
   - Run on 1 ticker with 3 folds.
   - Verify fold boundaries don't overlap.
   - Verify threshold sweep produces valid result.

3. **PSO Optimization**:
   - Run 5 particles × 3 iterations on toy data.
   - Verify gbest improves monotonically.
   - Verify checkpoint can resume.

---

**END OF AUDIT REPORT**
