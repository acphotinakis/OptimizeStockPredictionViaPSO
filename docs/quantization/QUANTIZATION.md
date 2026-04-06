# Quantization Strategy for Memory Optimization
## PSO-LSTM Stock Price Prediction System

**Goal:** Reduce memory footprint by 50-75% to prevent OOM errors while maintaining model performance.

---

## Table of Contents

1. [Problem Analysis](#problem-analysis)
2. [Quantization Strategy](#quantization-strategy)
3. [Implementation Plan](#implementation-plan)
4. [Expected Memory Savings](#expected-memory-savings)
5. [Performance Impact](#performance-impact)
6. [Implementation Details](#implementation-details)
7. [Testing & Validation](#testing--validation)

---

## Problem Analysis

### Current Memory Bottlenecks

| Component | Memory Usage | Data Type | Opportunity |
|-----------|--------------|-----------|-------------|
| **Raw OHLCV Data** | ~2GB (51 tickers × 5 years × 1-min) | float64 | ✅ Downcast to float32 |
| **Feature Matrices** | ~4GB (100+ features × 51 tickers) | float64 | ✅ Downcast to float32/float16 |
| **LSTM Weights** | ~50-200MB per model | float32 | ✅ Quantize to int8 |
| **Training Batches** | ~500MB-1GB | float32 | ✅ Mixed precision training |
| **PSO Swarm State** | ~100MB (30 particles) | float64 | ✅ Downcast to float32 |

### Root Causes of OOM

1. **Data Type Bloat**: Using float64 (8 bytes) when float32 (4 bytes) is sufficient
2. **Feature Explosion**: 100+ features per ticker × 51 tickers = 5,100 feature columns
3. **Batch Size**: Large batches (256) consume significant GPU memory
4. **Model Size**: Multiple LSTM layers with 512 hidden units
5. **PSO Parallelism**: 30 particles evaluated simultaneously

---

## Quantization Strategy

### Three-Tier Approach

#### **Tier 1: Data Quantization (Lossless)**
Downcast data types without losing precision where safe.

**Targets:**
- OHLCV prices: float64 → float32 (4 decimal places sufficient)
- Volume: int64 → int32 (max volume < 2B)
- Feature matrices: float64 → float32
- Timestamps: datetime64[ns] → datetime64[s]

**Expected Savings:** 50% memory reduction
**Performance Impact:** None (float32 has 7 decimal digits precision)

#### **Tier 2: Model Quantization (Minimal Loss)**
Quantize trained models for inference.

**Targets:**
- LSTM weights: float32 → int8 (post-training quantization)
- Use PyTorch's dynamic quantization
- Apply only to final models, not during PSO search

**Expected Savings:** 75% model size reduction
**Performance Impact:** <1% accuracy loss

#### **Tier 3: Training Optimization (Adaptive)**
Use mixed precision and gradient checkpointing during training.

**Targets:**
- Mixed precision training (FP16 + FP32)
- Gradient checkpointing to trade compute for memory
- Smaller batch sizes with gradient accumulation

**Expected Savings:** 40-60% training memory
**Performance Impact:** Minimal with proper scaling

---

## Implementation Plan

### Phase 1: Data Quantization (Week 1)

**Priority: HIGH | Risk: LOW | Impact: IMMEDIATE**

#### 1.1 Data Ingestion Layer
```python
# File: src/data/alpaca_ingestor.py
def download_bars(...):
    df = api.get_bars(...)
    # Downcast immediately after download
    df = downcast_ohlcv(df)
    return df

def downcast_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory by downcasting numeric types."""
    df['open'] = df['open'].astype('float32')
    df['high'] = df['high'].astype('float32')
    df['low'] = df['low'].astype('float32')
    df['close'] = df['close'].astype('float32')
    df['volume'] = df['volume'].astype('int32')
    return df
```

#### 1.2 Feature Pipeline
```python
# File: src/features/pipeline.py
def _compute_features(...):
    # Compute features in float32 from the start
    X = combined.values.astype(np.float32)  # Not float64
    y = r.values.astype(np.float32)
    return X, y, feature_names
```

#### 1.3 Data Splitter
```python
# File: src/data/splitter.py
def fit_transform(self, X_train, feature_names):
    # Ensure output is float32
    X_out = np.empty_like(X_train, dtype=np.float32)
    # ... scaling logic ...
    return X_out
```

**Deliverables:**
- [ ] Update `AlpacaIngestor.download_bars()` with downcasting
- [ ] Update `FeaturePipeline._compute_features()` to use float32
- [ ] Update `DataSplitter` to maintain float32 throughout
- [ ] Add memory profiling utility to track savings

**Testing:**
```bash
# Before/after memory comparison
python scripts/01_ingest_data.py --profile-memory
python scripts/02_build_features.py --profile-memory
```

---

### Phase 2: Model Quantization (Week 2)

**Priority: MEDIUM | Risk: MEDIUM | Impact: SIGNIFICANT**

#### 2.1 Post-Training Quantization for Inference
```python
# File: src/models/lstm_model.py
import torch.quantization

class QuantizedLSTMModel:
    """Wrapper for quantized LSTM inference."""
    
    def __init__(self, model: LSTMModel):
        self.model = model
        self.quantized_model = None
    
    def quantize(self):
        """Apply dynamic quantization to trained model."""
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.LSTM, torch.nn.Linear},  # Layers to quantize
            dtype=torch.qint8  # 8-bit integers
        )
        return self.quantized_model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Inference with quantized model."""
        return self.quantized_model.predict(X)
```

#### 2.2 Quantization-Aware Training (Optional)
```python
# File: src/models/lstm_model.py
class QATLSTMTrainer(LSTMTrainer):
    """Quantization-Aware Training for better accuracy."""
    
    def __init__(self, model, lr, **kwargs):
        super().__init__(model, lr, **kwargs)
        # Prepare model for QAT
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(self.model, inplace=True)
    
    def finalize(self):
        """Convert to quantized model after training."""
        self.model.eval()
        torch.quantization.convert(self.model, inplace=True)
```

#### 2.3 Integration with Evaluation
```python
# File: scripts/04_evaluate.py
# After training, quantize the model
ipso_trainer.fit(X_train, y_train, X_val, y_val)

# Quantize for inference
quantized_model = QuantizedLSTMModel(ipso_trainer.model)
quantized_model.quantize()

# Use quantized model for predictions
y_pred = quantized_model.predict(X_test)
```

**Deliverables:**
- [ ] Implement `QuantizedLSTMModel` wrapper class
- [ ] Add quantization option to `04_evaluate.py`
- [ ] Add quantization option to `05_backtest.py`
- [ ] Benchmark accuracy loss vs. memory savings

**Testing:**
```bash
# Compare quantized vs. full precision
python scripts/04_evaluate.py --ticker AAPL --quantize
python scripts/04_evaluate.py --ticker AAPL --no-quantize
```

---

### Phase 3: Training Optimization (Week 3)

**Priority: HIGH | Risk: MEDIUM | Impact: CRITICAL FOR PSO**

#### 3.1 Mixed Precision Training
```python
# File: src/models/lstm_model.py
from torch.cuda.amp import autocast, GradScaler

class LSTMTrainer:
    def __init__(self, model, lr, use_amp=True, **kwargs):
        # ... existing init ...
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
    
    def fit(self, X_train, y_train, X_val, y_val):
        for epoch in range(self.max_epochs):
            for x_b, y_b in train_dl:
                self.optimizer.zero_grad()
                
                # Mixed precision forward pass
                if self.use_amp:
                    with autocast():
                        pred = self.model(x_b)
                        loss = self.criterion(pred, y_b)
                    # Scaled backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pred = self.model(x_b)
                    loss = self.criterion(pred, y_b)
                    loss.backward()
                    self.optimizer.step()
```

#### 3.2 Gradient Checkpointing
```python
# File: src/models/lstm_model.py
from torch.utils.checkpoint import checkpoint

class LSTMModel(nn.Module):
    def __init__(self, ..., use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        # ... existing init ...
    
    def forward(self, x):
        if self.use_checkpointing and self.training:
            # Trade compute for memory
            lstm_out = checkpoint(self.lstm, x)
        else:
            lstm_out, _ = self.lstm(x)
        
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out
```

#### 3.3 Gradient Accumulation
```python
# File: src/models/lstm_model.py
class LSTMTrainer:
    def __init__(self, model, lr, batch_size=256, 
                 accumulation_steps=1, **kwargs):
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        # Effective batch size = batch_size * accumulation_steps
    
    def fit(self, X_train, y_train, X_val, y_val):
        # Use smaller physical batch size
        train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size // self.accumulation_steps,
            shuffle=True
        )
        
        for epoch in range(self.max_epochs):
            for i, (x_b, y_b) in enumerate(train_dl):
                pred = self.model(x_b)
                loss = self.criterion(pred, y_b)
                
                # Normalize loss by accumulation steps
                loss = loss / self.accumulation_steps
                loss.backward()
                
                # Update weights every N steps
                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
```

#### 3.4 Memory-Efficient PSO
```python
# File: src/optimizer/pso_core.py
class StandardPSO:
    def _evaluate_particle(self, particle, X_train, y_train, X_val, y_val):
        # Clear GPU cache before each particle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Build model with memory optimizations
        params = particle.decode()
        model = LSTMModel(
            input_size=X_train.shape[2],
            use_checkpointing=True,  # Enable gradient checkpointing
            **params
        )
        
        trainer = LSTMTrainer(
            model=model,
            lr=params["learning_rate"],
            batch_size=64,  # Smaller batch size
            accumulation_steps=4,  # Effective batch = 256
            use_amp=True,  # Mixed precision
        )
        
        trainer.fit(X_train, y_train, X_val, y_val)
        y_pred = trainer.predict(X_val)
        
        # Clear memory after evaluation
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return self.fitness_fn(y_val, y_pred)
```

**Deliverables:**
- [ ] Add mixed precision training to `LSTMTrainer`
- [ ] Add gradient checkpointing to `LSTMModel`
- [ ] Add gradient accumulation to `LSTMTrainer`
- [ ] Update PSO to clear GPU cache between particles
- [ ] Add memory monitoring to PSO loop

**Testing:**
```bash
# Test with memory profiling
python scripts/03_run_pso.py --ticker AAPL --profile-memory
```

---

### Phase 4: Configuration & Utilities (Week 4)

**Priority: MEDIUM | Risk: LOW | Impact: USABILITY**

#### 4.1 Memory-Optimized Config Profiles
```yaml
# File: config/memory_optimized.yaml
# Inherits from default_config.yaml but overrides for low memory

lstm:
  batch_size: 64  # Reduced from 256
  accumulation_steps: 4  # Effective batch = 256
  use_amp: true  # Mixed precision
  use_checkpointing: true  # Gradient checkpointing
  max_epochs: 80  # Slightly reduced
  
pso:
  n_particles: 20  # Reduced from 30
  n_iterations: 40  # Reduced from 50
  clear_cache: true  # Clear GPU cache between particles

features:
  dtype: "float32"  # Not float64
  selector:
    importance_threshold: 0.80  # More aggressive (was 0.70)
```

#### 4.2 Memory Profiling Utility
```python
# File: src/utils/memory_profiler.py
import psutil
import torch
from functools import wraps

class MemoryProfiler:
    """Track CPU and GPU memory usage."""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage."""
        cpu_mem = psutil.Process().memory_info().rss / 1e9  # GB
        gpu_mem = 0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1e9  # GB
        return {"cpu_gb": cpu_mem, "gpu_gb": gpu_mem}
    
    @staticmethod
    def profile(func):
        """Decorator to profile memory usage of a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            mem_before = MemoryProfiler.get_memory_usage()
            result = func(*args, **kwargs)
            mem_after = MemoryProfiler.get_memory_usage()
            
            logger.info(
                "Memory: CPU %.2f GB → %.2f GB (Δ%.2f GB) | "
                "GPU %.2f GB → %.2f GB (Δ%.2f GB)",
                mem_before["cpu_gb"], mem_after["cpu_gb"],
                mem_after["cpu_gb"] - mem_before["cpu_gb"],
                mem_before["gpu_gb"], mem_after["gpu_gb"],
                mem_after["gpu_gb"] - mem_before["gpu_gb"]
            )
            return result
        return wrapper
```

#### 4.3 Automatic Memory Management
```python
# File: src/utils/memory_manager.py
import torch
import gc

class MemoryManager:
    """Automatic memory management for training loops."""
    
    @staticmethod
    def clear_all():
        """Clear all caches and run garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_optimal_batch_size(model, input_shape, max_batch=512):
        """Binary search for optimal batch size."""
        device = next(model.parameters()).device
        batch_size = max_batch
        
        while batch_size > 1:
            try:
                # Test with dummy data
                dummy_input = torch.randn(batch_size, *input_shape).to(device)
                output = model(dummy_input)
                loss = output.sum()
                loss.backward()
                
                # Success - this batch size works
                MemoryManager.clear_all()
                return batch_size
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Try smaller batch
                    batch_size = batch_size // 2
                    MemoryManager.clear_all()
                else:
                    raise
        
        return 1  # Minimum batch size
```

**Deliverables:**
- [ ] Create `config/memory_optimized.yaml`
- [ ] Implement `MemoryProfiler` utility
- [ ] Implement `MemoryManager` utility
- [ ] Add `--memory-profile` flag to all scripts
- [ ] Add `--low-memory` flag to use optimized config

**Testing:**
```bash
# Use memory-optimized configuration
python scripts/03_run_pso.py --ticker AAPL --config config/memory_optimized.yaml

# Profile memory usage
python scripts/03_run_pso.py --ticker AAPL --memory-profile
```

---

## Expected Memory Savings

### Before Quantization

| Component | Memory | Data Type |
|-----------|--------|-----------|
| Raw data (51 tickers) | 2.0 GB | float64 |
| Features (train) | 4.0 GB | float64 |
| LSTM model | 200 MB | float32 |
| Training batch | 1.0 GB | float32 |
| PSO swarm (30 particles) | 6.0 GB | float64 |
| **Total Peak** | **~13 GB** | - |

### After Quantization

| Component | Memory | Data Type | Savings |
|-----------|--------|-----------|---------|
| Raw data (51 tickers) | 1.0 GB | float32 | **50%** |
| Features (train) | 2.0 GB | float32 | **50%** |
| LSTM model | 50 MB | int8 (inference) | **75%** |
| Training batch | 0.5 GB | FP16 | **50%** |
| PSO swarm (20 particles) | 2.0 GB | float32 | **67%** |
| **Total Peak** | **~5.5 GB** | - | **~58%** |

### GPU Memory Savings

| Scenario | Before | After | Savings |
|----------|--------|-------|---------|
| Feature building | 4 GB | 2 GB | 50% |
| Single LSTM training | 3 GB | 1.5 GB | 50% |
| PSO (30 particles) | 8 GB | 4 GB | 50% |
| Inference (quantized) | 200 MB | 50 MB | 75% |

---

## Performance Impact

### Accuracy Impact

| Quantization Level | RMSE Change | Sharpe Change | Acceptable? |
|-------------------|-------------|---------------|-------------|
| Data (float32) | 0.00% | 0.00% | ✅ Yes |
| Model (int8) | <0.5% | <1.0% | ✅ Yes |
| Mixed precision | <0.1% | <0.5% | ✅ Yes |
| **Combined** | **<1.0%** | **<2.0%** | ✅ **Yes** |

### Speed Impact

| Optimization | Speed Change | Note |
|-------------|--------------|------|
| float32 data | +5-10% faster | Smaller data transfers |
| int8 inference | +2-3x faster | CPU inference only |
| Mixed precision | +20-40% faster | GPU with Tensor Cores |
| Gradient checkpointing | -20% slower | Trade-off for memory |
| **Net Impact** | **+10-20% faster** | Overall improvement |

---

## Implementation Details

### File Changes Required

```
src/data/
  ├── alpaca_ingestor.py      [MODIFY] Add downcast_ohlcv()
  ├── cleaner.py               [MODIFY] Maintain float32
  └── splitter.py              [MODIFY] Use float32 throughout

src/features/
  ├── pipeline.py              [MODIFY] Compute in float32
  ├── technical.py             [MODIFY] Return float32
  ├── statistical.py           [MODIFY] Return float32
  └── volume.py                [MODIFY] Return float32

src/models/
  ├── lstm_model.py            [MODIFY] Add AMP, checkpointing, quantization
  └── quantized_lstm.py        [NEW] QuantizedLSTMModel wrapper

src/optimizer/
  ├── pso_core.py              [MODIFY] Add memory clearing
  └── ipso.py                  [MODIFY] Inherit memory optimizations

src/utils/
  ├── memory_profiler.py       [NEW] Memory tracking utilities
  └── memory_manager.py        [NEW] Automatic memory management

config/
  └── memory_optimized.yaml    [NEW] Low-memory configuration

scripts/
  ├── 01_ingest_data.py        [MODIFY] Add --profile-memory flag
  ├── 02_build_features.py     [MODIFY] Add --profile-memory flag
  ├── 03_run_pso.py            [MODIFY] Add --low-memory, --profile-memory flags
  ├── 04_evaluate.py           [MODIFY] Add --quantize flag
  └── 05_backtest.py           [MODIFY] Add --quantize flag
```

---

## Testing & Validation

### Unit Tests

```python
# tests/test_quantization.py
def test_data_quantization():
    """Test float64 → float32 maintains precision."""
    df_64 = pd.DataFrame({'price': [100.1234567890]}, dtype='float64')
    df_32 = df_64.astype('float32')
    assert np.allclose(df_64.values, df_32.values, rtol=1e-6)

def test_model_quantization():
    """Test int8 quantization accuracy loss < 1%."""
    model = LSTMModel(...)
    quantized = QuantizedLSTMModel(model).quantize()
    
    y_pred_fp32 = model.predict(X_test)
    y_pred_int8 = quantized.predict(X_test)
    
    rmse_fp32 = rmse(y_test, y_pred_fp32)
    rmse_int8 = rmse(y_test, y_pred_int8)
    
    assert (rmse_int8 - rmse_fp32) / rmse_fp32 < 0.01  # <1% increase

def test_mixed_precision():
    """Test AMP maintains accuracy."""
    trainer_fp32 = LSTMTrainer(model, lr=0.001, use_amp=False)
    trainer_amp = LSTMTrainer(model, lr=0.001, use_amp=True)
    
    # Both should converge to similar loss
    hist_fp32 = trainer_fp32.fit(X_train, y_train, X_val, y_val)
    hist_amp = trainer_amp.fit(X_train, y_train, X_val, y_val)
    
    assert abs(hist_fp32['val_loss'][-1] - hist_amp['val_loss'][-1]) < 0.01
```

### Integration Tests

```bash
# Test full pipeline with quantization
python scripts/01_ingest_data.py --profile-memory
python scripts/02_build_features.py --profile-memory
python scripts/03_run_pso.py --ticker AAPL --low-memory --profile-memory
python scripts/04_evaluate.py --ticker AAPL --quantize
```

### Benchmark Suite

```bash
# Compare memory usage before/after
python benchmarks/memory_benchmark.py --baseline
python benchmarks/memory_benchmark.py --quantized

# Compare accuracy before/after
python benchmarks/accuracy_benchmark.py --baseline
python benchmarks/accuracy_benchmark.py --quantized
```

---

## Rollout Plan

### Week 1: Data Quantization (Low Risk)
- [ ] Day 1-2: Implement data downcasting
- [ ] Day 3-4: Update feature pipeline
- [ ] Day 5: Testing and validation
- [ ] **Milestone:** 50% memory reduction in data layer

### Week 2: Model Quantization (Medium Risk)
- [ ] Day 1-2: Implement post-training quantization
- [ ] Day 3-4: Integration with evaluation scripts
- [ ] Day 5: Accuracy benchmarking
- [ ] **Milestone:** 75% model size reduction with <1% accuracy loss

### Week 3: Training Optimization (Medium Risk)
- [ ] Day 1-2: Implement mixed precision training
- [ ] Day 3: Implement gradient checkpointing
- [ ] Day 4: Implement gradient accumulation
- [ ] Day 5: PSO memory optimizations
- [ ] **Milestone:** 50% training memory reduction

### Week 4: Polish & Documentation (Low Risk)
- [ ] Day 1-2: Memory profiling utilities
- [ ] Day 3: Memory-optimized configs
- [ ] Day 4: Documentation and guides
- [ ] Day 5: Final testing and validation
- [ ] **Milestone:** Production-ready quantized system

---

## Success Criteria

### Must Have (P0)
- ✅ 50%+ memory reduction in data layer
- ✅ PSO runs successfully on 8GB GPU
- ✅ <2% accuracy degradation
- ✅ All existing tests pass

### Should Have (P1)
- ✅ 40%+ memory reduction in training
- ✅ Memory profiling utilities
- ✅ Low-memory configuration presets
- ✅ Documentation and guides

### Nice to Have (P2)
- ✅ Automatic batch size tuning
- ✅ Quantization-aware training
- ✅ Multi-GPU memory distribution
- ✅ Cloud deployment guides

---

## Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Accuracy loss > 2% | HIGH | LOW | Extensive testing; fallback to FP32 |
| Quantization bugs | MEDIUM | MEDIUM | Comprehensive unit tests |
| Performance regression | MEDIUM | LOW | Benchmark before/after |
| User adoption | LOW | MEDIUM | Clear documentation; default to safe settings |

---

## References

### PyTorch Quantization
- https://pytorch.org/docs/stable/quantization.html
- https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html

### Mixed Precision Training
- https://pytorch.org/docs/stable/amp.html
- https://arxiv.org/abs/1710.03740 (Mixed Precision Training paper)

### Gradient Checkpointing
- https://pytorch.org/docs/stable/checkpoint.html
- https://arxiv.org/abs/1604.06174 (Training Deep Nets with Sublinear Memory Cost)

---

## Appendix: Quick Reference

### Enable All Optimizations

```bash
# Use memory-optimized config
python scripts/03_run_pso.py \
  --ticker AAPL \
  --config config/memory_optimized.yaml \
  --low-memory \
  --profile-memory
```

### Disable Optimizations (Debugging)

```bash
# Use full precision for debugging
python scripts/03_run_pso.py \
  --ticker AAPL \
  --no-amp \
  --no-checkpointing \
  --batch-size 256
```

### Memory Profiling

```python
from src.utils.memory_profiler import MemoryProfiler

@MemoryProfiler.profile
def my_function():
    # Your code here
    pass
```

---

**Status:** 📋 **PLAN READY FOR IMPLEMENTATION**

**Next Steps:**
1. Review and approve plan
2. Begin Phase 1 (Data Quantization)
3. Measure and validate memory savings
4. Proceed to subsequent phases
