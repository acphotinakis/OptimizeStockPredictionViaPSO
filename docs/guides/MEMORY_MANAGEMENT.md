# Memory Management Guide - Preventing CUDA OOM Errors

**Last Updated:** April 6, 2026

---

## Issue: CUDA Out of Memory Error

### Error Message
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 23.17 GiB. 
GPU 0 has a total capacity of 7.92 GiB of which 6.15 GiB is free.
```

### Root Cause

The error occurred in `src/models/lstm_model.py` during LSTM training because:

1. **Entire validation set loaded to GPU** - Lines 214-215 moved the full validation set to GPU memory at once
2. **No batching for validation** - Validation forward pass processed all data in one go
3. **Memory accumulation** - Training batches + gradients + optimizer state already in GPU memory
4. **Large dataset** - Validation set too large to fit in remaining GPU memory

---

## Solutions Implemented

### Fix 1: Batch Validation Processing ✅

**Changed:** `src/models/lstm_model.py` LSTMTrainer.fit()

**Before (BROKEN):**
```python
# Load entire validation set to GPU (BAD!)
X_val_t = torch.FloatTensor(X_val).to(self.device)
y_val_t = torch.FloatTensor(y_val).unsqueeze(-1).to(self.device)

# Validation
self.model.eval()
with torch.no_grad():
    val_pred = self.model(X_val_t)  # OOM HERE!
    val_loss = self.criterion(val_pred, y_val_t).item()
```

**After (FIXED):**
```python
# Create validation DataLoader to batch processing
val_ds = TensorDataset(
    torch.FloatTensor(X_val),
    torch.FloatTensor(y_val).unsqueeze(-1),
)
val_dl = DataLoader(
    val_ds,
    batch_size=physical_batch_size,
    shuffle=False,
)

# Validation with batching
self.model.eval()
val_loss = 0.0
with torch.no_grad():
    for x_val_b, y_val_b in val_dl:
        x_val_b = x_val_b.to(self.device)
        y_val_b = y_val_b.to(self.device)
        val_pred_b = self.model(x_val_b)
        val_loss += self.criterion(val_pred_b, y_val_b).item() * len(x_val_b)
val_loss = val_loss / len(val_ds)
```

**Benefits:**
- ✅ Processes validation in batches (same size as training)
- ✅ Only loads one batch to GPU at a time
- ✅ Automatically handles any validation set size
- ✅ No change to validation loss computation

### Fix 2: Device Selection Option ✅

**Changed:** `scripts/run_lstm_baseline.py`

Added `--device` argument to force CPU if needed:

```python
parser.add_argument("--device", type=str, default=None, 
                   help="Device to use (cpu/cuda). Auto-detect if not specified.")
```

**Usage:**
```bash
# Force CPU (slower but no OOM)
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --device cpu

# Force CUDA (default)
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --device cuda
```

---

## Prevention Strategies

### 1. Always Batch Large Tensors ✅

**Rule:** Never load entire dataset to GPU at once.

**Bad Pattern:**
```python
# DON'T DO THIS
X_all = torch.FloatTensor(X).to(device)  # OOM if X is large!
predictions = model(X_all)
```

**Good Pattern:**
```python
# DO THIS
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
predictions = []
for x_batch in dataloader:
    x_batch = x_batch.to(device)
    pred_batch = model(x_batch)
    predictions.append(pred_batch.cpu())
predictions = torch.cat(predictions)
```

### 2. Use Appropriate Batch Sizes

**Guidelines:**
- **Small GPU (4-8 GB):** batch_size = 64-128
- **Medium GPU (8-16 GB):** batch_size = 128-256
- **Large GPU (16+ GB):** batch_size = 256-512

**Adjust based on:**
- Model size (layers × hidden_units)
- Sequence length (lookback)
- Number of features

### 3. Enable Memory Optimizations

**Mixed Precision Training (FP16):**
```python
# Reduces memory by ~50%
trainer = LSTMTrainer(
    model=model,
    use_amp=True,  # Enable mixed precision
    ...
)
```

**Gradient Accumulation:**
```python
# Simulate larger batches without more memory
trainer = LSTMTrainer(
    model=model,
    batch_size=64,           # Physical batch
    accumulation_steps=4,    # Effective batch = 256
    ...
)
```

**Gradient Checkpointing:**
```python
# Trade compute for memory
model = LSTMModel(
    use_checkpointing=True,  # Saves memory during backward pass
    ...
)
```

### 4. Clear GPU Cache Regularly

```python
import torch

# After training/evaluation
torch.cuda.empty_cache()

# Before loading new data
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 5. Monitor Memory Usage

```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

# Call periodically
print_gpu_memory()
```

---

## Troubleshooting Guide

### Issue: OOM during training

**Symptoms:**
```
torch.OutOfMemoryError: CUDA out of memory
```

**Solutions (in order):**

1. **Reduce batch size:**
   ```bash
   python scripts/run_lstm_baseline.py \
       --ticker AAPL \
       --mode train \
       --batch-size 64  # Reduce from 256
   ```

2. **Enable gradient accumulation:**
   ```yaml
   # config/default_config.yaml
   lstm_baseline:
     batch_size: 64
     accumulation_steps: 4  # Effective batch = 256
   ```

3. **Reduce model size:**
   ```bash
   python scripts/run_lstm_baseline.py \
       --ticker AAPL \
       --mode train \
       --num-layers 1 \
       --hidden-units 64
   ```

4. **Use CPU:**
   ```bash
   python scripts/run_lstm_baseline.py \
       --ticker AAPL \
       --mode train \
       --device cpu
   ```

### Issue: OOM during validation

**Symptoms:**
```
OutOfMemoryError in validation loop
```

**Solution:**
This is now fixed in `src/models/lstm_model.py`. If you still see this:

1. **Update code** to latest version
2. **Reduce batch size** (affects both train and val)
3. **Use CPU** for validation only (modify code)

### Issue: OOM during prediction

**Symptoms:**
```
OutOfMemoryError in model.predict()
```

**Solution:**
Ensure `predict()` uses batching:

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """Predict with batching to avoid OOM."""
    self.eval()
    predictions = []
    
    # Create DataLoader
    dataset = TensorDataset(torch.FloatTensor(X))
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    with torch.no_grad():
        for (x_batch,) in dataloader:
            x_batch = x_batch.to(self.device)
            pred_batch = self.forward(x_batch).squeeze(-1)
            predictions.append(pred_batch.cpu())
    
    return torch.cat(predictions).numpy()
```

### Issue: Gradual memory leak

**Symptoms:**
- Memory usage increases over epochs
- Eventually OOM after many epochs

**Solutions:**

1. **Clear cache between epochs:**
   ```python
   for epoch in range(max_epochs):
       # Training...
       
       # Clear cache
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
   ```

2. **Detach tensors in history:**
   ```python
   # Don't keep computation graph
   self.history["train_loss"].append(loss.item())  # .item() detaches
   ```

3. **Move best state to CPU:**
   ```python
   self._best_state = {
       k: v.clone().cpu() for k, v in self.model.state_dict().items()
   }
   ```

---

## Best Practices

### 1. Design for Memory Efficiency

**Always:**
- ✅ Use DataLoader for all data processing
- ✅ Process data in batches
- ✅ Move tensors to CPU when not needed
- ✅ Use `.item()` to extract scalars
- ✅ Delete large tensors explicitly with `del`

**Never:**
- ❌ Load entire dataset to GPU
- ❌ Keep unnecessary tensors in GPU memory
- ❌ Accumulate gradients without clearing
- ❌ Store computation graphs in history

### 2. Test with Small Data First

```bash
# Test with small subset first
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --max-epochs 2 \
    --batch-size 32
```

### 3. Profile Memory Usage

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model.fit(X_train, y_train, X_val, y_val)

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

### 4. Use Memory-Optimized Config

Create `config/memory_optimized.yaml`:

```yaml
lstm_baseline:
  num_layers: 1
  hidden_units: 64
  batch_size: 64
  accumulation_steps: 4
  use_amp: true
  use_checkpointing: true
```

Use it:
```bash
python scripts/run_lstm_baseline.py \
    --ticker AAPL \
    --mode train \
    --config config/memory_optimized.yaml
```

---

## Memory Estimation

### Formula

Approximate GPU memory needed:

```
Memory (GB) ≈ (
    batch_size × lookback × features × 4 bytes  # Input
    + batch_size × hidden_units × num_layers × 4 bytes × 4  # LSTM states
    + num_layers × hidden_units × hidden_units × 4 bytes × 4  # LSTM weights
    + batch_size × hidden_units × 4 bytes  # Activations
) × 2  # Forward + Backward
```

### Example Calculation

For LSTM baseline defaults:
- batch_size = 256
- lookback = 30
- features = 100
- hidden_units = 128
- num_layers = 2

```
Input: 256 × 30 × 100 × 4 = 3.07 MB
States: 256 × 128 × 2 × 4 × 4 = 1.05 MB
Weights: 2 × 128 × 128 × 4 × 4 = 0.52 MB
Activations: 256 × 128 × 4 = 0.13 MB

Total (forward): ~4.77 MB
Total (forward + backward): ~9.54 MB
With optimizer states: ~14.31 MB per batch

Safe estimate: ~2 GB for full training
```

### Quick Reference

| Configuration | Estimated Memory | Recommended GPU |
|---------------|------------------|-----------------|
| Small (1 layer, 64 hidden) | ~1 GB | 4 GB |
| Default (2 layers, 128 hidden) | ~2 GB | 8 GB |
| Large (3 layers, 256 hidden) | ~4 GB | 12 GB |
| Extra Large (4 layers, 512 hidden) | ~8 GB | 16 GB |

---

## Environment Variables

Set these to help with memory issues:

```bash
# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Limit PyTorch memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable memory debugging
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6

# Combine multiple settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

---

## Summary

### Key Takeaways

1. **Always batch large tensors** - Never load entire dataset to GPU
2. **Use DataLoader** - For both training and validation
3. **Monitor memory** - Profile and track usage
4. **Start small** - Test with small batches/models first
5. **Enable optimizations** - Mixed precision, gradient accumulation, checkpointing

### Fixed Issues ✅

- ✅ Validation batching in `src/models/lstm_model.py`
- ✅ Device selection in `scripts/run_lstm_baseline.py`
- ✅ Memory-efficient validation loop
- ✅ Proper tensor cleanup

### Quick Fix Commands

```bash
# If you get OOM, try these in order:

# 1. Reduce batch size
--batch-size 64

# 2. Use CPU
--device cpu

# 3. Reduce model size
--num-layers 1 --hidden-units 64

# 4. Use memory-optimized config
--config config/memory_optimized.yaml
```

---

**For more information:**
- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
