# Peer Selection Implementation - Command Reference

Quick reference for all commands needed to implement the peer selection system.

---

## Phase 1: Data Ingestion (New Symbols)

### Step 1: Ingest 17 New Context Symbols

```bash
# Ingest market indices, sector ETFs, and market internals
python pipelines/ingest_data.py \
    --mode ingest \
    --tickers config/tickers_new_context.txt \
    --config config/default_config.yaml \
    --raw-output data/raw
```

**Expected Output:**
- 17 new parquet files in `data/raw/`
- QQQ.parquet, IWM.parquet, DIA.parquet
- XLK.parquet, SOXX.parquet, XLF.parquet, XLV.parquet, XLE.parquet, XLY.parquet, XLP.parquet, XLI.parquet, XLC.parquet, XLB.parquet, XLU.parquet
- UVXY.parquet, GLD.parquet, TLT.parquet

**Time:** ~30-60 minutes (depends on network speed)

---

### Step 2: Clean New Symbols

```bash
# Clean the 17 new symbols
python pipelines/ingest_data.py \
    --mode clean \
    --tickers config/tickers_new_context.txt \
    --config config/default_config.yaml \
    --raw-output data/raw \
    --cleaned-output data/cleaned
```

**Expected Output:**
- 17 cleaned parquet files in `data/cleaned/`
- Session start/end markers added
- Outliers clipped
- Gaps filled

**Time:** ~10-20 minutes

---

### Step 3: Re-Align Full Universe (69 Symbols)

```bash
# Align all 69 symbols (52 existing + 17 new)
python pipelines/ingest_data.py \
    --mode align \
    --tickers config/tickers_full.txt \
    --config config/default_config.yaml \
    --cleaned-output data/cleaned \
    --processed-output data/processed
```

**Expected Output:**
- 69 aligned parquet files in `data/processed/`
- Common timestamp grid across all symbols
- Inner join alignment (only timestamps where all symbols have data)

**Time:** ~20-40 minutes

---

### Verification Commands

```bash
# Check that all 69 symbols were processed
ls data/processed/*.parquet | wc -l
# Expected: 69

# Check alignment quality for a few symbols
python -c "
import pandas as pd

spy = pd.read_parquet('data/processed/SPY.parquet')
qqq = pd.read_parquet('data/processed/QQQ.parquet')
xlk = pd.read_parquet('data/processed/XLK.parquet')
aapl = pd.read_parquet('data/processed/AAPL.parquet')

print(f'SPY:  {len(spy)} rows, {spy.index.min()} to {spy.index.max()}')
print(f'QQQ:  {len(qqq)} rows, {qqq.index.min()} to {qqq.index.max()}')
print(f'XLK:  {len(xlk)} rows, {xlk.index.min()} to {xlk.index.max()}')
print(f'AAPL: {len(aapl)} rows, {aapl.index.min()} to {aapl.index.max()}')

# Check if indices are identical (should be True)
print(f'\\nIndices aligned: {spy.index.equals(qqq.index) and spy.index.equals(xlk.index) and spy.index.equals(aapl.index)}')
"
```

---

## Phase 2: Feature Building (With Universe Builder)

### Step 1: Build Features for Test Subset (3-5 Stocks)

First, test on a small subset to verify everything works:

```bash
# Temporarily modify config/symbol_universe.yaml to only include:
# prediction_targets:
#   - AAPL
#   - NVDA
#   - JPM

python pipelines/run_build_features.py \
    --config config/default_config.yaml \
    --input data/processed/aligned_universe.parquet \
    --output data/features \
    --tickers config/tickers_full.txt \
    --n-jobs 1
```

**Expected Output:**
- `data/features/AAPL/` directory with:
  - X_train.npy, y_train.npy
  - X_val.npy, y_val.npy
  - X_test.npy, y_test.npy
  - metadata.pkl
  - scaler.pkl
- `data/features/NVDA/` directory (same structure)
- `data/features/JPM/` directory (same structure)
- `data/features/fitted_peers.json` with peer selections

**Time:** ~10-20 minutes for 3 stocks

---

### Step 2: Inspect Fitted Peers

```bash
# View fitted peers
cat data/features/fitted_peers.json | python -m json.tool
```

**Expected Output:**
```json
{
  "AAPL": ["MSFT", "GOOGL", "META"],
  "NVDA": ["AMD", "INTC"],
  "JPM": ["BAC", "GS", "MS"]
}
```

**Validation:**
- AAPL should have tech peers (MSFT, GOOGL, META, CRM, ADBE)
- NVDA should have semiconductor peers (AMD, INTC)
- JPM should have financial peers (BAC, GS, MS, C, WFC)

---

### Step 3: Check Feature Counts

```bash
# Check feature metadata
python -c "
import pickle
import numpy as np

for ticker in ['AAPL', 'NVDA', 'JPM']:
    with open(f'data/features/{ticker}/metadata.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    X_train = np.load(f'data/features/{ticker}/X_train.npy')
    
    print(f'{ticker}:')
    print(f'  Features: {meta[\"n_features\"]}')
    print(f'  Train samples: {meta[\"train_samples\"]}')
    print(f'  Val samples: {meta[\"val_samples\"]}')
    print(f'  Test samples: {meta[\"test_samples\"]}')
    print(f'  Feature names (first 10): {meta[\"feature_names\"][:10]}')
    print(f'  X_train shape: {X_train.shape}')
    print()
"
```

**Expected Output:**
- Features: 40-60 (after selection)
- Train samples: ~300k-400k (depends on date range)
- Feature names should include cross-ticker features like:
  - `beta_spy_60`
  - `corr_spy_20`
  - `peer_corr_MSFT` (for AAPL)
  - `peer_corr_AMD` (for NVDA)
  - `peer_corr_BAC` (for JPM)

---

### Step 4: Build Features for All Targets

Once validated, build for all prediction targets:

```bash
# Update config/symbol_universe.yaml to include all desired targets
# Then run:

python pipelines/run_build_features.py \
    --config config/default_config.yaml \
    --input data/processed/aligned_universe.parquet \
    --output data/features \
    --tickers config/tickers_full.txt \
    --n-jobs 1
```

**Time:** ~2-4 hours for 51 stocks (sequential processing)

**Note:** Use `--n-jobs 1` to avoid memory issues. Parallel processing can be added later if memory permits.

---

## Phase 3: Testing

### Run Unit Tests

```bash
# Run universe builder tests
pytest tests/test_universe_builder.py -v

# Run pipeline integration tests
pytest tests/test_pipeline_with_universe.py -v

# Run all tests
pytest tests/ -v
```

---

### Manual Feature Inspection

```bash
# Compare old vs. new feature counts
python -c "
import numpy as np
import pickle

ticker = 'AAPL'

# Load new features
with open(f'data/features/{ticker}/metadata.pkl', 'rb') as f:
    meta_new = pickle.load(f)

X_new = np.load(f'data/features/{ticker}/X_train.npy')

print(f'New features: {meta_new[\"n_features\"]}')
print(f'New shape: {X_new.shape}')
print(f'\\nFeature names:')
for i, name in enumerate(meta_new['feature_names'][:20], 1):
    print(f'  {i:2d}. {name}')
"
```

---

## Phase 4: Model Training

### Train LSTM on New Features

```bash
# Train LSTM baseline on new features
python pipelines/run_lstm_baseline.py \
    --config config/default_config.yaml \
    --ticker AAPL \
    --features-dir data/features
```

---

### Compare Old vs. New Performance

```bash
# Train on old features (if available)
python pipelines/run_lstm_baseline.py \
    --config config/default_config.yaml \
    --ticker AAPL \
    --features-dir data/features.backup

# Train on new features
python pipelines/run_lstm_baseline.py \
    --config config/default_config.yaml \
    --ticker AAPL \
    --features-dir data/features

# Compare results
python -c "
import json

# Load old results
with open('results/AAPL_old/metrics.json') as f:
    old = json.load(f)

# Load new results
with open('results/AAPL_new/metrics.json') as f:
    new = json.load(f)

print('Performance Comparison:')
print(f'RMSE:   {old[\"rmse\"]:.4f} → {new[\"rmse\"]:.4f} ({(new[\"rmse\"]/old[\"rmse\"]-1)*100:+.1f}%)')
print(f'Sharpe: {old[\"sharpe\"]:.4f} → {new[\"sharpe\"]:.4f} ({(new[\"sharpe\"]/old[\"sharpe\"]-1)*100:+.1f}%)')
print(f'Drawdown: {old[\"max_dd\"]:.4f} → {new[\"max_dd\"]:.4f} ({(new[\"max_dd\"]/old[\"max_dd\"]-1)*100:+.1f}%)')
"
```

---

## Troubleshooting Commands

### Check Data Quality

```bash
# Check for missing data
python -c "
import pandas as pd

for ticker in ['SPY', 'QQQ', 'XLK', 'AAPL']:
    df = pd.read_parquet(f'data/processed/{ticker}.parquet')
    missing_pct = df.isna().mean().mean() * 100
    print(f'{ticker:6s}: {len(df):7d} rows, {missing_pct:.2f}% missing')
"
```

---

### Check Peer Correlations

```bash
# Verify peer correlations are reasonable
python -c "
import pandas as pd
import json

# Load fitted peers
with open('data/features/fitted_peers.json') as f:
    peers = json.load(f)

# Check AAPL peers
ticker = 'AAPL'
aapl = pd.read_parquet(f'data/processed/{ticker}.parquet')

print(f'{ticker} peer correlations:')
for peer in peers[ticker]:
    peer_df = pd.read_parquet(f'data/processed/{peer}.parquet')
    corr = aapl['log_return'].corr(peer_df['log_return'])
    print(f'  {peer:6s}: {corr:.3f}')
"
```

---

### Memory Profiling

```bash
# Profile memory usage during feature building
python -m memory_profiler pipelines/run_build_features.py \
    --config config/default_config.yaml \
    --input data/processed/aligned_universe.parquet \
    --output data/features \
    --tickers config/tickers_full.txt \
    --n-jobs 1
```

---

## Backup & Restore Commands

### Backup Current Features

```bash
# Before running new feature building
cp -r data/features data/features.backup
cp config/tickers.txt config/tickers.old
```

---

### Restore Old Features

```bash
# If something goes wrong
rm -rf data/features
cp -r data/features.backup data/features
```

---

## Quick Reference: File Locations

| File | Purpose |
|------|---------|
| `config/symbol_universe.yaml` | Sector mappings, peer settings |
| `config/tickers_new_context.txt` | 17 new symbols to ingest |
| `config/tickers_full.txt` | All 69 symbols |
| `config/mass_tickers.txt` | Original 51 stocks + SPY |
| `data/raw/*.parquet` | Raw OHLCV data |
| `data/cleaned/*.parquet` | Cleaned data |
| `data/processed/*.parquet` | Aligned data |
| `data/features/*/` | Feature matrices per ticker |
| `data/features/fitted_peers.json` | Peer selections |
| `src/features/universe_builder.py` | Core builder class |
| `tests/test_universe_builder.py` | Unit tests |
| `docs/plans/PEERS_PLAN.md` | Full implementation plan |
| `docs/plans/PEERS_CHECKLIST.md` | Implementation checklist |

---

## Common Issues & Solutions

### Issue: "Symbol X not found in dfs"
**Solution:** Ensure symbol was ingested and aligned. Check `data/processed/X.parquet` exists.

```bash
ls data/processed/X.parquet
```

---

### Issue: "No peers found for ticker"
**Solution:** Lower `min_correlation` in `symbol_universe.yaml` or check that candidate peers exist.

```yaml
peer_selection:
  min_correlation: 0.2  # Lower from 0.3
```

---

### Issue: "Memory error during feature building"
**Solution:** Process tickers sequentially (n-jobs=1) and increase swap space.

```bash
# Check memory usage
free -h

# Process one at a time
python pipelines/run_build_features.py --n-jobs 1
```

---

### Issue: "Alignment produces too few timestamps"
**Solution:** Check for data quality issues in individual symbols. Some ETFs may have gaps.

```bash
# Check timestamp counts
python -c "
import pandas as pd

for ticker in ['SPY', 'QQQ', 'XLK', 'UVXY']:
    df = pd.read_parquet(f'data/cleaned/{ticker}.parquet')
    print(f'{ticker:6s}: {len(df):7d} timestamps')
"
```

---

## Next Steps After Implementation

1. **Validate peer selections** - Check `fitted_peers.json` makes sense
2. **Compare feature counts** - Should increase by 20-40 features
3. **Train baseline models** - Compare old vs. new performance
4. **Profile memory usage** - Ensure scalability to 51 targets
5. **Document results** - Update README with findings

---

## Questions?

See full documentation:
- `docs/plans/PEERS_PLAN.md` - Complete implementation plan
- `docs/plans/PEERS_ARCHITECTURE.md` - Architecture diagrams
- `docs/plans/PEERS_SUMMARY.md` - Quick summary
- `docs/plans/PEERS_CHECKLIST.md` - Implementation checklist
