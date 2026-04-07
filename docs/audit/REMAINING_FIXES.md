# Remaining Audit Fixes

**Status:** 24/87 issues fixed (28% complete)  
**Priority:** Continue with HIGH priority issues

---

## ✅ Completed (24 issues)

### CRITICAL (7/7) ✅
1. DataCleaner._init_stats() call
2. API keys in stdout
3. Import order
4. Backtester index bounds
5. Undefined CLI arguments
6. Ichimoku look-ahead bias
7. Peer selection look-ahead bias

### HIGH Priority (17/39)
- Session filter memory optimization
- Outlier clipping vectorized
- Empty DataFrame handling
- Division by zero guards
- Deprecated datetime
- Unconditional CUDA device
- Hard-coded column names
- Naive timezone handling
- SPY bfill removed
- Backtester bounds check
- Output directory ordering
- Gradient accumulation tail
- Hardcoded DataLoader seed
- NaN/Inf checks in training
- Optimizer uses self.model.parameters()

---

## 🔄 Quick Wins Remaining (High Impact, Low Effort)

### Model Implementation
1. **Add map_location to torch.load** (5 locations)
   - `pipelines/run_lstm_baseline.py` lines 382, 481
   - `src/models/quantized_lstm.py` line 124
   - Add: `torch.load(path, map_location="cpu", weights_only=True)`

2. **Fix JSON serialization** (1 location)
   - `pipelines/run_lstm_baseline.py` lines 604-620
   - Use proper `default` handler for numpy types

3. **Thread seed parameter** (3 locations)
   - `pipelines/run_lstm_baseline.py` line 656
   - `scripts/run_pso.py` lines 37-78
   - Pass `seed=args.seed` to LSTMTrainer

### Evaluation & Metrics
4. **Parameterize annualization** (4 functions)
   - `src/evaluation/metrics.py` - sharpe_ratio, sortino_ratio, information_ratio
   - `src/models/rl/trading_env.py` - get_episode_stats
   - Add `bars_per_year` parameter (default 252*390)

5. **Fix empty fold metrics** (1 location)
   - `pipelines/run_lstm_baseline.py` lines 414-422
   - Add: `if not fold_metrics: return`

---

## 📋 MEDIUM Priority Batch Fixes

### Remove Commented Code (10 files, ~1000 lines)
```bash
# Files to clean:
- src/data/aligner.py (lines 86-116)
- pipelines/build_features.py (lines 166-651)
- src/features/pipeline.py (lines 166-179, 247-273, 295-586)
- src/features/technical.py (lines 96-98, 149)
- src/features/statistical.py (lines 44-48, 79-89)
- src/features/ttm_squeeze.py (lines 159-168, 307-379)
```

### Fix Duplicate Imports (5 files)
```bash
- src/data/splitter.py (lines 13, 113)
- src/models/lstm_model.py (lines 20, 123)
- pipelines/run_lstm_baseline.py (lines 36-39)
```

### Update Module Docstrings (10+ files)
```bash
# Fix paths in docstrings:
- pipelines/ingest_data.py: "scripts/01_..." → "pipelines/..."
- src/models/rl/*.py: "src/rl/..." → "src/models/rl/..."
- src/database/*.py: "src/utils/..." → "src/database/..."
```

---

## 📚 Documentation Updates (Major Task)

### README.md
- [ ] Fix script paths (scripts/01_... → pipelines/...)
- [ ] Update ticker count (51 → 6 or fix tickers.txt)
- [ ] Fix CLI flags (--start/--end don't exist)
- [ ] Update repo layout tree
- [ ] Fix Python API example
- [ ] Update artifact names (best_params → pso_results)

### SETUP.md
- [ ] Remove corrupted line 1
- [ ] Update script paths
- [ ] Fix config key names
- [ ] Add TA-Lib installation instructions

### docs/README.md
- [ ] Same fixes as root README

### docs/overviews/reproducibility.md
- [ ] Update all commands
- [ ] Fix artifact names
- [ ] Remove missing scripts

### docs/guides/*.md
- [ ] Update script paths
- [ ] Fix line number references
- [ ] Add missing dependencies

---

## 🎯 Automated Fix Script

Create `scripts/apply_audit_fixes.py`:

```python
#!/usr/bin/env python3
"""Apply remaining audit fixes automatically."""

import re
from pathlib import Path

def remove_commented_blocks(file_path, start_line, end_line):
    """Remove commented code blocks."""
    lines = file_path.read_text().splitlines()
    new_lines = lines[:start_line-1] + lines[end_line:]
    file_path.write_text('\n'.join(new_lines) + '\n')

def fix_duplicate_imports(file_path, duplicate_line):
    """Remove duplicate import statements."""
    lines = file_path.read_text().splitlines()
    del lines[duplicate_line-1]
    file_path.write_text('\n'.join(lines) + '\n')

def update_docstring_paths(file_path, old_path, new_path):
    """Update paths in module docstrings."""
    content = file_path.read_text()
    content = content.replace(old_path, new_path)
    file_path.write_text(content)

def add_map_location(file_path, pattern):
    """Add map_location to torch.load calls."""
    content = file_path.read_text()
    content = re.sub(
        r'torch\.load\(([^,)]+)\)',
        r'torch.load(\1, map_location="cpu", weights_only=True)',
        content
    )
    file_path.write_text(content)

# Apply fixes
fixes = [
    # Remove commented code
    ('src/data/aligner.py', 'remove_commented', 86, 116),
    ('pipelines/build_features.py', 'remove_commented', 166, 651),
    # ... etc
]

for fix in fixes:
    apply_fix(*fix)
```

---

## ⏱️ Estimated Time Remaining

| Category | Issues | Time |
|----------|--------|------|
| HIGH priority | 22 | 2-3 hours |
| MEDIUM priority | 35 | 3-4 hours |
| LOW priority | 22 | 1-2 hours |
| Documentation | 1 major | 2-3 hours |
| **TOTAL** | **80** | **8-12 hours** |

---

## 🚀 Recommended Approach

1. **Batch fix quick wins** (1 hour)
   - Add map_location (5 locations)
   - Fix JSON serialization (1 location)
   - Thread seed parameter (3 locations)
   - Parameterize annualization (4 functions)
   - Fix empty fold metrics (1 location)

2. **Remove commented code** (30 min)
   - Use automated script
   - ~1000 lines across 10 files

3. **Fix duplicate imports** (15 min)
   - Simple deletions

4. **Update docstrings** (30 min)
   - Find/replace operations

5. **Documentation overhaul** (2-3 hours)
   - README.md
   - SETUP.md
   - docs/ files

6. **Testing** (2 hours)
   - Run smoke tests
   - Verify no regressions
   - Test on 2-3 tickers

---

## 📝 Notes

- Many MEDIUM/LOW priority issues are code quality improvements
- Documentation is the largest remaining task
- After fixes, comprehensive testing is critical
- Consider creating unit tests for fixed bugs

---

**Next Action:** Continue with quick wins, then batch process MEDIUM priority issues.
