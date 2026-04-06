# XGBoost Codebase Audit Report

**Date:** April 6, 2026  
**Scope:** XGBoost model implementation and training/validation/testing scripts  
**Auditor:** Code Review System  

---

## Executive Summary

This audit evaluates four Python files implementing XGBoost functionality for stock return prediction. The codebase demonstrates **high overall quality** with strong adherence to Python best practices, comprehensive documentation, and thoughtful design patterns. However, several areas require attention for improved maintainability, type safety, and error handling.

**Overall Grade:** B+ (87/100)

### Key Strengths
- Excellent documentation with clear docstrings
- Consistent interface design (mirrors LSTM implementation)
- Comprehensive error handling in most areas
- Well-structured code with clear separation of concerns
- Good use of logging throughout

### Critical Issues
- Missing type hints in several functions
- Inconsistent data loading paths between scripts
- Potential runtime errors from unvalidated assumptions
- Some code duplication across scripts
- Mixed responsibility in training scripts

---

## File-by-File Analysis

---

## 1. `src/models/xgboost_model.py`

**Overall Grade:** A- (91/100)

### Strengths

#### 1.1 Documentation Quality ⭐⭐⭐⭐⭐
- Exceptional module-level docstring explaining design decisions
- Comprehensive class and method docstrings with Args/Returns sections
- Clear explanation of design rationale and references to literature

#### 1.2 Interface Design ⭐⭐⭐⭐⭐
- Clean API that mirrors `LSTMTrainer` for consistency
- Well-defined `fit()`, `predict()`, `get_params()` interface
- Proper separation between model and tuner classes

#### 1.3 Error Handling ⭐⭐⭐⭐
- Good use of assertions and runtime checks
- Clear error messages with actionable guidance
- Proper exception handling in `load()` method

#### 1.4 Code Organization ⭐⭐⭐⭐⭐
- Logical grouping of methods with clear section comments
- Private helpers properly prefixed with underscore
- Constants defined at module level

### Issues & Recommendations

#### 🔴 CRITICAL: Type Hints Incomplete

**Lines 125-129, 193-223, 389-450**

```python
# Current (line 125)
self._model: Optional[xgb.XGBRegressor] = None

# Issue: Many methods lack return type annotations
def train_default(
    cfg: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple["XGBoostModel", dict]:  # Should be -> Tuple[XGBoostModel, Dict[str, Any]]
```

**Recommendation:**
```python
from typing import Any, Dict, List, Optional, Tuple

def fit(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, List[float]]:  # ✓ Good - already present
    ...
```

**Impact:** Medium - Reduces IDE support and type checking effectiveness

---

#### 🟡 WARNING: Redundant Import

**Line 301**

```python
def load(self, path: str) -> "XGBoostModel":
    try:
        import xgboost as xgb  # ← Already imported at line 29
    except ImportError as exc:
        raise ImportError("xgboost not installed.") from exc
```

**Recommendation:**
```python
def load(self, path: str) -> "XGBoostModel":
    # xgboost already imported at module level
    self._model = xgb.XGBRegressor(**self._xgb_params)
    assert self._model is not None
    self._model.load_model(path)
    logger.info("XGBoost model loaded from %s", path)
    return self
```

**Impact:** Low - Minor code smell, no functional impact

---

#### 🟡 WARNING: Commented Code

**Line 125**

```python
# self._model = None  # type: Optional[xgb.XGBRegressor]
self._model: Optional[xgb.XGBRegressor] = None
```

**Recommendation:** Remove commented line - it's superseded by the line below.

**Impact:** Low - Reduces code cleanliness

---

#### 🟡 WARNING: Magic Number in Feature Importance Parsing

**Lines 244-248**

```python
try:
    idx = int(feat_name.lstrip("f"))  # Assumes format 'f0', 'f1', etc.
    importances[idx] = float(score)
except (ValueError, IndexError):
    pass  # Silently ignoring errors
```

**Recommendation:**
```python
try:
    # XGBoost uses 'f{idx}' format for unnamed features
    if feat_name.startswith('f') and feat_name[1:].isdigit():
        idx = int(feat_name[1:])
        if 0 <= idx < n_flat:
            importances[idx] = float(score)
        else:
            logger.warning("Feature index %d out of range [0, %d)", idx, n_flat)
    else:
        logger.debug("Skipping non-standard feature name: %s", feat_name)
except (ValueError, IndexError) as e:
    logger.warning("Failed to parse feature name '%s': %s", feat_name, e)
```

**Impact:** Medium - Silent failures can hide bugs

---

#### 🟢 SUGGESTION: Improve Version Check

**Lines 334-343**

```python
@staticmethod
def _supports_eval_names() -> bool:
    """Check if the installed xgboost version accepts eval_names kwarg."""
    try:
        import xgboost as xgb
        major = int(xgb.__version__.split(".")[0])
        return major >= 2
    except Exception:  # Too broad
        return False
```

**Recommendation:**
```python
@staticmethod
def _supports_eval_names() -> bool:
    """Check if the installed xgboost version accepts eval_names kwarg."""
    try:
        import xgboost as xgb
        major = int(xgb.__version__.split(".")[0])
        return major >= 2
    except (AttributeError, ValueError, IndexError) as e:
        logger.debug("Failed to parse xgboost version: %s", e)
        return False
```

**Impact:** Low - More precise exception handling

---

#### 🟢 SUGGESTION: Add Input Validation

**Lines 154-156**

```python
def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
        X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, List[float]]:
    X_tr_flat = self._flatten(X_train)
    X_vl_flat = self._flatten(X_val)
```

**Recommendation:**
```python
def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
        X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, List[float]]:
    # Validate inputs
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}")
    if len(X_val) != len(y_val):
        raise ValueError(f"X_val and y_val length mismatch: {len(X_val)} vs {len(y_val)}")
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3D [N, T, F], got shape {X_train.shape}")
    
    X_tr_flat = self._flatten(X_train)
    X_vl_flat = self._flatten(X_val)
```

**Impact:** Medium - Prevents cryptic errors downstream

---

#### 🟢 SUGGESTION: XGBoostTuner Results Storage

**Line 387**

```python
self.results_: List[Dict[str, Any]] = []
```

**Recommendation:** Add a method to export results to DataFrame for easier analysis:

```python
def get_results_df(self) -> pd.DataFrame:
    """Return tuning results as a pandas DataFrame for analysis."""
    if not self.results_:
        raise RuntimeError("No tuning results available. Call fit() first.")
    return pd.DataFrame(self.results_).sort_values('val_rmse')
```

**Impact:** Low - Quality of life improvement

---

### Summary: `xgboost_model.py`

| Category | Score | Notes |
|----------|-------|-------|
| Documentation | 10/10 | Excellent docstrings and comments |
| Type Safety | 7/10 | Missing some type hints |
| Error Handling | 8/10 | Good coverage, some silent failures |
| Code Quality | 9/10 | Clean, well-organized |
| Performance | 9/10 | Efficient implementation |
| Maintainability | 9/10 | Easy to understand and modify |

**Action Items:**
1. ✅ Add missing type hints to all methods
2. ✅ Remove commented code and redundant imports
3. ✅ Improve error logging in feature importance parsing
4. ✅ Add input validation to `fit()` method

---

## 2. `scripts/06_train_xgboost.py`

**Overall Grade:** B+ (86/100)

### Strengths

#### 2.1 Script Structure ⭐⭐⭐⭐
- Clear separation of argument parsing, data loading, and training
- Good use of helper functions
- Comprehensive output artifacts

#### 2.2 Documentation ⭐⭐⭐⭐⭐
- Excellent module docstring with usage examples
- Clear explanation of operating modes

#### 2.3 Error Messages ⭐⭐⭐⭐
- Helpful error messages with actionable guidance (lines 142-145)

### Issues & Recommendations

#### 🔴 CRITICAL: Inconsistent Data Loading Path

**Lines 127-136 vs Lines 111-120**

```python
# load_windows() expects:
prefix = features_dir / ticker
file_map = {
    "X_train": prefix / "X_train.npy",  # ← Expects data/features/AAPL/X_train.npy
    ...
}

# But script 07 expects:
path = features_dir / f"{ticker}_{key}.npy"  # ← Expects data/features/AAPL_X_train.npy
```

**This is a critical inconsistency that will cause runtime failures!**

**Recommendation:** Standardize on one path convention across all scripts:

```python
def load_windows(features_dir: Path, ticker: str) -> Tuple[np.ndarray, ...]:
    """Load pre-built numpy arrays from script 02.
    
    Expected structure:
        features_dir/
            {ticker}/
                X_train.npy
                X_val.npy
                X_test.npy
                y_train.npy
                y_val.npy
                y_test.npy
                metadata.pkl
    """
    prefix = features_dir / ticker
    
    file_map = {
        "X_train": prefix / "X_train.npy",
        "X_val": prefix / "X_val.npy",
        "X_test": prefix / "X_test.npy",
        "y_train": prefix / "y_train.npy",
        "y_val": prefix / "y_val.npy",
        "y_test": prefix / "y_test.npy",
    }
    
    arrays = {}
    for key, path in file_map.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run script 02 first:\n"
                f"  python scripts/02_build_features.py --target {ticker}"
            )
        arrays[key] = np.load(path)
    
    return (
        arrays["X_train"], arrays["y_train"],
        arrays["X_val"], arrays["y_val"],
        arrays["X_test"], arrays["y_test"],
    )
```

**Impact:** CRITICAL - Will cause script failures

---

#### 🔴 CRITICAL: Commented Debug Code

**Lines 312-314**

```python
print(f"X_train Data: {X_train.shape}")

# import sys
# sys.exit(0)

feature_names = load_feature_names(features_dir, ticker)
```

**Recommendation:** Remove debug code before production:

```python
print(f"X_train Data: {X_train.shape}")
feature_names = load_feature_names(features_dir, ticker)
```

**Impact:** High - Debug code should never be committed

---

#### 🟡 WARNING: Incorrect Shape Access

**Line 317**

```python
feature_names = load_feature_names(features_dir, ticker)
F = X_train.shape[1]  # ← WRONG! Should be shape[2] for [N, T, F]
```

**This is a bug!** For 3D arrays `[N, T, F]`, feature dimension is `shape[2]`.

**Recommendation:**
```python
feature_names = load_feature_names(features_dir, ticker)
N, T, F = X_train.shape  # Unpack all dimensions
print(f"  Train : {X_train.shape}  ({N} samples, {T} timesteps, {F} features)")
```

**Impact:** HIGH - Incorrect feature count will cause confusion

---

#### 🟡 WARNING: Missing Type Hints

**Lines 187-223**

```python
def train_default(
    cfg: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple["XGBoostModel", dict]:  # Should use Tuple from typing
```

**Recommendation:**
```python
from typing import Dict, Any, Tuple

def train_default(
    cfg: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[XGBoostModel, Dict[str, Any]]:
    """Train one XGBoostModel with hyperparameters from the config."""
```

**Impact:** Medium - Reduces type checking effectiveness

---

#### 🟡 WARNING: Hardcoded Magic Numbers

**Lines 258-260, 269**

```python
best_iter = best_model._best_iteration
final_n_est = max(best_iter + 20, int(best_iter * 1.1), 50)  # Magic numbers!
...
n_pv = max(100, len(X_tv) // 100)  # Magic numbers!
```

**Recommendation:** Extract to named constants:

```python
# At module level
FINAL_ESTIMATOR_BUFFER = 20
FINAL_ESTIMATOR_MULTIPLIER = 1.1
MIN_FINAL_ESTIMATORS = 50
PSEUDO_VAL_MIN_SIZE = 100
PSEUDO_VAL_FRACTION = 0.01  # 1%

# In function
final_n_est = max(
    best_iter + FINAL_ESTIMATOR_BUFFER,
    int(best_iter * FINAL_ESTIMATOR_MULTIPLIER),
    MIN_FINAL_ESTIMATORS
)
n_pv = max(PSEUDO_VAL_MIN_SIZE, int(len(X_tv) * PSEUDO_VAL_FRACTION))
```

**Impact:** Medium - Improves maintainability

---

#### 🟡 WARNING: Direct Access to Private Attribute

**Lines 258, 340, 369**

```python
best_iter = best_model._best_iteration  # Accessing private attribute
```

**Recommendation:** Add public property to `XGBoostModel`:

```python
# In xgboost_model.py
@property
def best_iteration(self) -> int:
    """Return the best iteration from training."""
    return self._best_iteration

# In script
best_iter = best_model.best_iteration  # Use public property
```

**Impact:** Medium - Violates encapsulation

---

#### 🟢 SUGGESTION: Improve Config Handling

**Lines 196-216**

```python
hp = getattr(cfg, "xgboost", {})
assert hp is not None

hyperparams = {
    "objective": hp.get("objective", "multi:softprob"),
    "num_class": hp.get("num_class", 3),
    # ... many more lines
}
```

**Recommendation:** Create a helper function:

```python
def extract_hyperparameters(cfg: Config) -> Dict[str, Any]:
    """Extract XGBoost hyperparameters from config with defaults."""
    hp = getattr(cfg, "xgboost", {})
    
    defaults = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 1e-5,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 50,
        "tree_method": "hist",
        "max_bin": 128,
    }
    
    return {key: hp.get(key, default) for key, default in defaults.items()}
```

**Impact:** Low - Improves code organization

---

#### 🟢 SUGGESTION: Add Progress Tracking

**Lines 331-336**

```python
if args.mode == "default":
    model, hyperparams = train_default(cfg, X_train, y_train, X_val, y_val)
else:
    model, hyperparams = train_tune(args, X_train, y_train, X_val, y_val)
```

**Recommendation:** Add progress indicators for long-running operations:

```python
from tqdm import tqdm

if args.mode == "default":
    print("Training with default hyperparameters...")
    model, hyperparams = train_default(cfg, X_train, y_train, X_val, y_val)
else:
    print(f"Running hyperparameter search ({args.n_trials} trials)...")
    with tqdm(total=args.n_trials, desc="Tuning") as pbar:
        model, hyperparams = train_tune(args, X_train, y_train, X_val, y_val, pbar)
```

**Impact:** Low - Quality of life improvement

---

### Summary: `06_train_xgboost.py`

| Category | Score | Notes |
|----------|-------|-------|
| Documentation | 10/10 | Excellent module docstring |
| Type Safety | 6/10 | Missing type hints |
| Error Handling | 8/10 | Good error messages |
| Code Quality | 7/10 | Some issues (debug code, wrong shape) |
| Consistency | 6/10 | Path inconsistency with other scripts |
| Maintainability | 8/10 | Generally well-structured |

**Action Items:**
1. 🚨 **CRITICAL:** Fix data loading path inconsistency
2. 🚨 **CRITICAL:** Fix feature dimension access (shape[2] not shape[1])
3. 🚨 **CRITICAL:** Remove debug code (sys.exit)
4. ✅ Add type hints to all functions
5. ✅ Extract magic numbers to constants
6. ✅ Use public property instead of private attribute access

---

## 3. `scripts/07_validate_xgboost.py`

**Overall Grade:** B+ (85/100)

### Strengths

#### 3.1 Comprehensive Validation ⭐⭐⭐⭐⭐
- Walk-forward validation implementation
- Threshold optimization
- Feature importance analysis
- Multiple evaluation metrics

#### 3.2 Documentation ⭐⭐⭐⭐
- Clear module docstring
- Good function documentation

### Issues & Recommendations

#### 🔴 CRITICAL: Path Inconsistency (Same as Script 06)

**Lines 111-128**

```python
def load_windows(features_dir: Path, ticker: str):
    prefix = features_dir
    arrays = {}
    for split in ("train", "val", "test"):
        for kind in ("X", "y"):
            key = f"{kind}_{split}"
            path = prefix / f"{ticker}_{key}.npy"  # ← Different from script 06!
```

**Recommendation:** Use the same path structure as script 06:

```python
def load_windows(features_dir: Path, ticker: str) -> Tuple[np.ndarray, ...]:
    """Load pre-built numpy arrays from script 02.
    
    Args:
        features_dir: Base features directory
        ticker: Stock ticker symbol
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    prefix = features_dir / ticker
    
    file_map = {
        "X_train": prefix / "X_train.npy",
        "X_val": prefix / "X_val.npy",
        "X_test": prefix / "X_test.npy",
        "y_train": prefix / "y_train.npy",
        "y_val": prefix / "y_val.npy",
        "y_test": prefix / "y_test.npy",
    }
    
    arrays = {}
    for key, path in file_map.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run script 02 first.")
        arrays[key] = np.load(path)
    
    return (
        arrays["X_train"], arrays["y_train"],
        arrays["X_val"], arrays["y_val"],
        arrays["X_test"], arrays["y_test"],
    )
```

**Impact:** CRITICAL - Causes runtime failures

---

#### 🔴 CRITICAL: Unsafe Dictionary Access

**Lines 98-108**

```python
hparams = meta["hyperparameters"]
model = XGBoostModel(
    **{
        k: v
        for k, v in hparams.items()
        if k in XGBoostModel.__init__.__code__.co_varnames  # Fragile!
    }
)
```

**This is fragile and will break if `__init__` signature changes!**

**Recommendation:**
```python
import inspect

def filter_valid_kwargs(cls: type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to only include valid parameters for a class constructor."""
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    return {k: v for k, v in kwargs.items() if k in valid_params}

# Usage
hparams = meta["hyperparameters"]
valid_hparams = filter_valid_kwargs(XGBoostModel, hparams)
model = XGBoostModel(**valid_hparams)
```

**Impact:** HIGH - Brittle code that breaks easily

---

#### 🟡 WARNING: Magic Numbers

**Lines 66-67, 200**

```python
--wfv-fold-size", type=int, default=21 * 390,  # Magic calculation
--wfv-folds", type=int, default=6,
...
n_pv = max(100, len(X_tr) // 20)  # Magic numbers
```

**Recommendation:**
```python
# At module level with clear documentation
TRADING_DAYS_PER_MONTH = 21
BARS_PER_TRADING_DAY = 390  # 1-minute bars in 6.5 hour trading day
DEFAULT_WFV_FOLD_SIZE = TRADING_DAYS_PER_MONTH * BARS_PER_TRADING_DAY
DEFAULT_WFV_FOLDS = 6
PSEUDO_VAL_MIN_SIZE = 100
PSEUDO_VAL_FRACTION = 0.05  # 5% of training data

# In argparse
parser.add_argument(
    "--wfv-fold-size",
    type=int,
    default=DEFAULT_WFV_FOLD_SIZE,
    help=f"Bars per walk-forward fold (default: {TRADING_DAYS_PER_MONTH} days × "
         f"{BARS_PER_TRADING_DAY} bars/day = {DEFAULT_WFV_FOLD_SIZE} bars)",
)
```

**Impact:** Medium - Improves code clarity

---

#### 🟡 WARNING: Potential Division by Zero

**Line 200**

```python
n_pv = max(100, len(X_tr) // 20)
```

If `len(X_tr) < 100`, this works, but the logic is unclear.

**Recommendation:**
```python
# Calculate pseudo-validation size (5% of training, minimum 100 samples)
n_pv = max(PSEUDO_VAL_MIN_SIZE, int(len(X_tr) * PSEUDO_VAL_FRACTION))

# Ensure we have enough data
if len(X_tr) < n_pv + 100:  # Need at least 100 training samples
    raise ValueError(
        f"Insufficient training data: {len(X_tr)} samples. "
        f"Need at least {n_pv + 100} for pseudo-validation split."
    )
```

**Impact:** Medium - Prevents edge case failures

---

#### 🟢 SUGGESTION: Extract Walk-Forward Logic to Utility

**Lines 144-219**

The walk-forward validation logic is complex and could be reused. Consider extracting to `src/evaluation/walk_forward.py`:

```python
# src/evaluation/walk_forward.py
from typing import List, Dict, Any, Callable
import numpy as np

class WalkForwardValidator:
    """Expanding-window walk-forward validation for time series models."""
    
    def __init__(
        self,
        fold_size: int,
        max_folds: int = 10,
        min_train_size: int = 1000,
    ):
        self.fold_size = fold_size
        self.max_folds = max_folds
        self.min_train_size = min_train_size
    
    def validate(
        self,
        model_factory: Callable[[], Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Run walk-forward validation."""
        # Implementation here
        ...
```

**Impact:** Low - Improves code reusability

---

#### 🟢 SUGGESTION: Add Validation for Empty Results

**Lines 327-338**

```python
if wfv_results:
    metric_keys = [
        k for k in wfv_results[0] if k not in ("fold", "start_bar", "end_bar")
    ]
    # ...
else:
    wfv_summary = {}
```

**Recommendation:** Add warning if no folds were processed:

```python
if wfv_results:
    metric_keys = [
        k for k in wfv_results[0] if k not in ("fold", "start_bar", "end_bar")
    ]
    # ... existing code ...
else:
    logger.warning("Walk-forward validation produced no results. Check fold_size and data length.")
    wfv_summary = {}
```

**Impact:** Low - Better debugging experience

---

### Summary: `07_validate_xgboost.py`

| Category | Score | Notes |
|----------|-------|-------|
| Documentation | 9/10 | Good overall, could use more inline comments |
| Type Safety | 6/10 | Missing type hints |
| Error Handling | 7/10 | Some edge cases not handled |
| Code Quality | 8/10 | Well-structured but fragile in places |
| Consistency | 6/10 | Path inconsistency |
| Maintainability | 8/10 | Could extract reusable components |

**Action Items:**
1. 🚨 **CRITICAL:** Fix data loading path inconsistency
2. 🚨 **CRITICAL:** Replace fragile `__code__.co_varnames` with `inspect`
3. ✅ Extract magic numbers to named constants
4. ✅ Add input validation for edge cases
5. ✅ Consider extracting walk-forward logic to utility module

---

## 4. `scripts/08_test_xgboost.py`

**Overall Grade:** B (84/100)

### Strengths

#### 4.1 Comprehensive Testing ⭐⭐⭐⭐⭐
- Statistical metrics
- Backtesting with realistic costs
- Cross-model comparison
- Multiple output formats

#### 4.2 Integration ⭐⭐⭐⭐
- Good integration with existing model comparison framework
- Consistent with script 05 (LSTM testing)

### Issues & Recommendations

#### 🔴 CRITICAL: Same Path Inconsistency

**Lines 127-143**

```python
def load_windows(features_dir: Path, ticker: str):
    arrays = {}
    for split in ("train", "val", "test"):
        for kind in ("X", "y"):
            key = f"{kind}_{split}"
            path = features_dir / f"{ticker}_{key}.npy"  # ← Inconsistent!
```

**Recommendation:** Use the same fix as scripts 06 and 07. Consider extracting to a shared utility module:

```python
# src/utils/data_loading.py
from pathlib import Path
from typing import Tuple
import numpy as np

def load_split_arrays(
    features_dir: Path,
    ticker: str,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, np.ndarray]:
    """Load train/val/test arrays for a given ticker.
    
    Args:
        features_dir: Base features directory
        ticker: Stock ticker symbol
        splits: Tuple of split names to load
        
    Returns:
        Dictionary mapping keys like 'X_train', 'y_val' to arrays
        
    Raises:
        FileNotFoundError: If any required file is missing
    """
    prefix = features_dir / ticker
    arrays = {}
    
    for split in splits:
        for kind in ("X", "y"):
            key = f"{kind}_{split}"
            path = prefix / f"{key}.npy"
            
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Run script 02 first:\n"
                    f"  python scripts/02_build_features.py --target {ticker}"
                )
            
            arrays[key] = np.load(path)
    
    return arrays

# Usage in all scripts
from src.utils.data_loading import load_split_arrays

arrays = load_split_arrays(features_dir, ticker)
X_train, y_train = arrays["X_train"], arrays["y_train"]
X_val, y_val = arrays["X_val"], arrays["y_val"]
X_test, y_test = arrays["X_test"], arrays["y_test"]
```

**Impact:** CRITICAL - Eliminates code duplication and inconsistency

---

#### 🔴 CRITICAL: Fragile Dictionary Filtering (Same as Script 07)

**Lines 103-104**

```python
valid_keys = set(XGBoostModel.__init__.__code__.co_varnames)
model = XGBoostModel(**{k: v for k, v in hparams.items() if k in valid_keys})
```

**Recommendation:** Use the same `inspect`-based solution as script 07.

**Impact:** HIGH - Brittle code

---

#### 🟡 WARNING: Synthetic Price Generation Without Warning

**Lines 150-157**

```python
if not raw_path.exists():
    print("  WARNING: Raw price file not found — using synthetic prices.")
    y_test = np.load(Path(features_dir) / f"{ticker}_y_test.npy")
    prices = 100.0 * np.exp(np.cumsum(y_test))  # Synthetic!
    ts = pd.date_range(
        "2023-01-03 14:30", periods=len(prices), freq="1min", tz="UTC"
    )
    return prices, prices, ts  # Returns same array for open and close!
```

**Issues:**
1. Synthetic prices may not reflect realistic market behavior
2. Open and close prices are identical (unrealistic)
3. Hardcoded date "2023-01-03" may not match actual test period

**Recommendation:**
```python
if not raw_path.exists():
    logger.warning(
        "Raw price file not found: %s. Generating synthetic prices. "
        "Backtest results may not reflect realistic trading conditions.",
        raw_path
    )
    y_test_path = Path(features_dir) / ticker / "y_test.npy"
    y_test = np.load(y_test_path)
    
    # Generate synthetic prices with realistic open/close spread
    base_price = 100.0
    closes = base_price * np.exp(np.cumsum(y_test))
    
    # Add realistic open prices (small random offset from previous close)
    rng = np.random.default_rng(42)
    opens = np.roll(closes, 1)
    opens[0] = base_price
    opens *= (1 + rng.normal(0, 0.0001, len(opens)))  # 1bp spread
    
    # Use metadata for actual date range if available
    metadata_path = Path(features_dir) / ticker / "metadata.pkl"
    if metadata_path.exists():
        import pickle
        with open(metadata_path, 'rb') as f:
            meta = pickle.load(f)
            start_date = meta.get('test_start_date', '2023-01-03 14:30')
    else:
        start_date = '2023-01-03 14:30'
    
    ts = pd.date_range(start_date, periods=len(closes), freq="1min", tz="UTC")
    
    return opens, closes, ts
```

**Impact:** HIGH - Affects backtest realism

---

#### 🟡 WARNING: Silent Metric Mismatches

**Lines 232-236**

```python
for col in stat_cols:
    v = stat_results.get(model_name, {}).get(col, float("nan"))
    row += f"{_fmt(v):>{col_w}}"
```

If a metric is missing, it's silently replaced with NaN. This could hide bugs.

**Recommendation:**
```python
for col in stat_cols:
    model_data = stat_results.get(model_name, {})
    if model_name in stat_results and col not in model_data:
        logger.warning("Metric '%s' missing for model '%s'", col, model_name)
    v = model_data.get(col, float("nan"))
    row += f"{_fmt(v):>{col_w}}"
```

**Impact:** Medium - Better debugging

---

#### 🟢 SUGGESTION: Add Model Comparison Visualization

The comparison table is text-only. Consider adding a visualization function:

```python
def plot_model_comparison(
    stat_results: Dict[str, Dict[str, float]],
    trading_results: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Generate comparison plots for all models."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    models = list(stat_results.keys())
    metrics_to_plot = [
        ('rmse', 'RMSE (lower is better)', True),
        ('directional_accuracy', 'Directional Accuracy', False),
        ('f1_ternary', 'F1 Score (Ternary)', False),
        ('sharpe', 'Sharpe Ratio', False),
        ('max_drawdown', 'Max Drawdown (lower is better)', True),
        ('calmar', 'Calmar Ratio', False),
    ]
    
    for ax, (metric, title, lower_is_better) in zip(axes.flat, metrics_to_plot):
        # Extract values
        if metric in ['sharpe', 'max_drawdown', 'calmar']:
            values = [trading_results.get(m, {}).get(metric, np.nan) for m in models]
        else:
            values = [stat_results.get(m, {}).get(metric, np.nan) for m in models]
        
        # Plot
        colors = ['green' if not lower_is_better else 'red' for _ in values]
        ax.bar(models, values, color=colors, alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")
```

**Impact:** Low - Quality of life improvement

---

#### 🟢 SUGGESTION: Add Statistical Significance Testing

When comparing models, add statistical tests:

```python
from scipy import stats

def compare_models_statistically(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Perform pairwise statistical tests between model predictions."""
    from itertools import combinations
    
    results = []
    model_names = list(predictions.keys())
    
    for model1, model2 in combinations(model_names, 2):
        pred1 = predictions[model1]
        pred2 = predictions[model2]
        
        # Compute squared errors
        se1 = (y_true - pred1) ** 2
        se2 = (y_true - pred2) ** 2
        
        # Diebold-Mariano test
        diff = se1 - se2
        t_stat, p_value = stats.ttest_1samp(diff, 0)
        
        results.append({
            'model_1': model1,
            'model_2': model2,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
        })
    
    return pd.DataFrame(results)
```

**Impact:** Low - Adds scientific rigor

---

### Summary: `08_test_xgboost.py`

| Category | Score | Notes |
|----------|-------|-------|
| Documentation | 9/10 | Good module docstring |
| Type Safety | 6/10 | Missing type hints |
| Error Handling | 7/10 | Some silent failures |
| Code Quality | 8/10 | Well-structured |
| Consistency | 6/10 | Path inconsistency |
| Maintainability | 8/10 | Could reduce duplication |

**Action Items:**
1. 🚨 **CRITICAL:** Extract data loading to shared utility
2. 🚨 **CRITICAL:** Fix fragile dictionary filtering
3. ✅ Improve synthetic price generation
4. ✅ Add warnings for missing metrics
5. ✅ Consider adding visualization utilities

---

## Cross-Cutting Concerns

### 1. Code Duplication

**Issue:** The `load_windows()` function is duplicated across all three scripts with slight variations.

**Impact:** HIGH - Maintenance burden, inconsistency bugs

**Recommendation:** Create shared utility module:

```python
# src/utils/data_loading.py
"""Shared data loading utilities for XGBoost and LSTM scripts."""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pickle
import logging

logger = logging.getLogger(__name__)

def load_split_arrays(
    features_dir: Path,
    ticker: str,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, np.ndarray]:
    """Load train/val/test arrays for a given ticker.
    
    Expected directory structure:
        features_dir/
            {ticker}/
                X_train.npy
                X_val.npy
                X_test.npy
                y_train.npy
                y_val.npy
                y_test.npy
                metadata.pkl
    
    Args:
        features_dir: Base features directory
        ticker: Stock ticker symbol
        splits: Tuple of split names to load (default: train, val, test)
        
    Returns:
        Dictionary mapping keys like 'X_train', 'y_val' to numpy arrays
        
    Raises:
        FileNotFoundError: If any required file is missing
        
    Example:
        >>> arrays = load_split_arrays(Path("data/features"), "AAPL")
        >>> X_train = arrays["X_train"]
        >>> y_train = arrays["y_train"]
    """
    prefix = features_dir / ticker
    
    if not prefix.exists():
        raise FileNotFoundError(
            f"Ticker directory not found: {prefix}\n"
            f"Run script 02 first:\n"
            f"  python scripts/02_build_features.py --target {ticker}"
        )
    
    arrays = {}
    missing_files = []
    
    for split in splits:
        for kind in ("X", "y"):
            key = f"{kind}_{split}"
            path = prefix / f"{key}.npy"
            
            if not path.exists():
                missing_files.append(str(path))
            else:
                arrays[key] = np.load(path)
                logger.debug("Loaded %s: shape=%s", key, arrays[key].shape)
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing {len(missing_files)} required files:\n" +
            "\n".join(f"  - {f}" for f in missing_files[:5]) +
            ("\n  ..." if len(missing_files) > 5 else "")
        )
    
    return arrays


def load_feature_metadata(
    features_dir: Path,
    ticker: str,
) -> Dict[str, any]:
    """Load feature metadata (names, dates, etc.) for a ticker.
    
    Args:
        features_dir: Base features directory
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary containing metadata (feature_names, dates, etc.)
        
    Raises:
        FileNotFoundError: If metadata file is missing
    """
    path = features_dir / ticker / "metadata.pkl"
    
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}\n"
            f"Run script 02 first."
        )
    
    try:
        with open(path, "rb") as f:
            metadata = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata from {path}: {e}")
    
    # Validate required fields
    if "feature_names" not in metadata:
        logger.warning("'feature_names' missing from metadata")
        metadata["feature_names"] = []
    
    return metadata


def load_feature_names(features_dir: Path, ticker: str) -> list:
    """Load feature names for a ticker (convenience wrapper).
    
    Args:
        features_dir: Base features directory
        ticker: Stock ticker symbol
        
    Returns:
        List of feature names, or empty list if not available
    """
    try:
        metadata = load_feature_metadata(features_dir, ticker)
        return metadata.get("feature_names", [])
    except FileNotFoundError:
        logger.warning("Could not load feature names for %s", ticker)
        return []
```

**Then update all scripts:**

```python
# In scripts/06_train_xgboost.py, 07_validate_xgboost.py, 08_test_xgboost.py
from src.utils.data_loading import load_split_arrays, load_feature_names

# Replace load_windows() with:
arrays = load_split_arrays(features_dir, ticker)
X_train, y_train = arrays["X_train"], arrays["y_train"]
X_val, y_val = arrays["X_val"], arrays["y_val"]
X_test, y_test = arrays["X_test"], arrays["y_test"]

feature_names = load_feature_names(features_dir, ticker)
```

---

### 2. Type Hints

**Issue:** Inconsistent use of type hints across all files.

**Impact:** MEDIUM - Reduces IDE support and type checking

**Recommendation:** Add comprehensive type hints:

```python
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from numpy.typing import NDArray

# Use NDArray for numpy arrays
def fit(
    self,
    X_train: NDArray[np.float32],
    y_train: NDArray[np.float32],
    X_val: NDArray[np.float32],
    y_val: NDArray[np.float32],
) -> Dict[str, List[float]]:
    ...
```

---

### 3. Logging vs Print Statements

**Issue:** Mix of `print()` and `logger.info()` throughout scripts.

**Impact:** LOW - Inconsistent logging behavior

**Recommendation:** Standardize on logging:

```python
# Replace print statements with appropriate log levels
print(f"Training complete in {elapsed:.1f}s")  # ← Before
logger.info("Training complete in %.1fs", elapsed)  # ← After

# Use different levels appropriately
logger.debug("Detailed diagnostic info")
logger.info("Normal progress updates")
logger.warning("Potential issues")
logger.error("Errors that need attention")
```

---

### 4. Configuration Management

**Issue:** Hyperparameters scattered across multiple places (defaults in model, config file, CLI args).

**Impact:** MEDIUM - Confusion about precedence

**Recommendation:** Document precedence clearly:

```python
"""
Hyperparameter precedence (highest to lowest):
1. Explicit CLI arguments (--learning-rate 0.01)
2. Config file values (config/default_config.yaml)
3. Model defaults (_DEFAULT_PARAMS in xgboost_model.py)

Example:
    # Model defaults
    _DEFAULT_PARAMS = {"learning_rate": 1e-5, ...}
    
    # Config file overrides
    xgboost:
      learning_rate: 0.01
    
    # CLI overrides everything
    python script.py --learning-rate 0.001
"""
```

---

## Actionable Recommendations Summary

### 🚨 Critical (Must Fix Before Production)

1. **Data Loading Path Inconsistency** (All Scripts)
   - Extract to shared utility: `src/utils/data_loading.py`
   - Use consistent path structure: `features_dir/{ticker}/{split}.npy`
   - Estimated effort: 2 hours

2. **Feature Dimension Bug** (Script 06, Line 317)
   - Change `F = X_train.shape[1]` to `F = X_train.shape[2]`
   - Estimated effort: 5 minutes

3. **Remove Debug Code** (Script 06, Lines 312-314)
   - Delete commented `sys.exit(0)`
   - Estimated effort: 1 minute

4. **Fragile Dictionary Filtering** (Scripts 07, 08)
   - Replace `__code__.co_varnames` with `inspect.signature()`
   - Estimated effort: 30 minutes

### ✅ High Priority (Should Fix Soon)

5. **Add Comprehensive Type Hints** (All Files)
   - Add type hints to all functions
   - Use `numpy.typing.NDArray` for arrays
   - Estimated effort: 3 hours

6. **Extract Magic Numbers** (Scripts 06, 07, 08)
   - Move to module-level constants with documentation
   - Estimated effort: 1 hour

7. **Improve Error Handling** (Model File)
   - Add input validation to `fit()` method
   - Improve feature importance error logging
   - Estimated effort: 1 hour

8. **Fix Private Attribute Access** (Script 06)
   - Add `@property` for `best_iteration`
   - Estimated effort: 15 minutes

### 🟢 Nice to Have (Future Improvements)

9. **Standardize Logging** (All Scripts)
   - Replace `print()` with `logger.info()`
   - Estimated effort: 1 hour

10. **Add Visualization Utilities** (Script 08)
    - Model comparison plots
    - Estimated effort: 2 hours

11. **Extract Walk-Forward Logic** (Script 07)
    - Create reusable `WalkForwardValidator` class
    - Estimated effort: 2 hours

12. **Add Statistical Tests** (Script 08)
    - Diebold-Mariano test for model comparison
    - Estimated effort: 1 hour

---

## Testing Recommendations

### Unit Tests Needed

```python
# tests/test_xgboost_model.py
import pytest
import numpy as np
from src.models.xgboost_model import XGBoostModel, XGBoostTuner

class TestXGBoostModel:
    def test_fit_predict_shape(self):
        """Test that fit and predict work with correct shapes."""
        model = XGBoostModel(lookback=10, n_estimators=10)
        X_train = np.random.randn(100, 10, 5).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)
        X_val = np.random.randn(20, 10, 5).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32)
        
        history = model.fit(X_train, y_train, X_val, y_val)
        assert "train_rmse" in history
        assert "val_rmse" in history
        
        y_pred = model.predict(X_val)
        assert y_pred.shape == (20,)
    
    def test_lookback_validation(self):
        """Test that lookback validation works."""
        model = XGBoostModel(lookback=20)
        X = np.random.randn(10, 10, 5).astype(np.float32)  # T=10 < lookback=20
        y = np.random.randn(10).astype(np.float32)
        
        with pytest.raises(ValueError, match="lookback"):
            model.fit(X, y, X, y)
    
    def test_save_load(self, tmp_path):
        """Test model persistence."""
        model = XGBoostModel(lookback=10, n_estimators=10)
        X = np.random.randn(50, 10, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        
        model.fit(X, y, X[:10], y[:10])
        
        path = tmp_path / "model.ubj"
        model.save(str(path))
        
        model2 = XGBoostModel(lookback=10)
        model2.load(str(path))
        
        y_pred1 = model.predict(X[:5])
        y_pred2 = model2.predict(X[:5])
        np.testing.assert_allclose(y_pred1, y_pred2)
    
    def test_feature_importances(self):
        """Test feature importance extraction."""
        model = XGBoostModel(lookback=10, n_estimators=10)
        X = np.random.randn(100, 10, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        
        model.fit(X, y, X[:20], y[:20])
        
        importances = model.get_feature_importances()
        assert importances.shape == (10 * 5,)  # lookback × F
        
        orig_importances = model.get_per_original_feature_importances(5)
        assert orig_importances.shape == (5,)


class TestXGBoostTuner:
    def test_tuner_returns_best_model(self):
        """Test that tuner returns a trained model."""
        tuner = XGBoostTuner(n_trials=3, seed=42)
        X_train = np.random.randn(100, 10, 5).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)
        X_val = np.random.randn(20, 10, 5).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32)
        
        model, params = tuner.fit(X_train, y_train, X_val, y_val, lookback=10)
        
        assert model is not None
        assert "lookback" in params
        assert len(tuner.results_) == 3
```

### Integration Tests Needed

```python
# tests/integration/test_xgboost_pipeline.py
import pytest
from pathlib import Path
import numpy as np

def test_full_xgboost_pipeline(tmp_path):
    """Test complete train -> validate -> test pipeline."""
    # Setup: Create synthetic data
    features_dir = tmp_path / "features"
    results_dir = tmp_path / "results"
    ticker = "TEST"
    
    # ... create synthetic data files ...
    
    # Run training
    from scripts.06_train_xgboost import main as train_main
    # ... configure args ...
    train_main()
    
    # Run validation
    from scripts.07_validate_xgboost import main as validate_main
    validate_main()
    
    # Run testing
    from scripts.08_test_xgboost import main as test_main
    test_main()
    
    # Verify outputs exist
    assert (results_dir / f"xgb_model_{ticker}_default_seed42.ubj").exists()
    assert (results_dir / f"xgb_test_metrics_{ticker}_default_seed42.json").exists()
```

---

## Performance Considerations

### 1. GPU Acceleration

**Current:** `tree_method: "gpu_hist"` in defaults

**Recommendation:** Add fallback logic:

```python
def _get_tree_method() -> str:
    """Determine best tree method based on available hardware."""
    try:
        import xgboost as xgb
        # Check if GPU is available
        if xgb.get_config()['use_rmm'] or torch.cuda.is_available():
            logger.info("GPU detected, using 'gpu_hist' tree method")
            return "gpu_hist"
    except Exception as e:
        logger.debug("GPU check failed: %s", e)
    
    logger.info("No GPU detected, using 'hist' tree method")
    return "hist"

# In _DEFAULT_PARAMS
"tree_method": _get_tree_method(),
```

### 2. Memory Optimization

**Issue:** Large feature arrays loaded into memory all at once

**Recommendation:** For very large datasets, consider memory-mapped arrays:

```python
def load_split_arrays_mmap(
    features_dir: Path,
    ticker: str,
) -> Dict[str, np.ndarray]:
    """Load arrays using memory mapping for large datasets."""
    arrays = {}
    prefix = features_dir / ticker
    
    for split in ("train", "val", "test"):
        for kind in ("X", "y"):
            key = f"{kind}_{split}"
            path = prefix / f"{key}.npy"
            # Use mmap_mode='r' for read-only memory mapping
            arrays[key] = np.load(path, mmap_mode='r')
    
    return arrays
```

---

## Security Considerations

### 1. Path Traversal

**Issue:** User-provided ticker could contain path traversal characters

**Recommendation:**

```python
def validate_ticker(ticker: str) -> str:
    """Validate ticker symbol to prevent path traversal."""
    import re
    
    if not re.match(r'^[A-Z0-9]{1,10}$', ticker):
        raise ValueError(
            f"Invalid ticker: {ticker}. "
            "Ticker must be 1-10 uppercase alphanumeric characters."
        )
    
    return ticker

# In parse_args()
args = parser.parse_args()
args.ticker = validate_ticker(args.ticker)
```

### 2. Pickle Safety

**Issue:** Loading pickled metadata files can execute arbitrary code

**Recommendation:**

```python
import pickle

def safe_load_pickle(path: Path) -> Any:
    """Safely load pickle file with restrictions."""
    # Option 1: Use restricted unpickler
    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Only allow safe classes
            if module in ("builtins", "numpy", "pandas"):
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"Forbidden class: {module}.{name}")
    
    with open(path, 'rb') as f:
        return RestrictedUnpickler(f).load()
    
    # Option 2: Use JSON instead of pickle for metadata
    # (Preferred for new code)
```

---

## Documentation Improvements

### 1. Add Docstring Examples

```python
def get_feature_importances(
    self,
    importance_type: Optional[str] = None,
) -> np.ndarray:
    """Return per-input-feature importance scores.
    
    The scores are computed over the *flattened* feature vector
    (length = lookback × F).  If you want per-original-feature
    importances, aggregate across the lookback axis with np.mean.
    
    Args:
        importance_type: One of 'gain', 'weight', 'cover',
            'total_gain', 'total_cover'.  Defaults to the value
            passed at construction time.
    
    Returns:
        Float32 array of shape [lookback × F].
    
    Example:
        >>> model = XGBoostModel(lookback=10)
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> importances = model.get_feature_importances()
        >>> print(importances.shape)
        (50,)  # 10 timesteps × 5 features
        >>> 
        >>> # Get per-original-feature importances
        >>> orig_imp = model.get_per_original_feature_importances(5)
        >>> print(orig_imp.shape)
        (5,)
    """
```

### 2. Add Architecture Diagram

Create `docs/xgboost_architecture.md`:

```markdown
# XGBoost Implementation Architecture

## Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                  XGBoost Pipeline                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐      ┌──────────────┐                │
│  │   Script 06  │      │   Script 07  │                │
│  │    Train     │─────▶│   Validate   │                │
│  └──────────────┘      └──────────────┘                │
│         │                      │                         │
│         │                      │                         │
│         ▼                      ▼                         │
│  ┌──────────────────────────────────┐                  │
│  │      XGBoostModel Class          │                  │
│  ├──────────────────────────────────┤                  │
│  │  - fit()                          │                  │
│  │  - predict()                      │                  │
│  │  - get_feature_importances()     │                  │
│  │  - save() / load()                │                  │
│  └──────────────────────────────────┘                  │
│         │                      │                         │
│         │                      │                         │
│         ▼                      ▼                         │
│  ┌──────────────┐      ┌──────────────┐                │
│  │  Saved Model │      │  Validation  │                │
│  │   (.ubj)     │      │   Metrics    │                │
│  └──────────────┘      └──────────────┘                │
│         │                                                │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐                                       │
│  │   Script 08  │                                       │
│  │     Test     │                                       │
│  └──────────────┘                                       │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐                                       │
│  │   Backtest   │                                       │
│  │   Results    │                                       │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Input:** `[N, T, F]` sliding window tensors
2. **Flattening:** `[N, T×F]` for XGBoost
3. **Training:** XGBoost with early stopping
4. **Output:** Predictions `[N]`

## Key Design Decisions

- **Interface Consistency:** Mirrors `LSTMTrainer` API
- **Early Stopping:** Uses validation set for stopping
- **Feature Importances:** Supports multiple types (gain, weight, cover)
- **Persistence:** XGBoost binary format (.ubj)
```

---

## Conclusion

The XGBoost implementation demonstrates strong software engineering practices with comprehensive documentation, thoughtful design, and good error handling. However, several critical issues must be addressed before production use:

### Must Fix (Critical)
1. Data loading path inconsistency across scripts
2. Feature dimension access bug (shape[2] not shape[1])
3. Fragile dictionary filtering using `__code__.co_varnames`
4. Remove debug code

### Should Fix (High Priority)
5. Add comprehensive type hints
6. Extract magic numbers to constants
7. Improve error handling and validation
8. Standardize logging

### Nice to Have (Future)
9. Extract shared utilities to reduce duplication
10. Add visualization capabilities
11. Implement statistical model comparison
12. Add comprehensive test suite

**Estimated Total Effort:**
- Critical fixes: 3 hours
- High priority: 6 hours
- Nice to have: 8 hours
- **Total: ~17 hours**

### Next Steps

1. Create GitHub issues for each critical item
2. Implement fixes in order of priority
3. Add unit and integration tests
4. Update documentation
5. Perform code review before merging

---

**Report Generated:** April 6, 2026  
**Review Status:** Complete  
**Recommended Action:** Address critical issues before production deployment
