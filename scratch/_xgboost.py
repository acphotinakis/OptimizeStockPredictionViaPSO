import os
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from typing import Tuple
from src.utils.data_storage import _load_parquet
from src.utils.logger import LogFileMode, setup_logger
import logging
import uuid
from _metrics import *

logger = logging.getLogger(__name__)

# FIX THIS LATER
setup_logger(
    log_file="build_features_v3.log",
    level="INFO",
    mode=LogFileMode.OVERWRITE,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "lookback": 20,
    "name": "XGBoost-v1.0",
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "max_bin": 256,
    "n_jobs": -1,
    "importance_type": "gain",
    "verbosity": 0,
    "early_stopping_rounds": 50,
    "feature_type": "lag_based",
    "random_state": 42,
}

SEED = CONFIG["random_state"]
np.random.seed(SEED)

logger.info(f"Config: {json.dumps(CONFIG, indent=2)}\n")


# ARGUMENTS
ticker = "SPY"
timeframe = "1Day"

# target_method = "next_close"
target_method = "log_return"

feature_data_dir = "data/features_v4"


# output_model_dir =

# =============================================================================
# CACHE PATHS (FEATURE PIPELINE)
# =============================================================================
FEATURE_CACHE_DIR = Path(f"{feature_data_dir}/{ticker}/{timeframe}/{target_method}")
# FEATURE_CACHE_DIR = Path(f"data/fake_selection/{ticker}/{timeframe}/{target_method}")
FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_FEATURES_PATH = FEATURE_CACHE_DIR / "selected_features.json"
SELECTOR_PATH = FEATURE_CACHE_DIR / "feature_selector.joblib"
FEATURE_SCALER_PATH = FEATURE_CACHE_DIR / "feature_scaler.joblib"
TARGET_SCALER_PATH = FEATURE_CACHE_DIR / "target_scaler.joblib"
PIPELINE_META_PATH = FEATURE_CACHE_DIR / "pipeline_meta.json"


# create output directory
OUTPUT_DIR = FEATURE_CACHE_DIR / "experiments" / f"xgboost_{uuid.uuid4()}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_DIR = OUTPUT_DIR / "xgboost" / "model"

PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(MODEL_DIR, exist_ok=True)
model_path = MODEL_DIR / "xgboost.pkl"


# =============================================================================
# CACHE VALIDATION (EXPLICIT + DEBUGGABLE)
# =============================================================================

REQUIRED_FILES = {
    "selected_features": SELECTED_FEATURES_PATH,
    "selector": SELECTOR_PATH,
    "feature_scaler": FEATURE_SCALER_PATH,
    "target_scaler": TARGET_SCALER_PATH,
    "X_train": FEATURE_CACHE_DIR / "X_train.npy",
    "y_train": FEATURE_CACHE_DIR / "y_train.npy",
    "X_val": FEATURE_CACHE_DIR / "X_val.npy",
    "y_val": FEATURE_CACHE_DIR / "y_val.npy",
    "X_test": FEATURE_CACHE_DIR / "X_test.npy",
    "y_test": FEATURE_CACHE_DIR / "y_test.npy",
    "train_index": FEATURE_CACHE_DIR / "train_index.npy",
    "val_index": FEATURE_CACHE_DIR / "val_index.npy",
    "test_index": FEATURE_CACHE_DIR / "test_index.npy",
}

missing = []
existing = []

for name, path in REQUIRED_FILES.items():
    if path.exists():
        existing.append(name)
    else:
        missing.append((name, str(path)))

if missing:
    logger.error(" Feature cache validation FAILED")
    logger.error(f"Cache directory: {FEATURE_CACHE_DIR}")

    logger.error("Missing files:")
    for name, path in missing:
        logger.error(f"  - {name:<15} -> {path}")

    logger.info("Found files:")
    for name in existing:
        logger.info(f"  - {name}")

    raise FileNotFoundError(
        f"{len(missing)} required cache files missing. See logs for details."
    )

logger.info("All required cache artifacts found. Loading pipeline...")

cache_exists = (
    SELECTED_FEATURES_PATH.exists()
    and SELECTOR_PATH.exists()
    and FEATURE_SCALER_PATH.exists()
    and TARGET_SCALER_PATH.exists()
)

if cache_exists:
    logger.info("Loading cached feature pipeline...")

    with open(SELECTED_FEATURES_PATH, "r") as f:
        selected_names = json.load(f)

    selector = joblib.load(SELECTOR_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    logger.info(f"Loaded {len(selected_names)} selected features")

    # =============================================================================
    # LOAD FEATURES / TARGETS
    # =============================================================================
    X_train_sel = np.load(FEATURE_CACHE_DIR / "X_train.npy")
    y_train_scaled = np.load(FEATURE_CACHE_DIR / "y_train.npy")

    X_val_sel = np.load(FEATURE_CACHE_DIR / "X_val.npy")
    y_val_scaled = np.load(FEATURE_CACHE_DIR / "y_val.npy")

    X_test_sel = np.load(FEATURE_CACHE_DIR / "X_test.npy")
    y_test_scaled = np.load(FEATURE_CACHE_DIR / "y_test.npy")

    # =============================================================================
    # LOAD INDICES (FIXED SAFE CASTING)
    # =============================================================================
    train_index = pd.DatetimeIndex(np.load(FEATURE_CACHE_DIR / "train_index.npy"))

    val_index = pd.DatetimeIndex(np.load(FEATURE_CACHE_DIR / "val_index.npy"))

    test_index = pd.DatetimeIndex(np.load(FEATURE_CACHE_DIR / "test_index.npy"))

    logger.info(f"Loaded cached dataset:")
    logger.info(f"  Features: {len(selected_names)}")
    logger.info(f"  Train: {X_train_sel.shape}")
    logger.info(f"  Val:   {X_val_sel.shape}")
    logger.info(f"  Test:  {X_test_sel.shape}")

else:
    raise ValueError("Features not created")


logger.info(f"Selected features: {len(selected_names)}")


# =============================================================================
# 5. CREATE LAG-BASED FEATURES (NOT flattened sequences)
# =============================================================================
def create_lag_features(
    X: np.ndarray, y: np.ndarray, lookback: int, feature_names: list = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create lag-based features for XGBoost.
    Each row contains: [feat1_t, feat1_t-1, ..., feat1_t-lookback+1, feat2_t, ...]
    This is NOT a flattened sequence—it's a wide feature vector with explicit lags.
    """
    n_samples = len(X) - lookback
    n_features = X.shape[1]

    # Build column names if provided
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_features)]

    lagged_data = []
    targets = []

    for i in range(lookback, len(X)):
        # Current value + lags for each feature
        row_features = []
        for j in range(n_features):
            for lag in range(lookback):
                row_features.append(X[i - lag, j])

        lagged_data.append(row_features)
        targets.append(y[i])

    # Create descriptive column names
    columns = []
    for fname in feature_names:
        for lag in range(lookback):
            if lag == 0:
                columns.append(f"{fname}")
            else:
                columns.append(f"{fname}_lag{lag}")

    X_lagged = pd.DataFrame(lagged_data, columns=columns)
    y_lagged = np.array(targets)

    return X_lagged, y_lagged


# X_raw = _load_parquet(
#     FEATURE_CACHE_DIR / "splits" / "raw" / f"{ticker}_X_train_raw.parquet"
# )

# X_train_scaled = _load_parquet(
#     FEATURE_CACHE_DIR / "splits" / "scaled" / f"{ticker}_X_train_scaled.parquet"
# )
# X_val_scaled = _load_parquet(
#     FEATURE_CACHE_DIR / "splits" / "scaled" / f"{ticker}_X_val_scaled.parquet"
# )
# X_test_scaled = _load_parquet(
#     FEATURE_CACHE_DIR / "splits" / "scaled" / f"{ticker}_X_test_scaled.parquet"
# )

# feature_cols = list(X_raw.columns)

X_train_lag, y_train_lag = create_lag_features(
    X_train_sel, y_train_scaled, CONFIG["lookback"], selected_names
)
X_val_lag, y_val_lag = create_lag_features(
    X_val_sel, y_val_scaled, CONFIG["lookback"], selected_names
)
X_test_lag, y_test_lag = create_lag_features(
    X_test_sel, y_test_scaled, CONFIG["lookback"], selected_names
)

logger.info(f"Lag feature shapes (samples, lag_features):")
logger.info(f"  Train:      {X_train_lag.shape}")
logger.info(f"  Validation: {X_val_lag.shape}")
logger.info(f"  Test:       {X_test_lag.shape}")
logger.info(
    f"\nSample columns: {list(X_train_lag.columns[:6])}... (total: {len(X_train_lag.columns)})"
)
logger.info(f"Feature type: {CONFIG['feature_type']}\n")

# =============================================================================
# 6. XGBOOST MODEL
# =============================================================================
import xgboost as xgb

logger.info("=" * 60)
logger.info("TRAINING XGBOOST")
logger.info("=" * 60)

model = xgb.XGBRegressor(
    objective=CONFIG["objective"],
    max_depth=CONFIG["max_depth"],
    learning_rate=CONFIG["learning_rate"],
    n_estimators=CONFIG["n_estimators"],
    subsample=CONFIG["subsample"],
    colsample_bytree=CONFIG["colsample_bytree"],
    min_child_weight=CONFIG["min_child_weight"],
    gamma=CONFIG["gamma"],
    reg_alpha=CONFIG["reg_alpha"],
    reg_lambda=CONFIG["reg_lambda"],
    tree_method=CONFIG["tree_method"],
    max_bin=CONFIG["max_bin"],
    n_jobs=CONFIG["n_jobs"],
    importance_type=CONFIG["importance_type"],
    verbosity=CONFIG["verbosity"],
    random_state=CONFIG["random_state"],
    early_stopping_rounds=CONFIG["early_stopping_rounds"],
)

# Fit with validation set for early stopping
model.fit(X_train_lag, y_train_lag, eval_set=[(X_val_lag, y_val_lag)], verbose=False)

best_iteration = (
    model.best_iteration if hasattr(model, "best_iteration") else CONFIG["n_estimators"]
)
logger.info(f"Best iteration: {best_iteration}")

# =============================================================================
# 7. SAVE MODEL
# =============================================================================
model.save_model(model_path)

# Also save config and feature names
meta_path = MODEL_DIR / "xgboost_meta.pkl"
joblib.dump(
    {
        "config": CONFIG,
        "feature_names": list(X_train_lag.columns),
        "feature_scaler_path": str(FEATURE_SCALER_PATH),
        "target_scaler_path": str(TARGET_SCALER_PATH),
    },
    meta_path,
)

logger.info(f"\nModel saved to {model_path}")
logger.info(f"Metadata saved to {meta_path}")

# =============================================================================
# 8. FEATURE IMPORTANCE
# =============================================================================
importance = model.get_booster().get_score(importance_type=CONFIG["importance_type"])
top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

logger.info(f"\nTop 10 Features by {CONFIG['importance_type']}:")
for feat, score in top_features:
    logger.info(f"  {feat:20s}: {score:.4f}")


# =============================================================================
# 10. EVALUATION
# =============================================================================
logger.info("\n" + "=" * 60)
logger.info("EVALUATION ON TEST SET")
logger.info("=" * 60)

predictions_scaled = model.predict(X_test_lag)

# Load scalers for inverse transform
loaded_target_scaler = joblib.load(TARGET_SCALER_PATH)

predictions_real = loaded_target_scaler.inverse_transform(
    predictions_scaled.reshape(-1, 1)
).flatten()
targets_real = loaded_target_scaler.inverse_transform(
    y_test_lag.reshape(-1, 1)
).flatten()

logger.info("\n" + "-" * 50)
logger.info("METRICS IN SCALED SPACE [-1, 1]")
logger.info("-" * 50)
scaled_metrics = {
    "RMSE": rmse(y_test_lag, predictions_scaled),
    "MAE": mae_metric(y_test_lag, predictions_scaled),
    "MAPE": mape(y_test_lag, predictions_scaled),
    "R²": r_squared(y_test_lag, predictions_scaled),
    "Directional Accuracy (excl. zeros)": directional_accuracy(
        y_test_lag, predictions_scaled, exclude_zeros=True
    ),
    "Directional Accuracy (ternary)": directional_accuracy(
        y_test_lag, predictions_scaled, exclude_zeros=False
    ),
    "F1 Ternary": f1_ternary(y_test_lag, predictions_scaled),
    "AUC Ternary": auc_ternary(y_test_lag, predictions_scaled),
}
for name, value in scaled_metrics.items():
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        logger.info(f"  {name:35s}: N/A")
    else:
        logger.info(f"  {name:35s}: {value:10.6f}")

logger.info("\n" + "-" * 50)
logger.info("METRICS IN REAL-WORLD SPACE")
logger.info("-" * 50)
real_metrics = {
    "RMSE": rmse(targets_real, predictions_real),
    "MAE": mae_metric(targets_real, predictions_real),
    "MAPE": mape(targets_real, predictions_real),
    "R²": r_squared(targets_real, predictions_real),
    "Directional Accuracy (excl. zeros)": directional_accuracy(
        targets_real, predictions_real, exclude_zeros=True
    ),
    "Directional Accuracy (ternary)": directional_accuracy(
        targets_real, predictions_real, exclude_zeros=False
    ),
    "F1 Ternary": f1_ternary(targets_real, predictions_real),
    "AUC Ternary": auc_ternary(targets_real, predictions_real),
}
for name, value in real_metrics.items():
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        logger.info(f"  {name:35s}: N/A")
    else:
        logger.info(f"  {name:35s}: {value:10.6f}")

logger.info("\n" + "-" * 60)
logger.info("SAMPLE PREDICTIONS (Real-World Values)")
logger.info("-" * 60)
logger.info(
    f"{'Index':>6} | {'Actual':>12} | {'Predicted':>12} | {'Error':>12} | {'Dir Match':>9}"
)
logger.info("-" * 60)
for i in range(min(15, len(predictions_real))):
    err = targets_real[i] - predictions_real[i]
    dir_match = (
        "YES" if np.sign(targets_real[i]) == np.sign(predictions_real[i]) else "NO"
    )
    logger.info(
        f"{i:6d} | {targets_real[i]:12.4f} | {predictions_real[i]:12.4f} | {err:12.4f} | {dir_match:>9}"
    )
