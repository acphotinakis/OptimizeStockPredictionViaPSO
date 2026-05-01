import os
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from _metrics import *
from src.data.windowing import build_lstm_windows
from src.models.financial_dataset import FinancialDataset
from src.utils.data_storage import _load_parquet
from src.utils.logger import LogFileMode, setup_logger
import logging
import uuid

logger = logging.getLogger(__name__)

# FIX THIS LATER
setup_logger(
    log_file="build_features_v3.log",
    level="INFO",
    mode=LogFileMode.OVERWRITE,
)

# =============================================================================
# CONFIGURATION (from your YAML spec)
# =============================================================================
CONFIG = {
    "name": "LSTM-Baseline-v1.0",
    "framework": "pytorch",
    "lstm_units_1": 128,
    "lstm_units_2": 64,
    "dropout_rate": 0.1,
    "activation": "tanh",
    "output_units": 1,
    "output_activation": "linear",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "loss": "mse",
    "epochs": 100,
    # "batch_size": 32,
    # "batch_size": 4096,
    "batch_size": 512,
    "shuffle": False,
    "lookback": 20,
    # "prediction_horizon": 1,
    "prediction_horizon": 6,
    "random_seed": 42,
    "deterministic": True,
    "validation_split": 0.10,
    "early_stopping": {
        "enabled": True,
        "monitor": "val_loss",
        "patience": 10,
        "restore_best_weights": True,
    },
    "grad_clip": 1.0,
    "use_amp": True,
    "accumulation_steps": 1,
}

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = CONFIG["random_seed"]
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
if CONFIG["deterministic"]:
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")
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
OUTPUT_DIR = FEATURE_CACHE_DIR / "experiments" / f"lstm_baseline_{uuid.uuid4()}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_DIR = OUTPUT_DIR / "lstm_baseline" / "model"

PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(MODEL_DIR, exist_ok=True)
model_path = MODEL_DIR / "lstm_baseline.pt"


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
# 5. CREATE SEQUENCES (Lookback Windows for LSTM)
# =============================================================================


def take_fraction(X, y, frac: float):
    n = int(len(X) * frac)
    return X[:n], y[:n]


LOOKBACK = CONFIG["lookback"]
HORIZON = CONFIG["prediction_horizon"]

FRACTION = 0.10  # 10%


X_train_seq, y_train_seq = build_lstm_windows(
    X_train_sel, y_train_scaled, LOOKBACK, HORIZON
)
X_val_seq, y_val_seq = build_lstm_windows(X_val_sel, y_val_scaled, LOOKBACK, HORIZON)
X_test_seq, y_test_seq = build_lstm_windows(
    X_test_sel, y_test_scaled, LOOKBACK, HORIZON
)

# X_train_seq, y_train_seq = take_fraction(X_train_seq, y_train_seq, FRACTION)
# X_val_seq, y_val_seq = take_fraction(X_val_seq, y_val_seq, FRACTION)
# X_test_seq, y_test_seq = take_fraction(X_test_seq, y_test_seq, FRACTION)

logger.info(f"Sequence shapes (samples, timesteps, features):")
logger.info(f"  Train:      {X_train_seq.shape}")
logger.info(f"  Validation: {X_val_seq.shape}")
logger.info(f"  Test:       {X_test_seq.shape}\n")


# =============================================================================
# 6. PYTORCH DATASET & DATALOADER
# =============================================================================
train_dataset = FinancialDataset(X_train_seq, y_train_seq)
val_dataset = FinancialDataset(X_val_seq, y_val_seq)
test_dataset = FinancialDataset(X_test_seq, y_test_seq)

# train_loader = DataLoader(
#     train_dataset, batch_size=CONFIG["batch_size"], shuffle=CONFIG["shuffle"]
# )
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=CONFIG["shuffle"],
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)


# =============================================================================
# 7. LSTM MODEL DEFINITION
# =============================================================================
class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int, config: Dict):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_dim, hidden_size=config["lstm_units_1"], batch_first=True
        )
        self.dropout1 = nn.Dropout(config["dropout_rate"])

        self.lstm2 = nn.LSTM(
            input_size=config["lstm_units_1"],
            hidden_size=config["lstm_units_2"],
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(config["dropout_rate"])

        self.fc = nn.Linear(config["lstm_units_2"], config["output_units"])

        # Activation mapping
        self.activation = nn.Tanh() if config["activation"] == "tanh" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM layer 1
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        # LSTM layer 2 (use final timestep output)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        # Take last timestep
        out = out[:, -1, :]

        # Final output (linear activation by default)
        out = self.fc(out)

        return out


model = LSTMBaseline(input_dim=X_train_seq.shape[2], config=CONFIG).to(DEVICE)
logger.info(model)
logger.info(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# =============================================================================
# 8. TRAINING SETUP
# =============================================================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

# AMP for mixed precision training
scaler = (
    GradScaler(device=str(DEVICE))
    if CONFIG["use_amp"] and torch.cuda.is_available()
    else None
)

# Early stopping state
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None

# =============================================================================
# 9. TRAINING LOOP
# =============================================================================
logger.info("=" * 60)
logger.info("TRAINING STARTED")
logger.info("=" * 60)

train_losses_history = []
val_losses_history = []

train_preds_history = []
train_targets_history = []

val_preds_history = []
val_targets_history = []

# BEST SNAPSHOT STORAGE
best_train_preds = None
best_train_targets = None
best_val_preds = None
best_val_targets = None


for epoch in tqdm(range(CONFIG["epochs"]), desc="Training Epochs"):
    # ---- Training Phase ----
    model.train()

    train_losses = []
    train_preds = []
    train_targets = []

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass with AMP
        if scaler:
            with autocast(device_type="cuda", enabled=scaler is not None):
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
        else:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if batch_idx % 2 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if batch_idx % 2 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()

        train_losses.append(loss.item())
        train_preds.append(y_pred.detach().cpu().numpy())
        train_targets.append(y_batch.detach().cpu().numpy())

    avg_train_loss = float(np.mean(train_losses))
    train_losses_history.append(avg_train_loss)

    train_preds = np.concatenate(train_preds)
    train_targets = np.concatenate(train_targets).flatten()

    train_preds_history.append(train_preds)
    train_targets_history.append(train_targets)

    # ---- Validation Phase ----
    model.eval()

    val_losses = []
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_loader, desc="Validation", leave=False):
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            if scaler:
                with autocast(device_type=str(DEVICE)):
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
            else:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)

            val_losses.append(loss.item())
            val_preds.append(y_pred.detach().cpu().numpy())
            val_targets.append(y_batch.detach().cpu().numpy())

    avg_val_loss = float(np.mean(val_losses))
    val_losses_history.append(avg_val_loss)

    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()

    val_preds_history.append(val_preds)
    val_targets_history.append(val_targets)

    # ---- Logging ----
    if (epoch + 1) % 2 == 0 or epoch == 0:
        logger.info(
            f"Epoch [{epoch+1:3d}/{CONFIG['epochs']}]  "
            f"Train Loss: {avg_train_loss:.10f}  "
            f"Val Loss: {avg_val_loss:.10f}"
        )

    # ---- Early Stopping ----
    if CONFIG["early_stopping"]["enabled"]:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if CONFIG["early_stopping"]["restore_best_weights"]:
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            best_train_preds = train_preds.copy()
            best_train_targets = train_targets.copy()
            best_val_preds = val_preds.copy()
            best_val_targets = val_targets.copy()
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping"]["patience"]:
                logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

# Restore best weights
if (
    CONFIG["early_stopping"]["enabled"]
    and CONFIG["early_stopping"]["restore_best_weights"]
    and best_model_state
):
    model.load_state_dict(best_model_state)
    logger.info(f"Restored best model (Val Loss: {best_val_loss})")


# =============================================================================
# LOAD ARTIFACTS
# =============================================================================
assert PREDICTIONS_DIR.exists(), f"Missing directory: {PREDICTIONS_DIR}"
assert TARGET_SCALER_PATH.exists(), f"Missing scaler: {TARGET_SCALER_PATH}"

train_losses = np.array(train_losses_history)
val_losses = np.array(val_losses_history)

preds = np.array(best_val_preds)
targets = np.array(best_val_targets)

assert preds.shape == targets.shape, "Pred/target shape mismatch"

# =============================================================================
# INVERSE TRANSFORM
# =============================================================================
target_scaler = joblib.load(TARGET_SCALER_PATH)

preds_real = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
targets_real = target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

# =============================================================================
# METRICS (SCALED)
# =============================================================================
logger.info("-" * 50)
logger.info("METRICS (SCALED)")
logger.info("-" * 50)

scaled_metrics = {
    "RMSE": rmse(targets, preds),
    "MAE": mae_metric(targets, preds),
    "MAPE": mape(targets, preds),
    "R²": r_squared(targets, preds),
    "Directional Accuracy (excl. zeros)": directional_accuracy(targets, preds, True),
    "Directional Accuracy (ternary)": directional_accuracy(targets, preds, False),
    "F1 Ternary": f1_ternary(targets, preds),
    "AUC Ternary": auc_ternary(targets, preds),
    "IC (Pearson)": information_coefficient(targets, preds),
    "Hit Rate": hit_rate(targets, preds, threshold=1e-4),
    "Sharpe": sharpe_ratio(targets, preds),
}

for k, v in scaled_metrics.items():
    logger.info(f"{k:35s}: {v:10.6f}" if np.isfinite(v) else f"{k:35s}: N/A")


logger.info("-" * 60)
logger.info("SAMPLE PREDICTIONS (SCALED)")
logger.info("-" * 60)

for i in range(min(15, len(preds))):
    err = targets[i] - preds[i]
    match = np.sign(targets[i]) == np.sign(preds[i])
    logger.info(
        f"{i:4d} | actual={targets[i]: .10f} "
        f"pred={preds[i]: .10f} err={err: .10f} "
        f"dir={'Y' if match else 'N'}"
    )


# =============================================================================
# METRICS (REAL)
# =============================================================================
logger.info("-" * 50)
logger.info("METRICS (REAL)")
logger.info("-" * 50)

real_metrics = {
    "RMSE": rmse(targets_real, preds_real),
    "MAE": mae_metric(targets_real, preds_real),
    "MAPE": mape(targets_real, preds_real),
    "R²": r_squared(targets_real, preds_real),
    "Directional Accuracy (excl. zeros)": directional_accuracy(
        targets_real, preds_real, True
    ),
    "Directional Accuracy (ternary)": directional_accuracy(
        targets_real, preds_real, False
    ),
    "F1 Ternary": f1_ternary(targets_real, preds_real),
    "AUC Ternary": auc_ternary(targets_real, preds_real),
    "IC (Pearson)": information_coefficient(targets_real, preds_real),
    "Hit Rate": hit_rate(targets_real, preds_real, threshold=1e-4),
    "Sharpe": sharpe_ratio(targets_real, preds_real),
}

for k, v in real_metrics.items():
    logger.info(f"{k:35s}: {v:10.6f}" if np.isfinite(v) else f"{k:35s}: N/A")

# =============================================================================
# SAMPLE TABLE
# =============================================================================
logger.info("-" * 60)
logger.info("SAMPLE PREDICTIONS (REAL)")
logger.info("-" * 60)

for i in range(min(15, len(preds_real))):
    err = targets_real[i] - preds_real[i]
    match = np.sign(targets_real[i]) == np.sign(preds_real[i])
    logger.info(
        f"{i:4d} | actual={targets_real[i]: .10f} "
        f"pred={preds_real[i]: .10f} err={err: .10f} "
        f"dir={'Y' if match else 'N'}"
    )

# =============================================================================
# PLOTS
# =============================================================================
import matplotlib.pyplot as plt

# ---- LOSS CURVE ----
plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig(PLOTS_DIR / "loss_curve.png")
plt.close()

# ---- PRED VS ACTUAL (SCALED) ----
plt.figure()
plt.plot(targets[:500], label="Actual")
plt.plot(preds[:500], label="Pred")
plt.title("Pred vs Actual (Scaled)")
plt.legend()
plt.grid()
plt.savefig(PLOTS_DIR / "pred_vs_actual_scaled.png")
plt.close()

# ---- PRED VS ACTUAL (REAL) ----
plt.figure()
plt.plot(targets_real[:500], label="Actual")
plt.plot(preds_real[:500], label="Pred")
plt.title("Pred vs Actual (Real)")
plt.legend()
plt.grid()
plt.savefig(PLOTS_DIR / "pred_vs_actual_real.png")
plt.close()

# ---- SCATTER ----
plt.figure()
plt.scatter(targets, preds, alpha=0.3)
min_v = min(targets.min(), preds.min())
max_v = max(targets.max(), preds.max())
plt.plot([min_v, max_v], [min_v, max_v])
plt.title("Scatter (Scaled)")
plt.xlabel("Actual")
plt.ylabel("Pred")
plt.grid()
plt.savefig(PLOTS_DIR / "scatter_scaled.png")
plt.close()

# ---- SCATTER REAL ----
plt.figure()
plt.scatter(targets_real, preds_real, alpha=0.3)
min_v = min(targets_real.min(), preds_real.min())
max_v = max(targets_real.max(), preds_real.max())
plt.plot([min_v, max_v], [min_v, max_v])
plt.title("Scatter (Real)")
plt.xlabel("Actual")
plt.ylabel("Pred")
plt.grid()
plt.savefig(PLOTS_DIR / "scatter_real.png")
plt.close()

# ---- RESIDUALS ----
residuals = targets - preds
plt.figure()
plt.hist(residuals, bins=100)
plt.title("Residuals (Scaled)")
plt.grid()
plt.savefig(PLOTS_DIR / "residuals_scaled.png")
plt.close()

# ---- RESIDUALS REAL ----
residuals_real = targets_real - preds_real
plt.figure()
plt.hist(residuals_real, bins=100)
plt.title("Residuals (Real)")
plt.grid()
plt.savefig(PLOTS_DIR / "residuals_real.png")
plt.close()

# ---- DIRECTIONAL ACCURACY  ----
dir_true = np.sign(targets)
dir_pred = np.sign(preds)
correct = (dir_true == dir_pred).astype(int)

plt.figure()
plt.plot(correct[:500])
plt.title("Directional Accuracy")
plt.ylim(-0.1, 1.1)
plt.grid()
plt.savefig(PLOTS_DIR / "directional_accuracy_scaled.png")
plt.close()

logger.info(f"Directional Accuracy: {correct.mean():.6f}")

# ---- DIRECTIONAL ACCURACY REAL ----
dir_true = np.sign(targets_real)
dir_pred = np.sign(preds_real)
correct = (dir_true == dir_pred).astype(int)

plt.figure()
plt.plot(correct[:500])
plt.title("Directional Accuracy")
plt.ylim(-0.1, 1.1)
plt.grid()
plt.savefig(PLOTS_DIR / "directional_accuracy_real.png")
plt.close()

logger.info(f"Directional Accuracy: {correct.mean():.6f}")

# ---- ROLLING CORRELATION ----
window = 100
rolling_corr = pd.Series(preds).rolling(window).corr(pd.Series(targets))

plt.figure()
plt.plot(rolling_corr)
plt.title(f"Rolling Corr (window={window})")
plt.grid()
plt.savefig(PLOTS_DIR / "rolling_corr_scaled.png")
plt.close()

# ---- ROLLING CORRELATION REAL ----
window = 100
rolling_corr = pd.Series(preds_real).rolling(window).corr(pd.Series(targets_real))

plt.figure()
plt.plot(rolling_corr)
plt.title(f"Rolling Corr (window={window})")
plt.grid()
plt.savefig(PLOTS_DIR / "rolling_corr_real.png")
plt.close()


logger.info("Plots + metrics complete.")
# import sys

# sys.exit(0)


# =============================================================================
# 10. SAVE MODEL
# =============================================================================

training_history = {
    "train_losses_history": train_losses_history,
    "val_losses_history": val_losses_history,
}
feature_contract = {
    "lookback": CONFIG["lookback"],
    "horizon": CONFIG["prediction_horizon"],
    "input_dim": X_train_seq.shape[2],
    "num_features_selected": (
        len(selected_names) if "selected_names" in globals() else None
    ),
}

torch.save(
    {
        "config": CONFIG,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "feature_scaler_path": str(FEATURE_SCALER_PATH),
        "target_scaler_path": str(TARGET_SCALER_PATH),
        "feature_names": selected_names,
        "feature_contract": feature_contract,
        "best_val_loss": best_val_loss,
        "training_history": training_history,
    },
    model_path,
)
logger.info(f"\nModel saved to {model_path}")

# =============================================================================
# 11. EVALUATION & INVERSE TRANSFORM
# =============================================================================
logger.info("=" * 60)
logger.info("EVALUATION ON TEST SET")
logger.info("=" * 60)

model.eval()
all_predictions_scaled = []
all_targets_scaled = []

with torch.no_grad():
    # for X_batch, y_batch in test_loader:
    for X_batch, y_batch in tqdm(test_loader, desc="Evaluation", leave=False):
        X_batch = X_batch.to(DEVICE)
        y_pred = model(X_batch)
        all_predictions_scaled.append(y_pred.cpu().numpy())
        all_targets_scaled.append(y_batch.cpu().numpy())

# Concatenate batches
predictions_scaled = np.concatenate(all_predictions_scaled).flatten()
targets_scaled = np.concatenate(all_targets_scaled).flatten()

# ---- LOAD SCALERS (simulating inference environment) ----
loaded_target_scaler = joblib.load(TARGET_SCALER_PATH)

# ---- INVERSE TRANSFORM TO REAL-WORLD VALUES ----
predictions_real = loaded_target_scaler.inverse_transform(
    predictions_scaled.reshape(-1, 1)
).flatten()
targets_real = loaded_target_scaler.inverse_transform(
    targets_scaled.reshape(-1, 1)
).flatten()

# ---- COMPUTE ALL METRICS ----
logger.info("-" * 50)
logger.info("METRICS IN SCALED SPACE [-1, 1]")
logger.info("-" * 50)
scaled_metrics = {
    "RMSE": rmse(targets_scaled, predictions_scaled),
    "MAE": mae_metric(targets_scaled, predictions_scaled),
    "MAPE": mape(targets_scaled, predictions_scaled),
    "R²": r_squared(targets_scaled, predictions_scaled),
    "Directional Accuracy (excl. zeros)": directional_accuracy(
        targets_scaled, predictions_scaled, exclude_zeros=True
    ),
    "Directional Accuracy (ternary)": directional_accuracy(
        targets_scaled, predictions_scaled, exclude_zeros=False
    ),
    "F1 Ternary": f1_ternary(targets_scaled, predictions_scaled),
    "AUC Ternary": auc_ternary(targets_scaled, predictions_scaled),
    "IC (Pearson)": information_coefficient(targets_scaled, predictions_scaled),
    "Hit Rate": hit_rate(targets_scaled, predictions_scaled, threshold=1e-4),
    "Sharpe": sharpe_ratio(targets_scaled, predictions_scaled),
}

for k, v in scaled_metrics.items():
    logger.info(f"{k:35s}: {v:10.6f}" if np.isfinite(v) else f"{k:35s}: N/A")

logger.info("-" * 50)
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
    "IC (Pearson)": information_coefficient(targets_real, predictions_real),
    "Hit Rate": hit_rate(targets_real, predictions_real, threshold=1e-4),
    "Sharpe": sharpe_ratio(targets_real, predictions_real),
}
for k, v in real_metrics.items():
    logger.info(f"{k:35s}: {v:10.6f}" if np.isfinite(v) else f"{k:35s}: N/A")
# ---- SAMPLE COMPARISON ----
logger.info("-" * 60)
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
