import os
import json
from pathlib import Path
import uuid
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict
from src.data.windowing import build_lstm_windows
from src.models.financial_dataset import FinancialDataset
from src.utils.logger import LogFileMode, setup_logger
import logging
import warnings
from _metrics import *

logger = logging.getLogger(__name__)

# FIX THIS LATER
setup_logger(
    log_file="build_features_v3.log",
    level="INFO",
    mode=LogFileMode.OVERWRITE,
)

# =============================================================================
# CONFIGURATION (PSO-LSTM)
# =============================================================================
CONFIG = {
    "name": "PSO-LSTM-v1.0",
    "framework": "pytorch",
    "enabled": True,
    "n_particles": 20,
    "n_iterations": 50,
    "inertia_min": 0.4,
    "inertia_max": 0.9,
    "c1": 1.5,
    "c2": 1.5,
    "v_clamp_fraction": 0.20,
    "search_space": {
        "lstm_units_1": {"min": 50, "max": 300},
        "lstm_units_2": {"min": 20, "max": 200},
        "dropout_rate": {"min": 0.0, "max": 0.5},
        "learning_rate": {"min": 0.001, "max": 0.01, "scale": "log"},
        "batch_size": {"choices": [32, 64]},
        "epochs": {"min": 50, "max": 300},
    },
    "lookback": 20,
    "shuffle": False,
    "deterministic": True,
    "random_seed": 42,
    "checkpoint_dir": "results/checkpoints",
    "fitness": {"gamma": 0.9},
    "prediction_horizon": 1,
}

SEED = CONFIG["random_seed"]
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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
OUTPUT_DIR = FEATURE_CACHE_DIR / "experiments" / f"pso_lstm_{uuid.uuid4()}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_DIR = OUTPUT_DIR / "pso_lstm" / "model"

PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(MODEL_DIR, exist_ok=True)
model_path = MODEL_DIR / "pso_lstm.pt"


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
# 4. CREATE SEQUENCES (Lookback Windows for LSTM)
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
logger.info(f"Sequence shapes (samples, timesteps, features):")
logger.info(f"  Train:      {X_train_seq.shape}")
logger.info(f"  Validation: {X_val_seq.shape}")
logger.info(f"  Test:       {X_test_seq.shape}\n")


# =============================================================================
# 5. PYTORCH DATASET
# =============================================================================
train_ds = FinancialDataset(X_train_seq, y_train_seq)
val_ds = FinancialDataset(X_val_seq, y_val_seq)
test_ds = FinancialDataset(X_test_seq, y_test_seq)


# =============================================================================
# 6. DYNAMIC LSTM (Architecture changes per particle)
# =============================================================================
class DynamicLSTM(nn.Module):
    def __init__(self, input_dim: int, units_1: int, units_2: int, dropout: float):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, units_1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(units_1, units_2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(units_2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out)
        out = out[:, -1, :]
        return self.fc(out)


# =============================================================================
# 7. MSW & FITNESS EVALUATION (Deng & Peng penalized objective)
# =============================================================================
def compute_msw(model: nn.Module) -> float:
    """Mean Square Weight: penalizes large network weights."""
    total_sq = 0.0
    total_n = 0
    for p in model.parameters():
        total_sq += (p**2).sum().item()
        total_n += p.numel()
    return total_sq / total_n if total_n > 0 else 0.0


def evaluate_particle(
    pos: np.ndarray, train_ds, val_ds, input_dim: int, device: torch.device
):
    """
    Decode particle position, train a candidate LSTM, and return fitness.
    Fitness = gamma * MSE + (1-gamma) * MSW
    """
    ss = CONFIG["search_space"]

    # Decode hyperparameters from particle position
    u1 = int(
        np.clip(round(pos[0]), ss["lstm_units_1"]["min"], ss["lstm_units_1"]["max"])
    )
    u2 = int(
        np.clip(round(pos[1]), ss["lstm_units_2"]["min"], ss["lstm_units_2"]["max"])
    )
    dr = float(np.clip(pos[2], ss["dropout_rate"]["min"], ss["dropout_rate"]["max"]))
    lr = 10 ** float(
        np.clip(
            pos[3],
            np.log10(ss["learning_rate"]["min"]),
            np.log10(ss["learning_rate"]["max"]),
        )
    )
    bs_idx = int(np.clip(round(pos[4]), 0, len(ss["batch_size"]["choices"]) - 1))
    bs = ss["batch_size"]["choices"][bs_idx]
    eps = int(np.clip(round(pos[5]), ss["epochs"]["min"], ss["epochs"]["max"]))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    model = DynamicLSTM(input_dim, u1, u2, dr).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience = 0
    best_state = None

    for epoch in range(eps):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                val_losses.append(criterion(pred, yb).item())
        vloss = np.mean(val_losses)

        if vloss < best_val_loss:
            best_val_loss = vloss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:  # early stopping to keep PSO fast
                break

    if best_state:
        model.load_state_dict(best_state)

    msw = compute_msw(model)
    gamma = CONFIG["fitness"]["gamma"]
    fitness = gamma * best_val_loss + (1 - gamma) * msw

    return fitness, {
        "mse": best_val_loss,
        "msw": msw,
        "fitness": fitness,
        "units_1": u1,
        "units_2": u2,
        "dropout": dr,
        "lr": lr,
        "batch_size": bs,
        "epochs": eps,
    }


# =============================================================================
# 8. IPSO IMPLEMENTATION (Ji et al. improvements)
# =============================================================================
class IPSO:
    def __init__(self, config: Dict):
        self.n_particles = config["n_particles"]
        self.n_iterations = config["n_iterations"]
        self.inertia_min = config["inertia_min"]
        self.inertia_max = config["inertia_max"]
        self.c1 = config["c1"]
        self.c2 = config["c2"]
        self.v_frac = config["v_clamp_fraction"]

        ss = config["search_space"]
        # Bounds: [units_1, units_2, dropout, log10(lr), batch_idx, epochs]
        self.bounds_low = np.array(
            [
                ss["lstm_units_1"]["min"],
                ss["lstm_units_2"]["min"],
                ss["dropout_rate"]["min"],
                np.log10(ss["learning_rate"]["min"]),
                0,
                ss["epochs"]["min"],
            ],
            dtype=np.float32,
        )

        self.bounds_high = np.array(
            [
                ss["lstm_units_1"]["max"],
                ss["lstm_units_2"]["max"],
                ss["dropout_rate"]["max"],
                np.log10(ss["learning_rate"]["max"]),
                len(ss["batch_size"]["choices"]) - 1,
                ss["epochs"]["max"],
            ],
            dtype=np.float32,
        )

        self.ranges = self.bounds_high - self.bounds_low
        self.v_max = self.v_frac * self.ranges

        # Initialize swarm
        self.positions = np.random.uniform(
            self.bounds_low, self.bounds_high, (self.n_particles, 6)
        )
        self.velocities = np.random.uniform(
            -self.v_max, self.v_max, (self.n_particles, 6)
        )

        self.pbest_pos = self.positions.copy()
        self.pbest_fit = np.full(self.n_particles, float("inf"))
        self.gbest_pos = None
        self.gbest_fit = float("inf")
        self.gbest_info = None

    def inertia_weight(self, iteration: int) -> float:
        """Non-linear decreasing inertia via hyperbolic tangent (Ji et al.)."""
        prog = iteration / max(1, self.n_iterations - 1)
        return self.inertia_min + (self.inertia_max - self.inertia_min) * (
            1 - np.tanh(2 * prog)
        )

    def mutate(self, pos: np.ndarray, iteration: int) -> np.ndarray:
        """Adaptive mutation factor - probability increases with iterations."""
        mutation_prob = 0.1 + 0.4 * (iteration / max(1, self.n_iterations - 1))
        if np.random.random() < mutation_prob:
            return np.random.uniform(self.bounds_low, self.bounds_high)
        return pos

    def optimize(self, train_ds, val_ds, input_dim: int, device: torch.device):
        logger.info("=" * 60)
        logger.info("IPSO OPTIMIZATION STARTED")
        logger.info("=" * 60)
        logger.info(
            "WARNING: PSO trains many LSTM models; this may take several minutes.\n"
        )

        for it in range(self.n_iterations):
            w = self.inertia_weight(it)

            # Evaluate each particle
            for i in range(self.n_particles):
                fit, info = evaluate_particle(
                    self.positions[i], train_ds, val_ds, input_dim, device
                )

                if fit < self.pbest_fit[i]:
                    self.pbest_fit[i] = fit
                    self.pbest_pos[i] = self.positions[i].copy()

                    if fit < self.gbest_fit:
                        self.gbest_fit = fit
                        self.gbest_pos = self.positions[i].copy()
                        self.gbest_info = info

            # Update velocities and positions (standard PSO equations)
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (
                    w * self.velocities[i]
                    + self.c1 * r1 * (self.pbest_pos[i] - self.positions[i])
                    + self.c2 * r2 * (self.gbest_pos - self.positions[i])
                )
                # Velocity clamping to 20% of search range
                self.velocities[i] = np.clip(
                    self.velocities[i], -self.v_max, self.v_max
                )

                # Update position
                self.positions[i] = self.positions[i] + self.velocities[i]
                self.positions[i] = np.clip(
                    self.positions[i], self.bounds_low, self.bounds_high
                )

                # Adaptive mutation (IPSO)
                self.positions[i] = self.mutate(self.positions[i], it)

            if (it + 1) % 5 == 0 or it == 0:
                logger.info(
                    f"Iter {it+1:3d}/{self.n_iterations} | w={w:.3f} | gbest_fitness={self.gbest_fit:.6f}"
                )
                if self.gbest_info:
                    logger.info(
                        f"  Best: units=({self.gbest_info['units_1']},{self.gbest_info['units_2']}), "
                        f"drop={self.gbest_info['dropout']:.3f}, lr={self.gbest_info['lr']:.5f}, "
                        f"bs={self.gbest_info['batch_size']}, epochs={self.gbest_info['epochs']}"
                    )
                    logger.info(
                        f"  Components: MSE={self.gbest_info['mse']:.6f}, MSW={self.gbest_info['msw']:.6f}"
                    )

        return self.gbest_pos, self.gbest_info


# =============================================================================
# 9. RUN IPSO OPTIMIZATION
# =============================================================================
ipso = IPSO(CONFIG)
best_pos, best_info = ipso.optimize(train_ds, val_ds, X_train_seq.shape[2], DEVICE)

logger.info("\n" + "=" * 60)
logger.info("PSO OPTIMIZATION COMPLETE")
logger.info("=" * 60)
logger.info(f"Global Best Fitness: {ipso.gbest_fit:.6f}")
logger.info(f"Optimal Hyperparameters:")
for k, v in best_info.items():
    logger.info(f"  {k:15s}: {v}")

# =============================================================================
# 10. TRAIN FINAL MODEL WITH OPTIMAL HYPERPARAMETERS
# =============================================================================
logger.info("\n" + "=" * 60)
logger.info("TRAINING FINAL MODEL WITH OPTIMAL PARAMETERS")
logger.info("=" * 60)

# Combine train+val for final model training
X_train_val = np.concatenate([X_train_seq, X_val_seq], axis=0)
y_train_val = np.concatenate([y_train_seq, y_val_seq], axis=0)
train_val_ds = FinancialDataset(X_train_val, y_train_val)

final_train_loader = DataLoader(
    train_val_ds, batch_size=best_info["batch_size"], shuffle=False
)
final_test_loader = DataLoader(
    test_ds, batch_size=best_info["batch_size"], shuffle=False
)

final_model = DynamicLSTM(
    input_dim=X_train_seq.shape[2],
    units_1=best_info["units_1"],
    units_2=best_info["units_2"],
    dropout=best_info["dropout"],
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=best_info["lr"])

for epoch in range(best_info["epochs"]):
    final_model.train()
    for Xb, yb in final_train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = final_model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        logger.info(f"Final Model Epoch {epoch+1}/{best_info['epochs']}")

# =============================================================================
# 11. SAVE MODEL & CHECKPOINT
# =============================================================================

training_history = {
    "train_losses_history": [],
    "val_losses_history": [],
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
        "best_info": best_info,
        "model_state_dict": final_model.state_dict(),
        "feature_scaler_path": str(FEATURE_SCALER_PATH),
        "target_scaler_path": str(TARGET_SCALER_PATH),
        "feature_names": selected_names,
        "feature_contract": feature_contract,
        "best_val_loss": 0.0,
        "training_history": training_history,
    },
    model_path,
)
logger.info(f"\nModel saved to {model_path}")


# =============================================================================
# 12. STATISTICAL METRICS
# =============================================================================


# =============================================================================
# 13. EVALUATION ON TEST SET
# =============================================================================
logger.info("\n" + "=" * 60)
logger.info("EVALUATION ON TEST SET")
logger.info("=" * 60)

final_model.eval()
all_pred_s = []
all_true_s = []

with torch.no_grad():
    for Xb, yb in final_test_loader:
        Xb = Xb.to(DEVICE)
        pred = final_model(Xb).cpu().numpy()
        all_pred_s.append(pred)
        all_true_s.append(yb.numpy())

pred_s = np.concatenate(all_pred_s).flatten()
true_s = np.concatenate(all_true_s).flatten()

loaded_target_scaler = joblib.load("./scalers_pso/target_scaler.pkl")
pred_real = loaded_target_scaler.inverse_transform(pred_s.reshape(-1, 1)).flatten()
true_real = loaded_target_scaler.inverse_transform(true_s.reshape(-1, 1)).flatten()

logger.info("\n" + "-" * 50)
logger.info("METRICS IN SCALED SPACE [-1, 1]")
logger.info("-" * 50)
scaled_metrics = {
    "RMSE": rmse(true_s, pred_s),
    "MAE": mae_metric(true_s, pred_s),
    "MAPE": mape(true_s, pred_s),
    "R²": r_squared(true_s, pred_s),
    "Directional Accuracy (excl. zeros)": directional_accuracy(
        true_s, pred_s, exclude_zeros=True
    ),
    "Directional Accuracy (ternary)": directional_accuracy(
        true_s, pred_s, exclude_zeros=False
    ),
    "F1 Ternary": f1_ternary(true_s, pred_s),
    "AUC Ternary": auc_ternary(true_s, pred_s),
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
    "RMSE": rmse(true_real, pred_real),
    "MAE": mae_metric(true_real, pred_real),
    "MAPE": mape(true_real, pred_real),
    "R²": r_squared(true_real, pred_real),
    "Directional Accuracy (excl. zeros)": directional_accuracy(
        true_real, pred_real, exclude_zeros=True
    ),
    "Directional Accuracy (ternary)": directional_accuracy(
        true_real, pred_real, exclude_zeros=False
    ),
    "F1 Ternary": f1_ternary(true_real, pred_real),
    "AUC Ternary": auc_ternary(true_real, pred_real),
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
for i in range(min(15, len(pred_real))):
    err = true_real[i] - pred_real[i]
    dir_match = "YES" if np.sign(true_real[i]) == np.sign(pred_real[i]) else "NO"
    logger.info(
        f"{i:6d} | {true_real[i]:12.4f} | {pred_real[i]:12.4f} | {err:12.4f} | {dir_match:>9}"
    )
