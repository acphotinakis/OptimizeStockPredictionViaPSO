# ── Standard library imports ───────────────────────────────────────────
import sys
from pathlib import Path
import logging
import seaborn as sns

# ── Third-party imports ───────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
from hydra import initialize, compose

# ── Add project root to sys.path for local module imports ─────────────
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# ── Local imports ─────────────────────────────────────────────────────
from src.features.build_features import calculate_indicators
from src.utils.logger import setup_logger

# ── Hydra configuration ──────────────────────────────────────────────
cfg_path = root_path / "configs"
initialize(config_path=str(cfg_path), version_base="1.3")
cfg = compose(config_name="config")

# ── Logger setup ─────────────────────────────────────────────────────
logger = setup_logger("04_feature_engineering_validation", cfg.paths.log_dir)

# ── Load raw data (e.g., SPY benchmark) ───────────────────────────────
raw_path = Path(cfg.paths.data_storage.raw) / "SPY_1Min.parquet"
logger.info("raw_path --> " + str(raw_path))

df = pd.read_parquet(raw_path)
logger.info(f"Dataset Shape: {df.shape}")
logger.info(f"Time Range: {df.index.min()} to {df.index.max()}")
logger.info(f"\n{df.head()}")

# ── Calculate indicators ─────────────────────────────────────────────
df_feats = calculate_indicators(df, cfg.features)

# ── Wavelet Denoising Validation ────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 5))
df_feats["close"].iloc[:500].plot(ax=ax, label="Raw Close", alpha=0.5)

if cfg.features.denoising.enabled:
    df_feats["close_denoised"].iloc[:500].plot(
        ax=ax, label="Wavelet Denoised", color="red"
    )

plt.legend()
plt.title("Wavelet Denoising Validation (First 500 Ticks)")

# ── Technical Indicator Overlay ─────────────────────────────────────
fig, ax = plt.subplots(2, 1, figsize=(15, 8))
df_feats[["close", "bb_high", "bb_low"]].iloc[:1000].plot(ax=ax[0])
df_feats["rsi"].iloc[:1000].plot(ax=ax[1], color="purple")

ax[1].axhline(70, color="red", linestyle="--")
ax[1].axhline(30, color="green", linestyle="--")
plot_path = Path(cfg.paths.data_storage.plots) / "04_feature_engineering_validation.png"
plt.savefig(plot_path)
plt.show()
