import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.features.build_features import calculate_indicators

from hydra import initialize, compose

# Add project root to sys.path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# Hydra config
cfg_path = root_path / "configs"
initialize(config_path=str(cfg_path), version_base="1.3")
cfg = compose(config_name="config")

# 04_feature_engineering_validation.py
# Purpose: Validate that engineered features (RSI, Wavelets) are correctly transformed

import matplotlib.pyplot as plt
from src.features.build_features import calculate_indicators
import pandas as pd

# ── Assume df and cfg are already loaded ───────────────────────────────
df = pd.read_parquet("../data/raw/SPY_1Min.parquet")

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

plt.show()
