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


# 01_data_overview.py
# High-level summary of the raw high-frequency market data

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Load raw data (e.g., SPY benchmark) ────────────────────────────────
raw_path = Path("../data/raw/SPY_1Min.parquet")
df = pd.read_parquet(raw_path)

print(f"Dataset Shape: {df.shape}")
print(f"Time Range: {df.index.min()} to {df.index.max()}")
print(df.head())

# ── Summary Statistics ────────────────────────────────────────────────
print("\nSummary Statistics:")
print(df.describe())

print("\nTotal Missing Values:")
print(df.isnull().sum())

# ── Price and Volume Visualization ───────────────────────────────────
fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

df["close"].plot(ax=ax[0], color="blue", title="SPY Close Price")
df["volume"].plot(
    ax=ax[1], kind="area", color="gray", alpha=0.3, title="Transaction Volume"
)

plt.tight_layout()
plt.show()
