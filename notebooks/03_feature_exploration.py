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

# 03_feature_exploration.py
# Purpose: Analyze linear and non-linear relationships in the time-series before model training

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
import pandas as pd

# ── Assume df is already loaded (from previous steps) ────────────────
df = pd.read_parquet("../data/raw/SPY_1Min.parquet")

# ── Feature Correlation Heatmap ─────────────────────────────────────
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

# ── Autocorrelation Check ───────────────────────────────────────────
autocorrelation_plot(df["close"].resample("1H").mean())
plt.title("Hourly Price Autocorrelation")
plt.show()
