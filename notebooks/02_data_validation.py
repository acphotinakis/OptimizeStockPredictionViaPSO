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

# 02_data_validation.py
# Validating data against thresholds for missing values, gaps, and outliers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Assume df is already loaded (from 01_data_overview.py) ────────────
# df = pd.read_parquet("../data/raw/SPY_1Min.parquet")
# cfg = your config object with validation thresholds

# ── Load raw data (e.g., SPY benchmark) ────────────────────────────────
raw_path = Path("../data/raw/SPY_1Min.parquet")
df = pd.read_parquet(raw_path)

# ── Gap Detection (Missing 1-Min Intervals) ─────────────────────────────
expected_intervals = pd.date_range(
    start=df.index.min(), end=df.index.max(), freq="1min"
)
missing_intervals = expected_intervals.difference(df.index)

print(
    f"Missing Time Steps: {len(missing_intervals)} ({len(missing_intervals)/len(expected_intervals):.2%})"
)

# Check against config threshold
max_allowed = cfg.data.validation.max_missing_pct
print(f"Config Threshold: {max_allowed * 100}%")

# ── Outlier Analysis (Price Returns) ───────────────────────────────────
returns = df["close"].pct_change().dropna()

sns.boxplot(x=returns)
plt.title("Distribution of 1-Min Returns (Outlier Detection)")
plt.show()
