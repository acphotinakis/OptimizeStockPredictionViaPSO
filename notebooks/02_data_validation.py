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
logger = setup_logger("data_overview", cfg.paths.log_dir)

# ── Load raw data (e.g., SPY benchmark) ───────────────────────────────
raw_path = Path(cfg.paths.data_storage.raw) / "SPY_1Min.parquet"
logger.info("raw_path --> " + str(raw_path))

df = pd.read_parquet(raw_path)
logger.info(f"Dataset Shape: {df.shape}")
logger.info(f"Time Range: {df.index.min()} to {df.index.max()}")
logger.info(f"\n{df.head()}")

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
plot_path = Path(cfg.paths.data_storage.plots) / "02_data_validation.png"
plt.savefig(plot_path)
plt.show()
