# ── Standard library imports ───────────────────────────────────────────
import sys
from pathlib import Path
import logging
import seaborn as sns
from pandas.plotting import autocorrelation_plot

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

# ── Feature Correlation Heatmap ─────────────────────────────────────
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

# ── Autocorrelation Check ───────────────────────────────────────────
autocorrelation_plot(df["close"].resample("1H").mean())
plt.title("Hourly Price Autocorrelation")
plot_path = Path(cfg.paths.data_storage.plots) / "03_feature_exploration.png"
plt.savefig(plot_path)
plt.show()
