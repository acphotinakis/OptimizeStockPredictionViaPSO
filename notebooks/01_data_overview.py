# ── Standard library imports ───────────────────────────────────────────
import sys
from pathlib import Path
import logging

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
logger = setup_logger("01_data_overview", cfg.paths.log_dir)

# ── Load raw data (e.g., SPY benchmark) ───────────────────────────────
raw_path = Path(cfg.paths.data_storage.raw) / "SPY_1Min.parquet"
logger.info("raw_path --> " + str(raw_path))

df = pd.read_parquet(raw_path)
logger.info(f"Dataset Shape: {df.shape}")
logger.info(f"Time Range: {df.index.min()} to {df.index.max()}")
logger.info(f"\n{df.head()}")

# ── Summary Statistics ───────────────────────────────────────────────
logger.info("\nSummary Statistics:")
logger.info(f"\n{df.describe()}")

logger.info("\nTotal Missing Values:")
logger.info(f"\n{df.isnull().sum()}")

# Additional descriptive statistics
logger.info("\nSkewness of each column:")
logger.info(f"\n{df.skew()}")

logger.info("\nKurtosis of each column:")
logger.info(f"\n{df.kurtosis()}")

logger.info("\nMedian values of each column:")
logger.info(f"\n{df.median()}")

logger.info("\nQuantiles (5%, 25%, 50%, 75%, 95%):")
logger.info(f"\n{df.quantile([0.05, 0.25, 0.5, 0.75, 0.95])}")

# ── Price and Volume Visualization ───────────────────────────────────
fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
df["close"].plot(ax=ax[0], color="blue", title="SPY Close Price")
df["volume"].plot(ax=ax[1], kind="area", color="gray", alpha=0.3, title="Transaction Volume")

plt.tight_layout()
plot_path = Path(cfg.paths.data_storage.plots) / "01_data_overview.png"
plt.savefig(plot_path)
plt.show()
