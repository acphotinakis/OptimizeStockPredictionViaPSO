# Constants

from pathlib import Path


FEATURES_DIR = "data/features/"
RESULTS_DIR = "results/"

# Data Ingestion Constants
MAX_MISSING_FRACTION = 0.05  # Drop timestamps where >5 % of tickers have NaN
MAX_FFILL_BARS = 5  # Forward-fill at most 5 bars per ticker during alignment

# Data Cleaner Constants
SESSION_START = "09:30"
SESSION_END = "16:00"
MAX_GAP_FILL_BARS = 5  # Forward-fill at most this many missing bars
OUTLIER_ZSCORE_THRESHOLD = 5  # Clip returns beyond ±5σ
OUTLIER_ROLLING_WINDOW = 60  # Rolling window (bars) for z-score computation

# Validation Constants
DEFAULT_WFV_FOLD_SIZE = 252 * 390  # 1 month of 1-minute bars
DEFAULT_WFV_FOLDS = 6


DEFAULT_CONFIG_PATH = Path("config/default_config.yaml")
