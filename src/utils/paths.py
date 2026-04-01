import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class ProjectPaths:
    """
    Manages and validates the directory structure for the PSO-LSTM Stock Tuner.
    """

    def __init__(self, cfg):
        """
        Initializes path mappings from Hydra configuration.
        """
        self.cfg = cfg
        # Mapping Hydra config paths to Path objects
        self.raw_data = Path(cfg.paths.data_storage.raw)
        self.interim_data = Path(cfg.paths.data_storage.interim)
        self.processed_data = Path(cfg.paths.data_storage.processed)
        self.checkpoints = Path(cfg.paths.model_save_path)
        self.results = Path(cfg.paths.results_dir)

    def ensure_directories(self):
        """
        Physically creates the directory tree if it does not exist.
        """
        directories = [
            self.raw_data,
            self.interim_data,
            self.processed_data,
            self.checkpoints,
            self.results,
        ]

        for directory in directories:
            if not directory.exists():
                logger.info(f"Creating directory: {directory}")
                directory.mkdir(parents=True, exist_ok=True)
            else:
                logger.debug(f"Verified directory exists: {directory}")

    def get_raw_file_path(self, ticker: str) -> Path:
        """Constructs the path for a specific raw ticker file."""
        return self.raw_data / f"{ticker}_{self.cfg.data.timeframe}.parquet"

    def get_checkpoint_path(self, filename: str = "best_model.pt") -> Path:
        """Constructs the path for a model checkpoint."""
        return self.checkpoints / filename
