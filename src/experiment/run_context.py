"""
src/experiment/run_context.py

RuntimeContext — single object passed through train / val / test pipelines.

Responsibilities:
  1. Load config.
  2. Set random seeds.
  3. Instantiate ArtifactStore.
  4. Set up logging.

No model creation, no data loading — those belong in the pipelines.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..utils.config_loader import Config, load_config
from ..utils.seed import set_all_seeds
from .artifact_store import ArtifactStore

logger = logging.getLogger(__name__)


@dataclass
class RuntimeContext:
    """Immutable context object shared across pipeline stages.

    Args:
        model:        'lstm' or 'xgboost'.
        ticker:       Equity ticker symbol.
        seed:         Global random seed.
        config_path:  Path to YAML config file.
        run_id:       Existing run ID (None = create new).
        features_dir: Root of feature arrays directory.
        results_dir:  Root of results directory.
        log_file:     Optional log file path.
        args:         Raw argparse namespace (forwarded to pipelines for CLI overrides).
    """

    model: str
    ticker: str
    seed: int
    config_path: str = "config/default_config.yaml"
    run_id: Optional[str] = None
    features_dir: str = "data/features"
    results_dir: str = "results"
    log_file: Optional[str] = None
    args: Optional[argparse.Namespace] = None

    # populated in __post_init__
    cfg: Optional[Config] = field(default=None, init=False, repr=False)
    store: Optional[ArtifactStore] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        set_all_seeds(self.seed)
        self._load_config()
        self._setup_logging()
        self._init_store()

    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        object.__setattr__(self, "cfg", load_config(self.config_path))

    def _setup_logging(self) -> None:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"
        handlers = [logging.StreamHandler()]
        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(self.log_file))
        logging.basicConfig(
            level=logging.INFO, format=fmt, handlers=handlers, force=True
        )

    def _init_store(self) -> None:
        store = ArtifactStore(
            model=self.model,
            ticker=self.ticker,
            seed=self.seed,
            run_id=self.run_id,
            base_dir=self.results_dir,
            features_dir=self.features_dir,
        )
        object.__setattr__(self, "store", store)
        # Propagate the (possibly new) run_id back so callers can chain stages
        object.__setattr__(self, "run_id", store.run_id)
