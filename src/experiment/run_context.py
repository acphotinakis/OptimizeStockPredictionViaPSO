from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import logging

from src.utils.config_loader import Config, load_config
from src.utils.seed import set_all_seeds
from .usage_enums import ModelType, RunMode, Phase
from .experiment_tracker import ExperimentTracker


# =========================================================
# RUNTIME CONTEXT
# =========================================================


@dataclass(frozen=True)
class RuntimeContext:
    args: argparse.Namespace

    model: ModelType
    ticker: str
    seed: int
    config_path: str

    features_dir: str = "data/features"
    results_dir: str = "results"
    log_file: str | None = None

    run_mode: RunMode = RunMode.CREATE
    run_id: str | None = None

    # populated in __post_init__
    cfg: Config = None
    features_path: Path = None
    results_path: Path = None
    tracker: ExperimentTracker = None

    # =====================================================
    # ENTRY POINT
    # =====================================================
    def __post_init__(self):
        self._bootstrap()

    # =====================================================
    # BOOTSTRAP PIPELINE (single source of truth)
    # =====================================================
    def _bootstrap(self) -> None:
        self._setup_seed()
        self._setup_config()
        self._setup_paths()
        self._setup_tracker()
        self._setup_logging()

    # =====================================================
    # CONFIG
    # =====================================================
    def _setup_config(self) -> None:
        object.__setattr__(self, "cfg", load_config(self.config_path))

    # =====================================================
    # SEEDING
    # =====================================================
    def _setup_seed(self) -> None:
        set_all_seeds(self.seed)

    # =====================================================
    # PATHS
    # =====================================================
    def _setup_paths(self) -> None:
        features_path = Path(self.features_dir) / self.ticker
        results_path = Path(self.results_dir)

        results_path.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)

        if not features_path.exists():
            raise FileNotFoundError(f"Missing features: {features_path}")

        object.__setattr__(self, "features_path", features_path)
        object.__setattr__(self, "results_path", results_path)

    # =====================================================
    # TRACKER
    # =====================================================
    def _setup_tracker(self) -> None:
        tracker = ExperimentTracker(
            model_type=self.model,
            mode=self.run_mode,
            phase=Phase.TRAIN,
            ticker=self.ticker,
            seed=self.seed,
            features_path=self.features_path,
            run_id=self.run_id,
            base_dir=self.results_dir,
        )

        object.__setattr__(self, "tracker", tracker)

    # =====================================================
    # LOGGING
    # =====================================================
    def _setup_logging(self) -> None:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

        # ensure tracker exists first
        log_path = self.tracker.log_path("train.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(),
            ],
            force=True,  # important: overrides previous configs
        )
