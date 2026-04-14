from __future__ import annotations

import json
import uuid
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from .usage_enums import ModelType, Phase, RunMode, ArtifactType


# =========================================================
# EXPERIMENT TRACKER
# =========================================================


class ExperimentTracker:
    def __init__(
        self,
        model_type: ModelType,
        mode: RunMode,
        phase: Phase,
        ticker: str,
        seed: int,
        features_path: Path,
        run_id: str | None = None,
        base_dir: str = "results",
    ):
        self.model_type = model_type
        self.mode = mode
        self.phase = phase
        self.ticker = ticker
        self.seed = seed
        self.features_path = Path(features_path)

        self.base_dir = Path(base_dir)

        # -----------------------------------------------------
        # RUN ID MANAGEMENT
        # -----------------------------------------------------
        if self.mode == RunMode.CREATE:
            self.run_id = run_id or str(uuid.uuid4())
        elif self.mode == RunMode.ATTACH:
            if run_id is None:
                raise ValueError("RunMode.ATTACH requires run_id")
            self.run_id = run_id
            self._verify_run_exists()
        else:
            raise ValueError(f"Invalid RunMode: {self.mode}")

        self.run_dir = self.base_dir / self.run_id

        # -----------------------------------------------------
        # DIRECTORY STRUCTURE
        # -----------------------------------------------------
        self.paths = {
            "train": self.run_dir / "train",
            "val": self.run_dir / "val",
            "test": self.run_dir / "test",
            "artifacts": self.run_dir / "artifacts",
            "logs": self.run_dir / "logs",
            "metadata": self.run_dir / "metadata",
        }

        self._init_dirs()

        if self.mode == RunMode.CREATE:
            self._init_metadata()

    # =========================================================
    # INIT
    # =========================================================

    def _init_dirs(self):
        for p in self.paths.values():
            p.mkdir(parents=True, exist_ok=True)

    def _init_metadata(self):
        self.save_json(
            self.run_dir / "metadata" / "run.json",
            {
                "run_id": self.run_id,
                "model_type": self.model_type.value,
                "ticker": self.ticker,
                "seed": self.seed,
                "start_time": datetime.utcnow().isoformat(),
            },
        )

    def _verify_run_exists(self):
        required = ["train", "val", "test", "artifacts", "metadata"]

        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run not found: {self.run_dir}")

        missing = [d for d in required if not (self.run_dir / d).exists()]

        if missing:
            raise ValueError(f"Corrupted run directory. Missing: {missing}")

    # =========================================================
    # PATH RESOLUTION
    # =========================================================

    def _resolve(self, phase: Phase, filename: str) -> Path:
        return self.paths[phase.value] / filename

    def artifact_path(self, filename: str) -> Path:
        return self.paths["artifacts"] / filename

    def log_path(self, filename: str) -> Path:
        return self.paths["logs"] / filename

    # =========================================================
    # FILE NAMING ENGINE (CRITICAL)
    # =========================================================

    def make_filename(
        self,
        phase: Phase,
        artifact: ArtifactType,
        ext: str = "json",
    ) -> str:
        tag = f"{self.ticker}_{self.model_type.value}_seed{self.seed}"
        return f"{artifact.value}_{phase.value}_{tag}.{ext}"

    # =========================================================
    # SAVE API
    # =========================================================

    def save_json(self, path: Path, obj: dict):
        path.parent.mkdir(parents=True, exist_ok=True)

        def _default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        path.write_text(json.dumps(obj, indent=2, default=_default))

    def save_npy(self, phase: Phase, arr: np.ndarray, artifact: ArtifactType):
        filename = self.make_filename(phase, artifact, ext="npy")
        path = self._resolve(phase, filename)
        np.save(path, arr)

    def save_torch(
        self, phase: Phase, obj, artifact: ArtifactType = ArtifactType.MODEL
    ):
        filename = self.make_filename(phase, artifact, ext="pth")
        path = self.artifact_path(filename)
        torch.save(obj, path)

    def save_plot(self, phase: Phase, fig, artifact: ArtifactType = ArtifactType.PLOT):
        filename = self.make_filename(phase, artifact, ext="png")
        path = self._resolve(phase, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def save_metrics(self, phase: Phase, metrics: dict):
        filename = self.make_filename(phase, ArtifactType.METRICS)
        path = self._resolve(phase, filename)
        self.save_json(path, metrics)

    def save_history(self, phase: Phase, history: dict):
        filename = self.make_filename(phase, ArtifactType.HISTORY)
        path = self._resolve(phase, filename)
        self.save_json(path, history)

    def save_predictions(self, phase: Phase, arr: np.ndarray):
        filename = self.make_filename(phase, ArtifactType.PREDICTIONS, ext="npy")
        path = self._resolve(phase, filename)
        np.save(path, arr)

    def save_truth(self, phase: Phase, arr: np.ndarray):
        filename = self.make_filename(phase, ArtifactType.TRUTH, ext="npy")
        path = self._resolve(phase, filename)
        np.save(path, arr)

    # =========================================================
    # LOAD API
    # =========================================================

    def load_npy(self, phase: Phase, artifact: ArtifactType):
        filename = self.make_filename(phase, artifact, ext="npy")
        path = self._resolve(phase, filename)
        return np.load(path)

    def load_json(self, path: Path | str):
        path = self.run_dir / path if isinstance(path, str) else path
        return json.loads(path.read_text())

    def load_torch(self, filename: str):
        return torch.load(self.artifact_path(filename), map_location="cpu")

    def load_split(self, phase: Phase):
        X_path = self.features_path / f"X_{phase}.npy"
        y_path = self.features_path / f"y_{phase}.npy"

        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Missing split files for '{phase}' in {self.features_path}"
            )

        X = np.load(X_path)
        y = np.load(y_path)

        logger.info(
            "Loaded split='%s' | X=%s | y=%s",
            phase,
            X.shape,
            y.shape,
        )

        return X, y

    # =========================================================
    # HELPERS
    # =========================================================

    def phase_dir(self, phase: Phase) -> Path:
        return self.paths[phase.value]
