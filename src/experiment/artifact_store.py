"""
src/experiment/artifact_store.py

ArtifactStore — lightweight artifact manager.

Replaces ExperimentTracker.  Key design goals:
  1. No "RunMode" complexity — just read/write files.
  2. Structured JSON schemas so visualization scripts can load
     training, validation, and test outputs without guessing paths.
  3. Single source of truth for where every file lives.

Directory layout for a run:
  results/
    <run_id>/
      meta.json                   # run metadata
      artifacts/
        model_lstm_<tag>.pth      # model weights
        model_xgb_<tag>.ubj       # xgboost booster
        params_<tag>.json         # hyperparameters
      train/
        history_<tag>.json        # epoch losses
        metrics_<tag>.json        # val metrics after training
        predictions_<tag>.npy
        truth_<tag>.npy
      val/
        fold_metrics_<tag>.json   # per-fold metrics
        aggregate_<tag>.json      # mean ± std across folds
        threshold_<tag>.json      # optimal signal threshold + sweep
      test/
        stat_metrics_<tag>.json   # RMSE, DA, F1, R2
        trading_metrics_<tag>.json # Sharpe, Sortino, MDD, CAGR …
        trade_log_<tag>.csv       # individual trades
        equity_curve_<tag>.npy    # portfolio value over time
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# JSON serialiser that handles numpy types
# ======================================================================


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (datetime,)):
        return o.isoformat()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


# ======================================================================
# ArtifactStore
# ======================================================================


class ArtifactStore:
    """Manages all file I/O for a single model training run.

    Args:
        model:       Model type string ('lstm' or 'xgboost').
        ticker:      Ticker symbol.
        seed:        Random seed used in this run.
        run_id:      Existing run ID to attach to.  If None, a new UUID is created.
        base_dir:    Root directory for all runs (default 'results').
        features_dir: Root directory for feature arrays (default 'data/features').
    """

    def __init__(
        self,
        model: str,
        ticker: str,
        seed: int,
        run_id: Optional[str] = None,
        base_dir: str = "results",
        features_dir: str = "data/features",
    ) -> None:
        self.model = model
        self.ticker = ticker
        self.seed = seed
        self.run_id = run_id or str(uuid.uuid4())
        self.base_dir = Path(base_dir)
        self.features_dir = Path(features_dir) / ticker

        # File-system tag used in all artifact names
        self.tag = f"{ticker}_{model}_seed{seed}"

        # Directory handles
        self.run_dir = self.base_dir / self.run_id
        self._dirs = {
            "artifacts": self.run_dir / "artifacts",
            "train": self.run_dir / "train",
            "val": self.run_dir / "val",
            "test": self.run_dir / "test",
        }
        self._init_dirs()
        self._ensure_meta()

        logger.info(
            "[ArtifactStore] run_id=%s  tag=%s  dir=%s",
            self.run_id,
            self.tag,
            self.run_dir,
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_dirs(self) -> None:
        for d in self._dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def _ensure_meta(self) -> None:
        meta_path = self.run_dir / "meta.json"
        if not meta_path.exists():
            _write_json(
                meta_path,
                {
                    "run_id": self.run_id,
                    "model": self.model,
                    "ticker": self.ticker,
                    "seed": self.seed,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def artifact_path(self, filename: str) -> Path:
        return self._dirs["artifacts"] / filename

    def phase_path(self, phase: str, filename: str) -> Path:
        assert phase in ("train", "val", "test"), f"Unknown phase: {phase}"
        return self._dirs[phase] / filename

    # ------------------------------------------------------------------
    # Data loading (features)
    # ------------------------------------------------------------------

    def load_split(self, phase: str):
        """Load X_{phase}.npy and y_{phase}.npy from the features directory."""
        X = np.load(self.features_dir / f"X_{phase}.npy")
        y = np.load(self.features_dir / f"y_{phase}.npy")
        logger.info(
            "[ArtifactStore] Loaded split='%s'  X=%s  y=%s", phase, X.shape, y.shape
        )
        return X, y

    # ------------------------------------------------------------------
    # Model weights
    # ------------------------------------------------------------------

    def save_model(self, model, ext: str) -> Path:
        """Save model weights.  Returns the saved path.

        Args:
            model:  Object with .save(path) method.
            ext:    File extension ('pth' for LSTM, 'ubj' for XGBoost).
        """
        path = self.artifact_path(f"model_{self.tag}.{ext}")
        model.save(str(path))
        return path

    def model_path(self, ext: str) -> Path:
        return self.artifact_path(f"model_{self.tag}.{ext}")

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------

    def save_params(self, params: Dict[str, Any], extra: Optional[Dict] = None) -> None:
        """Save hyperparameters + optional metadata (runtime, val metrics)."""
        obj = {
            "run_id": self.run_id,
            "model": self.model,
            "ticker": self.ticker,
            "seed": self.seed,
            "hyperparameters": params,
        }
        if extra:
            obj.update(extra)
        _write_json(self.artifact_path(f"params_{self.tag}.json"), obj)

    def load_params(self) -> Dict[str, Any]:
        return _read_json(self.artifact_path(f"params_{self.tag}.json"))

    # ------------------------------------------------------------------
    # Training outputs
    # ------------------------------------------------------------------

    def save_train_history(self, history: Dict[str, list]) -> None:
        """Save epoch-level training history for loss curves."""
        _write_json(
            self.phase_path("train", f"history_{self.tag}.json"),
            {"tag": self.tag, "history": history},
        )

    def save_train_predictions(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        np.save(self.phase_path("train", f"predictions_{self.tag}.npy"), y_pred)
        np.save(self.phase_path("train", f"truth_{self.tag}.npy"), y_true)

    # ------------------------------------------------------------------
    # Validation outputs
    # ------------------------------------------------------------------

    def save_val_fold_metrics(self, fold_metrics: list) -> None:
        """Save per-fold validation metrics (list of dicts)."""
        _write_json(
            self.phase_path("val", f"fold_metrics_{self.tag}.json"),
            {"tag": self.tag, "folds": fold_metrics},
        )

    def save_val_aggregate(self, aggregate: Dict[str, Any]) -> None:
        """Save mean ± std metrics aggregated across all folds."""
        _write_json(
            self.phase_path("val", f"aggregate_{self.tag}.json"),
            {"tag": self.tag, **aggregate},
        )

    def save_val_threshold(self, threshold_result: Dict[str, Any]) -> None:
        """Save signal threshold sweep results + optimal threshold."""
        _write_json(
            self.phase_path("val", f"threshold_{self.tag}.json"),
            {"tag": self.tag, **threshold_result},
        )

    def load_val_threshold(self) -> float:
        """Load the optimal signal threshold, falling back to 1e-4."""
        path = self.phase_path("val", f"threshold_{self.tag}.json")
        if path.exists():
            data = _read_json(path)
            theta = float(data.get("optimal_threshold", 1e-4))
            logger.info("[ArtifactStore] Loaded threshold=%.5f", theta)
            return theta
        logger.warning("[ArtifactStore] Threshold file not found, using θ=1e-4")
        return 1e-4

    # ------------------------------------------------------------------
    # Test outputs
    # ------------------------------------------------------------------

    def save_test_stat_metrics(self, metrics: Dict[str, float]) -> None:
        """Save RMSE, DA, F1, R² from statistical evaluation."""
        _write_json(
            self.phase_path("test", f"stat_metrics_{self.tag}.json"),
            {"tag": self.tag, **metrics},
        )

    def save_test_trading_metrics(
        self,
        metrics: Dict[str, Any],
        theta: float,
    ) -> None:
        """Save Sharpe, Sortino, MDD, CAGR, etc. from backtesting."""
        _write_json(
            self.phase_path("test", f"trading_metrics_{self.tag}.json"),
            {"tag": self.tag, "signal_threshold": theta, **metrics},
        )

    def save_test_equity_curve(self, equity_curve: np.ndarray) -> None:
        """Save portfolio equity curve for visualization."""
        np.save(
            self.phase_path("test", f"equity_curve_{self.tag}.npy"),
            equity_curve,
        )

    def save_test_trade_log(self, trade_log: pd.DataFrame) -> None:
        """Save trade-level log as CSV."""
        if not trade_log.empty:
            trade_log.to_csv(
                self.phase_path("test", f"trade_log_{self.tag}.csv"),
                index=False,
            )

    def save_test_predictions(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        np.save(self.phase_path("test", f"predictions_{self.tag}.npy"), y_pred)
        np.save(self.phase_path("test", f"truth_{self.tag}.npy"), y_true)

    # ------------------------------------------------------------------
    # Convenience: save all backtest outputs in one call
    # ------------------------------------------------------------------

    def save_backtest_result(self, result, stat_metrics: Dict, theta: float) -> None:
        """Save all outputs produced by a Backtester.run() call."""
        self.save_test_trading_metrics(
            {
                "sharpe": result.sharpe,
                "sortino": result.sortino,
                "max_drawdown": result.mdd,
                "cagr": result.cagr_,
                "calmar": result.calmar,
                "profit_factor": result.profit_factor_,
                "win_rate": result.win_rate_,
                "n_trades": result.n_trades,
                "turnover": result.turnover,
                "final_equity": float(result.equity_curve[-1]),
            },
            theta=theta,
        )
        self.save_test_stat_metrics(stat_metrics)
        self.save_test_equity_curve(result.equity_curve)
        self.save_test_trade_log(result.trade_log)

    # ------------------------------------------------------------------
    # Summary index (for dashboards / visualization tooling)
    # ------------------------------------------------------------------

    def write_index(self) -> None:
        """Write/update a top-level index.json describing all artifacts."""
        index = {
            "run_id": self.run_id,
            "model": self.model,
            "ticker": self.ticker,
            "seed": self.seed,
            "tag": self.tag,
            "paths": {
                "model_lstm": str(self.model_path("pth")),
                "model_xgb": str(self.model_path("ubj")),
                "params": str(self.artifact_path(f"params_{self.tag}.json")),
                "train_history": str(
                    self.phase_path("train", f"history_{self.tag}.json")
                ),
                "val_fold_metrics": str(
                    self.phase_path("val", f"fold_metrics_{self.tag}.json")
                ),
                "val_aggregate": str(
                    self.phase_path("val", f"aggregate_{self.tag}.json")
                ),
                "val_threshold": str(
                    self.phase_path("val", f"threshold_{self.tag}.json")
                ),
                "test_stat_metrics": str(
                    self.phase_path("test", f"stat_metrics_{self.tag}.json")
                ),
                "test_trading_metrics": str(
                    self.phase_path("test", f"trading_metrics_{self.tag}.json")
                ),
                "test_equity_curve": str(
                    self.phase_path("test", f"equity_curve_{self.tag}.npy")
                ),
                "test_trade_log": str(
                    self.phase_path("test", f"trade_log_{self.tag}.csv")
                ),
            },
        }
        _write_json(self.run_dir / "index.json", index)
