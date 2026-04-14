"""
src/models/xgboost/trainer.py

XGBoostTuner — random hyperparameter search on the validation set.
Decoupled from XGBoostWrapper so the tuner can be used independently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import XGBoostWrapper, _DEFAULT_LOOKBACK

logger = logging.getLogger(__name__)


class XGBoostTuner:
    """Random search over XGBoost hyperparameters.

    Args:
        n_trials:   Number of random combinations to try.
        param_grid: Dict of param name → list of candidates.
        seed:       RNG seed.
    """

    DEFAULT_GRID: Dict[str, List[Any]] = {
        "n_estimators": [200, 300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.10],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [0.5, 1.0, 2.0],
        "min_child_weight": [1, 3, 5],
    }

    def __init__(
        self,
        n_trials: int = 20,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        seed: int = 42,
    ) -> None:
        self.n_trials = n_trials
        self.param_grid = param_grid or self.DEFAULT_GRID
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.results_: List[Dict[str, Any]] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        lookback: int = _DEFAULT_LOOKBACK,
    ) -> Tuple[XGBoostWrapper, Dict[str, Any]]:
        """Run random search.  Returns (best_model, best_params)."""
        best_rmse = float("inf")
        best_model: Optional[XGBoostWrapper] = None
        best_params: Dict[str, Any] = {}

        for trial in range(self.n_trials):
            sampled = {k: self._rng.choice(v) for k, v in self.param_grid.items()}
            sampled["random_state"] = self.seed + trial
            logger.info("Trial %d/%d: %s", trial + 1, self.n_trials, sampled)

            model = XGBoostWrapper(lookback=lookback, **sampled)
            model.fit(X_train, y_train, X_val, y_val)

            val_rmse = (
                min(model.history["val_rmse"])
                if model.history["val_rmse"]
                else float("inf")
            )
            self.results_.append({"trial": trial + 1, "val_rmse": val_rmse, **sampled})
            logger.info("  val_rmse=%.6f", val_rmse)

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_model = model
                best_params = {**sampled, "lookback": lookback}

        if best_model is None:
            raise RuntimeError("All XGBoost tuning trials failed.")

        logger.info("Best val_rmse=%.6f  params=%s", best_rmse, best_params)
        return best_model, best_params
