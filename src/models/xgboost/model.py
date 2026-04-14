"""
src/models/xgboost/model.py

XGBoostWrapper — implements BaseModel.
Accepts [N, T, F] tensors (same contract as LSTMWrapper),
flattens them to [N, T×F] internally.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import xgboost as xgb

from ..base import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.01,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "colsample_bylevel": 1.0,
    "min_child_weight": 5,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.1,
    "tree_method": "hist",  # gpu_hist if GPU is available
    "max_bin": 128,
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
}

_EARLY_STOPPING_ROUNDS = 50
_DEFAULT_LOOKBACK = 30


class XGBoostWrapper(BaseModel):
    """XGBoost regressor wrapping the BaseModel interface.

    Args:
        lookback:               Number of timesteps to keep (last T steps of window).
        early_stopping_rounds:  Stop if val RMSE does not improve for N rounds.
        importance_type:        Feature importance metric ('gain', 'weight', 'cover').
        **xgb_params:           XGBoost booster parameters (merged with defaults).
    """

    def __init__(
        self,
        lookback: int = _DEFAULT_LOOKBACK,
        early_stopping_rounds: int = _EARLY_STOPPING_ROUNDS,
        importance_type: str = "gain",
        **xgb_params,
    ) -> None:
        self.lookback = lookback
        self.early_stopping_rounds = early_stopping_rounds
        self.importance_type = importance_type
        self._xgb_params: Dict[str, Any] = {**DEFAULT_PARAMS, **xgb_params}
        self._model: Optional[xgb.XGBRegressor] = None
        self._best_iteration: int = 0
        self.history: Dict[str, List[float]] = {"train_rmse": [], "val_rmse": []}
        self._setup_device()

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Train with early stopping; return history."""
        if X_train.ndim != 3:
            raise ValueError(f"X_train must be 3D [N,T,F], got {X_train.shape}")

        X_tr = self._flatten(X_train)
        X_vl = self._flatten(X_val)

        self._model = xgb.XGBRegressor(
            **self._xgb_params,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric="rmse",
        )
        self._model.fit(
            X_tr,
            y_train,
            eval_set=[(X_tr, y_train), (X_vl, y_val)],
            verbose=False,
        )

        self._best_iteration = int(getattr(self._model, "best_iteration", 0))
        evals = self._model.evals_result()
        self.history["train_rmse"] = evals.get("validation_0", {}).get("rmse", [])
        self.history["val_rmse"] = evals.get("validation_1", {}).get("rmse", [])

        best_val = (
            min(self.history["val_rmse"]) if self.history["val_rmse"] else float("nan")
        )
        logger.info(
            "[XGBoost] Training complete — best_iter=%d  best_val_rmse=%.6f",
            self._best_iteration,
            best_val,
        )
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(self._flatten(X)).astype(np.float32)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Nothing to save — model not fitted.")
        self._model.save_model(path)
        logger.info("[XGBoost] Model saved → %s", path)

    @classmethod
    def load(
        cls, path: str, lookback: int = _DEFAULT_LOOKBACK, **params
    ) -> "XGBoostWrapper":
        wrapper = cls(lookback=lookback, **params)
        wrapper._model = xgb.XGBRegressor(**wrapper._xgb_params)
        wrapper._model.load_model(path)
        logger.info("[XGBoost] Model loaded ← %s", path)
        return wrapper

    def get_params(self) -> Dict[str, Any]:
        return {
            "lookback": self.lookback,
            "early_stopping_rounds": self.early_stopping_rounds,
            "importance_type": self.importance_type,
            **self._xgb_params,
        }

    # ------------------------------------------------------------------
    # Feature importances
    # ------------------------------------------------------------------

    def feature_importances(self) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        return self._model.feature_importances_

    @property
    def best_iteration(self) -> int:
        return self._best_iteration

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        N, T_full, F = X.shape
        if T_full < self.lookback:
            raise ValueError(f"Window T={T_full} < lookback={self.lookback}")
        return X[:, -self.lookback :, :].reshape(N, -1).astype(np.float32)

    def _setup_device(self) -> None:
        if self._xgb_params.get("tree_method") == "gpu_hist":
            try:
                import cupy  # noqa: F401

                logger.info("[XGBoost] GPU mode enabled (gpu_hist)")
            except ImportError:
                logger.info("[XGBoost] CuPy unavailable, falling back to hist")
                self._xgb_params["tree_method"] = "hist"
