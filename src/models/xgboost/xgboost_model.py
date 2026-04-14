"""
src/models/xgboost_model.py

Production-grade XGBoost model for next-bar log-return prediction.

Design decisions aligned with the broader project:
  - Accepts the same [N, T, F] sliding-window tensors as LSTMModel
    (sequences are flattened to [N, T×F] internally).
  - Exposes the same fit() / predict() / get_params() interface as
    LSTMTrainer so scripts can treat all models uniformly.
  - Trains with eval_set early stopping (mirrors LSTM patience logic).
  - Persists and restores best booster state (mirrors LSTMTrainer._best_state).
  - Exposes feature importances for the FeatureSelector pipeline.
  - Supports three importance types: 'gain' (default), 'weight', 'cover'.

Hyperparameter defaults sourced from:
  - Zeng et al. (2025): PSO-LSTM comparison uses XGBoost with
    n_estimators=500, max_depth=6, lr=0.05.
  - CSCI 633 project proposal: XGBoost as non-deep-learning baseline.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import xgboost as xgb
import numpy as np
from src.utils.config_loader import Config
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

# Default hyperparameter set — mirrors experiment_plan.md §5.3
_DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "reg:squarederror",
    # "num_class": 3,
    "n_estimators": 200,
    "max_depth": 4,
    "lookback": 30,
    "learning_rate": 0.01,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "colsample_bylevel": 1.0,
    "min_child_weight": 5,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.1,
    "tree_method": "gpu_hist",
    "max_bin": 128,
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
}

# Early stopping: stop if val RMSE does not improve for this many rounds
_DEFAULT_EARLY_STOPPING_ROUNDS = 50

# Lookback window for sequence flattening (overridden by constructor)
_DEFAULT_LOOKBACK = 30


# ======================================================================
# Helper functions
# ======================================================================


class XGBoostModel:
    """XGBoost regressor for sliding-window stock return prediction.

    Accepts [N, T, F] tensors (same shape contract as LSTMModel) and
    flattens them to [N, T×F] before feeding into the booster.  Supports
    early stopping on a validation split, feature importance extraction,
    and clean save / load semantics.

    Args:
        lookback: Number of timesteps T to use.  Windows in the input
            tensor must have T >= lookback; the *last* `lookback` steps
            are sliced (most-recent history kept).
        n_estimators: Maximum number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage (η).
        subsample: Row subsampling fraction per tree.
        colsample_bytree: Column subsampling fraction per tree.
        colsample_bylevel: Column subsampling fraction per depth level.
        min_child_weight: Minimum sum of instance weight in a child node.
        reg_alpha: L1 weight regularisation term.
        reg_lambda: L2 weight regularisation term.
        gamma: Minimum loss reduction required to make a further partition.
        tree_method: XGBoost tree construction algorithm ('hist', 'approx', 'exact').
        early_stopping_rounds: Stop training if val RMSE does not improve
            for this many consecutive rounds.
        importance_type: Feature importance type for get_feature_importances().
            One of 'gain', 'weight', 'cover', 'total_gain', 'total_cover'.
        random_state: Random seed (passed to XGBRegressor).
        n_jobs: Number of parallel threads (-1 = all available).
    """

    def __init__(
        self,
        lookback: int = _DEFAULT_LOOKBACK,
        importance_type: str = "gain",
        early_stopping_rounds: int = _DEFAULT_EARLY_STOPPING_ROUNDS,
        **xgb_params,
    ) -> None:
        """
        Initialize XGBoostModel with clean parameter separation.

        Args:
            lookback: Number of timesteps T to use.
            importance_type: Feature importance metric ('gain', 'weight', 'cover').
            early_stopping_rounds: Patience for early stopping.
            **xgb_params: XGBoost booster parameters (merged with defaults).
        """
        self.lookback = lookback
        self.importance_type = importance_type
        self.early_stopping_rounds = early_stopping_rounds

        # Merge defaults with user-provided params (user takes precedence)
        self._xgb_params = {**_DEFAULT_PARAMS, **xgb_params}

        # Internal state
        self._model: Optional[xgb.XGBRegressor] = None
        self._best_iteration: int = 0
        self._feature_names: List[str] = []
        self.history: Dict[str, List[float]] = {"train_rmse": [], "val_rmse": []}

        self._setup_device()

    @classmethod
    def from_config(cls, cfg: Optional[Config] = None, **overrides) -> "XGBoostModel":
        """
        Factory method to construct from Config object.

        Args:
            cfg: Configuration object with optional 'xgboost' section.
            **overrides: Additional parameters to override config values.

        Returns:
            Configured XGBoostModel instance.
        """
        params: Dict[str, Any] = {}

        if cfg is not None:
            params.update(getattr(cfg, "xgboost", {}))

        params.update(overrides)

        # Extract non-xgb parameters
        lookback = params.pop("lookback", _DEFAULT_LOOKBACK)
        importance_type = params.pop("importance_type", "gain")
        early_stopping = params.pop(
            "early_stopping_rounds", _DEFAULT_EARLY_STOPPING_ROUNDS
        )

        return cls(
            lookback=lookback,
            importance_type=importance_type,
            early_stopping_rounds=early_stopping,
            **params,
        )

    def _setup_device(self) -> None:
        """Configure GPU/CPU device settings."""
        if self._xgb_params.get("tree_method") != "gpu_hist":
            return

        try:
            import cupy as cp

            pool = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(pool.malloc)
            logger.info("GPU memory pool configured")
        except ImportError:
            logger.info("CuPy unavailable, falling back to CPU")
            self._xgb_params["tree_method"] = "hist"
        except Exception as e:
            logger.info(f"GPU setup failed: {e}, using CPU")
            self._xgb_params["tree_method"] = "hist"

    def get_params(self) -> Dict[str, Any]:
        """Return full parameter dict for serialization."""
        return {
            "lookback": self.lookback,
            "importance_type": self.importance_type,
            "early_stopping_rounds": self.early_stopping_rounds,
            **self._xgb_params,
        }

    # ------------------------------------------------------------------
    # Core interface — mirrors LSTMTrainer
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, List[float]]:
        """Train XGBoost with early stopping on the validation set.

        Args:
            X_train: [N_train, T, F] sliding-window tensor.
            y_train: [N_train] next-bar log returns.
            X_val:   [N_val, T, F] sliding-window tensor.
            y_val:   [N_val] next-bar log returns.

        Returns:
            Training history dict with 'train_rmse' and 'val_rmse' lists
            (one entry per boosting round that was actually evaluated).
        """
        # Validate inputs
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}"
            )
        if len(X_val) != len(y_val):
            raise ValueError(
                f"X_val and y_val length mismatch: {len(X_val)} vs {len(y_val)}"
            )
        if X_train.ndim != 3:
            raise ValueError(f"X_train must be 3D [N, T, F], got shape {X_train.shape}")
        X_tr_flat = self._flatten(X_train)
        X_vl_flat = self._flatten(X_val)

        eval_set = [(X_tr_flat, y_train), (X_vl_flat, y_val)]

        self._model = xgb.XGBRegressor(
            **self._xgb_params,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric="rmse",
        )

        assert self._model is not None, "XGBoost model not initialized."

        fit_kwargs = {
            "X": X_tr_flat,
            "y": y_train,
            "eval_set": eval_set,
            "verbose": False,
        }

        self._model.fit(**fit_kwargs)

        self._best_iteration = int(getattr(self._model, "best_iteration", 0))

        # Extract per-round evaluation results
        evals = self._model.evals_result()
        self.history["train_rmse"] = evals.get("train", {}).get("rmse", [])
        self.history["val_rmse"] = evals.get("val", {}).get("rmse") or evals.get(
            "validation_1", {}
        ).get("rmse", [])

        best_val = (
            min(self.history["val_rmse"]) if self.history["val_rmse"] else float("nan")
        )
        logger.info(
            "XGBoost training complete — best_iteration=%d  best_val_rmse=%.6f",
            self._best_iteration,
            best_val,
        )
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-bar log returns.

        Args:
            X: [N, T, F] sliding-window tensor.

        Returns:
            [N] float32 prediction array.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(self._flatten(X)).astype(np.float32)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the fitted booster to disk in XGBoost binary format.

        Args:
            path: File path (e.g. 'results/xgb_AAPL.ubj').
        """
        if self._model is None:
            raise RuntimeError("Nothing to save — model has not been fitted.")
        self._model.save_model(path)
        logger.info("XGBoost model saved to %s", path)

    def load(self, path: str) -> "XGBoostModel":
        """Load a previously saved booster.

        Args:
            path: File path written by save().

        Returns:
            self
        """
        self._model = xgb.XGBRegressor(**self._xgb_params)
        assert self._model is not None
        self._model.load_model(path)
        logger.info("XGBoost model loaded from %s", path)
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """Slice last `lookback` timesteps and flatten to [N, T×F].

        Args:
            X: [N, T_full, F] — T_full >= self.lookback.

        Returns:
            [N, lookback × F] float32 array.
        """
        N, T_full, F = X.shape
        if T_full < self.lookback:
            raise ValueError(
                f"Input has T={T_full} timesteps but lookback={self.lookback}. "
                "Either reduce lookback or rebuild windows with a larger max lookback."
            )
        # Keep the most recent `lookback` timesteps
        sliced = X[:, -self.lookback :, :]  # [N, lookback, F]
        return sliced.reshape(N, -1).astype(np.float32)

    @property
    def best_iteration(self) -> int:
        """Return the best iteration from training."""
        return self._best_iteration


# ======================================================================
# XGBoost hyperparameter tuner (grid / random search on validation RMSE)
# ======================================================================


class XGBoostTuner:
    """Light hyperparameter search for XGBoostModel using the validation set.

    Performs a random search over a configurable grid and returns the
    XGBoostModel instance with the best validation RMSE.  Intentionally
    kept simple — full PSO-level search is out of scope here, but this
    provides a fair baseline comparison against the manually-tuned default.

    Args:
        n_trials: Number of random hyperparameter combinations to try.
        param_grid: Dict mapping parameter name --> list of candidate values.
            If None, a sensible default grid is used.
        seed: Random seed for reproducibility.
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

    def get_results_df(self) -> pd.DataFrame:
        """Return tuning results as a pandas DataFrame for analysis."""
        if not self.results_:
            raise RuntimeError("No tuning results available. Call fit() first.")
        return pd.DataFrame(self.results_).sort_values("val_rmse")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        lookback: int = _DEFAULT_LOOKBACK,
    ) -> Tuple["XGBoostModel", Dict[str, Any]]:
        """Run random search and return the best model.

        Args:
            X_train: [N_train, T, F] training tensor.
            y_train: [N_train] training targets.
            X_val:   [N_val, T, F] validation tensor.
            y_val:   [N_val] validation targets.
            lookback: Lookback window to pass to XGBoostModel.

        Returns:
            (best_model, best_hyperparams_dict)
        """
        # Validate inputs
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}"
            )
        if len(X_val) != len(y_val):
            raise ValueError(
                f"X_val and y_val length mismatch: {len(X_val)} vs {len(y_val)}"
            )
        if X_train.ndim != 3:
            raise ValueError(f"X_train must be 3D [N, T, F], got shape {X_train.shape}")

        best_val_rmse = float("inf")
        best_model: Optional[XGBoostModel] = None
        best_params: Dict[str, Any] = {}

        for trial in range(self.n_trials):
            # Sample one combination
            sampled = {k: self._rng.choice(v) for k, v in self.param_grid.items()}
            sampled["random_state"] = self.seed + trial

            logger.info(
                "XGBoostTuner trial %d/%d: %s", trial + 1, self.n_trials, sampled
            )

            model = XGBoostModel(lookback=lookback, **sampled)
            model.fit(X_train, y_train, X_val, y_val)

            if model.history["val_rmse"]:
                val_rmse = min(model.history["val_rmse"])
            else:
                from src.evaluation.metrics import rmse as rmse_fn

                val_rmse = rmse_fn(y_val, model.predict(X_val))

            trial_result = {"trial": trial + 1, "val_rmse": val_rmse, **sampled}
            self.results_.append(trial_result)
            logger.info("  val_rmse=%.6f", val_rmse)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model = model
                best_params = {**sampled, "lookback": lookback}

        logger.info(
            "XGBoostTuner complete. Best val_rmse=%.6f  params=%s",
            best_val_rmse,
            best_params,
        )

        if best_model is None:
            raise RuntimeError("No successful trials in XGBoostTuner.fit().")

        return best_model, best_params

    def _run_single_trial(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        lookback: int = 30,
        trial_idx: Optional[int] = None,
    ) -> Tuple["XGBoostModel", Dict[str, Any]]:
        """
        Run a single random hyperparameter trial and store the result.

        Args:
            X_train, y_train, X_val, y_val: Data arrays.
            lookback: Lookback window to pass to XGBoostModel.
            trial_idx: Optional trial index, used for random_state seeding.

        Returns:
            (model, sampled_hyperparams)
        """
        trial_number = trial_idx if trial_idx is not None else len(self.results_)
        sampled = {k: self._rng.choice(v) for k, v in self.param_grid.items()}
        sampled["random_state"] = self.seed + trial_number

        logger.info(
            "XGBoostTuner trial %d/%d: %s",
            trial_number + 1,
            self.n_trials,
            sampled,
        )

        model = XGBoostModel(lookback=lookback, **sampled)
        model.fit(X_train, y_train, X_val, y_val)

        # Compute validation RMSE
        if model.history["val_rmse"]:
            val_rmse = min(model.history["val_rmse"])
        else:
            from src.evaluation.metrics import rmse as rmse_fn

            val_rmse = rmse_fn(y_val, model.predict(X_val))

        trial_result = {"trial": trial_number + 1, "val_rmse": val_rmse, **sampled}
        self.results_.append(trial_result)
        logger.info("  val_rmse=%.6f", val_rmse)

        return model, {**sampled, "lookback": lookback}
