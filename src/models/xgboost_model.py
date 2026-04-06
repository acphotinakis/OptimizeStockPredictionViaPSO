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
import xgboost as xgb
import numpy as np
from src.utils.config_loader import Config, load_config

logger = logging.getLogger(__name__)

# Default hyperparameter set — mirrors experiment_plan.md §5.3
_DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 1e-5,
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
        cfg: Optional[Config] = None,
        lookback: int = _DEFAULT_LOOKBACK,
        early_stopping_rounds: int = _DEFAULT_EARLY_STOPPING_ROUNDS,
        importance_type: str = "gain",
        **kwargs,
    ) -> None:
        # Start with default YAML-based params
        self._xgb_params: Dict[str, Any] = _DEFAULT_PARAMS.copy()

        # Override from user config if provided
        if cfg is not None:
            self._xgb_params.update(getattr(cfg, "xgboost", {}))

        self.cfg = cfg
        # Override with explicit kwargs
        self._xgb_params.update(kwargs)

        self.lookback = lookback
        self.early_stopping_rounds = early_stopping_rounds
        self.importance_type = importance_type

        # Store constructor kwargs for get_params()
        self._init_kwargs: Dict[str, Any] = {
            "lookback": lookback,
            "early_stopping_rounds": early_stopping_rounds,
            "importance_type": importance_type,
            **self._xgb_params,
        }

        self._model = None  # type: Optional[xgb.XGBRegressor]
        self._best_iteration: int = 0
        self._feature_names: List[str] = []
        self.history: Dict[str, List[float]] = {"train_rmse": [], "val_rmse": []}

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
        X_tr_flat = self._flatten(X_train)
        X_vl_flat = self._flatten(X_val)

        eval_set = [(X_tr_flat, y_train), (X_vl_flat, y_val)]
        eval_names = ["train", "val"]

        self._model = xgb.XGBRegressor(
            **self._xgb_params,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric="rmse",
        )

        self._model.fit(
            X_tr_flat,
            y_train,
            eval_set=eval_set,
            eval_names=eval_names if self._supports_eval_names() else None,
            verbose=False,
        )

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

    def get_params(self) -> Dict[str, Any]:
        """Return the full constructor parameter dict (mirrors LSTM interface)."""
        return dict(self._init_kwargs)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importances(
        self,
        importance_type: Optional[str] = None,
    ) -> np.ndarray:
        """Return per-input-feature importance scores.

        The scores are computed over the *flattened* feature vector
        (length = lookback × F).  If you want per-original-feature
        importances, aggregate across the lookback axis with np.mean.

        Args:
            importance_type: One of 'gain', 'weight', 'cover',
                'total_gain', 'total_cover'.  Defaults to the value
                passed at construction time.

        Returns:
            Float32 array of shape [lookback × F].
        """
        if self._model is None:
            raise RuntimeError("Call fit() before get_feature_importances().")
        itype = importance_type or self.importance_type
        booster = self._model.get_booster()
        scores_dict = booster.get_score(importance_type=itype)
        # XGBoost may omit features with zero importance; fill with 0.0
        n_flat = self._model.n_features_in_
        importances = np.zeros(n_flat, dtype=np.float32)
        for feat_name, score in scores_dict.items():
            # Feature names are 'f0', 'f1', ... when no names are set
            try:
                idx = int(feat_name.lstrip("f"))
                importances[idx] = float(score)
            except (ValueError, IndexError):
                pass
        return importances

    def get_per_original_feature_importances(
        self,
        n_original_features: int,
        importance_type: Optional[str] = None,
    ) -> np.ndarray:
        """Aggregate flattened importances back to original feature axes.

        Each original feature appears `lookback` times in the flattened
        vector (once per timestep).  This method returns the mean
        importance across all timesteps for each original feature.

        Args:
            n_original_features: F (number of features before flattening).
            importance_type: See get_feature_importances().

        Returns:
            Float32 array of shape [F].
        """
        flat_imp = self.get_feature_importances(importance_type)
        # Reshape to [lookback, F] then average across timestep axis
        reshaped = flat_imp[: self.lookback * n_original_features].reshape(
            self.lookback, n_original_features
        )
        return reshaped.mean(axis=0)

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
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost not installed.") from exc
        self._model = xgb.XGBRegressor(**self._xgb_params)
        self._model.load_model(path)
        logger.info("XGBoost model loaded from %s", path)
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def load_model(self, path: str) -> "XGBoostModel":
        """Load a previously saved XGBoost booster from disk.

        Args:
            path: File path written by save().
            cfg: Optional configuration object containing xgboost defaults.

        Returns:
            self
        """
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost not installed. Run: pip install xgboost"
            ) from exc

        # Start with default YAML memory-safe params
        xgb_params = _DEFAULT_PARAMS.copy()

        # Override from cfg if provided
        if self.cfg is not None:
            xgb_params.update(getattr(self.cfg, "xgboost", {}))

        # Keep the previously used constructor overrides if any
        xgb_params.update(getattr(self, "_xgb_params", {}))

        # Initialize the XGBRegressor
        self._model = xgb.XGBRegressor(**xgb_params)

        # Load the booster state from file
        self._model.load_model(path)

        # Store params for get_params()
        self._xgb_params = xgb_params
        self._init_kwargs = {
            "lookback": getattr(self, "lookback", _DEFAULT_LOOKBACK),
            **xgb_params,
        }

        logger.info("XGBoost model loaded from %s", path)
        return self

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

    @staticmethod
    def _supports_eval_names() -> bool:
        """Check if the installed xgboost version accepts eval_names kwarg."""
        try:
            import xgboost as xgb

            major = int(xgb.__version__.split(".")[0])
            return major >= 2
        except Exception:
            return False


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
        param_grid: Dict mapping parameter name → list of candidate values.
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
