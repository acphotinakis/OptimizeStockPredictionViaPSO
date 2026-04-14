"""
src/models/baselines.py

Three baseline models for benchmarking against IPSO-LSTM:
  1. PersistenceModel — predicts ŷ_{t+1} = y_t (naive / random walk)
  2. XGBoostBaseline  — flattened-sequence XGBoost regressor
  3. VanillaLSTM      — LSTM with manually set "reasonable" hyperparameters
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .lstm.lstm_model import LSTMModel, LSTMTrainer


# ======================================================================
# Persistence (Naive / Random Walk)
# ======================================================================


class PersistenceModel:
    """Predicts next-bar return = current-bar return (pure random walk baseline)."""

    def fit(self, X: np.ndarray, y: np.ndarray, **_) -> "PersistenceModel":
        return self  # No fitting needed

    def predict(self, X: np.ndarray, y_prev: np.ndarray) -> np.ndarray:
        """Return y_prev as the forecast.

        Args:
            X: Feature array (ignored).
            y_prev: The actual return at t (i.e. predict y_{t+1} = y_t).

        Returns:
            y_prev unchanged.
        """
        return y_prev.copy()

    @staticmethod
    def predict_from_returns(y: np.ndarray) -> np.ndarray:
        """Convenience: shift returns by one step."""
        preds = np.empty_like(y)
        preds[0] = 0.0
        preds[1:] = y[:-1]
        return preds


# ======================================================================
# XGBoost baseline
# ======================================================================


class XGBoostBaseline:
    """XGBoost regressor on flattened (T × F) sequences.

    Args:
        lookback: Number of timesteps T to flatten. Sequences shorter than T
            will be zero-padded on the left.
        n_estimators: Number of boosting rounds.
        max_depth: Tree max depth.
        learning_rate: Boosting learning rate.
        random_state: Random seed.
    """

    def __init__(
        self,
        lookback: int = 30,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.lookback = lookback
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        """Fit on [N, T, F] input (flattened internally).

        Args:
            X: [N, T, F] array.
            y: [N] target.

        Returns:
            self
        """
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost not installed. Run: pip install xgboost") from e

        X_flat = self._flatten(X)
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=self.random_state,
            verbosity=0,
        )
        self._model.fit(X_flat, y, eval_set=[(X_flat, y)], verbose=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.predict(self._flatten(X))

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        N, T, F = X.shape
        return X.reshape(N, T * F)


# ======================================================================
# Vanilla LSTM with fixed "sensible" hyperparameters
# ======================================================================


import logging

logger = logging.getLogger(__name__)


class VanillaLSTM:
    """LSTM with manually chosen hyperparameters (no PSO optimization).

    Serves as the 'standard LSTM' baseline from the experiment plan.
    Default values reflect common choices in the literature.

    Args:
        input_size: Feature dimension F.
        num_layers: 2 (Ji et al. default; Zeng et al. best result).
        hidden_units: 128.
        dropout: 0.2.
        learning_rate: 0.001 (Adam default).
        lookback: 30 minutes (Lanbouri & Achchab optimal short-term window).
        max_epochs: 100.
        patience: 10.
        batch_size: 256.
        device: Torch device.
    """

    DEFAULT_PARAMS = {
        "num_layers": 2,
        "hidden_units": 128,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "lookback": 30,
        "max_epochs": 100,
        "patience": 10,
        "batch_size": 256,
    }

    def __init__(
        self,
        input_size: int,
        device: Optional[str] = None,
        **overrides,
    ) -> None:
        self.params = {**self.DEFAULT_PARAMS, **overrides}
        self.lookback = self.params["lookback"]

        # ------------------------------------------------------------------
        # Log all resolved hyperparameters
        # ------------------------------------------------------------------
        logger.info(
            "[LSTM INIT] input_size=%d | device=%s | num_layers=%d | hidden_units=%d | "
            "dropout=%.3f | learning_rate=%.6f | lookback=%d | max_epochs=%d | "
            "patience=%d | batch_size=%d",
            input_size,
            device,
            self.params["num_layers"],
            self.params["hidden_units"],
            self.params["dropout"],
            self.params["learning_rate"],
            self.params["lookback"],
            self.params["max_epochs"],
            self.params["patience"],
            self.params["batch_size"],
        )

        model = LSTMModel(
            input_size=input_size,
            num_layers=self.params["num_layers"],
            hidden_units=self.params["hidden_units"],
            dropout=self.params["dropout"],
        )
        self._trainer = LSTMTrainer(
            model=model,
            lr=self.params["learning_rate"],
            max_epochs=self.params["max_epochs"],
            patience=self.params["patience"],
            batch_size=self.params["batch_size"],
            device=device,
        )

    def fit(self, X_train, y_train, X_val, y_val):
        return self._trainer.fit(X_train, y_train, X_val, y_val)

    def predict(self, X) -> np.ndarray:
        return self._trainer.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 256):
        return self._trainer.evaluate(X=X, y=y, batch_size=self.params["batch_size"])
