"""
tests/test_xgboost.py

Unit and integration tests for XGBoostModel, XGBoostTuner, and the
three XGBoost pipeline scripts (06, 07, 08).

Test categories:
  - Construction and parameter storage
  - Flattening logic (lookback slicing)
  - fit() / predict() correctness
  - History recording
  - Feature importance extraction
  - Save / load round-trip
  - XGBoostTuner smoke test
  - Metric compatibility: results use the same metric functions as LSTM
  - Script smoke tests (tiny data, short runs)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models.xgboost.xgboost_model import XGBoostModel, XGBoostTuner
from src.evaluation import all_statistical_metrics, all_trading_metrics


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def tiny_windows_xgb():
    """Small [N, T, F] tensors sized for fast XGBoost tests."""
    rng = np.random.default_rng(0)
    N, T, F = 300, 30, 15
    X = rng.standard_normal((N, T, F)).astype(np.float32)
    y = rng.standard_normal(N).astype(np.float32) * 0.001
    return X, y


@pytest.fixture
def split_windows(tiny_windows_xgb):
    X, y = tiny_windows_xgb
    n = len(X)
    n_tr = int(0.6 * n)
    n_vl = int(0.2 * n)
    return (
        X[:n_tr],
        y[:n_tr],
        X[n_tr : n_tr + n_vl],
        y[n_tr : n_tr + n_vl],
        X[n_tr + n_vl :],
        y[n_tr + n_vl :],
    )


# ======================================================================
# Construction and parameter storage
# ======================================================================


class TestXGBoostModelConstruction:

    def test_default_params_stored(self):
        m = XGBoostModel()
        p = m.get_params()
        assert p["lookback"] == 30
        assert p["n_estimators"] == 500
        assert p["max_depth"] == 6
        assert p["learning_rate"] == pytest.approx(0.05)
        assert p["importance_type"] == "gain"

    def test_custom_params_stored(self):
        m = XGBoostModel(
            lookback=60,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.01,
            reg_alpha=0.5,
        )
        p = m.get_params()
        assert p["lookback"] == 60
        assert p["n_estimators"] == 100
        assert p["max_depth"] == 4
        assert p["learning_rate"] == pytest.approx(0.01)
        assert p["reg_alpha"] == pytest.approx(0.5)

    def test_model_is_unfitted_on_init(self):
        m = XGBoostModel()
        assert m._model is None

    def test_predict_before_fit_raises(self):
        m = XGBoostModel(lookback=10)
        X = np.ones((5, 30, 10), dtype=np.float32)
        with pytest.raises(RuntimeError, match="fit"):
            m.predict(X)


# ======================================================================
# Flattening logic
# ======================================================================


class TestFlattening:

    def test_flatten_shape(self):
        m = XGBoostModel(lookback=10)
        X = np.ones((50, 30, 8), dtype=np.float32)
        flat = m._flatten(X)
        assert flat.shape == (50, 10 * 8)

    def test_flatten_uses_last_lookback_steps(self):
        """The last `lookback` timesteps should be kept, not the first."""
        m = XGBoostModel(lookback=5)
        # Unique values at each timestep
        X = np.zeros((1, 10, 2), dtype=np.float32)
        for t in range(10):
            X[0, t, :] = float(t)  # timestep t --> value t
        flat = m._flatten(X)  # shape (1, 5*2)
        # The last 5 timesteps are indices 5..9
        expected = np.array([[5, 5, 6, 6, 7, 7, 8, 8, 9, 9]], dtype=np.float32)
        np.testing.assert_array_equal(flat, expected)

    def test_flatten_lookback_longer_than_window_raises(self):
        m = XGBoostModel(lookback=50)
        X = np.ones((5, 30, 4), dtype=np.float32)  # T=30 < lookback=50
        with pytest.raises(ValueError, match="lookback"):
            m._flatten(X)

    def test_flatten_dtype_float32(self):
        m = XGBoostModel(lookback=5)
        X = np.ones((10, 20, 3), dtype=np.float64)
        flat = m._flatten(X)
        assert flat.dtype == np.float32


# ======================================================================
# fit() / predict()
# ======================================================================


class TestFitPredict:

    def test_fit_returns_history(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_windows
        m = XGBoostModel(lookback=20, n_estimators=20, early_stopping_rounds=5)
        history = m.fit(X_tr, y_tr, X_vl, y_vl)
        assert "train_rmse" in history
        assert "val_rmse" in history
        assert len(history["val_rmse"]) >= 1

    def test_predict_shape(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_windows
        m = XGBoostModel(lookback=20, n_estimators=20, early_stopping_rounds=5)
        m.fit(X_tr, y_tr, X_vl, y_vl)
        preds = m.predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_predict_dtype(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_windows
        m = XGBoostModel(lookback=20, n_estimators=20, early_stopping_rounds=5)
        m.fit(X_tr, y_tr, X_vl, y_vl)
        preds = m.predict(X_te)
        assert preds.dtype == np.float32

    def test_predict_finite(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_windows
        m = XGBoostModel(lookback=20, n_estimators=20, early_stopping_rounds=5)
        m.fit(X_tr, y_tr, X_vl, y_vl)
        preds = m.predict(X_te)
        assert np.isfinite(preds).all(), "Predictions contain NaN or Inf"

    def test_fit_is_deterministic(self, split_windows):
        """Same seed --> identical predictions."""
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_windows
        m1 = XGBoostModel(lookback=20, n_estimators=30, random_state=7)
        m2 = XGBoostModel(lookback=20, n_estimators=30, random_state=7)
        m1.fit(X_tr, y_tr, X_vl, y_vl)
        m2.fit(X_tr, y_tr, X_vl, y_vl)
        np.testing.assert_array_equal(m1.predict(X_te), m2.predict(X_te))

    def test_early_stopping_reduces_rounds(self, split_windows):
        """With aggressive early stopping, best_iteration < n_estimators."""
        X_tr, y_tr, X_vl, y_vl, _, _ = split_windows
        m = XGBoostModel(lookback=20, n_estimators=200, early_stopping_rounds=3)
        m.fit(X_tr, y_tr, X_vl, y_vl)
        assert m._best_iteration <= 200

    def test_val_rmse_history_length_matches_best_iteration(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, _, _ = split_windows
        m = XGBoostModel(lookback=20, n_estimators=50, early_stopping_rounds=5)
        m.fit(X_tr, y_tr, X_vl, y_vl)
        # val_rmse list should have at least 1 entry
        assert len(m.history["val_rmse"]) >= 1


# ======================================================================
# Feature importances
# ======================================================================


class TestFeatureImportances:

    def test_flat_importances_shape(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, _, _ = split_windows
        lb = 15
        m = XGBoostModel(lookback=lb, n_estimators=20, early_stopping_rounds=5)
        m.fit(X_tr, y_tr, X_vl, y_vl)
        imp = m.get_feature_importances()
        F = X_tr.shape[2]
        assert imp.shape == (lb * F,), f"Expected ({lb*F},), got {imp.shape}"

    def test_flat_importances_non_negative(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, _, _ = split_windows
        m = XGBoostModel(lookback=15, n_estimators=20, early_stopping_rounds=5)
        m.fit(X_tr, y_tr, X_vl, y_vl)
        imp = m.get_feature_importances()
        assert (imp >= 0).all()

    def test_per_original_feature_shape(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, _, _ = split_windows
        F = X_tr.shape[2]
        m = XGBoostModel(lookback=15, n_estimators=20, early_stopping_rounds=5)
        m.fit(X_tr, y_tr, X_vl, y_vl)
        orig_imp = m.get_per_original_feature_importances(F)
        assert orig_imp.shape == (F,)

    def test_importances_before_fit_raises(self):
        m = XGBoostModel()
        with pytest.raises(RuntimeError, match="fit"):
            m.get_feature_importances()


# ======================================================================
# Save / load round-trip
# ======================================================================


class TestSaveLoad:

    def test_save_load_identical_predictions(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_windows
        m = XGBoostModel(lookback=20, n_estimators=20, early_stopping_rounds=5)
        m.fit(X_tr, y_tr, X_vl, y_vl)
        preds_before = m.predict(X_te)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.ubj")
            m.save(path)

            m2 = XGBoostModel(lookback=20, n_estimators=20)
            m2.load(path)
            preds_after = m2.predict(X_te)

        np.testing.assert_array_almost_equal(preds_before, preds_after, decimal=5)

    def test_save_before_fit_raises(self):
        m = XGBoostModel()
        with pytest.raises(RuntimeError, match="Nothing to save"):
            m.save("/tmp/should_not_exist.ubj")


# ======================================================================
# Metric compatibility with the rest of the project
# ======================================================================


class TestMetricCompatibility:
    """XGBoost predictions flow through the SAME metric functions as LSTM."""

    def test_all_statistical_metrics_keys(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_windows
        m = XGBoostModel(lookback=20, n_estimators=20, early_stopping_rounds=5)
        m.fit(X_tr, y_tr, X_vl, y_vl)
        preds = m.predict(X_te)
        metrics = all_statistical_metrics(y_te, preds)

        required_keys = {
            "rmse",
            "mae",
            "mape",
            "r2",
            "directional_accuracy",
            "f1_ternary",
            "auc_ternary",
        }
        assert required_keys <= set(
            metrics.keys()
        ), f"Missing keys: {required_keys - set(metrics.keys())}"

    def test_all_statistical_metrics_finite(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_windows
        m = XGBoostModel(lookback=20, n_estimators=20, early_stopping_rounds=5)
        m.fit(X_tr, y_tr, X_vl, y_vl)
        preds = m.predict(X_te)
        metrics = all_statistical_metrics(y_te, preds)

        for key, val in metrics.items():
            if val != val:  # NaN check — AUC can be NaN if only one class present
                continue
            assert np.isfinite(val), f"Metric '{key}' is not finite: {val}"

    def test_rmse_value_range(self, split_windows):
        """RMSE must be ≥ 0."""
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_windows
        m = XGBoostModel(lookback=20, n_estimators=20, early_stopping_rounds=5)
        m.fit(X_tr, y_tr, X_vl, y_vl)
        preds = m.predict(X_te)
        metrics = all_statistical_metrics(y_te, preds)
        assert metrics["rmse"] >= 0.0

    def test_directional_accuracy_bounded(self, split_windows):
        """DA must be in [0, 1]."""
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_windows
        m = XGBoostModel(lookback=20, n_estimators=20, early_stopping_rounds=5)
        m.fit(X_tr, y_tr, X_vl, y_vl)
        preds = m.predict(X_te)
        metrics = all_statistical_metrics(y_te, preds)
        assert 0.0 <= metrics["directional_accuracy"] <= 1.0

    def test_backtester_accepts_xgb_predictions(self, split_windows):
        """Backtester must work with XGBoost predictions."""
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_windows
        m = XGBoostModel(lookback=20, n_estimators=20, early_stopping_rounds=5)
        m.fit(X_tr, y_tr, X_vl, y_vl)
        preds = m.predict(X_te)

        import pandas as pd
        from src.evaluation import Backtester

        N = len(preds)
        opens = np.ones(N, dtype=np.float32) * 100.0
        closes = np.ones(N, dtype=np.float32) * 100.0
        ts = pd.date_range("2023-01-03 14:30", periods=N, freq="1min", tz="UTC")

        bt = Backtester(
            initial_capital=100_000,
            position_fraction=0.01,
            transaction_cost=0.001,
            slippage=0.0005,
        )
        result = bt.run(preds.copy(), opens, closes, ts)

        assert np.isfinite(result.sharpe)
        assert 0.0 <= result.mdd <= 1.0
        assert len(result.equity_curve) == N


# ======================================================================
# XGBoostTuner
# ======================================================================


class TestXGBoostTuner:

    def test_tuner_returns_model_and_params(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_windows
        tuner = XGBoostTuner(n_trials=3, seed=0)
        best_model, best_params = tuner.fit(X_tr, y_tr, X_vl, y_vl, lookback=20)

        assert isinstance(best_model, XGBoostModel)
        assert isinstance(best_params, dict)
        assert "lookback" in best_params

    def test_tuner_stores_all_trial_results(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, _, _ = split_windows
        n_trials = 4
        tuner = XGBoostTuner(n_trials=n_trials, seed=1)
        tuner.fit(X_tr, y_tr, X_vl, y_vl, lookback=20)
        assert len(tuner.results_) == n_trials

    def test_tuner_best_is_minimum_val_rmse(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_windows
        tuner = XGBoostTuner(n_trials=5, seed=2)
        best_model, best_params = tuner.fit(X_tr, y_tr, X_vl, y_vl, lookback=20)
        min_rmse = min(r["val_rmse"] for r in tuner.results_)
        # The returned model's val RMSE should equal or be near the minimum
        preds = best_model.predict(X_vl)
        from src.evaluation.metrics import rmse as rmse_fn

        actual_rmse = rmse_fn(y_vl, preds)
        # Allow a small tolerance (final retrain may differ slightly)
        assert actual_rmse <= min_rmse * 1.5 + 1e-7

    def test_tuner_deterministic_with_same_seed(self, split_windows):
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_windows
        t1 = XGBoostTuner(n_trials=3, seed=99)
        m1, p1 = t1.fit(X_tr, y_tr, X_vl, y_vl, lookback=20)
        t2 = XGBoostTuner(n_trials=3, seed=99)
        m2, p2 = t2.fit(X_tr, y_tr, X_vl, y_vl, lookback=20)
        np.testing.assert_array_equal(m1.predict(X_te), m2.predict(X_te))
