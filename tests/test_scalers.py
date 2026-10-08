"""Tests verifying FrozenStandardScaler and FrozenMinMaxScaler prevent lookahead leakage."""

import numpy as np
import pytest

from src.features.scaler import FrozenMinMaxScaler, FrozenStandardScaler


def test_frozen_standard_scaler_train_only_fitting():
    """Verify FrozenStandardScaler fits only on train data and stays frozen."""
    scaler = FrozenStandardScaler()
    assert not scaler.is_fitted

    # Attempting to transform before fitting must raise RuntimeError
    dummy_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(RuntimeError, match="Scaler not fitted"):
        scaler.transform(dummy_data)

    # Fit on training data
    train_data = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    feature_names = ["feat1", "feat2"]
    scaler.fit(train_data, feature_names=feature_names)

    assert scaler.is_fitted
    params_before = dict(scaler.get_params())

    # Transform test data with different distribution
    test_data = np.array([[50.0, 500.0], [100.0, 1000.0]])
    scaled_test = scaler.transform(test_data)
    assert scaled_test.shape == test_data.shape

    # Crucial leakage test: params must remain completely identical
    params_after = dict(scaler.get_params())
    assert params_before == params_after, "Scaler parameters drifted after transforming test data!"


def test_frozen_minmax_scaler_train_only_fitting():
    """Verify FrozenMinMaxScaler fits only on train data and stays frozen."""
    scaler = FrozenMinMaxScaler(feature_range=(-1.0, 1.0))
    assert not scaler.is_fitted

    dummy_data = np.array([[1.0], [2.0]])
    with pytest.raises(RuntimeError, match="Scaler not fitted"):
        scaler.transform(dummy_data)

    train_data = np.array([[0.0], [50.0], [100.0]])
    scaler.fit(train_data, feature_names=["f1"])
    assert scaler.is_fitted

    scaled_train = scaler.transform(train_data)
    assert np.isclose(scaled_train.min(), -1.0)
    assert np.isclose(scaled_train.max(), 1.0)

    params_before = dict(scaler.get_params())

    # Transform out-of-range test data
    test_data = np.array([[-50.0], [200.0]])
    scaled_test = scaler.transform(test_data)
    # Clip to bounds [-1.0, 1.0]
    assert np.all(scaled_test >= -1.0)
    assert np.all(scaled_test <= 1.0)

    params_after = dict(scaler.get_params())
    assert params_before == params_after, (
        "MinMax scaler parameters modified during test transformation!"
    )
