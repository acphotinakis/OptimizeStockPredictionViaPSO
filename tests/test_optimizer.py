"""Tests for IPSO particle encoding, bounds, inertia schedule, and fitness."""

import numpy as np
import pytest

from src.optimizer.particle import LB, UB, decode, random_position


def test_particle_bounds_and_decoding():
    """Verify particle decoding preserves valid hyperparameter ranges."""
    rng = np.random.default_rng(42)

    for _ in range(20):
        pos = random_position(rng)
        params = decode(pos)

        assert 50 <= params["units_1"] <= 300
        assert 20 <= params["units_2"] <= 200
        assert 0.0 <= params["dropout"] <= 0.5
        assert 0.001 - 1e-6 <= params["learning_rate"] <= 0.01 + 1e-6
        assert params["batch_size"] in [32, 64]
        assert 50 <= params["epochs"] <= 300
        assert params["lookback"] == 20


def test_particle_boundary_clipping():
    """Verify out-of-bounds positions are clipped cleanly."""
    extreme_low = LB - 100.0
    params_low = decode(extreme_low)
    assert params_low["units_1"] == 50
    assert params_low["units_2"] == 20
    assert params_low["dropout"] == 0.0
    assert np.isclose(params_low["learning_rate"], 0.001)
    assert params_low["batch_size"] == 32
    assert params_low["epochs"] == 50

    extreme_high = UB + 100.0
    params_high = decode(extreme_high)
    assert params_high["units_1"] == 300
    assert params_high["units_2"] == 200
    assert params_high["dropout"] == 0.5
    assert np.isclose(params_high["learning_rate"], 0.01)
    assert params_high["batch_size"] == 64
    assert params_high["epochs"] == 300


def test_ipso_tanh_inertia_schedule():
    """Verify tanh nonlinear inertia decay behaves per Ji et al. 2021."""
    w_max = 0.9
    w_min = 0.4
    max_iter = 50

    omegas = [w_max - (w_max - w_min) * np.tanh(4.0 * t / max_iter) for t in range(max_iter + 1)]

    # Monotonically non-increasing
    for i in range(len(omegas) - 1):
        assert omegas[i] >= omegas[i + 1]

    # Initial value is w_max
    assert np.isclose(omegas[0], w_max)
    # Final value approaches w_min
    assert omegas[-1] < 0.45


def test_spec_compliant_fitness_if_torch_available():
    """Test composite fitness calculation and bias exclusion when torch is installed."""
    torch = pytest.importorskip("torch")
    from src.optimizer.fitness import SpecCompliantFitness, compute_msw

    # Create a small dummy model with weights and biases
    class DummyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(5, 5, bias=True)
            with torch.no_grad():
                self.linear.weight.fill_(2.0)
                self.linear.bias.fill_(100.0)  # Large bias should be excluded

    model = DummyNet()
    # MSW should reflect only weights (2.0^2 = 4.0), not bias (100.0)
    msw = compute_msw(model)
    assert np.isclose(msw, 4.0), f"Expected MSW=4.0, got {msw}"

    fitness_fn = SpecCompliantFitness(gamma=0.9, msw_scale=0.01)
    val_preds = np.array([0.1, 0.2])
    val_targets = np.array([0.1, 0.2])  # MSE = 0.0
    f_val = fitness_fn(model, val_preds, val_targets)
    expected_f = 0.9 * 0.0 + 0.1 * 0.01 * 4.0
    assert np.isclose(f_val, expected_f)
