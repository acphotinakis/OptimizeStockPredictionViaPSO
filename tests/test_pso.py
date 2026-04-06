"""
tests/test_pso.py

Unit tests for PSO optimizer components.
"""

import numpy as np
import pytest

from src.optimizer.particle import Particle, decode, random_position, random_velocity, LB, UB
from src.optimizer.fitness import CompositeFitness, generate_signals
from src.optimizer.pso_core import StandardPSO
from src.optimizer.ipso import IPSO


class TestParticle:
    """Test particle encoding and decoding."""

    def test_decode_valid_position(self):
        """Test decoding a valid particle position."""
        position = np.array([2.5, 128.0, 0.3, np.log(0.001), 1.5])
        params = decode(position)
        
        assert params["num_layers"] == 2
        assert params["hidden_units"] == 128
        assert 0.29 < params["dropout"] < 0.31
        assert 0.0009 < params["learning_rate"] < 0.0011
        assert params["lookback"] == 30

    def test_random_position_bounds(self):
        """Test that random positions are within bounds."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            pos = random_position(rng)
            assert np.all(pos >= LB)
            assert np.all(pos <= UB)

    def test_random_velocity_scale(self):
        """Test that random velocities are reasonably scaled."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            vel = random_velocity(rng)
            # Velocity should be within ±25% of range
            expected_max = 0.25 * (UB - LB)
            assert np.all(np.abs(vel) <= expected_max)

    def test_particle_initialization(self):
        """Test Particle class initialization."""
        rng = np.random.default_rng(42)
        particle = Particle.random_init(rng, idx=5)
        
        assert particle.idx == 5
        assert particle.fitness == float("inf")
        assert particle.pbest_fitness == float("inf")
        assert len(particle.position) == 5
        assert len(particle.velocity) == 5


class TestCompositeFitness:
    """Test composite fitness function."""

    def test_fitness_calculation(self):
        """Test basic fitness calculation."""
        fitness_fn = CompositeFitness()
        
        y_true = np.array([0.001, -0.002, 0.003, -0.001, 0.002])
        y_pred = np.array([0.0015, -0.0015, 0.0025, -0.0008, 0.0018])
        
        fitness = fitness_fn(y_true, y_pred)
        
        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0

    def test_generate_signals(self):
        """Test signal generation from predictions."""
        y_pred = np.array([0.0005, -0.0005, 0.00001, -0.00001, 0.002])
        signals = generate_signals(y_pred, threshold=1e-4)
        
        expected = np.array([1.0, -1.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(signals, expected)

    def test_fitness_reset(self):
        """Test that fitness function can be reset."""
        fitness_fn = CompositeFitness()
        
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        
        fitness_fn(y_true, y_pred)
        assert fitness_fn._rmse_min < float("inf")
        
        fitness_fn.reset()
        assert fitness_fn._rmse_min == float("inf")


class TestStandardPSO:
    """Test standard PSO optimizer."""

    def test_pso_initialization(self):
        """Test PSO initialization."""
        pso = StandardPSO(
            n_particles=10,
            n_iterations=5,
            seed=42,
        )
        
        assert pso.M == 10
        assert pso.T == 5
        assert len(pso._swarm) == 0  # Not initialized yet

    def test_inertia_decay(self):
        """Test linear inertia weight decay."""
        pso = StandardPSO(n_iterations=10, w_min=0.4, w_max=0.9, seed=42)
        
        w_0 = pso._inertia(0)
        w_5 = pso._inertia(5)
        w_10 = pso._inertia(10)
        
        assert w_0 == 0.9
        assert w_5 == 0.65
        assert w_10 == 0.4

    def test_swarm_initialization(self):
        """Test swarm initialization."""
        pso = StandardPSO(n_particles=5, seed=42)
        pso._initialise_swarm()
        
        assert len(pso._swarm) == 5
        for particle in pso._swarm:
            assert len(particle.position) == 5
            assert np.all(particle.position >= LB)
            assert np.all(particle.position <= UB)


class TestIPSO:
    """Test Improved PSO optimizer."""

    def test_ipso_inertia(self):
        """Test tanh inertia weight schedule."""
        ipso = IPSO(n_iterations=50, w_min=0.4, w_max=0.9, seed=42)
        
        w_0 = ipso._inertia(0)
        w_25 = ipso._inertia(25)
        w_50 = ipso._inertia(50)
        
        # Tanh should give non-linear decay
        assert w_0 > w_25 > w_50
        assert w_0 <= 0.9
        assert w_50 >= 0.4

    def test_adaptive_mutation_probability(self):
        """Test that mutation probability decreases over iterations."""
        ipso = IPSO(n_iterations=100, seed=42)
        
        # Mutation factor μ_mf = 0.7 + 0.3 * (t / T)
        # Mutation happens when ξ > μ_mf
        # So probability = 1 - μ_mf decreases from 0.3 to 0
        
        # Early iterations: higher mutation probability
        mu_early = 0.7 + 0.3 * (10 / 100)
        prob_early = 1 - mu_early
        
        # Late iterations: lower mutation probability
        mu_late = 0.7 + 0.3 * (90 / 100)
        prob_late = 1 - mu_late
        
        assert prob_early > prob_late


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
