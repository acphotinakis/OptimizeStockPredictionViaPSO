"""
Unit tests for PSO optimization module.
"""
import pytest
import numpy as np
from omegaconf import OmegaConf
from src.optimization.pso import Particle, ImprovedPSO


@pytest.fixture
def simple_search_space():
    """Create a simple search space for testing."""
    return OmegaConf.create({
        "param1": {
            "type": "float",
            "range": [0.0, 1.0],
        },
        "param2": {
            "type": "int",
            "range": [1, 10],
        },
        "param3": {
            "type": "categorical",
            "values": ["a", "b", "c"],
        },
    })


def test_particle_initialization(simple_search_space):
    """Test particle initialization."""
    np.random.seed(42)
    particle = Particle(simple_search_space)
    
    # Check position initialized
    assert "param1" in particle.position
    assert "param2" in particle.position
    assert "param3" in particle.position
    
    # Check types
    assert isinstance(particle.position["param1"], (float, np.floating))
    assert isinstance(particle.position["param2"], (int, np.integer))
    assert particle.position["param3"] in ["a", "b", "c"]
    
    # Check ranges
    assert 0.0 <= particle.position["param1"] <= 1.0
    assert 1 <= particle.position["param2"] <= 10
    
    # Check velocity initialized
    assert "param1" in particle.velocity
    assert "param2" in particle.velocity
    assert "param3" in particle.velocity
    
    # Check personal best initialized
    assert particle.pbest_position == particle.position
    assert particle.pbest_fitness == -float("inf")


def test_particle_multiple_initializations(simple_search_space):
    """Test that multiple particles have different initial positions."""
    np.random.seed(42)
    particle1 = Particle(simple_search_space)
    particle2 = Particle(simple_search_space)
    
    # Should have different positions (with high probability)
    assert particle1.position["param1"] != particle2.position["param1"]


def test_pso_swarm_initialization(simple_search_space, mock_config):
    """Test PSO swarm initialization."""
    mock_config.optimization.swarm.particles = 5
    mock_config.optimization.swarm.iterations = 3
    
    # Create minimal PSO instance (without data loaders for unit test)
    pso = ImprovedPSO(
        cfg=mock_config,
        search_space=simple_search_space,
        train_loader=None,
        val_loader=None,
        device=torch.device("cpu"),
    )
    
    assert len(pso.swarm) == 5
    assert pso.n_particles == 5
    assert pso.iterations == 3
    assert pso.gbest_fitness == -float("inf")


def test_pso_inertia_decay(simple_search_space, mock_config):
    """Test non-linear inertia decay."""
    import torch
    
    pso = ImprovedPSO(
        cfg=mock_config,
        search_space=simple_search_space,
        train_loader=None,
        val_loader=None,
        device=torch.device("cpu"),
    )
    
    # Calculate inertia at different iterations
    w_start = pso._calculate_inertia(0)
    w_mid = pso._calculate_inertia(pso.iterations // 2)
    w_end = pso._calculate_inertia(pso.iterations - 1)
    
    # Inertia should decay over time
    assert w_start > w_mid > w_end
    
    # Should be within reasonable bounds
    assert 0.4 <= w_end <= w_start <= 0.9


def test_pso_mutation(simple_search_space, mock_config):
    """Test adaptive mutation."""
    import torch
    
    np.random.seed(42)
    mock_config.optimization.improvements.mutation_factor = 1.0  # Always mutate
    
    pso = ImprovedPSO(
        cfg=mock_config,
        search_space=simple_search_space,
        train_loader=None,
        val_loader=None,
        device=torch.device("cpu"),
    )
    
    particle = pso.swarm[0]
    original_position = particle.position.copy()
    
    # Apply mutation
    pso._apply_mutation(particle)
    
    # At least one parameter should have changed
    changed = any(
        particle.position[key] != original_position[key]
        for key in particle.position.keys()
    )
    assert changed


def test_pso_on_simple_function():
    """Test PSO on a simple optimization problem (sphere function)."""
    # This is a simplified test without actual LSTM training
    # Tests the PSO mechanics on a known function
    
    # Define simple sphere function: f(x) = -sum(x^2)
    # Global minimum at x = [0, 0, ...]
    
    search_space = OmegaConf.create({
        "x1": {"type": "float", "range": [-5.0, 5.0]},
        "x2": {"type": "float", "range": [-5.0, 5.0]},
    })
    
    # Create minimal config
    cfg = OmegaConf.create({
        "optimization": {
            "swarm": {
                "particles": 10,
                "iterations": 20,
                "inertia_weight": 0.9,
                "cognitive_coeff": 1.5,
                "social_coeff": 2.0,
            },
            "improvements": {
                "mutation_factor": 0.05,
            },
        },
    })
    
    # Note: Full PSO test would require mocking the fitness evaluation
    # This test validates initialization only
    import torch
    pso = ImprovedPSO(cfg, search_space, None, None, torch.device("cpu"))
    
    assert len(pso.swarm) == 10
    assert pso.gbest_fitness == -float("inf")


# Import torch for PSO tests
import torch
