"""
src/optimizer/particle.py

Particle encoding and decoding for the 6-dimensional LSTM hyperparameter
search space per IPSO-LSTM specification.

SPEC-COMPLIANT ENCODING:
  dim 0: lstm_units_1   [50, 300]     --> int  (first LSTM layer size)
  dim 1: lstm_units_2   [20, 200]     --> int  (second LSTM layer size)
  dim 2: dropout_rate   [0.0, 0.5]    --> float
  dim 3: log_lr         [ln0.001, ln0.01] --> float exp(x)
  dim 4: batch_size_idx [0, 1.99]     --> int  index into {32, 64}
  dim 5: epochs         [50, 300]     --> int

FIXED (NOT OPTIMIZED):
  lookback: 20 (fixed per specification)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

# FIXED PARAMETER (not in PSO search space)
LOOKBACK_FIXED = 20

# Discrete batch size choices
BATCH_SIZE_CHOICES = [32, 64]

# Search space bounds (continuous encoding)
# [units_1, units_2, dropout, log_lr, batch_idx, epochs]
LB = np.array([50.0, 20.0, 0.0, np.log(0.001), 0.0, 50.0], dtype=np.float64)
UB = np.array([300.0, 200.0, 0.5, np.log(0.01), 1.99, 300.0], dtype=np.float64)


def decode(position: np.ndarray) -> Dict[str, Any]:
    """Map a continuous particle position to LSTM hyperparameters.

    Args:
        position: Length-6 float array (clipped to [LB, UB]).

    Returns:
        Dict with keys: units_1, units_2, dropout, learning_rate,
                        batch_size, epochs, lookback (fixed at 20).
    """
    x = np.clip(position, LB, UB)

    # Decode each dimension
    units_1 = int(np.round(x[0]))  # [50, 300]
    units_2 = int(np.round(x[1]))  # [20, 200]
    dropout = float(x[2])  # [0.0, 0.5]
    learning_rate = float(np.exp(x[3]))  # [0.001, 0.01]
    batch_size = BATCH_SIZE_CHOICES[int(x[4])]  # {32, 64}
    epochs = int(np.round(x[5]))  # [50, 300]

    return {
        "units_1": units_1,
        "units_2": units_2,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "lookback": LOOKBACK_FIXED,  # Always 20
    }


def random_position(rng: np.random.Generator) -> np.ndarray:
    """Sample a random particle position uniformly inside [LB, UB].

    Returns:
        Length-6 array for 6D search space.
    """
    return rng.uniform(LB, UB)


def random_velocity(rng: np.random.Generator) -> np.ndarray:
    """Sample initial velocity as ±25% of the search range.

    Returns:
        Length-6 array for 6D search space.
    """
    half_range = (UB - LB) * 0.25
    return rng.uniform(-half_range, half_range)


@dataclass
class Particle:
    """Single PSO particle.

    Attributes:
        position:       Current continuous position in R^6.
        velocity:       Current velocity in R^6.
        pbest_position: Best position found by this particle.
        pbest_fitness:  Fitness value at pbest_position (lower is better).
        fitness:        Current fitness value.
        idx:            Particle index within the swarm (for logging).
    """

    position: np.ndarray
    velocity: np.ndarray
    pbest_position: np.ndarray
    pbest_fitness: float = float("inf")
    fitness: float = float("inf")
    idx: int = 0

    # Convenience
    def decode(self) -> Dict[str, Any]:
        return decode(self.position)

    def decode_pbest(self) -> Dict[str, Any]:
        return decode(self.pbest_position)

    @classmethod
    def random_init(cls, rng: np.random.Generator, idx: int = 0) -> "Particle":
        pos = random_position(rng)
        vel = random_velocity(rng)
        return cls(
            position=pos,
            velocity=vel,
            pbest_position=pos.copy(),
            idx=idx,
        )
