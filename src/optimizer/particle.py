"""
src/optimizer/particle.py

Particle encoding and decoding for the 5-dimensional LSTM hyperparameter
search space defined in pso_mathematical_spec.md.

Continuous encoding → decoded hyperparameters:
  dim 0: num_layers    [1.0, 4.99] → int  {1,2,3,4}
  dim 1: hidden_units  [32, 512]   → int  multiple of 32
  dim 2: dropout       [0.0, 0.5]  → float
  dim 3: log_lr        [ln1e-5, ln1e-1] → float exp(x)
  dim 4: lookback_idx  [0, 3.99]   → int  index into {10,30,60,120}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

LOOKBACK_CHOICES = [10, 30, 60, 120]

# Search space bounds (continuous encoding)
LB = np.array([1.0, 32.0, 0.0, np.log(1e-5), 0.0], dtype=np.float64)
UB = np.array([4.99, 512.0, 0.5, np.log(1e-1), 3.99], dtype=np.float64)


def decode(position: np.ndarray) -> Dict[str, Any]:
    """Map a continuous particle position to LSTM hyperparameters.

    Args:
        position: Length-5 float array (clipped to [LB, UB]).

    Returns:
        Dict with keys: num_layers, hidden_units, dropout,
                        learning_rate, lookback.
    """
    x = np.clip(position, LB, UB)
    num_layers = int(x[0])  # floor → {1,2,3,4}
    hidden_raw = int(round(x[1] / 32.0)) * 32  # round to nearest 32
    hidden_units = int(np.clip(hidden_raw, 32, 512))
    dropout = float(x[2])
    learning_rate = float(np.exp(x[3]))
    lookback = LOOKBACK_CHOICES[int(x[4])]
    return {
        "num_layers": num_layers,
        "hidden_units": hidden_units,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "lookback": lookback,
    }


def random_position(rng: np.random.Generator) -> np.ndarray:
    """Sample a random particle position uniformly inside [LB, UB]."""
    return rng.uniform(LB, UB)


def random_velocity(rng: np.random.Generator) -> np.ndarray:
    """Sample initial velocity as ±25 % of the search range."""
    half_range = (UB - LB) * 0.25
    return rng.uniform(-half_range, half_range)


@dataclass
class Particle:
    """Single PSO particle.

    Attributes:
        position:       Current continuous position in R^5.
        velocity:       Current velocity in R^5.
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
