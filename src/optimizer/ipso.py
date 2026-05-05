"""
src/optimizer/ipso.py

Improved PSO (IPSO) - Ji, Liew & Yang, IEEE Access 2021.

Two enhancements over standard PSO:

1. Non-linear (tanh) inertia weight (per-particle):
       ω_i^t = ω_max − (ω_max − ω_min) · tanh(4t / T_max)

2. Adaptive mutation factor:
       μ_mf^t = 0.7 + 0.3 · (t / T_max)          ∈ (0.7, 1]
   When ξ ~ U(0,1) > μ_mf  -->  particle mutates (random reinit).
   Mutation probability decays from 30 % --> 0 % over iterations.
"""

from __future__ import annotations

import logging

import numpy as np

from .pso_core import StandardPSO
from .particle import Particle, random_position, random_velocity

logger = logging.getLogger(__name__)


class IPSO(StandardPSO):
    """Improved PSO with tanh inertia weight and adaptive mutation.

    All constructor arguments are identical to StandardPSO. The
    optimisation loop is inherited from StandardPSO; only ``_inertia``
    and ``_pre_update_hook`` are overridden.
    """

    # ---- Inertia (override) ---------------------------------------------

    def _inertia(self, t: int) -> float:
        """Non-linear tanh inertia weight (eq. 5 in pso_mathematical_spec.md).

        Args:
            t: Current iteration index (1-based).

        Returns:
            The shared inertia schedule value for iteration ``t``.
            Per-particle diversity arises from the stochastic
            ``r1``/``r2`` velocity updates rather than per-particle
            inertia state.
        """
        ratio = 4.0 * t / max(self.T, 1)
        return self.w_max - (self.w_max - self.w_min) * float(np.tanh(ratio))

    # ---- Adaptive mutation (override) -----------------------------------

    def _pre_update_hook(self, particle: Particle, t: int) -> bool:
        """Apply adaptive mutation with decreasing probability.

        Args:
            particle: The particle being considered for mutation.
            t: Current iteration index (1-based).

        Returns:
            True when the particle is mutated; the caller then skips
            the standard velocity/position update for this iteration.
        """
        mu_mf = 0.7 + 0.3 * (t / max(self.T, 1))  # ∈ (0.7, 1]
        xi = self._rng.uniform()
        if xi > mu_mf:
            # Mutate: reinitialise position and velocity randomly. The personal
            # best is also reset so the cognitive force does not drag the
            # freshly-mutated particle back toward an irrelevant region.
            particle.position = random_position(self._rng)
            particle.velocity = random_velocity(self._rng)
            particle.pbest_position = particle.position.copy()
            particle.pbest_fitness = float("inf")
            logger.info("Particle %d mutated at iteration %d", particle.idx, t)
            return True
        return False
