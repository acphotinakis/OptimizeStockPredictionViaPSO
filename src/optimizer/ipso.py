"""
src/optimizer/ipso.py

Improved PSO (IPSO) — Ji, Liew & Yang, IEEE Access 2021.

Two enhancements over standard PSO:

1. Non-linear (tanh) inertia weight (per-particle):
       ω_i^t = ω_max − (ω_max − ω_min) · tanh(4t / T_max)

2. Adaptive mutation factor:
       μ_mf^t = 0.7 + 0.3 · (t / T_max)          ∈ (0.7, 1]
   When ξ ~ U(0,1) > μ_mf  →  particle mutates (random reinit).
   Mutation probability decays from 30 % → 0 % over iterations.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .pso_core import StandardPSO
from .particle import LB, UB, Particle, random_position, random_velocity
from .fitness import CompositeFitness

logger = logging.getLogger(__name__)


class IPSO(StandardPSO):
    """Improved PSO with tanh inertia weight and adaptive mutation.

    All constructor arguments are identical to StandardPSO.
    The optimisation loop is inherited; only _inertia and
    _pre_update_hook are overridden.
    """

    # ---- Inertia (override) ---------------------------------------------

    def _inertia(self, t: int) -> float:
        """Non-linear tanh inertia weight (eq. 5 in pso_mathematical_spec.md).

        Each particle could theoretically track its own ω; here we compute
        the shared schedule value for iteration t.  Individual diversity
        arises naturally from the stochastic velocity updates (r1, r2).
        """
        ratio = 4.0 * t / max(self.T, 1)
        return self.w_max - (self.w_max - self.w_min) * float(np.tanh(ratio))

    # ---- Adaptive mutation (override) -----------------------------------

    def _pre_update_hook(self, particle: Particle, t: int) -> bool:
        """Apply adaptive mutation with decreasing probability.

        Returns True when the particle is mutated (caller skips normal update).
        """
        mu_mf = 0.7 + 0.3 * (t / max(self.T, 1))  # ∈ (0.7, 1]
        xi = self._rng.uniform()
        if xi > mu_mf:
            # Mutate: reinitialise position and velocity randomly
            particle.position = random_position(self._rng)
            particle.velocity = random_velocity(self._rng)
            logger.debug("Particle %d mutated at iteration %d", particle.idx, t)
            return True  # Signal: skip standard velocity/position update
        return False

    # ---- Override _update_particle to thread t through -----------------

    def _update_particle_with_t(self, particle: Particle, w: float, t: int) -> None:
        """Velocity/position update with mutation check."""
        if self._pre_update_hook(particle, t):
            return

        r1 = self._rng.uniform(0.0, 1.0, size=5)
        r2 = self._rng.uniform(0.0, 1.0, size=5)

        cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
        social = self.c2 * r2 * (self._gbest_position - particle.position)
        particle.velocity = w * particle.velocity + cognitive + social
        particle.velocity = np.clip(particle.velocity, -self.v_clamp, self.v_clamp)
        particle.position = np.clip(particle.position + particle.velocity, LB, UB)

    # ---- Full run loop (overrides parent to pass t to hook) ---------------

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Dict[str, Any], float]:
        """Execute IPSO optimisation.

        Args:
            X_train: [N_train, T_max_lookback, F]
            y_train: [N_train]
            X_val:   [N_val, T_max_lookback, F]
            y_val:   [N_val]

        Returns:
            (best_params_dict, best_fitness)
        """
        self.fitness_fn.reset()
        self._initialise_swarm()
        logger.info(f"Initialized Swarm")
        self._evaluate_all(X_train, y_train, X_val, y_val, iteration=0)

        for t in range(1, self.T + 1):
            logger.info(f"Running {t} iteration")
            w = self._inertia(t)
            for particle in self._swarm:
                self._update_particle_with_t(particle, w, t)
            self._evaluate_all(X_train, y_train, X_val, y_val, iteration=t)

            self.fitness_history.append(self._gbest_fitness)
            self.diversity_history.append(self._swarm_diversity())

            logger.info(
                "[IPSO] iter %3d/%d | w=%.4f | gbest=%.6f | diversity=%.4f | %s",
                t,
                self.T,
                w,
                self._gbest_fitness,
                self.diversity_history[-1],
                self._gbest_params(),
            )

            if self.checkpoint_dir and t % 10 == 0:
                self._save_checkpoint(t)

        from .particle import decode

        best_params = decode(self._gbest_position)
        logger.info(
            "IPSO complete. Best params: %s  Fitness: %.6f",
            best_params,
            self._gbest_fitness,
        )
        return best_params, self._gbest_fitness
