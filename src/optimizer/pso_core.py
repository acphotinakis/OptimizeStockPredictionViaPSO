"""
src/optimizer/pso_core.py

Standard PSO base class (linear inertia, no mutation).
Used both as a standalone baseline and as the foundation for IPSO.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import numpy as np
from tqdm.auto import tqdm

from .particle import LB, UB, Particle
from .fitness import CompositeFitness

logger = logging.getLogger(__name__)


def _format_postfix(
    gbest_fitness: float, best_params: Dict[str, Any], diversity: float
) -> Dict[str, str]:
    """Compact tqdm postfix dict showing the swarm's current best.

    Keeps the bar narrow enough to fit in a typical terminal while
    surfacing the architecturally important hyperparameters.
    """
    if not np.isfinite(gbest_fitness):
        gbest_str = "+inf"
    elif abs(gbest_fitness) < 1e-2:
        gbest_str = f"{gbest_fitness:.4e}"
    else:
        gbest_str = f"{gbest_fitness:.4f}"

    return {
        "gbest": gbest_str,
        "u1": f"{int(best_params.get('units_1', 0))}",
        "u2": f"{int(best_params.get('units_2', 0))}",
        "drop": f"{float(best_params.get('dropout', 0.0)):.2f}",
        "lr": f"{float(best_params.get('learning_rate', 0.0)):.1e}",
        "bs": f"{int(best_params.get('batch_size', 0))}",
        "div": f"{diversity:.2f}",
    }


class StandardPSO:
    """Standard Particle Swarm Optimization with linear inertia decay.

    Minimises a fitness function F: R^5 --> R that requires
    building and evaluating an LSTM model for each particle.

    Args:
        n_particles: Swarm size M.
        n_iterations: Maximum iterations T.
        fitness_fn: CompositeFitness instance (callable).
        model_builder: Function that takes hyperparameter dict and
            (X_train, y_train, X_val, y_val) and returns y_pred.
        w_max: Maximum (initial) inertia weight.
        w_min: Minimum (final) inertia weight.
        c1: Cognitive coefficient.
        c2: Social coefficient.
        v_clamp_fraction: Velocity clamped to ±this × (UB−LB).
        seed: Random seed.
        checkpoint_dir: If set, saves checkpoints every 10 iterations.
        n_workers: Number of parallel processes for particle evaluation.
    """

    def __init__(
        self,
        n_particles: int = 30,
        n_iterations: int = 50,
        fitness_fn: Optional[CompositeFitness] = None,
        model_builder: Optional[Callable] = None,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        v_clamp_fraction: float = 0.20,
        seed: int = 42,
        checkpoint_dir: Optional[str | Path] = None,
        n_workers: int = 1,
    ) -> None:
        self.M = n_particles
        self.T = n_iterations
        self.fitness_fn = fitness_fn or CompositeFitness()
        self.model_builder = model_builder
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_clamp = v_clamp_fraction * (UB - LB)
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.n_workers = n_workers

        self._rng = np.random.default_rng(seed)
        self._swarm: List[Particle] = []
        self._gbest_position: np.ndarray = np.zeros(6)  # 6D search space
        self._gbest_fitness: float = float("inf")
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Dict[str, Any], float]:
        """Execute the PSO optimisation loop.

        Args:
            X_train: [N_train, T, F]
            y_train: [N_train]
            X_val:   [N_val, T, F]
            y_val:   [N_val]

        Returns:
            (best_params_dict, best_fitness)
        """
        self.fitness_fn.reset()
        self._initialise_swarm()

        # Single startup banner (replaces the per-iteration "Evaluating
        # swarms..." line that fired T+1 times and corrupted the tqdm bar).
        logger.info(
            "%s starting: n_particles=%d n_iterations=%d n_workers=%d",
            self.__class__.__name__,
            self.M,
            self.T,
            self.n_workers,
        )

        # Progress bar covers iteration 0 (initial random-position
        # evaluation, ~M particles trained from scratch) AND iterations
        # 1..T (velocity update + re-evaluation each step). Without
        # iteration 0 in the bar, the first 5-15 minutes show nothing.
        pbar = tqdm(
            range(0, self.T + 1),
            desc=f"{self.__class__.__name__}",
            total=self.T + 1,
            unit="iter",
            dynamic_ncols=True,
            leave=True,
            file=sys.stderr,
        )

        try:
            for t in pbar:
                if t == 0:
                    # Initial random-position evaluation
                    self._evaluate_all(X_train, y_train, X_val, y_val, iteration=0)
                else:
                    w = self._inertia(t)
                    for particle in self._swarm:
                        self._update_particle(particle, w, t)
                    self._evaluate_all(X_train, y_train, X_val, y_val, iteration=t)
                    self.fitness_history.append(self._gbest_fitness)
                    self.diversity_history.append(self._swarm_diversity())

                # Audit trail at DEBUG level so it ends up in the log file
                # (when setup_logger is at DEBUG) but doesn't corrupt the
                # tqdm bar on the console at INFO. The full per-iteration
                # history is also persisted to ``pso_phase1_results.yaml``
                # at the end of phase1_pso_search.
                logger.debug(
                    "[%s] iter %3d/%d | gbest=%.6f | diversity=%.4f | params=%s",
                    self.__class__.__name__,
                    t,
                    self.T,
                    self._gbest_fitness,
                    self._swarm_diversity(),
                    self._gbest_params(),
                )

                pbar.set_postfix(
                    _format_postfix(
                        self._gbest_fitness,
                        self._gbest_params(),
                        self._swarm_diversity(),
                    )
                )

                if self.checkpoint_dir and t > 0 and t % 10 == 0:
                    self._save_checkpoint(t)
        finally:
            pbar.close()

        from .particle import decode

        return decode(self._gbest_position), self._gbest_fitness

    # ------------------------------------------------------------------
    # Overridable hooks (used by IPSO subclass)
    # ------------------------------------------------------------------

    def _inertia(self, t: int) -> float:
        """Linear inertia weight decay."""
        return self.w_max - (self.w_max - self.w_min) * t / self.T

    def _pre_update_hook(self, particle: Particle, t: int) -> bool:
        """Return True to skip standard velocity/position update (mutation)."""
        return False

    # ------------------------------------------------------------------
    # Core mechanics
    # ------------------------------------------------------------------

    def _initialise_swarm(self) -> None:
        self._swarm = [Particle.random_init(self._rng, idx=i) for i in range(self.M)]
        self._gbest_position = self._swarm[0].position.copy()
        self._gbest_fitness = float("inf")

    def _update_particle(self, particle: Particle, w: float, t: int) -> None:
        """Apply velocity and position update equations (eqs. 1-2).

        Args:
            particle: The particle to update.
            w: Inertia weight for this iteration.
            t: Current iteration index (1-based), threaded into the
                pre-update hook so mutating subclasses can schedule
                operators against the iteration count.
        """
        # Allow subclass to mutate instead
        if self._pre_update_hook(particle, t):
            return

        r1 = self._rng.uniform(0.0, 1.0, size=6)  # 6D search space
        r2 = self._rng.uniform(0.0, 1.0, size=6)  # 6D search space

        cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
        social = self.c2 * r2 * (self._gbest_position - particle.position)
        particle.velocity = w * particle.velocity + cognitive + social

        # Velocity clamping
        particle.velocity = np.clip(particle.velocity, -self.v_clamp, self.v_clamp)

        # Position update + boundary enforcement. Per-dim velocity components
        # that drove the particle past the wall are zeroed so that subsequent
        # iterations do not stagnate against the boundary.
        proposed = particle.position + particle.velocity
        clipped = np.clip(proposed, LB, UB)
        particle.velocity = np.where(proposed != clipped, 0.0, particle.velocity)
        particle.position = clipped

    def _evaluate_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        iteration: int,
    ) -> None:
        """Evaluate every particle and update pbest / gbest.

        Per-particle / per-iteration chatter is demoted to DEBUG so the
        parent's tqdm progress bar stays the only console UI. The PSO
        startup banner and the per-iteration audit-trail line in
        ``run()`` remain at INFO so the log file still tells the full
        story.
        """
        logger.debug("Evaluating swarm at iteration %d", iteration)
        if self.n_workers > 1:
            self._evaluate_parallel(X_train, y_train, X_val, y_val)
        else:
            for particle in self._swarm:
                fitness = self._evaluate_particle(
                    particle, X_train, y_train, X_val, y_val
                )
                self._update_bests(particle, fitness)

    def _evaluate_particle(
        self,
        particle: Particle,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Build model, train, predict on val, return composite fitness.

        Failed evaluations (OOM, NaN loss, divergent training) are logged and
        return ``+inf`` so the swarm can continue without losing the entire run.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            params = particle.decode()
            lookback = params["lookback"]

            if X_train.shape[1] < lookback:
                raise ValueError(
                    f"X_train lookback {X_train.shape[1]} < required {lookback}"
                )

            X_tr = X_train[:, :lookback, :]
            X_vl = X_val[:, :lookback, :]

            if self.model_builder is None:
                raise RuntimeError("model_builder must be set before calling run().")

            result = self.model_builder(params, X_tr, y_train, X_vl, y_val)

            if isinstance(result, tuple):
                y_pred, model = result
                fitness = self.fitness_fn(y_val, y_pred, model)
            else:
                y_pred = result
                fitness = self.fitness_fn(y_val, y_pred, None)

            if not np.isfinite(fitness):
                logger.warning(
                    "Particle %d returned non-finite fitness (%s); using +inf",
                    particle.idx,
                    fitness,
                )
                return float("inf")

            return fitness
        except Exception as exc:
            logger.warning(
                "Particle %d evaluation failed: %s",
                particle.idx,
                exc,
            )
            return float("inf")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _evaluate_parallel(self, X_train, y_train, X_val, y_val) -> None:
        """Evaluate particles in parallel using ProcessPoolExecutor.

        ``model_builder`` and ``fitness_fn`` must be picklable. A common
        regression is a nested-closure ``model_builder`` which fails
        ``pickle`` - historically this fell back to sequential silently;
        the fallback now surfaces at WARNING level so users can spot it.
        """
        from concurrent.futures import ProcessPoolExecutor

        try:
            with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
                futures = [
                    ex.submit(
                        self._evaluate_particle, p, X_train, y_train, X_val, y_val
                    )
                    for p in self._swarm
                ]
                for particle, future in zip(self._swarm, futures):
                    fitness = future.result()
                    self._update_bests(particle, fitness)
        except Exception as e:
            logger.warning(
                "Parallel evaluation failed (%s: %s); falling back to "
                "sequential. Common cause: model_builder is a nested closure "
                "and is not picklable. Use functools.partial of a "
                "module-level function instead.",
                type(e).__name__,
                e,
            )
            for particle in self._swarm:
                fitness = self._evaluate_particle(
                    particle, X_train, y_train, X_val, y_val
                )
                self._update_bests(particle, fitness)

    def _update_bests(self, particle: Particle, fitness: float) -> None:
        particle.fitness = fitness
        if fitness < particle.pbest_fitness:
            particle.pbest_fitness = fitness
            particle.pbest_position = particle.position.copy()
        if fitness < self._gbest_fitness:
            self._gbest_fitness = fitness
            self._gbest_position = particle.position.copy()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _swarm_diversity(self) -> float:
        positions = np.stack([p.position for p in self._swarm])
        centroid = positions.mean(axis=0)
        return float(np.mean(np.linalg.norm(positions - centroid, axis=1)))

    def _gbest_params(self) -> Dict[str, Any]:
        from .particle import decode

        return decode(self._gbest_position)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, iteration: int) -> None:
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "iteration": iteration,
            "gbest_fitness": self._gbest_fitness,
            "gbest_position": self._gbest_position.tolist(),
            "fitness_history": self.fitness_history,
            "diversity_history": self.diversity_history,
        }
        path = self.checkpoint_dir / f"checkpoint_iter_{iteration:04d}.json"
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("Checkpoint saved: %s", path)
