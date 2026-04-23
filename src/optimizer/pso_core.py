"""
src/optimizer/pso_core.py

Standard PSO base class (linear inertia, no mutation).
Used both as a standalone baseline and as the foundation for IPSO.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import numpy as np

from .particle import LB, UB, Particle
from .fitness import CompositeFitness

logger = logging.getLogger(__name__)


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
        self._evaluate_all(X_train, y_train, X_val, y_val, iteration=0)

        for t in range(1, self.T + 1):
            w = self._inertia(t)
            for particle in self._swarm:
                self._update_particle(particle, w)
            self._evaluate_all(X_train, y_train, X_val, y_val, iteration=t)

            self.fitness_history.append(self._gbest_fitness)
            self.diversity_history.append(self._swarm_diversity())

            logger.info(
                "[%s] iter %3d/%d | gbest=%.6f | diversity=%.4f | params=%s",
                self.__class__.__name__,
                t,
                self.T,
                self._gbest_fitness,
                self.diversity_history[-1],
                self._gbest_params(),
            )

            if self.checkpoint_dir and t % 10 == 0:
                self._save_checkpoint(t)

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

    def _update_particle(self, particle: Particle, w: float) -> None:
        """Apply velocity and position update equations (eqs. 1–2)."""
        # Allow subclass to mutate instead
        if self._pre_update_hook(particle, 0):
            return

        r1 = self._rng.uniform(0.0, 1.0, size=6)  # 6D search space
        r2 = self._rng.uniform(0.0, 1.0, size=6)  # 6D search space

        cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
        social = self.c2 * r2 * (self._gbest_position - particle.position)
        particle.velocity = w * particle.velocity + cognitive + social

        # Velocity clamping
        particle.velocity = np.clip(particle.velocity, -self.v_clamp, self.v_clamp)

        # Position update + boundary enforcement
        particle.position = np.clip(particle.position + particle.velocity, LB, UB)

    def _evaluate_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        iteration: int,
    ) -> None:
        """Evaluate every particle and update pbest / gbest."""
        logger.info(f"Evaludating swarms...")
        logger.info(f"number of works = {self.n_workers}")
        if self.n_workers > 1:
            self._evaluate_parallel(X_train, y_train, X_val, y_val)
        else:
            for particle in self._swarm:
                fitness = self._evaluate_particle(
                    particle, X_train, y_train, X_val, y_val
                )
                logger.info(f"Updating weights")
                self._update_bests(particle, fitness)

    def _evaluate_particle(
        self,
        particle: Particle,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Build model, train, predict on val, return composite fitness."""
        # Phase 3: Clear GPU cache before each particle evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        params = particle.decode()
        lookback = params["lookback"]  # Always 20 (fixed)

        # Validate lookback matches data
        if X_train.shape[1] < lookback:
            raise ValueError(f"X_train lookback {X_train.shape[1]} < required {lookback}")

        # Slice windows to the required lookback length (should be 20)
        X_tr = X_train[:, :lookback, :]
        X_vl = X_val[:, :lookback, :]

        if self.model_builder is None:
            raise RuntimeError("model_builder must be set before calling run().")

        # model_builder must now return (y_pred, model) tuple for MSW computation
        result = self.model_builder(params, X_tr, y_train, X_vl, y_val)
        
        if isinstance(result, tuple):
            y_pred, model = result
            fitness = self.fitness_fn(y_val, y_pred, model)
        else:
            # Backward compatibility: if only predictions returned
            y_pred = result
            fitness = self.fitness_fn(y_val, y_pred, None)

        # Phase 3: Clear GPU cache after evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return fitness

    def _evaluate_parallel(self, X_train, y_train, X_val, y_val) -> None:
        """Evaluate particles in parallel using ProcessPoolExecutor."""
        from concurrent.futures import ProcessPoolExecutor

        # Note: model_builder must be picklable for multiprocessing.
        # Fall back to sequential if it is not.
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
            logger.info(
                "Parallel evaluation failed (%s); falling back to sequential.", e
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
