"""
IPSO (Improved Particle Swarm Optimization) Algorithm

Implements IPSO for LSTM hyperparameter optimization as defined in FINAL_PLAN.md Section 4.2.

IPSO Features:
- Adaptive tanh inertia weight schedule
- Adaptive mutation
- Velocity clamping
- Nonlinear parameter decay

References:
- Ji et al. (2021): IPSO for LSTM hyperparameter tuning
- FINAL_PLAN.md Section 4.2: PSO Phase 1

Author: System Architect
Version: CANONICAL 1.0
"""

import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class IPSOOptimizer:
    """
    Improved Particle Swarm Optimization for LSTM hyperparameter search.

    FINAL_PLAN.md Section 4.2: PSO Phase 1 - Hyperparameter Search

    This optimizer finds optimal LSTM hyperparameters by:
    1. Initializing swarm in search space
    2. Evaluating fitness (MSE on validation set)
    3. Updating particle positions using IPSO rules
    4. Returning global best hyperparameters
    """

    def __init__(
        self,
        n_particles: int,
        n_iterations: int,
        search_space: Dict[str, Dict],
        fitness_func: Callable,
        inertia_min: float = 0.4,
        inertia_max: float = 0.9,
        c1: float = 1.5,
        c2: float = 1.5,
        v_clamp_fraction: float = 0.20,
        seed: int = 42,
    ):
        """
        Initialize IPSO optimizer.

        Args:
            n_particles: Number of particles in swarm
            n_iterations: Maximum iterations
            search_space: Dictionary defining parameter bounds
            fitness_func: Function(params) -> fitness (lower is better)
            inertia_min: Minimum inertia weight
            inertia_max: Maximum inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
            v_clamp_fraction: Velocity clamping fraction
            seed: Random seed
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.search_space = search_space
        self.fitness_func = fitness_func
        self.inertia_min = inertia_min
        self.inertia_max = inertia_max
        self.c1 = c1
        self.c2 = c2
        self.v_clamp_fraction = v_clamp_fraction
        self.seed = seed

        np.random.seed(seed)

        # Initialize particles
        self.positions = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_fitness = None
        self.gbest_position = None
        self.gbest_fitness = np.inf

        # History
        self.fitness_history = []

        logger.info("IPSO Optimizer initialized")
        logger.info(f"  Particles: {n_particles}")
        logger.info(f"  Iterations: {n_iterations}")
        logger.info(f"  Inertia: [{inertia_min}, {inertia_max}] (adaptive tanh)")
        logger.info(f"  C1 (cognitive): {c1}")
        logger.info(f"  C2 (social): {c2}")
        logger.info(f"  Velocity clamp: {v_clamp_fraction}")

    def _initialize_swarm(self) -> None:
        """Initialize particle positions and velocities."""
        logger.info("Initializing swarm...")

        # Get dimensionality
        param_names = list(self.search_space.keys())
        n_dims = len(param_names)

        # Initialize positions
        self.positions = np.zeros((self.n_particles, n_dims))

        for i, param_name in enumerate(param_names):
            param_spec = self.search_space[param_name]

            if "choices" in param_spec:
                # Discrete parameter
                choices = param_spec["choices"]
                self.positions[:, i] = np.random.choice(choices, self.n_particles)
            else:
                # Continuous parameter
                min_val = param_spec["min"]
                max_val = param_spec["max"]
                self.positions[:, i] = np.random.uniform(
                    min_val, max_val, self.n_particles
                )

        # Initialize velocities
        velocity_ranges = []
        for param_name in param_names:
            param_spec = self.search_space[param_name]
            if "choices" in param_spec:
                velocity_ranges.append(1.0)  # Nominal range for discrete
            else:
                param_range = param_spec["max"] - param_spec["min"]
                velocity_ranges.append(param_range * self.v_clamp_fraction)

        velocity_ranges = np.array(velocity_ranges)
        self.velocities = np.random.uniform(
            -velocity_ranges, velocity_ranges, (self.n_particles, n_dims)
        )

        # Initialize personal best
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.full(self.n_particles, np.inf)

        logger.info(
            f"Swarm initialized with {self.n_particles} particles in {n_dims}D space"
        )

    def _decode_position(self, position: np.ndarray) -> Dict[str, Any]:
        """
        Decode particle position to hyperparameter dict.

        Args:
            position: Position vector

        Returns:
            Dictionary of hyperparameters
        """
        params = {}
        param_names = list(self.search_space.keys())

        for i, param_name in enumerate(param_names):
            param_spec = self.search_space[param_name]
            value = position[i]

            if "choices" in param_spec:
                # Discrete parameter - round to nearest choice
                choices = param_spec["choices"]
                idx = int(
                    np.clip(
                        np.round(value / max(choices) * (len(choices) - 1)),
                        0,
                        len(choices) - 1,
                    )
                )
                params[param_name] = choices[idx]
            elif param_spec.get("scale") == "log":
                # Log scale parameter
                params[param_name] = float(value)
            else:
                # Continuous parameter - round if integer type expected
                if param_name in ["epochs", "units_1", "units_2"]:
                    params[param_name] = int(
                        np.clip(value, param_spec["min"], param_spec["max"])
                    )
                else:
                    params[param_name] = float(
                        np.clip(value, param_spec["min"], param_spec["max"])
                    )

        return params

    def _compute_inertia(self, iteration: int) -> float:
        """
        Compute adaptive inertia weight using tanh schedule.

        IPSO uses nonlinear inertia decay for better exploration/exploitation balance.

        Args:
            iteration: Current iteration

        Returns:
            Inertia weight
        """
        # Tanh schedule: starts high (exploration), decreases (exploitation)
        progress = iteration / self.n_iterations
        w = self.inertia_max - (self.inertia_max - self.inertia_min) * np.tanh(
            2 * progress
        )
        return w

    def _apply_mutation(self, iteration: int) -> None:
        """
        Apply adaptive mutation to prevent premature convergence.

        Args:
            iteration: Current iteration
        """
        # Mutation probability decreases over time
        mutation_prob = 0.1 * (1 - iteration / self.n_iterations)

        if np.random.rand() < mutation_prob:
            # Select random particle
            particle_idx = np.random.randint(self.n_particles)
            dim_idx = np.random.randint(len(self.search_space))

            # Reinitialize that dimension
            param_names = list(self.search_space.keys())
            param_name = param_names[dim_idx]
            param_spec = self.search_space[param_name]

            if "choices" in param_spec:
                choices = param_spec["choices"]
                self.positions[particle_idx, dim_idx] = np.random.choice(choices)
            else:
                min_val = param_spec["min"]
                max_val = param_spec["max"]
                self.positions[particle_idx, dim_idx] = np.random.uniform(
                    min_val, max_val
                )

            logger.info(
                f"Mutation applied to particle {particle_idx}, dimension {dim_idx}"
            )

    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """
        Run IPSO optimization.

        Returns:
            Tuple of (best_hyperparameters, best_fitness)
        """
        logger.info("=" * 80)
        logger.info("STARTING IPSO OPTIMIZATION")
        logger.info("=" * 80)

        # Initialize swarm
        self._initialize_swarm()

        # Optimization loop
        for iteration in range(self.n_iterations):
            logger.info(f"\nIteration {iteration + 1}/{self.n_iterations}")

            # Compute inertia for this iteration
            w = self._compute_inertia(iteration)
            logger.info(f"  Inertia weight: {w:.4f}")

            # Evaluate all particles
            for p in range(self.n_particles):
                # Decode position to hyperparameters
                params = self._decode_position(self.positions[p])

                # Evaluate fitness
                try:
                    fitness = self.fitness_func(params)
                except Exception as e:
                    logger.warning(f"  Particle {p} evaluation failed: {e}")
                    fitness = np.inf

                # Update personal best
                if fitness < self.pbest_fitness[p]:
                    self.pbest_fitness[p] = fitness
                    self.pbest_positions[p] = self.positions[p].copy()

                # Update global best
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.positions[p].copy()
                    logger.info(f"  ★ New global best: {fitness:.6f}")

            # Log iteration summary
            avg_fitness = np.mean(self.pbest_fitness[self.pbest_fitness < np.inf])
            logger.info(f"  Global best fitness: {self.gbest_fitness:.6f}")
            logger.info(f"  Average fitness: {avg_fitness:.6f}")

            self.fitness_history.append(
                {
                    "iteration": iteration + 1,
                    "gbest_fitness": float(self.gbest_fitness),
                    "avg_fitness": float(avg_fitness),
                }
            )

            # Update velocities and positions
            for p in range(self.n_particles):
                # Random coefficients
                r1 = np.random.rand(len(self.search_space))
                r2 = np.random.rand(len(self.search_space))

                # Velocity update (IPSO)
                cognitive = self.c1 * r1 * (self.pbest_positions[p] - self.positions[p])
                social = self.c2 * r2 * (self.gbest_position - self.positions[p])
                self.velocities[p] = w * self.velocities[p] + cognitive + social

                # Clamp velocity
                param_names = list(self.search_space.keys())
                for i, param_name in enumerate(param_names):
                    param_spec = self.search_space[param_name]
                    if "choices" in param_spec:
                        v_max = 1.0
                    else:
                        param_range = param_spec["max"] - param_spec["min"]
                        v_max = param_range * self.v_clamp_fraction

                    self.velocities[p, i] = np.clip(
                        self.velocities[p, i], -v_max, v_max
                    )

                # Position update
                self.positions[p] += self.velocities[p]

                # Enforce bounds
                for i, param_name in enumerate(param_names):
                    param_spec = self.search_space[param_name]
                    if "choices" not in param_spec:
                        self.positions[p, i] = np.clip(
                            self.positions[p, i], param_spec["min"], param_spec["max"]
                        )

            # Apply mutation
            self._apply_mutation(iteration)

        # Decode best parameters
        best_params = self._decode_position(self.gbest_position)

        logger.info("=" * 80)
        logger.info("IPSO OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best fitness: {self.gbest_fitness:.6f}")
        logger.info("Best hyperparameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)

        return best_params, self.gbest_fitness
