import logging
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from pathlib import Path
import json

# Internal imports from our project structure
from src.models.lstm import LSTMModel
from src.training.train import run_training

logger = logging.getLogger(__name__)


class Particle:
    """
    Represents a single 'agent' in the swarm searching for optimal LSTM settings.
    Each particle tracks its own position (hyperparameters), velocity, and personal best.
    """

    def __init__(self, search_space):
        self.position = {}
        self.velocity = {}
        self.pbest_position = {}
        self.pbest_fitness = -float("inf")
        self.fitness = -float("inf")

        # Initialize random position and velocity within search space bounds
        for param, config in search_space.items():
            if config.type == "int":
                self.position[param] = np.random.randint(
                    config.range[0], config.range[1] + 1
                )
                self.velocity[param] = np.random.uniform(-1, 1)
            elif config.type == "float":
                self.position[param] = np.random.uniform(
                    config.range[0], config.range[1]
                )
                self.velocity[param] = np.random.uniform(-0.1, 0.1)
            elif config.type == "categorical":
                self.position[param] = np.random.choice(config.values)
                self.velocity[param] = np.random.uniform(-1, 1)

        self.pbest_position = deepcopy(self.position)


class ImprovedPSO:
    """
    Orchestrates the 'Improved PSO' algorithm to tune LSTM hyperparameters.
    Implements non-linear inertia and mutation to enhance global search capability.
    """

    def __init__(self, cfg, search_space, train_loader, val_loader, device):
        self.cfg = cfg
        self.search_space = search_space
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Swarm parameters from config
        self.n_particles = cfg.optimization.swarm.particles
        self.iterations = cfg.optimization.swarm.iterations
        self.c1 = cfg.optimization.swarm.cognitive_coeff
        self.c2 = cfg.optimization.swarm.social_coeff

        # Global Best tracking
        self.gbest_position = {}
        self.gbest_fitness = -float("inf")

        # Initialize swarm
        self.swarm = [Particle(search_space) for _ in range(self.n_particles)]
        logger.info(f"Swarm initialized with {self.n_particles} particles.")

    def _calculate_inertia(self, iter_idx):
        """Calculates non-linear inertia weight to balance exploration and exploitation."""
        w_max = self.cfg.optimization.swarm.inertia_weight
        w_min = 0.4
        # Non-linear decay formula
        return w_min + (w_max - w_min) * np.exp(-iter_idx / self.iterations)

    def _apply_mutation(self, particle):
        """Adaptive mutation to maintain swarm diversity and escape local optima."""
        if np.random.rand() < self.cfg.optimization.improvements.mutation_factor:
            param_to_mutate = np.random.choice(list(self.search_space.keys()))
            config = self.search_space[param_to_mutate]

            if config.type == "int":
                particle.position[param_to_mutate] = np.random.randint(
                    config.range[0], config.range[1] + 1
                )
            elif config.type == "float":
                particle.position[param_to_mutate] = np.random.uniform(
                    config.range[0], config.range[1]
                )
            elif config.type == "categorical":
                particle.position[param_to_mutate] = np.random.choice(config.values)

            logger.debug(f"Particle mutated: {param_to_mutate} reset.")

    def evaluate_fitness(self, particle):
        """
        Calculates fitness by training an LSTM with the particle's parameters.
        Primary metric: Sharpe Ratio on the validation set.
        
        Returns:
            float: Fitness score (higher is better). Uses Sharpe ratio if configured,
                   otherwise uses negative validation loss.
        """
        # 1. Instantiate LSTM with current particle hyperparameters
        model = LSTMModel(
            input_size=self.cfg.model.architecture.input_size,
            hidden_size=int(particle.position["hidden_size"]),
            num_layers=int(particle.position["num_layers"]),
            dropout=particle.position["dropout"],
        ).to(self.device)

        # 2. Update training config with particle values (e.g., learning rate)
        current_cfg = deepcopy(self.cfg)
        current_cfg.model.training.learning_rate = particle.position["learning_rate"]

        # 3. Run training loop and get best validation loss
        model, val_loss = run_training(
            model, self.train_loader, self.val_loader, current_cfg, self.device
        )

        # 4. Calculate fitness based on configured metric
        fitness_metric = self.cfg.optimization.fitness.primary_metric
        
        if fitness_metric == "sharpe_ratio":
            # Generate predictions on validation set to calculate Sharpe ratio
            model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_x, batch_y in self.val_loader:
                    batch_x = batch_x.to(self.device)
                    outputs = model(batch_x)
                    predictions.extend(outputs.cpu().numpy().flatten())
                    actuals.extend(batch_y.numpy().flatten())
            
            # Calculate Sharpe ratio
            from src.evaluation.metrics import calculate_sharpe_ratio
            
            # Strategy returns = sign(prediction) * actual_return
            strategy_returns = np.sign(predictions) * np.array(actuals)
            fitness = calculate_sharpe_ratio(strategy_returns)
            
            # Apply risk penalty if configured
            if hasattr(self.cfg.optimization.fitness, 'risk_penalty'):
                from src.evaluation.metrics import calculate_max_drawdown
                cumulative_returns = (1 + strategy_returns).cumprod()
                mdd = abs(calculate_max_drawdown(cumulative_returns))
                fitness -= self.cfg.optimization.fitness.risk_penalty * mdd
            
            logger.info(f"Particle fitness (Sharpe): {fitness:.4f}")
            return fitness
        else:
            # Fallback: use negative validation loss
            logger.warning(f"Using validation loss as fitness (metric '{fitness_metric}' not implemented)")
            return -val_loss

    def search(self):
        """Main optimization loop."""
        for i in range(self.iterations):
            w = self._calculate_inertia(i)
            logger.info(f"Iteration {i+1}/{self.iterations} | Inertia: {w:.4f}")

            for particle in self.swarm:
                # 1. Evaluate Fitness
                particle.fitness = self.evaluate_fitness(particle)

                # 2. Update Personal Best
                if particle.fitness > particle.pbest_fitness:
                    particle.pbest_fitness = particle.fitness
                    particle.pbest_position = deepcopy(particle.position)

                # 3. Update Global Best
                if particle.fitness > self.gbest_fitness:
                    self.gbest_fitness = particle.fitness
                    self.gbest_position = deepcopy(particle.position)
                    logger.info(
                        f"New Global Best Found: Fitness {self.gbest_fitness:.6f}"
                    )

            # 4. Update Velocities and Positions
            for particle in self.swarm:
                for param, config in self.search_space.items():
                    r1, r2 = np.random.rand(), np.random.rand()

                    # Velocity update rule
                    # Categorical parameters use simplified logic or discrete mapping
                    if config.type != "categorical":
                        v_new = (
                            w * particle.velocity[param]
                            + self.c1
                            * r1
                            * (
                                particle.pbest_position[param]
                                - particle.position[param]
                            )
                            + self.c2
                            * r2
                            * (self.gbest_position[param] - particle.position[param])
                        )

                        particle.velocity[param] = v_new
                        particle.position[param] += v_new

                        # Boundary enforcement
                        particle.position[param] = np.clip(
                            particle.position[param], config.range[0], config.range[1]
                        )

                # 5. Improved PSO: Mutation step
                self._apply_mutation(particle)

        self.save_results()
        return self.gbest_position

    def save_results(self):
        """Saves the Global Best configuration to disk for reproduction ."""
        save_path = Path(self.cfg.paths.data_storage.processed).parent / "pso_best"
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "gbest_config.json", "w") as f:
            json.dump(
                {"fitness": self.gbest_fitness, "params": self.gbest_position},
                f,
                indent=4,
            )
        logger.info(f"Optimization complete. Global best saved to {save_path}")
