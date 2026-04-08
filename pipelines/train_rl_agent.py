#!/usr/bin/env python3
"""
scripts/train_rl_agent.py

Train RL agent for trading using PPO.

Usage:
    python scripts/train_rl_agent.py --ticker SPY --episodes 500
    python scripts/train_rl_agent.py --ticker AAPL --episodes 1000 --eval-freq 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.rl.trading_env import TradingEnv
from src.models.rl.ppo_agent import PPOAgent
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


class RLTrainer:
    """RL training manager with logging and visualization."""

    def __init__(
        self,
        ticker: str,
        features_dir: Path,
        results_dir: Path,
        config_path: str,
        seed: int = 42,
    ):
        self.ticker = ticker
        self.features_dir = features_dir
        self.results_dir = results_dir
        self.seed = seed

        # Load config
        self.cfg = load_config(config_path)

        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create results directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        # Training history
        self.history = {
            "episode": [],
            "train_reward": [],
            "train_return": [],
            "train_sharpe": [],
            "train_max_dd": [],
            "train_n_trades": [],
            "val_reward": [],
            "val_return": [],
            "val_sharpe": [],
            "val_max_dd": [],
            "val_n_trades": [],
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "kl_divergence": [],
        }

        logger.info(f"RLTrainer initialized for {ticker}")

    def load_data(self) -> Dict:
        """Load training and validation data."""
        ticker_dir = self.features_dir / self.ticker

        logger.info(f"Loading data from {ticker_dir}")

        # Load features
        X_train = np.load(ticker_dir / "X_train.npy")
        y_train = np.load(ticker_dir / "y_train.npy")
        X_val = np.load(ticker_dir / "X_val.npy")
        y_val = np.load(ticker_dir / "y_val.npy")

        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

        # Load OHLCV for prices
        try:
            ohlcv = pd.read_parquet(f"data/processed/{self.ticker}.parquet")
            prices_train = ohlcv["close"].values[: len(X_train)]
            prices_val = ohlcv["close"].values[len(X_train) : len(X_train) + len(X_val)]
            logger.info("Loaded OHLCV data")
        except Exception as e:
            logger.warning(f"Could not load OHLCV: {e}. Using dummy prices.")
            prices_train = np.ones(len(X_train))
            prices_val = np.ones(len(X_val))

        return {
            "X_train": X_train,
            "y_train": y_train,
            "prices_train": prices_train,
            "X_val": X_val,
            "y_val": y_val,
            "prices_val": prices_val,
        }

    def create_envs(self, data: Dict) -> tuple:
        """Create training and validation environments."""
        train_env = TradingEnv(
            features=data["X_train"],
            returns=data["y_train"],
            prices=data["prices_train"],
            initial_capital=100_000.0,
            transaction_cost=0.001,
            max_position=1.0,
            drawdown_penalty=0.1,
            hold_penalty=0.0,
        )

        val_env = TradingEnv(
            features=data["X_val"],
            returns=data["y_val"],
            prices=data["prices_val"],
            initial_capital=100_000.0,
            transaction_cost=0.001,
            max_position=1.0,
            drawdown_penalty=0.1,
            hold_penalty=0.0,
        )

        logger.info(
            f"Environments created: train={train_env.n_steps} steps, val={val_env.n_steps} steps"
        )

        return train_env, val_env

    def train_episode(self, env: TradingEnv, agent: PPOAgent) -> Dict:
        """Run one training episode."""
        state = env.reset()
        episode_reward = 0.0

        states, actions, log_probs, rewards, dones = [], [], [], [], []

        # Collect trajectory
        done = False
        while not done:
            action, log_prob = agent.select_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            episode_reward += reward

        # Compute values
        with torch.no_grad():
            values = []
            for s in states:
                s_tensor = torch.FloatTensor(s).unsqueeze(0).to(agent.device)
                _, _, v = agent.policy.forward(s_tensor)
                values.append(v.item())

        # Compute advantages and returns
        advantages, returns = agent.compute_gae(rewards, values, dones)

        # Update policy
        update_metrics = agent.update(
            np.array(states),
            np.array(actions),
            np.array(log_probs),
            np.array(returns),
            np.array(advantages),
            epochs=10,
            batch_size=64,
        )

        # Get episode stats
        episode_stats = env.get_episode_stats()
        episode_stats["episode_reward"] = episode_reward
        episode_stats.update(update_metrics)

        return episode_stats

    def evaluate(self, env: TradingEnv, agent: PPOAgent) -> Dict:
        """Evaluate agent on validation set."""
        state = env.reset()
        episode_reward = 0.0

        done = False
        while not done:
            action, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)

            state = next_state
            episode_reward += reward

        # Get episode stats
        episode_stats = env.get_episode_stats()
        episode_stats["episode_reward"] = episode_reward

        return episode_stats

    def train(
        self,
        n_episodes: int = 500,
        eval_freq: int = 50,
        save_freq: int = 100,
    ):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info(f"Starting RL training for {self.ticker}")
        logger.info(f"Episodes: {n_episodes}, Eval freq: {eval_freq}")
        logger.info("=" * 60)

        # Load data
        data = self.load_data()

        # Create environments
        train_env, val_env = self.create_envs(data)

        # Create agent
        state_dim = train_env.observation_space.shape[0]
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=1,
            hidden_dims=[256, 128, 64],
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
        )

        # Training loop
        best_val_sharpe = -np.inf
        start_time = time.time()

        pbar = tqdm(range(1, n_episodes + 1), desc=f"Training {self.ticker}")

        for episode in pbar:
            # Train
            train_stats = self.train_episode(train_env, agent)

            # Log training stats
            self.history["episode"].append(episode)
            self.history["train_reward"].append(train_stats["episode_reward"])
            self.history["train_return"].append(train_stats["total_return"])
            self.history["train_sharpe"].append(train_stats["sharpe_ratio"])
            self.history["train_max_dd"].append(train_stats["max_drawdown"])
            self.history["train_n_trades"].append(train_stats["n_trades"])
            self.history["actor_loss"].append(train_stats["actor_loss"])
            self.history["critic_loss"].append(train_stats["critic_loss"])
            self.history["entropy"].append(train_stats["entropy"])
            self.history["kl_divergence"].append(train_stats["kl_divergence"])

            # Evaluate
            if episode % eval_freq == 0:
                val_stats = self.evaluate(val_env, agent)

                self.history["val_reward"].append(val_stats["episode_reward"])
                self.history["val_return"].append(val_stats["total_return"])
                self.history["val_sharpe"].append(val_stats["sharpe_ratio"])
                self.history["val_max_dd"].append(val_stats["max_drawdown"])
                self.history["val_n_trades"].append(val_stats["n_trades"])

                # Log
                elapsed = time.time() - start_time
                logger.info(
                    f"Episode {episode}/{n_episodes} ({elapsed:.1f}s) | "
                    f"Train: R={train_stats['episode_reward']:.3f}, "
                    f"Return={train_stats['total_return']:+.2%}, "
                    f"Sharpe={train_stats['sharpe_ratio']:.3f} | "
                    f"Val: R={val_stats['episode_reward']:.3f}, "
                    f"Return={val_stats['total_return']:+.2%}, "
                    f"Sharpe={val_stats['sharpe_ratio']:.3f}"
                )

                # Save best model
                if val_stats["sharpe_ratio"] > best_val_sharpe:
                    best_val_sharpe = val_stats["sharpe_ratio"]
                    agent.save(
                        self.checkpoints_dir / f"rl_agent_{self.ticker}_best.pth"
                    )
                    logger.info(
                        f"✓ New best model saved (Sharpe={best_val_sharpe:.3f})"
                    )
            else:
                # Pad validation history
                self.history["val_reward"].append(np.nan)
                self.history["val_return"].append(np.nan)
                self.history["val_sharpe"].append(np.nan)
                self.history["val_max_dd"].append(np.nan)
                self.history["val_n_trades"].append(np.nan)

                # Log every 10 episodes
                if episode % 10 == 0:
                    logger.info(
                        f"Episode {episode}/{n_episodes} | "
                        f"Train: R={train_stats['episode_reward']:.3f}, "
                        f"Return={train_stats['total_return']:+.2%}, "
                        f"Sharpe={train_stats['sharpe_ratio']:.3f}"
                    )

            # Save checkpoint
            if episode % save_freq == 0:
                agent.save(
                    self.checkpoints_dir / f"rl_agent_{self.ticker}_ep{episode}.pth"
                )
            # --- UPDATE PROGRESS BAR ---
            postfix = {
                "R": f"{train_stats['episode_reward']:.2f}",
                "Ret": f"{train_stats['total_return']:+.2%}",
                "Sharpe": f"{train_stats['sharpe_ratio']:.2f}",
            }
            
            if val_stats is not None:
                postfix["ValSharpe"] = f"{val_stats['sharpe_ratio']:.2f}"
            
            pbar.set_postfix(postfix)
            
        # Save final model
        agent.save(self.checkpoints_dir / f"rl_agent_{self.ticker}_final.pth")

        # Save history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(
            self.results_dir / f"rl_training_history_{self.ticker}.csv", index=False
        )

        logger.info("=" * 60)
        logger.info("✓ Training complete!")
        logger.info(f"Best validation Sharpe: {best_val_sharpe:.3f}")
        logger.info("=" * 60)

        # Generate plots
        self.plot_training_curves()
        self.plot_final_evaluation(val_env, agent)

    def plot_training_curves(self):
        """Plot training curves."""
        logger.info("Generating training curve plots...")

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(
            f"{self.ticker} RL Training Curves", fontsize=16, fontweight="bold"
        )

        episodes = self.history["episode"]

        # Row 1: Rewards and Returns
        axes[0, 0].plot(
            episodes, self.history["train_reward"], label="Train", alpha=0.7
        )
        val_episodes = [
            e for e, v in zip(episodes, self.history["val_reward"]) if not np.isnan(v)
        ]
        val_rewards = [v for v in self.history["val_reward"] if not np.isnan(v)]
        axes[0, 0].plot(
            val_episodes, val_rewards, label="Val", marker="o", markersize=4
        )
        axes[0, 0].set_title("Episode Reward")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(
            episodes,
            [r * 100 for r in self.history["train_return"]],
            label="Train",
            alpha=0.7,
        )
        val_returns = [v * 100 for v in self.history["val_return"] if not np.isnan(v)]
        axes[0, 1].plot(
            val_episodes, val_returns, label="Val", marker="o", markersize=4
        )
        axes[0, 1].set_title("Total Return")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Return (%)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Row 2: Sharpe and Drawdown
        axes[1, 0].plot(
            episodes, self.history["train_sharpe"], label="Train", alpha=0.7
        )
        val_sharpes = [v for v in self.history["val_sharpe"] if not np.isnan(v)]
        axes[1, 0].plot(
            val_episodes, val_sharpes, label="Val", marker="o", markersize=4
        )
        axes[1, 0].axhline(
            y=1.0, color="g", linestyle="--", alpha=0.5, label="Target (1.0)"
        )
        axes[1, 0].set_title("Sharpe Ratio")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Sharpe Ratio")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(
            episodes,
            [d * 100 for d in self.history["train_max_dd"]],
            label="Train",
            alpha=0.7,
        )
        val_dds = [v * 100 for v in self.history["val_max_dd"] if not np.isnan(v)]
        axes[1, 1].plot(val_episodes, val_dds, label="Val", marker="o", markersize=4)
        axes[1, 1].set_title("Max Drawdown")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Max Drawdown (%)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Row 3: Training metrics
        axes[2, 0].plot(
            episodes, self.history["actor_loss"], label="Actor Loss", alpha=0.7
        )
        axes[2, 0].plot(
            episodes, self.history["critic_loss"], label="Critic Loss", alpha=0.7
        )
        axes[2, 0].set_title("Training Losses")
        axes[2, 0].set_xlabel("Episode")
        axes[2, 0].set_ylabel("Loss")
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].plot(episodes, self.history["entropy"], label="Entropy", alpha=0.7)
        axes[2, 1].plot(
            episodes, self.history["kl_divergence"], label="KL Divergence", alpha=0.7
        )
        axes[2, 1].set_title("Policy Metrics")
        axes[2, 1].set_xlabel("Episode")
        axes[2, 1].set_ylabel("Value")
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.plots_dir / f"rl_training_curves_{self.ticker}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"✓ Training curves saved to {plot_path}")
        plt.close()

    def plot_final_evaluation(self, env: TradingEnv, agent: PPOAgent):
        """Plot final evaluation on validation set."""
        logger.info("Generating final evaluation plots...")

        # Run evaluation
        state = env.reset()
        done = False
        while not done:
            action, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            state = next_state

        # Get history
        positions = np.array(env.history["positions"])
        equity = np.array(env.history["equity"])
        returns = np.array(env.history["returns"])
        actions = np.array(env.history["actions"])

        # Create 4-panel plot
        fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
        fig.suptitle(
            f"{self.ticker} RL Agent - Final Evaluation (Validation Set)",
            fontsize=16,
            fontweight="bold",
        )

        x = np.arange(len(positions))

        # Panel 1: Equity curve
        axes[0].plot(x, equity, linewidth=2, color="blue", label="Equity")
        axes[0].axhline(
            y=env.initial_capital,
            color="k",
            linestyle="--",
            alpha=0.5,
            label=f"Initial (${env.initial_capital:,.0f})",
        )
        axes[0].fill_between(
            x,
            env.initial_capital,
            equity,
            where=(equity >= env.initial_capital),
            color="green",
            alpha=0.2,
        )
        axes[0].fill_between(
            x,
            env.initial_capital,
            equity,
            where=(equity < env.initial_capital),
            color="red",
            alpha=0.2,
        )

        stats = env.get_episode_stats()
        axes[0].set_title(
            f"Equity Curve | Return: {stats['total_return']:+.2%} | "
            f"Sharpe: {stats['sharpe_ratio']:.3f} | Max DD: {stats['max_drawdown']:.2%}",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].set_ylabel("Equity ($)")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Panel 2: Positions
        axes[1].plot(x, positions, linewidth=1.5, color="purple", alpha=0.8)
        axes[1].fill_between(
            x,
            0,
            positions,
            where=(positions > 0),
            color="green",
            alpha=0.3,
            label="Long",
        )
        axes[1].fill_between(
            x,
            0,
            positions,
            where=(positions < 0),
            color="red",
            alpha=0.3,
            label="Short",
        )
        axes[1].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        axes[1].set_title(
            f"Position Sizing | Avg Position: {stats['avg_position']:.2f} | Trades: {stats['n_trades']}",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].set_ylabel("Position")
        axes[1].set_ylim(-1.1, 1.1)
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)

        # Panel 3: Returns
        strategy_returns = positions[:-1] * returns[1:]
        axes[2].bar(
            x[1:],
            strategy_returns * 100,
            color=["g" if r > 0 else "r" for r in strategy_returns],
            alpha=0.6,
            width=1.0,
        )
        axes[2].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        axes[2].set_title("Strategy Returns", fontsize=12, fontweight="bold")
        axes[2].set_ylabel("Return (%)")
        axes[2].grid(True, alpha=0.3, axis="y")

        # Panel 4: Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        axes[3].fill_between(x, 0, drawdown, color="red", alpha=0.5)
        axes[3].plot(x, drawdown, color="darkred", linewidth=1.5)
        axes[3].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        axes[3].set_title("Drawdown", fontsize=12, fontweight="bold")
        axes[3].set_xlabel("Time Step")
        axes[3].set_ylabel("Drawdown (%)")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.plots_dir / f"rl_final_evaluation_{self.ticker}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"✓ Final evaluation plot saved to {plot_path}")
        plt.close()

        # Save evaluation stats
        stats_path = self.results_dir / f"rl_final_stats_{self.ticker}.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"✓ Final stats saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agent for trading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument(
        "--episodes", type=int, default=500, help="Number of training episodes"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=50, help="Evaluation frequency"
    )
    parser.add_argument(
        "--save-freq", type=int, default=100, help="Checkpoint save frequency"
    )
    parser.add_argument(
        "--features-dir", default="data/features", help="Features directory"
    )
    parser.add_argument("--results-dir", default="results/rl", help="Results directory")
    parser.add_argument(
        "--config", default="config/default_config.yaml", help="Config file"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log-file", default="logs/train_rl_agent.log", help="Log file"
    )
    args = parser.parse_args()

    # Setup logging
    setup_logger(args.log_file, level="INFO")

    # Create trainer
    trainer = RLTrainer(
        ticker=args.ticker,
        features_dir=Path(args.features_dir),
        results_dir=Path(args.results_dir),
        config_path=args.config,
        seed=args.seed,
    )

    # Train
    trainer.train(
        n_episodes=args.episodes,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
    )


if __name__ == "__main__":
    main()
