"""
src/rl/ppo_agent.py

Proximal Policy Optimization (PPO) agent for trading.
Stable, sample-efficient, and well-suited for continuous action spaces.
"""

from __future__ import annotations

import logging
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Actor: Outputs mean and std of Gaussian policy
    Critic: Outputs value estimate V(s)
    """
    
    def __init__(
        self,
        state_dim: int = 122,
        action_dim: int = 1,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = "relu",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        layers = []
        in_dim = state_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU() if activation == "relu" else nn.Tanh(),
                nn.LayerNorm(h_dim),
            ])
            in_dim = h_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"ActorCritic initialized: state_dim={state_dim}, "
            f"action_dim={action_dim}, hidden_dims={hidden_dims}"
        )
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Small initialization for actor mean (encourages exploration)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: [batch, state_dim] tensor
        
        Returns:
            mean: [batch, action_dim] action mean
            std: [batch, action_dim] action std
            value: [batch, 1] state value
        """
        features = self.feature_net(state)
        
        # Actor: Gaussian policy
        mean = torch.tanh(self.actor_mean(features))  # Bounded to [-1, 1]
        std = torch.exp(self.actor_logstd).expand_as(mean).clamp(min=1e-3, max=1.0)
        
        # Critic: Value estimate
        value = self.critic(features)
        
        return mean, std, value
    
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: [batch, state_dim] or [state_dim] tensor
            deterministic: If True, return mean action (no sampling)
        
        Returns:
            action: [batch, action_dim] or [action_dim] sampled action
            log_prob: [batch] or scalar log probability
        """
        mean, std, _ = self.forward(state)
        
        if deterministic:
            action = mean
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy.
        
        Args:
            state: [batch, state_dim] tensor
            action: [batch, action_dim] tensor
        
        Returns:
            log_prob: [batch] log probability of actions
            entropy: [batch] entropy of policy
            value: [batch, 1] state values
        """
        mean, std, value = self.forward(state)
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value


class PPOAgent:
    """
    Proximal Policy Optimization agent.
    
    Features:
    - Clipped surrogate objective (stable training)
    - Generalized Advantage Estimation (GAE)
    - Multiple epochs per update (sample efficient)
    - Gradient clipping (prevents divergence)
    """
    
    def __init__(
        self,
        state_dim: int = 122,
        action_dim: int = 1,
        hidden_dims: List[int] = [256, 128, 64],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping threshold
            device: Device to use (cuda/cpu)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create policy network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Training statistics
        self.training_step = 0
        
        logger.info(
            f"PPOAgent initialized on {self.device}: "
            f"lr={lr}, gamma={gamma}, clip_epsilon={clip_epsilon}"
        )
    
    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Select action given state.
        
        Args:
            state: [state_dim] numpy array
            deterministic: If True, return mean action
        
        Returns:
            action: [action_dim] numpy array
            log_prob: scalar log probability
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy.act(state_tensor, deterministic=deterministic)
        
        return action.cpu().numpy()[0], log_prob.cpu().item()
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags
        
        Returns:
            advantages: List of advantages
            returns: List of returns (targets for value function)
        """
        advantages = []
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        # Returns = advantages + values
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        epochs: int = 10,
        batch_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Update policy using PPO objective.
        
        Args:
            states: [N, state_dim] array
            actions: [N, action_dim] array
            old_log_probs: [N] array
            returns: [N] array (value targets)
            advantages: [N] array
            epochs: Number of optimization epochs
            batch_size: Mini-batch size (None = full batch)
        
        Returns:
            metrics: Dictionary of training metrics
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        actor_losses = []
        critic_losses = []
        entropies = []
        kl_divs = []
        clip_fractions = []
        
        # Multiple epochs
        for epoch in range(epochs):
            # Mini-batch training
            if batch_size is None:
                batch_size = len(states)
            
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(states), batch_size):
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate current policy
                log_probs, entropy, values = self.policy.evaluate(batch_states, batch_actions)
                
                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss
                critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())
                
                # KL divergence (for monitoring)
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - log_probs).mean().item()
                    kl_divs.append(kl_div)
                    
                    # Clip fraction (how often we clip)
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    clip_fractions.append(clip_fraction)
        
        self.training_step += 1
        
        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy": np.mean(entropies),
            "kl_divergence": np.mean(kl_divs),
            "clip_fraction": np.mean(clip_fractions),
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        logger.info(f"Model loaded from {path}")
