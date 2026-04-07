"""
src/rl/trading_env.py

OpenAI Gym environment for RL trading.
Supports continuous position sizing with realistic transaction costs and risk penalties.
"""

from __future__ import annotations

import logging
from typing import Tuple, Dict, Optional

import gym
from gym import spaces
import numpy as np

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Trading environment for reinforcement learning.
    
    State: [N features + 5 context features]
        - LSTM/XGBoost features (technical, statistical, cross-ticker)
        - Current position (-1 to +1)
        - Unrealized PnL (normalized)
        - Recent volatility (20-bar)
        - Time since last trade
        - Current drawdown from peak
    
    Action: Continuous position ∈ [-1, +1]
        -1 = 100% short
         0 = flat (no position)
        +1 = 100% long
    
    Reward: PnL - transaction costs - drawdown penalty
    """
    
    metadata = {"render.modes": ["human"]}
    
    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        prices: Optional[np.ndarray] = None,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
        drawdown_penalty: float = 0.1,
        hold_penalty: float = 0.0,
        lookback_window: int = 20,
    ):
        """
        Args:
            features: [N, F] array of features per timestep
            returns: [N] array of log returns
            prices: [N] array of prices (optional, for visualization)
            initial_capital: Starting capital in USD
            transaction_cost: One-way transaction cost (e.g., 0.001 = 0.1%)
            max_position: Maximum position size (1.0 = 100%)
            drawdown_penalty: Penalty coefficient for drawdown^2
            hold_penalty: Small penalty for holding positions (encourages turnover)
            lookback_window: Window for volatility calculation
        """
        super().__init__()
        
        self.features = features.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.prices = prices if prices is not None else np.ones(len(returns))
        
        self.initial_capital = initial_capital
        self.tc = transaction_cost
        self.max_position = max_position
        self.drawdown_penalty = drawdown_penalty
        self.hold_penalty = hold_penalty
        self.lookback_window = lookback_window
        
        self.n_steps = len(self.features)
        self.n_features = self.features.shape[1]
        
        # State: features + 5 context features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features + 5,),
            dtype=np.float32
        )
        
        # Action: continuous position [-1, +1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.t = 0
        self.position = 0.0
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.last_trade_t = 0
        
        # History for logging
        self.history = {
            "positions": [],
            "equity": [],
            "returns": [],
            "rewards": [],
            "actions": [],
            "pnl": [],
        }
        
        logger.info(
            f"TradingEnv initialized: {self.n_steps} steps, "
            f"{self.n_features} features, initial_capital=${initial_capital:,.0f}"
        )
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.t = 0
        self.position = 0.0
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.last_trade_t = 0
        
        # Clear history
        self.history = {
            "positions": [0.0],
            "equity": [self.initial_capital],
            "returns": [0.0],
            "rewards": [0.0],
            "actions": [0.0],
            "pnl": [0.0],
        }
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Construct state vector."""
        # Base features
        feat = self.features[self.t]
        
        # Context features
        position = self.position
        unrealized_pnl = (self.equity - self.initial_capital) / self.initial_capital
        
        # Recent volatility
        start_idx = max(0, self.t - self.lookback_window)
        recent_returns = self.returns[start_idx:self.t+1]
        volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.0
        
        # Time since last trade (normalized)
        time_since_trade = (self.t - self.last_trade_t) / 100.0
        
        # Drawdown from peak
        drawdown = (self.equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        
        # Concatenate
        state = np.concatenate([
            feat,
            [position, unrealized_pnl, volatility, time_since_trade, drawdown]
        ])
        
        return state.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one timestep.
        
        Args:
            action: [1] array with target position ∈ [-1, +1]
        
        Returns:
            state: Next state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        # Clip action to valid range
        target_position = float(np.clip(action[0], -self.max_position, self.max_position))
        
        # Calculate position change
        position_change = target_position - self.position
        
        # Transaction cost (proportional to position change)
        tc_cost = abs(position_change) * self.tc * self.equity
        
        # Update position
        old_position = self.position
        self.position = target_position
        
        # Track if we traded
        if abs(position_change) > 0.01:
            self.last_trade_t = self.t
        
        # Move to next timestep
        self.t += 1
        done = self.t >= self.n_steps - 1
        
        if not done:
            # Calculate PnL from position × return
            ret = self.returns[self.t]
            position_pnl = old_position * ret * self.equity
            
            # Update equity
            self.equity += position_pnl - tc_cost
            
            # Update peak equity
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
            
            # Calculate drawdown
            drawdown = (self.equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
            
            # Reward = PnL - transaction cost - drawdown penalty - hold penalty
            reward = (
                position_pnl
                - tc_cost
                - self.drawdown_penalty * (drawdown ** 2) * self.equity
                - self.hold_penalty * abs(self.position) * self.equity
            )
            
            # Normalize reward by initial capital
            reward = reward / self.initial_capital
            
            # Get next state
            state = self._get_state()
            
            # Log history
            self.history["positions"].append(self.position)
            self.history["equity"].append(self.equity)
            self.history["returns"].append(ret)
            self.history["rewards"].append(reward)
            self.history["actions"].append(target_position)
            self.history["pnl"].append(self.equity - self.initial_capital)
            
            # Info dict
            info = {
                "position": self.position,
                "equity": self.equity,
                "pnl": self.equity - self.initial_capital,
                "drawdown": drawdown,
                "tc_cost": tc_cost,
                "position_pnl": position_pnl,
            }
        else:
            # Episode ended
            reward = 0.0
            state = self._get_state()
            info = {
                "final_equity": self.equity,
                "final_pnl": self.equity - self.initial_capital,
                "total_return": (self.equity / self.initial_capital) - 1,
            }
        
        return state, reward, done, info
    
    def render(self, mode: str = "human"):
        """Render environment state (for debugging)."""
        if mode == "human":
            print(f"Step {self.t}/{self.n_steps-1} | "
                  f"Position: {self.position:+.2f} | "
                  f"Equity: ${self.equity:,.2f} | "
                  f"PnL: ${self.equity - self.initial_capital:+,.2f}")
    
    def get_episode_stats(self) -> Dict:
        """Calculate episode statistics."""
        equity_curve = np.array(self.history["equity"])
        returns = np.array(self.history["returns"])
        positions = np.array(self.history["positions"])
        
        # Total return
        total_return = (equity_curve[-1] / self.initial_capital) - 1
        
        # Sharpe ratio (annualized, assuming 252*390 1-min bars per year)
        strategy_returns = positions[:-1] * returns[1:]
        sharpe = (
            np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10) * np.sqrt(252 * 390)
            if len(strategy_returns) > 1 else 0.0
        )
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_dd = np.min(drawdown)
        
        # Number of trades
        position_changes = np.diff(positions, prepend=0)
        n_trades = (np.abs(position_changes) > 0.01).sum()
        
        # Win rate
        trade_returns = strategy_returns[np.abs(position_changes[1:]) > 0.01]
        win_rate = (trade_returns > 0).sum() / len(trade_returns) if len(trade_returns) > 0 else 0.0
        
        return {
            "final_equity": float(equity_curve[-1]),
            "total_pnl": float(equity_curve[-1] - self.initial_capital),
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "n_trades": int(n_trades),
            "win_rate": float(win_rate),
            "avg_position": float(np.mean(np.abs(positions))),
        }
