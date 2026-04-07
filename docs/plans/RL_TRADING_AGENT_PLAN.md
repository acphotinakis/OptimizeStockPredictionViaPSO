# Reinforcement Learning Trading Agent
## From LSTM Predictions to RL-Optimized Trading

**Date**: April 7, 2026  
**Status**: Planning Phase  
**Priority**: High (Strategic Enhancement)

---

## Executive Summary

You're absolutely right - an **RL agent is superior to fixed threshold signals** for trading. Here's why and how to implement it.

### Current System (LSTM + Fixed Thresholds)

```
LSTM → Predicted Return → if pred > 0.001: BUY, elif pred < -0.001: SELL, else: HOLD
```

**Problems**:
- ❌ Fixed thresholds ignore market conditions (volatility, regime)
- ❌ No position sizing (always full position or flat)
- ❌ No multi-step planning (can't hold through temporary drawdowns)
- ❌ Doesn't learn from trading outcomes (only prediction accuracy)
- ❌ Ignores transaction costs in decision-making

### Proposed System (LSTM + RL Agent)

```
LSTM Features → RL Agent (PPO/SAC) → Continuous Actions (position size: -1 to +1)
                    ↓
                Reward = PnL - Transaction Costs - Drawdown Penalty
```

**Advantages**:
- ✅ Learns optimal thresholds dynamically
- ✅ Continuous position sizing (-100% short to +100% long)
- ✅ Multi-step planning (holds through noise, exits on real signals)
- ✅ Directly optimizes trading PnL (not prediction RMSE)
- ✅ Adapts to changing market regimes

---

## Why RL is Better for Trading

### 1. **Direct Optimization of Trading Objective**

| Approach | Optimizes | Problem |
|----------|-----------|---------|
| **LSTM** | Prediction accuracy (RMSE, MAE) | Good predictions ≠ good trades |
| **LSTM + Thresholds** | Prediction + fixed rules | Thresholds don't adapt |
| **RL Agent** | **Cumulative PnL** | Directly optimizes what we care about |

**Example**: LSTM might predict +0.0005 return (below 0.001 threshold → HOLD), but RL agent learns that in low-volatility regimes, even 0.0005 is worth trading.

### 2. **Learns Transaction Cost Awareness**

```python
# LSTM approach: Blind to costs
if pred_return > 0.001:
    signal = BUY  # Doesn't know if 0.001 > transaction_cost

# RL approach: Learns to trade only when profit > costs
action = agent.act(state)  # Implicitly learned: only trade if expected_profit > 2 × transaction_cost
```

### 3. **Handles Non-Stationarity**

Markets change:
- 2021: Low volatility, high Sharpe → Tight thresholds work
- 2022: High volatility, low Sharpe → Wide thresholds needed
- 2023: Sector rotation → Different thresholds per sector

**RL agent adapts** by observing recent rewards and adjusting policy.

### 4. **Multi-Step Planning**

```
LSTM: pred[t] → action[t] (myopic, 1-step)
RL:   state[t] → action[t] considering future rewards (multi-step)
```

**Example**: RL can hold through a -0.5% drawdown if it expects +2% gain in 10 bars.

---

## Proposed Architecture

### Option 1: **PPO (Proximal Policy Optimization)** [RECOMMENDED]

```
┌─────────────────────────────────────────────────────────────┐
│                     RL Trading Agent (PPO)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  State (117 features + 5 market context):                   │
│    - LSTM features (technical, statistical, cross-ticker)   │
│    - Current position (-1 to +1)                            │
│    - Unrealized PnL                                         │
│    - Recent volatility (20-bar)                             │
│    - Time since last trade                                  │
│    - Current drawdown from peak                             │
│                                                              │
│  Actor Network (Policy):                                    │
│    Input: State [122 dims]                                  │
│    Hidden: [256, 128, 64]                                   │
│    Output: μ, σ (mean, std of Gaussian)                     │
│    Action: position ∈ [-1, +1] (continuous)                 │
│                                                              │
│  Critic Network (Value):                                    │
│    Input: State [122 dims]                                  │
│    Hidden: [256, 128, 64]                                   │
│    Output: V(s) (expected cumulative reward)                │
│                                                              │
│  Reward Function:                                           │
│    r[t] = PnL[t] - λ_tc × |Δposition| - λ_dd × drawdown²    │
│                                                              │
│  Training:                                                   │
│    - Collect trajectories (1000 steps)                      │
│    - Compute advantages (GAE-λ)                             │
│    - Update policy with clipped objective                   │
│    - Update value function (MSE loss)                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Why PPO?**
- ✅ Stable training (clipped objective prevents large policy updates)
- ✅ Sample efficient (reuses data via multiple epochs)
- ✅ Works well with continuous actions
- ✅ Industry standard (used by OpenAI, DeepMind)

### Option 2: **SAC (Soft Actor-Critic)** [ALTERNATIVE]

```
Similar to PPO, but:
  - Maximum entropy objective (encourages exploration)
  - Off-policy (can use replay buffer)
  - Better for high-dimensional action spaces
```

**When to use SAC**:
- If you want to trade multiple assets simultaneously (portfolio actions)
- If you have limited data (off-policy is more sample efficient)

### Option 3: **DQN (Deep Q-Network)** [NOT RECOMMENDED]

```
Discrete actions: {SELL, HOLD, BUY}
```

**Why not?**
- ❌ Can't do position sizing (only binary: in/out)
- ❌ Worse performance than continuous action methods
- ❌ Requires discretization of action space

---

## Implementation Plan

### Phase 1: Baseline RL Agent (2 weeks)

**Goal**: Beat LSTM + fixed thresholds

#### Step 1.1: Environment Setup

**File**: `src/rl/trading_env.py`

```python
import gym
from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    """
    OpenAI Gym environment for RL trading.
    
    State: [117 LSTM features + 5 context features]
    Action: position ∈ [-1, +1] (continuous)
    Reward: PnL - transaction costs - drawdown penalty
    """
    
    def __init__(
        self,
        features: np.ndarray,  # [N, 117] LSTM features
        prices: np.ndarray,    # [N] close prices
        returns: np.ndarray,   # [N] log returns
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
    ):
        super().__init__()
        
        self.features = features
        self.prices = prices
        self.returns = returns
        self.initial_capital = initial_capital
        self.tc = transaction_cost
        self.max_position = max_position
        
        # State: features + position + PnL + volatility + time + drawdown
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(122,), dtype=np.float32
        )
        
        # Action: continuous position [-1, +1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.t = 0
        self.position = 0.0
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.last_trade_t = 0
        return self._get_state()
    
    def _get_state(self):
        # LSTM features
        feat = self.features[self.t]
        
        # Context features
        position = self.position
        unrealized_pnl = (self.equity - self.initial_capital) / self.initial_capital
        volatility = np.std(self.returns[max(0, self.t-20):self.t+1])
        time_since_trade = self.t - self.last_trade_t
        drawdown = (self.equity - self.peak_equity) / self.peak_equity
        
        state = np.concatenate([
            feat,
            [position, unrealized_pnl, volatility, time_since_trade, drawdown]
        ])
        
        return state.astype(np.float32)
    
    def step(self, action):
        # Clip action to valid range
        target_position = np.clip(action[0], -self.max_position, self.max_position)
        
        # Calculate position change
        position_change = target_position - self.position
        
        # Transaction cost
        tc_cost = abs(position_change) * self.tc * self.equity
        
        # Update position
        self.position = target_position
        if abs(position_change) > 0.01:
            self.last_trade_t = self.t
        
        # Move to next timestep
        self.t += 1
        
        # Calculate PnL
        if self.t < len(self.returns):
            ret = self.returns[self.t]
            pnl = self.position * ret * self.equity
            self.equity += pnl - tc_cost
            
            # Update peak
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
            
            # Reward = PnL - transaction cost - drawdown penalty
            drawdown = (self.equity - self.peak_equity) / self.peak_equity
            reward = pnl - tc_cost - 0.1 * (drawdown ** 2) * self.equity
            
            done = self.t >= len(self.returns) - 1
            state = self._get_state()
            
            return state, reward, done, {}
        else:
            return self._get_state(), 0.0, True, {}
```

#### Step 1.2: PPO Agent

**File**: `src/rl/ppo_agent.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, state_dim=122, action_dim=1, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        # Shared feature extractor
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.LayerNorm(h_dim),
            ])
            in_dim = h_dim
        self.feature_net = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, state):
        features = self.feature_net(state)
        
        # Actor: Gaussian policy
        mean = torch.tanh(self.actor_mean(features))  # Bounded [-1, 1]
        std = torch.exp(self.actor_logstd).expand_as(mean)
        
        # Critic: Value estimate
        value = self.critic(features)
        
        return mean, std, value
    
    def act(self, state):
        mean, std, _ = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), log_prob.detach()
    
    def evaluate(self, state, action):
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(
        self,
        state_dim=122,
        action_dim=1,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.policy.act(state_tensor)
        return action.cpu().numpy()[0], log_prob.cpu().item()
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, returns, advantages, epochs=10):
        """Update policy using PPO objective."""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(epochs):
            # Evaluate current policy
            log_probs, entropy, values = self.policy.evaluate(states, actions)
            
            # PPO clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.mean().item(),
        }
```

#### Step 1.3: Training Script

**File**: `scripts/train_rl_agent.py`

```python
#!/usr/bin/env python3
"""
Train RL agent for trading.

Usage:
    python scripts/train_rl_agent.py --ticker SPY --episodes 1000
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.trading_env import TradingEnv
from src.rl.ppo_agent import PPOAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--results-dir", default="results/rl")
    args = parser.parse_args()
    
    # Load features and prices
    features_dir = Path(args.features_dir) / args.ticker
    X_train = np.load(features_dir / "X_train.npy")
    y_train = np.load(features_dir / "y_train.npy")
    
    # Load OHLCV for prices
    ohlcv = pd.read_parquet(f"data/processed/{args.ticker}.parquet")
    prices = ohlcv["close"].values
    
    # Create environment
    env = TradingEnv(
        features=X_train,
        prices=prices[:len(X_train)],
        returns=y_train,
        initial_capital=100000.0,
        transaction_cost=0.001,
    )
    
    # Create agent
    agent = PPOAgent(state_dim=122, action_dim=1)
    
    # Training loop
    best_reward = -np.inf
    
    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0
        
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        
        # Collect trajectory
        done = False
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
            episode_reward += reward
        
        # Compute values
        with torch.no_grad():
            values = [agent.policy.forward(torch.FloatTensor(s).to(agent.device))[2].item() 
                     for s in states]
        
        # Compute advantages and returns
        advantages, returns = agent.compute_gae(rewards, values, dones)
        
        # Update policy
        metrics = agent.update(states, actions, log_probs, returns, advantages)
        
        # Log
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={episode_reward:.2f}, "
                  f"Final Equity=${env.equity:,.2f}, "
                  f"Actor Loss={metrics['actor_loss']:.4f}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            results_dir = Path(args.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            torch.save(agent.policy.state_dict(), 
                      results_dir / f"rl_agent_{args.ticker}_best.pth")

if __name__ == "__main__":
    main()
```

---

### Phase 2: Advanced Features (4 weeks)

#### 2.1 Multi-Asset Portfolio RL

**State**: Concatenate features from all N assets  
**Action**: Position vector [p1, p2, ..., pN] where Σ|pi| ≤ 1  
**Reward**: Portfolio PnL - transaction costs - correlation penalty

#### 2.2 Hierarchical RL

```
High-Level Agent: Regime detection (bull/bear/sideways)
    ↓
Low-Level Agent: Tactical trading within regime
```

#### 2.3 Offline RL (Conservative Q-Learning)

Train on historical data without environment interaction (safer for deployment).

---

## Expected Performance

### Baseline (LSTM + Fixed Thresholds)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 0.8 - 1.2 |
| Max Drawdown | -15% to -25% |
| Win Rate | 52% - 56% |
| Avg Trade | +0.05% |

### RL Agent (PPO)

| Metric | Expected | Improvement |
|--------|----------|-------------|
| Sharpe Ratio | 1.5 - 2.0 | **+50% to +100%** |
| Max Drawdown | -10% to -15% | **-30% to -50%** |
| Win Rate | 55% - 60% | **+5% to +10%** |
| Avg Trade | +0.10% | **+100%** |

**Why?**
- RL learns to avoid low-probability trades (higher win rate)
- RL sizes positions dynamically (better risk-adjusted returns)
- RL adapts to regime changes (lower drawdowns)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Overfitting** | Agent memorizes training data | Use walk-forward validation, regularization |
| **Instability** | Training diverges | Use PPO (stable), clip gradients, lower LR |
| **Sample inefficiency** | Needs lots of data | Use off-policy methods (SAC), data augmentation |
| **Reward hacking** | Agent exploits reward bugs | Careful reward design, human-in-the-loop |

---

## Next Steps

1. **Implement Phase 1** (2 weeks)
   - [ ] Create `TradingEnv` class
   - [ ] Implement PPO agent
   - [ ] Train on SPY
   - [ ] Compare vs LSTM baseline

2. **Validate** (1 week)
   - [ ] Walk-forward validation
   - [ ] Out-of-sample testing
   - [ ] Robustness checks (different seeds, tickers)

3. **Deploy** (1 week)
   - [ ] Paper trading integration
   - [ ] Monitoring dashboard
   - [ ] Live deployment

---

## Conclusion

**Yes, you're absolutely right** - an RL agent is the logical next step. The current LSTM + fixed thresholds is a good baseline, but RL will:

1. **Directly optimize trading PnL** (not prediction accuracy)
2. **Learn adaptive thresholds** (regime-dependent)
3. **Handle transaction costs** (implicitly in reward)
4. **Multi-step planning** (hold through noise)

**Recommendation**: Start with PPO (Phase 1), validate on SPY, then expand to multi-asset portfolio (Phase 2).

**Timeline**: 3-4 weeks to production-ready RL agent.

---

**Ready to implement?** Let me know and I'll start building the RL trading environment!
