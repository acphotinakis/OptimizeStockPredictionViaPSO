#!/usr/bin/env python3
"""
Quick test to verify RL implementation works correctly.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.rl.trading_env import TradingEnv
from src.models.rl.ppo_agent import PPOAgent


def test_environment():
    """Test trading environment."""
    print("Testing TradingEnv...")

    # Create dummy data
    n_steps = 1000
    n_features = 117

    features = np.random.randn(n_steps, n_features).astype(np.float32)
    returns = np.random.randn(n_steps).astype(np.float32) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    # Create environment
    env = TradingEnv(
        features=features,
        returns=returns,
        prices=prices,
        initial_capital=100_000.0,
        transaction_cost=0.001,
    )

    # Test reset
    state = env.reset()
    assert state.shape == (122,), f"Expected state shape (122,), got {state.shape}"
    print(f"  [SELECTED] State shape: {state.shape}")

    # Test step
    action = np.array([0.5])  # 50% long
    next_state, reward, done, info = env.step(action)

    assert next_state.shape == (
        122,
    ), f"Expected next_state shape (122,), got {next_state.shape}"
    assert isinstance(
        reward, (float, np.floating)
    ), f"Expected reward to be float, got {type(reward)}"
    assert isinstance(done, bool), f"Expected done to be bool, got {type(done)}"
    print(f"  [SELECTED] Step works: reward={reward:.6f}, done={done}")

    # Test full episode
    state = env.reset()
    total_reward = 0.0
    steps = 0

    while steps < 100:
        action = np.random.uniform(-1, 1, size=(1,))
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    print(
        f"  [SELECTED] Episode completed: {steps} steps, total_reward={total_reward:.3f}"
    )

    # Test episode stats
    stats = env.get_episode_stats()
    assert "sharpe_ratio" in stats, "Missing sharpe_ratio in stats"
    assert "total_return" in stats, "Missing total_return in stats"
    print(
        f"  [SELECTED] Episode stats: Sharpe={stats['sharpe_ratio']:.3f}, Return={stats['total_return']:.2%}"
    )

    print("[SELECTED] TradingEnv tests passed!\n")


def test_agent():
    """Test PPO agent."""
    print("Testing PPOAgent...")

    # Create agent
    agent = PPOAgent(
        state_dim=122,
        action_dim=1,
        hidden_dims=[64, 32],  # Smaller for testing
        lr=3e-4,
    )

    print(f"  [SELECTED] Agent created on device: {agent.device}")

    # Test action selection
    state = np.random.randn(122).astype(np.float32)
    action, log_prob = agent.select_action(state, deterministic=False)

    assert action.shape == (1,), f"Expected action shape (1,), got {action.shape}"
    assert -1.0 <= action[0] <= 1.0, f"Action out of bounds: {action[0]}"
    print(
        f"  [SELECTED] Action selection: action={action[0]:.3f}, log_prob={log_prob:.3f}"
    )

    # Test deterministic action
    action_det, _ = agent.select_action(state, deterministic=True)
    print(f"  [SELECTED] Deterministic action: {action_det[0]:.3f}")

    # Test update
    n_samples = 100
    states = np.random.randn(n_samples, 122).astype(np.float32)
    actions = np.random.uniform(-1, 1, size=(n_samples, 1)).astype(np.float32)
    old_log_probs = np.random.randn(n_samples).astype(np.float32)
    returns = np.random.randn(n_samples).astype(np.float32)
    advantages = np.random.randn(n_samples).astype(np.float32)

    metrics = agent.update(
        states, actions, old_log_probs, returns, advantages, epochs=2, batch_size=32
    )

    assert "actor_loss" in metrics, "Missing actor_loss in metrics"
    assert "critic_loss" in metrics, "Missing critic_loss in metrics"
    print(
        f"  [SELECTED] Update works: actor_loss={metrics['actor_loss']:.4f}, critic_loss={metrics['critic_loss']:.4f}"
    )

    # Test save/load
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        temp_path = f.name

    agent.save(temp_path)
    print(f"  [SELECTED] Model saved to {temp_path}")

    agent2 = PPOAgent(state_dim=122, action_dim=1, hidden_dims=[64, 32])
    agent2.load(temp_path)
    print(f"  [SELECTED] Model loaded successfully")

    # Cleanup
    Path(temp_path).unlink()

    print("[SELECTED] PPOAgent tests passed!\n")


def test_integration():
    """Test environment + agent integration."""
    print("Testing Environment + Agent Integration...")

    # Create dummy data
    n_steps = 500
    n_features = 117

    features = np.random.randn(n_steps, n_features).astype(np.float32)
    returns = np.random.randn(n_steps).astype(np.float32) * 0.01

    # Create environment
    env = TradingEnv(
        features=features,
        returns=returns,
        initial_capital=100_000.0,
        transaction_cost=0.001,
    )

    # Create agent
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=1,
        hidden_dims=[64, 32],
    )

    # Run one episode
    state = env.reset()
    done = False

    states, actions, log_probs, rewards, dones = [], [], [], [], []

    while not done:
        action, log_prob = agent.select_action(state, deterministic=False)
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)

        state = next_state

    print(f"  [SELECTED] Episode completed: {len(states)} steps")

    # Compute values
    import torch

    with torch.no_grad():
        values = []
        for s in states:
            s_tensor = torch.FloatTensor(s).unsqueeze(0).to(agent.device)
            _, _, v = agent.policy.forward(s_tensor)
            values.append(v.item())

    # Compute GAE
    advantages, returns_gae = agent.compute_gae(rewards, values, dones)
    print(f"  [SELECTED] GAE computed: {len(advantages)} advantages")

    # Update policy
    metrics = agent.update(
        np.array(states),
        np.array(actions),
        np.array(log_probs),
        np.array(returns_gae),
        np.array(advantages),
        epochs=2,
        batch_size=64,
    )

    print(f"  [SELECTED] Policy updated: actor_loss={metrics['actor_loss']:.4f}")

    # Get episode stats
    stats = env.get_episode_stats()
    print(
        f"  [SELECTED] Episode stats: Sharpe={stats['sharpe_ratio']:.3f}, Return={stats['total_return']:.2%}"
    )

    print("[SELECTED] Integration tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("RL Implementation Tests")
    print("=" * 60)
    print()

    try:
        test_environment()
        test_agent()
        test_integration()

        print("=" * 60)
        print("[SELECTED] All tests passed!")
        print("=" * 60)
        print("\nRL implementation is working correctly.")
        print("Ready to train on real data:")
        print("  ./scripts/run_rl_pipeline.sh SPY 500")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
