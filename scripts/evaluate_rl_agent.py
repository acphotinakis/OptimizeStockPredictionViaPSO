#!/usr/bin/env python3
"""
scripts/evaluate_rl_agent.py

Evaluate trained RL agent on test set and compare with LSTM baseline.

Usage:
    python scripts/evaluate_rl_agent.py --ticker SPY
    python scripts/evaluate_rl_agent.py --ticker AAPL --model-path results/rl/checkpoints/rl_agent_AAPL_best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rl.trading_env import TradingEnv
from src.rl.ppo_agent import PPOAgent
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def load_lstm_baseline(ticker: str, results_dir: Path) -> dict:
    """Load LSTM baseline results for comparison."""
    try:
        # Load aligned predictions
        csv_path = results_dir / f"lstm_aligned_{ticker}_test_seed42.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            # Calculate metrics
            if "equity" in df.columns:
                initial_capital = df["equity"].iloc[0] - df["pnl"].iloc[0]
                final_equity = df["equity"].iloc[-1]
                total_return = (final_equity / initial_capital) - 1
                
                if "strategy_return" in df.columns:
                    strategy_returns = df["strategy_return"].values
                    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10) * np.sqrt(252 * 390)
                else:
                    sharpe = 0.0
                
                running_max = np.maximum.accumulate(df["equity"].values)
                drawdown = (df["equity"].values - running_max) / running_max
                max_dd = np.min(drawdown)
                
                # Count trades
                if "signal" in df.columns:
                    position_changes = np.diff(df["signal"].values, prepend=0)
                    n_trades = (np.abs(position_changes) > 0).sum()
                else:
                    n_trades = 0
                
                return {
                    "final_equity": final_equity,
                    "total_return": total_return,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_dd,
                    "n_trades": n_trades,
                }
        
        logger.warning(f"LSTM baseline not found for {ticker}")
        return None
    except Exception as e:
        logger.error(f"Failed to load LSTM baseline: {e}")
        return None


def evaluate_agent(
    ticker: str,
    model_path: Path,
    features_dir: Path,
    results_dir: Path,
):
    """Evaluate RL agent on test set."""
    logger.info("="*60)
    logger.info(f"Evaluating RL Agent: {ticker}")
    logger.info("="*60)
    
    # Load test data
    ticker_dir = features_dir / ticker
    X_test = np.load(ticker_dir / "X_test.npy")
    y_test = np.load(ticker_dir / "y_test.npy")
    
    logger.info(f"Test data: {X_test.shape}")
    
    # Load prices
    try:
        ohlcv = pd.read_parquet(f"data/processed/{ticker}.parquet")
        prices_test = ohlcv["close"].values[-len(X_test):]
    except:
        prices_test = np.ones(len(X_test))
    
    # Create test environment
    test_env = TradingEnv(
        features=X_test,
        returns=y_test,
        prices=prices_test,
        initial_capital=100_000.0,
        transaction_cost=0.001,
        max_position=1.0,
    )
    
    # Load agent
    state_dim = test_env.observation_space.shape[0]
    agent = PPOAgent(state_dim=state_dim, action_dim=1)
    agent.load(str(model_path))
    
    logger.info(f"Model loaded from {model_path}")
    
    # Run evaluation
    state = test_env.reset()
    done = False
    
    while not done:
        action, _ = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = test_env.step(action)
        state = next_state
    
    # Get stats
    rl_stats = test_env.get_episode_stats()
    
    logger.info("\nRL Agent Performance:")
    logger.info(f"  Final Equity:  ${rl_stats['final_equity']:,.2f}")
    logger.info(f"  Total Return:  {rl_stats['total_return']:+.2%}")
    logger.info(f"  Sharpe Ratio:  {rl_stats['sharpe_ratio']:.3f}")
    logger.info(f"  Max Drawdown:  {rl_stats['max_drawdown']:.2%}")
    logger.info(f"  Trades:        {rl_stats['n_trades']}")
    logger.info(f"  Win Rate:      {rl_stats['win_rate']:.2%}")
    
    # Load LSTM baseline
    lstm_stats = load_lstm_baseline(ticker, Path("results"))
    
    if lstm_stats:
        logger.info("\nLSTM Baseline Performance:")
        logger.info(f"  Final Equity:  ${lstm_stats['final_equity']:,.2f}")
        logger.info(f"  Total Return:  {lstm_stats['total_return']:+.2%}")
        logger.info(f"  Sharpe Ratio:  {lstm_stats['sharpe_ratio']:.3f}")
        logger.info(f"  Max Drawdown:  {lstm_stats['max_drawdown']:.2%}")
        logger.info(f"  Trades:        {lstm_stats['n_trades']}")
        
        logger.info("\nImprovement (RL vs LSTM):")
        logger.info(f"  Return:  {(rl_stats['total_return'] - lstm_stats['total_return'])*100:+.2f}%")
        logger.info(f"  Sharpe:  {rl_stats['sharpe_ratio'] - lstm_stats['sharpe_ratio']:+.3f}")
        logger.info(f"  Max DD:  {(rl_stats['max_drawdown'] - lstm_stats['max_drawdown'])*100:+.2f}%")
    
    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "ticker": ticker,
        "rl_agent": rl_stats,
        "lstm_baseline": lstm_stats,
    }
    
    with open(results_dir / f"rl_test_results_{ticker}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison plot
    plot_comparison(ticker, test_env, rl_stats, lstm_stats, results_dir)
    
    logger.info("="*60)
    logger.info("✓ Evaluation complete!")
    logger.info("="*60)


def plot_comparison(
    ticker: str,
    test_env: TradingEnv,
    rl_stats: dict,
    lstm_stats: dict,
    results_dir: Path,
):
    """Create comparison plots."""
    logger.info("Generating comparison plots...")
    
    # Get RL history
    positions = np.array(test_env.history["positions"])
    equity = np.array(test_env.history["equity"])
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f"{ticker} - RL Agent vs LSTM Baseline (Test Set)", 
                fontsize=16, fontweight='bold')
    
    # Plot 1: Equity curves
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(equity))
    ax1.plot(x, equity, linewidth=2, color='blue', label='RL Agent', alpha=0.8)
    ax1.axhline(y=test_env.initial_capital, color='k', linestyle='--', alpha=0.5)
    ax1.fill_between(x, test_env.initial_capital, equity, 
                     where=(equity >= test_env.initial_capital), 
                     color='green', alpha=0.2)
    ax1.fill_between(x, test_env.initial_capital, equity, 
                     where=(equity < test_env.initial_capital), 
                     color='red', alpha=0.2)
    
    ax1.set_title(
        f"Equity Curve | RL Return: {rl_stats['total_return']:+.2%} | "
        f"RL Sharpe: {rl_stats['sharpe_ratio']:.3f}",
        fontsize=12, fontweight='bold'
    )
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Positions
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(x, positions, linewidth=1.5, color='purple', alpha=0.8)
    ax2.fill_between(x, 0, positions, where=(positions > 0), color='green', alpha=0.3)
    ax2.fill_between(x, 0, positions, where=(positions < 0), color='red', alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_title(f"RL Agent Position Sizing | Trades: {rl_stats['n_trades']}", 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Position")
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Metrics comparison (if LSTM available)
    if lstm_stats:
        ax3 = fig.add_subplot(gs[2, 0])
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        metric_labels = ['Return', 'Sharpe', 'Max DD']
        
        rl_values = [rl_stats[m] * 100 if 'return' in m or 'drawdown' in m else rl_stats[m] 
                    for m in metrics]
        lstm_values = [lstm_stats[m] * 100 if 'return' in m or 'drawdown' in m else lstm_stats[m] 
                      for m in metrics]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        ax3.bar(x_pos - width/2, rl_values, width, label='RL Agent', color='blue', alpha=0.7)
        ax3.bar(x_pos + width/2, lstm_values, width, label='LSTM Baseline', color='orange', alpha=0.7)
        
        ax3.set_title("Performance Comparison", fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(metric_labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Improvement percentages
        ax4 = fig.add_subplot(gs[2, 1])
        improvements = [
            (rl_stats['total_return'] - lstm_stats['total_return']) * 100,
            (rl_stats['sharpe_ratio'] - lstm_stats['sharpe_ratio']) / lstm_stats['sharpe_ratio'] * 100 if lstm_stats['sharpe_ratio'] != 0 else 0,
            (rl_stats['max_drawdown'] - lstm_stats['max_drawdown']) * 100,
        ]
        
        colors = ['g' if i > 0 else 'r' for i in improvements]
        ax4.barh(metric_labels, improvements, color=colors, alpha=0.7)
        ax4.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax4.set_title("Improvement (RL vs LSTM)", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Improvement (%)")
        ax4.grid(True, alpha=0.3, axis='x')
    else:
        # Just show RL stats
        ax3 = fig.add_subplot(gs[2, :])
        stats_text = (
            f"RL Agent Statistics:\n"
            f"  Return: {rl_stats['total_return']:+.2%}\n"
            f"  Sharpe: {rl_stats['sharpe_ratio']:.3f}\n"
            f"  Max DD: {rl_stats['max_drawdown']:.2%}\n"
            f"  Trades: {rl_stats['n_trades']}\n"
            f"  Win Rate: {rl_stats['win_rate']:.2%}"
        )
        ax3.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=14, 
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.axis('off')
    
    plt.tight_layout()
    plot_path = results_dir / "plots" / f"rl_test_comparison_{ticker}.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Comparison plot saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RL agent on test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument("--model-path", default=None, help="Path to trained model (default: auto-detect)")
    parser.add_argument("--features-dir", default="data/features", help="Features directory")
    parser.add_argument("--results-dir", default="results/rl", help="Results directory")
    parser.add_argument("--log-file", default="logs/evaluate_rl_agent.log", help="Log file")
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(args.log_file, level="INFO")
    
    # Auto-detect model path if not provided
    if args.model_path is None:
        model_path = Path(args.results_dir) / "checkpoints" / f"rl_agent_{args.ticker}_best.pth"
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            logger.error("Please train the model first or specify --model-path")
            sys.exit(1)
    else:
        model_path = Path(args.model_path)
    
    # Evaluate
    evaluate_agent(
        ticker=args.ticker,
        model_path=model_path,
        features_dir=Path(args.features_dir),
        results_dir=Path(args.results_dir),
    )


if __name__ == "__main__":
    main()
