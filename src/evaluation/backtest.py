"""
Canonical Backtesting Module

Implements trading simulation with transaction costs as defined in FINAL_PLAN.md Section 6.

CRITICAL RULES (FINAL_PLAN.md Section 6):
- Signal generation: sign(predicted_return)
- Transaction cost: 0.15% per trade (one-way)
- Round-trip cost: 0.30%
- Cost applied only on position changes
- Fixed position sizing (±1)

Author: System Architect
Version: CANONICAL 1.0
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CanonicalBacktest:
    """
    Backtesting engine with canonical signal generation and transaction costs.
    
    FINAL_PLAN.md Section 6: Backtesting Protocol
    
    Enforces:
    - Directional signal generation
    - Transaction cost application
    - Performance metric calculation
    - Benchmark comparison
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.0015,  # 0.15% one-way
        initial_capital: float = 1.0,
    ):
        """
        Initialize backtest engine.
        
        Args:
            transaction_cost: One-way transaction cost (default: 0.0015 = 0.15%)
            initial_capital: Starting capital (default: 1.0)
        """
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        
        logger.info("=" * 80)
        logger.info("CANONICAL BACKTEST ENGINE (FINAL_PLAN.md)")
        logger.info("=" * 80)
        logger.info(f"Transaction cost: {transaction_cost*100:.2f}% (one-way)")
        logger.info(f"Round-trip cost: {transaction_cost*2*100:.2f}%")
        logger.info(f"Initial capital: {initial_capital}")
        logger.info("=" * 80)
    
    def generate_signals(
        self,
        predictions: np.ndarray,
        threshold: float = 0.0,
    ) -> np.ndarray:
        """
        Convert return predictions to trading signals.
        
        FINAL_PLAN.md Section 6.1: Signal Generation
        
        Rule: signal = sign(predicted_return)
            +1: Long (if prediction > threshold)
            -1: Short (if prediction < -threshold)
             0: Neutral (if |prediction| <= threshold)
        
        Args:
            predictions: Predicted returns
            threshold: Minimum return to trigger signal (default: 0.0)
        
        Returns:
            Signals array (+1, 0, -1)
        """
        signals = np.zeros_like(predictions, dtype=np.int8)
        signals[predictions > threshold] = 1   # LONG
        signals[predictions < -threshold] = -1  # SHORT
        # |prediction| <= threshold → 0 (NEUTRAL)
        
        n_long = np.sum(signals == 1)
        n_short = np.sum(signals == -1)
        n_neutral = np.sum(signals == 0)
        
        logger.info(f"Signals generated: {len(signals)} total")
        logger.info(f"  Long: {n_long} ({100*n_long/len(signals):.1f}%)")
        logger.info(f"  Short: {n_short} ({100*n_short/len(signals):.1f}%)")
        logger.info(f"  Neutral: {n_neutral} ({100*n_neutral/len(signals):.1f}%)")
        
        return signals
    
    def apply_transaction_costs(
        self,
        signals: np.ndarray,
        actual_returns: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply transaction costs to strategy returns.
        
        FINAL_PLAN.md Section 6.2: Transaction Cost Application
        
        Cost applied on position changes only:
        - Long → Short: 2 × transaction_cost (exit + enter)
        - Long → Neutral: 1 × transaction_cost (exit only)
        - Same position: 0 (no cost)
        
        Args:
            signals: Trading signals (+1, 0, -1)
            actual_returns: Actual market returns
        
        Returns:
            Tuple of (strategy_returns, trade_costs)
        """
        n = len(signals)
        
        # Initialize position array (includes t=0)
        position = np.zeros(n + 1, dtype=np.int8)
        position[0] = 0  # Start neutral
        position[1:] = signals
        
        # Detect position changes
        position_changes = np.abs(np.diff(position))
        
        # Compute trade costs
        trade_costs = position_changes * self.transaction_cost
        
        # Strategy return = position × actual_return - trade_cost
        strategy_returns = signals * actual_returns - trade_costs
        
        total_costs = np.sum(trade_costs)
        n_trades = np.sum(position_changes > 0)
        
        logger.info(f"Transaction costs applied:")
        logger.info(f"  Total trades: {n_trades}")
        logger.info(f"  Total costs: {total_costs:.6f} ({100*total_costs:.4f}%)")
        logger.info(f"  Avg cost per trade: {total_costs/n_trades:.6f}" if n_trades > 0 else "  Avg cost per trade: N/A")
        
        return strategy_returns, trade_costs
    
    def run_backtest(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        dates: pd.DatetimeIndex = None,
    ) -> pd.DataFrame:
        """
        Execute full backtest with transaction costs.
        
        FINAL_PLAN.md Section 6.3: Backtest Execution
        
        Args:
            predictions: Predicted returns
            actual_returns: Actual returns
            dates: Optional datetime index
        
        Returns:
            DataFrame with daily backtest results
        """
        logger.info("Running canonical backtest...")
        
        # Generate signals
        signals = self.generate_signals(predictions, threshold=0.0)
        
        # Apply transaction costs
        strategy_returns, trade_costs = self.apply_transaction_costs(
            signals, actual_returns
        )
        
        # Compute cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        capital = self.initial_capital * cumulative_returns
        
        # Build results DataFrame
        if dates is None:
            dates = pd.RangeIndex(len(predictions))
        
        results = pd.DataFrame({
            "date": dates,
            "prediction": predictions,
            "actual_return": actual_returns,
            "signal": signals,
            "strategy_return": strategy_returns,
            "trade_cost": trade_costs,
            "cumulative_return": cumulative_returns - 1,
            "capital": capital,
        })
        
        logger.info(f"Backtest complete: {len(results)} days")
        
        return results
    
    def compute_performance_metrics(
        self,
        backtest_results: pd.DataFrame,
        risk_free_rate: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute financial performance metrics.
        
        FINAL_PLAN.md Section 6.4: Performance Metrics
        
        Args:
            backtest_results: DataFrame from run_backtest()
            risk_free_rate: Annual risk-free rate (default: 0.0)
        
        Returns:
            Dictionary of performance metrics
        """
        returns = backtest_results["strategy_return"].values
        capital = backtest_results["capital"].values
        
        # Total return
        total_return = capital[-1] / capital[0] - 1
        
        # Annualized return (252 trading days)
        n_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Volatility
        daily_vol = np.std(returns)
        annualized_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol != 0 else 0.0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(capital)
        drawdown = (capital - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Win rate
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.0
        
        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses != 0 else np.inf
        
        # Number of trades
        signals = backtest_results["signal"].values
        n_trades = np.sum(np.diff(signals) != 0)
        
        metrics = {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "total_trades": int(n_trades),
            "total_days": int(n_days),
        }
        
        logger.info("=" * 80)
        logger.info("PERFORMANCE METRICS")
        logger.info("=" * 80)
        logger.info(f"Total Return:        {metrics['total_return']:>10.2%}")
        logger.info(f"Annualized Return:   {metrics['annualized_return']:>10.2%}")
        logger.info(f"Annualized Vol:      {metrics['annualized_volatility']:>10.2%}")
        logger.info(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        logger.info(f"Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
        logger.info(f"Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")
        logger.info(f"Win Rate:            {metrics['win_rate']:>10.2%}")
        logger.info(f"Profit Factor:       {metrics['profit_factor']:>10.2f}")
        logger.info(f"Total Trades:        {metrics['total_trades']:>10d}")
        logger.info("=" * 80)
        
        return metrics
    
    def compute_buy_and_hold_benchmark(
        self,
        actual_returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute buy-and-hold benchmark (always long).
        
        FINAL_PLAN.md Section 7.1: Buy-and-Hold Benchmark
        
        Args:
            actual_returns: Actual market returns
        
        Returns:
            Dictionary of benchmark metrics
        """
        # Buy and hold = always long, no transaction costs
        cumulative_returns = (1 + actual_returns).cumprod()
        capital = self.initial_capital * cumulative_returns
        
        # Compute metrics
        total_return = capital[-1] / capital[0] - 1
        n_days = len(actual_returns)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        annualized_vol = actual_returns.std() * np.sqrt(252)
        sharpe = annualized_return / annualized_vol if annualized_vol != 0 else 0.0
        
        peak = np.maximum.accumulate(capital)
        drawdown = (capital - peak) / peak
        max_drawdown = drawdown.min()
        
        benchmark = {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
        }
        
        logger.info("Buy-and-Hold Benchmark:")
        logger.info(f"  Total Return:      {benchmark['total_return']:>10.2%}")
        logger.info(f"  Annualized Return: {benchmark['annualized_return']:>10.2%}")
        logger.info(f"  Sharpe Ratio:      {benchmark['sharpe_ratio']:>10.2f}")
        
        return benchmark
