import logging
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..backtesting.backtester import BacktestResult

logger = logging.getLogger(__name__)


def plot_equity_curve(
    backtest_results: Union[pd.DataFrame, BacktestResult],
    benchmark_returns: np.ndarray,
    model_type: str,
    output_path: Path,
) -> None:
    """
    Plot portfolio equity curve with buy-and-hold benchmark.

    Features:
    - Strategy equity line
    - Buy-and-hold benchmark line
    - Grid for readability

    Args:
        backtest_results: Results from backtesting engine
        benchmark_returns: Actual returns for buy-and-hold
        model_type: Model identifier for title
        output_path: Directory to save plot

    Saves to: output_path / "equity_curve.png"
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract strategy equity
    if isinstance(backtest_results, pd.DataFrame):
        dates = backtest_results["date"].values
        capital = backtest_results["capital"].values
    else:
        # BacktestResult from Backtester
        capital = backtest_results.equity_curve
        dates = np.arange(len(capital))

    # Plot strategy
    ax.plot(
        dates,
        capital,
        label=f"{model_type.upper()} Strategy",
        linewidth=1.5,
        color="blue",
    )

    # Compute and plot benchmark
    initial_capital = capital[0]
    benchmark_capital = initial_capital * np.cumprod(1 + benchmark_returns)
    ax.plot(
        dates,
        benchmark_capital,
        label="Buy & Hold",
        linestyle="--",
        alpha=0.7,
        color="orange",
    )

    ax.set_xlabel("Date" if isinstance(dates[0], (pd.Timestamp, str)) else "Time")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title(f"Equity Curve: {model_type.upper()}")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("   Equity curve plot saved")


def plot_drawdown_chart(
    backtest_results: Union[pd.DataFrame, BacktestResult],
    model_type: str,
    output_path: Path,
) -> None:
    """
    Plot drawdown curve over time.

    Shows peak-to-trough decline as percentage.

    Args:
        backtest_results: Results from backtesting engine
        model_type: Model identifier for title
        output_path: Directory to save plot

    Saves to: output_path / "drawdown.png"
    """
    # Extract equity
    if isinstance(backtest_results, pd.DataFrame):
        equity = backtest_results["capital"].values
        dates = backtest_results["date"].values
    else:
        equity = backtest_results.equity_curve
        dates = np.arange(len(equity))

    # Compute drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(
        dates,
        drawdown * 100,
        0,
        color="red",
        alpha=0.3,
        label="Drawdown",
    )
    ax.plot(dates, drawdown * 100, color="darkred", linewidth=1)

    ax.set_xlabel("Date" if isinstance(dates[0], (pd.Timestamp, str)) else "Time")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(f"Drawdown: {model_type.upper()}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(alpha=0.3)

    output_path = Path(output_path)
    plt.savefig(output_path / "drawdown.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("   Drawdown chart saved")


def plot_returns_distribution(
    backtest_results: Union[pd.DataFrame, BacktestResult],
    model_type: str,
    output_path: Path,
) -> None:
    """
    Plot distribution of strategy returns.

    Args:
        backtest_results: Results from backtesting engine
        model_type: Model identifier for title
        output_path: Directory to save plot

    Saves to: output_path / "returns_distribution.png"
    """
    # Extract returns
    if isinstance(backtest_results, pd.DataFrame):
        returns = backtest_results["strategy_return"].values
    else:
        returns = backtest_results.bar_returns

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns * 100, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Return")

    # Add statistics
    mean_ret = np.mean(returns) * 100
    std_ret = np.std(returns) * 100
    ax.axvline(
        mean_ret,
        color="green",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean: {mean_ret:.3f}%",
    )

    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Returns Distribution: {model_type.upper()}")
    ax.legend()
    ax.grid(alpha=0.3)

    output_path = Path(output_path)
    plt.savefig(output_path / "returns_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("   Returns distribution saved")


def plot_signal_analysis(
    backtest_results: Union[pd.DataFrame, BacktestResult],
    model_type: str,
    output_path: Path,
) -> None:
    """
    Plot signal distribution and trading activity.

    Args:
        backtest_results: Results from backtesting engine
        model_type: Model identifier for title
        output_path: Directory to save plot

    Saves to: output_path / "signal_analysis.png"
    """
    # Extract signals
    if isinstance(backtest_results, pd.DataFrame):
        signals = backtest_results["signal"].values
        dates = backtest_results["date"].values
    else:
        # BacktestResult doesn't store signals directly
        logger.warning("Signal analysis not available for BacktestResult format")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Signal distribution
    unique, counts = np.unique(signals, return_counts=True)
    colors = []
    for sig in unique:
        if sig == -1:
            colors.append("red")
        elif sig == 0:
            colors.append("gray")
        else:
            colors.append("green")

    ax1.bar(unique, counts, color=colors, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Signal")
    ax1.set_ylabel("Count")
    ax1.set_title("Signal Distribution")
    ax1.set_xticks([-1, 0, 1])
    ax1.set_xticklabels(["SHORT", "NEUTRAL", "LONG"])
    ax1.grid(alpha=0.3, axis="y")

    # Add percentages
    for sig, count in zip(unique, counts):
        pct = 100 * count / len(signals)
        ax1.text(sig, count, f"{pct:.1f}%", ha="center", va="bottom")

    # Signal transitions (trading activity)
    transitions = np.abs(np.diff(signals, prepend=signals[0]))
    ax2.plot(dates, transitions, alpha=0.6, linewidth=0.8, color="steelblue")
    ax2.fill_between(dates, transitions, alpha=0.3, color="steelblue")
    ax2.set_xlabel("Date" if isinstance(dates[0], (pd.Timestamp, str)) else "Time")
    ax2.set_ylabel("Position Change")
    ax2.set_title("Trading Activity (Position Changes)")
    ax2.grid(alpha=0.3)

    output_path = Path(output_path)
    plt.savefig(output_path / "signal_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("   Signal analysis saved")


def create_all_plots(
    backtest_results: Union[pd.DataFrame, BacktestResult],
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    model_type: str,
    output_dir: Path,
) -> None:
    """
    Generate all standard backtesting visualizations.

    Creates:
    - equity_curve.png
    - drawdown.png
    - returns_distribution.png
    - signal_analysis.png

    Args:
        backtest_results: Results from backtesting engine
        predictions: Model predictions
        actual_returns: True returns (for benchmark)
        model_type: Model identifier
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating visualizations...")

    try:
        plot_equity_curve(backtest_results, actual_returns, model_type, output_dir)
    except Exception as e:
        logger.warning(f"Failed to generate equity curve: {e}")

    try:
        plot_drawdown_chart(backtest_results, model_type, output_dir)
    except Exception as e:
        logger.warning(f"Failed to generate drawdown chart: {e}")

    try:
        plot_returns_distribution(backtest_results, model_type, output_dir)
    except Exception as e:
        logger.warning(f"Failed to generate returns distribution: {e}")

    try:
        plot_signal_analysis(backtest_results, model_type, output_dir)
    except Exception as e:
        logger.warning(f"Failed to generate signal analysis: {e}")

    logger.info("=" * 80)
    logger.info(f" All plots saved to {output_dir}")
    logger.info("=" * 80)
