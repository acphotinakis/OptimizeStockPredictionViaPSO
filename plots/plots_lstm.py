from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def plot_lstm_pnl(
    results_dir,
    pnl_stats,
    y_test_windows,
    y_pred_test,
    signal_threshold,
    ticker,
    aligned_df,
    initial_capital,
    n_sell_signals,
    n_hold_signals,
    n_buy_signals,
    signals,
    tag,
    results_dict,
):

    # Create comprehensive plots
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of subplots based on PnL availability
    n_plots = 3 if pnl_stats is not None else 2

    # Plot 1: Predictions vs Actual (with optional PnL)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 5 * n_plots))

    n_samples = min(1000, len(y_test_windows))

    if "equity" in aligned_df.columns:
        pnl_stats["total_pnl"] = aligned_df["equity"].iloc[-1] - initial_capital

    # Panel 1: Time series
    ax_idx = 0
    axes[ax_idx].plot(
        y_test_windows[:n_samples], label="Actual", alpha=0.7, linewidth=1.5
    )
    axes[ax_idx].plot(
        y_pred_test[:n_samples], label="Predicted", alpha=0.7, linewidth=1.5
    )
    axes[ax_idx].axhline(
        y=signal_threshold,
        color="g",
        linestyle="--",
        alpha=0.5,
        label=f"Buy Threshold ({signal_threshold})",
    )
    axes[ax_idx].axhline(
        y=-signal_threshold,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Sell Threshold ({-signal_threshold})",
    )
    axes[ax_idx].axhline(y=0, color="k", linestyle="-", alpha=0.3, linewidth=0.5)
    axes[ax_idx].set_title(
        f"{ticker} LSTM Baseline Test Predictions", fontsize=14, fontweight="bold"
    )
    axes[ax_idx].set_xlabel("Sample")
    axes[ax_idx].set_ylabel("Return")
    axes[ax_idx].legend(loc="upper left")
    axes[ax_idx].grid(True, alpha=0.3)

    # Panel 2: Scatter plot
    ax_idx = 1
    axes[ax_idx].scatter(
        y_test_windows[:n_samples], y_pred_test[:n_samples], alpha=0.3, s=10
    )
    axes[ax_idx].plot(
        [-0.05, 0.05], [-0.05, 0.05], "r--", label="Perfect Prediction", linewidth=2
    )
    axes[ax_idx].axhline(y=signal_threshold, color="g", linestyle="--", alpha=0.5)
    axes[ax_idx].axhline(y=-signal_threshold, color="r", linestyle="--", alpha=0.5)
    axes[ax_idx].axvline(x=signal_threshold, color="g", linestyle="--", alpha=0.5)
    axes[ax_idx].axvline(x=-signal_threshold, color="r", linestyle="--", alpha=0.5)
    axes[ax_idx].set_title(
        f"{ticker} Predicted vs Actual Returns", fontsize=14, fontweight="bold"
    )
    axes[ax_idx].set_xlabel("Actual Return")
    axes[ax_idx].set_ylabel("Predicted Return")
    axes[ax_idx].legend()
    axes[ax_idx].grid(True, alpha=0.3)

    # Panel 3: PnL / Equity Curve (if available)
    if pnl_stats is not None and "equity" in aligned_df.columns:
        ax_idx = 2
        equity = aligned_df["equity"].values

        # Plot equity curve
        axes[ax_idx].plot(equity, label="Equity Curve", linewidth=2, color="blue")
        axes[ax_idx].axhline(
            y=initial_capital,
            color="k",
            linestyle="--",
            alpha=0.5,
            label=f"Initial Capital (${initial_capital:,.0f})",
        )

        # Shade profitable/unprofitable regions
        axes[ax_idx].fill_between(
            range(len(equity)),
            initial_capital,
            equity,
            where=(equity >= initial_capital),
            color="green",
            alpha=0.2,
            label="Profit",
        )
        axes[ax_idx].fill_between(
            range(len(equity)),
            initial_capital,
            equity,
            where=(equity < initial_capital),
            color="red",
            alpha=0.2,
            label="Loss",
        )

        # Add final PnL annotation
        final_pnl = equity[-1] - initial_capital
        final_return = (equity[-1] / initial_capital - 1) if initial_capital > 0 else 0.0
        
        axes[ax_idx].annotate(
            f"Final PnL: ${final_pnl:,.2f} ({final_return:+.2%})",
            xy=(len(equity) - 1, equity[-1]),
            xytext=(len(equity) * 0.7, equity[-1]),
            fontsize=12,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="lightgreen" if final_pnl > 0 else "lightcoral",
                alpha=0.7,
            ),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )

        # Get Sharpe and Max DD from pnl_stats (handles both key formats)
        sharpe = pnl_stats.get('sharpe_ratio', pnl_stats.get('sharpe', 0.0))
        max_dd = pnl_stats.get('max_drawdown', 0.0)
        
        axes[ax_idx].set_title(
            f"{ticker} Equity Curve | Sharpe: {sharpe:.3f} | Max DD: {max_dd:.2%}",
            fontsize=14,
            fontweight="bold",
        )
        axes[ax_idx].set_xlabel("Sample")
        axes[ax_idx].set_ylabel("Equity ($)")
        axes[ax_idx].legend(loc="upper left")
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.0f}"))

    plt.tight_layout()
    plot_path = plots_dir / f"lstm_baseline_test_{tag}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved to {plot_path}")
    plt.close()

    # Plot 2: Signal distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Signal counts
    signal_labels = ["Sell", "Hold", "Buy"]
    signal_counts = [n_sell_signals, n_hold_signals, n_buy_signals]
    signal_colors = ["red", "gray", "green"]

    axes[0].bar(signal_labels, signal_counts, color=signal_colors, alpha=0.7)
    axes[0].set_title(f"{ticker} Signal Distribution")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Add percentage labels on bars
    for i, (label, count) in enumerate(zip(signal_labels, signal_counts)):
        pct = count / len(signals) * 100
        axes[0].text(i, count, f"{count}\n({pct:.1f}%)", ha="center", va="bottom")

    # Prediction distribution
    axes[1].hist(y_pred_test, bins=50, alpha=0.7, edgecolor="black")
    axes[1].axvline(
        x=signal_threshold,
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Buy Threshold",
    )
    axes[1].axvline(
        x=-signal_threshold,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Sell Threshold",
    )
    axes[1].axvline(x=0, color="k", linestyle="-", linewidth=1, alpha=0.5)
    axes[1].set_title(f"{ticker} Prediction Distribution")
    axes[1].set_xlabel("Predicted Return")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    signal_plot_path = plots_dir / f"lstm_signals_{tag}.png"
    plt.savefig(signal_plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Signal plot saved to {signal_plot_path}")
    plt.close()
