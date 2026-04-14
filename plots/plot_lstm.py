#!/usr/bin/env python3
"""
Visualize LSTM predictions overlaid on OHLCV charts.
Handles window alignment and shows trading signals.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from pathlib import Path


def load_and_align(ticker, results_dir, seed=42):
    """Load predictions and align to OHLCV."""
    # Load aligned predictions from run_test() - this already has OHLCV merged
    pred_csv = results_dir / f"lstm_aligned_{ticker}_test_seed{seed}.csv"

    if not pred_csv.exists():
        raise FileNotFoundError(f"Aligned predictions not found: {pred_csv}")

    df = pd.read_csv(pred_csv)

    # Parse date column if it exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


def plot_ohlcv_with_signals(df, ticker, save_path=None):
    """Create comprehensive chart with OHLCV, signals, predictions, and PnL."""

    # Check if OHLCV data is available
    has_ohlcv = all(
        col in df.columns for col in ["open", "high", "low", "close", "volume"]
    )
    has_pnl = "equity" in df.columns and "pnl" in df.columns

    if has_ohlcv:
        # Determine number of panels (add PnL if available)
        n_panels = 4 if has_pnl else 3
        height_ratios = [3, 1, 1.5, 1.5] if has_pnl else [3, 1, 1.5]

        # Full chart with OHLCV (+ optional PnL)
        fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=(20, 5 * n_panels),
            gridspec_kw={"height_ratios": height_ratios},
            sharex=True,
        )

        if has_pnl:
            ax_price, ax_vol, ax_pred, ax_pnl = axes
        else:
            ax_price, ax_vol, ax_pred = axes
        x = np.arange(len(df))

        # --- Panel 1: Candlesticks + Signals ---
        for i in range(len(df)):
            row = df.iloc[i]
            if pd.isna(row["open"]) or pd.isna(row["close"]):
                continue

            color = "green" if row["close"] >= row["open"] else "red"

            # Candle body
            height = abs(row["close"] - row["open"])
            bottom = min(row["open"], row["close"])
            rect = Rectangle(
                (i - 0.4, bottom),
                0.8,
                height,
                facecolor=color,
                edgecolor=color,
                alpha=0.8,
            )
            ax_price.add_patch(rect)

            # Wick
            ax_price.plot([i, i], [row["low"], row["high"]], color=color, lw=0.5)

        # Overlay prediction strength as background
        valid = df["predicted_return"].notna()
        if valid.any():
            max_ret = df.loc[valid, "predicted_return"].abs().max()
            if max_ret > 0:
                for i in range(len(df)):
                    if valid.iloc[i]:
                        row = df.iloc[i]
                        intensity = abs(row["predicted_return"]) / max_ret
                        if intensity > 0.1:
                            color = "green" if row["predicted_return"] > 0 else "red"
                            ax_price.axvspan(
                                i - 0.5, i + 0.5, alpha=intensity * 0.3, color=color
                            )

        # Signal markers
        buys = df[df["signal"] == 1]
        sells = df[df["signal"] == -1]

        if len(buys) > 0:
            buy_indices = [df.index.get_loc(idx) for idx in buys.index]
            ax_price.scatter(
                buy_indices,
                buys["low"].values * 0.995,
                marker="^",
                color="darkgreen",
                s=120,
                zorder=5,
                label=f"Buy ({len(buys)})",
                edgecolors="black",
                linewidths=0.5,
            )
        if len(sells) > 0:
            sell_indices = [df.index.get_loc(idx) for idx in sells.index]
            ax_price.scatter(
                sell_indices,
                sells["high"].values * 1.005,
                marker="v",
                color="darkred",
                s=120,
                zorder=5,
                label=f"Sell ({len(sells)})",
                edgecolors="black",
                linewidths=0.5,
            )

        ax_price.set_ylabel("Price ($)", fontsize=12)
        ax_price.set_title(
            f"{ticker} - LSTM Trading Signals on OHLCV Chart",
            fontsize=14,
            fontweight="bold",
        )
        ax_price.legend(loc="upper left", fontsize=10)
        ax_price.grid(True, alpha=0.3)
        ax_price.set_xlim(-1, len(df))

        # --- Panel 2: Volume ---
        colors = [
            "green" if df.iloc[i]["close"] >= df.iloc[i]["open"] else "red"
            for i in range(len(df))
            if not pd.isna(df.iloc[i]["close"]) and not pd.isna(df.iloc[i]["open"])
        ]
        valid_volume = df["volume"].notna()
        ax_vol.bar(
            x[valid_volume],
            df.loc[valid_volume, "volume"],
            color=colors,
            alpha=0.6,
            width=0.8,
        )
        ax_vol.set_ylabel("Volume", fontsize=12)
        ax_vol.grid(True, alpha=0.3)
        ax_vol.ticklabel_format(style="plain", axis="y")

        # --- Panel 3: Predicted Returns ---
        valid_pred = df["predicted_return"].notna()
        if valid_pred.any():
            ax_pred.plot(
                x[valid_pred],
                df.loc[valid_pred, "predicted_return"],
                "b-",
                lw=2,
                label="Predicted Return",
                alpha=0.8,
            )
            ax_pred.fill_between(
                x[valid_pred],
                0,
                df.loc[valid_pred, "predicted_return"],
                where=(df.loc[valid_pred, "predicted_return"] > 0),
                color="green",
                alpha=0.2,
            )
            ax_pred.fill_between(
                x[valid_pred],
                0,
                df.loc[valid_pred, "predicted_return"],
                where=(df.loc[valid_pred, "predicted_return"] < 0),
                color="red",
                alpha=0.2,
            )

        ax_pred.axhline(y=0, color="k", linestyle="-", lw=1, alpha=0.5)
        ax_pred.axhline(
            y=0.001,
            color="g",
            linestyle="--",
            alpha=0.7,
            lw=1.5,
            label="Buy Threshold (+0.1%)",
        )
        ax_pred.axhline(
            y=-0.001,
            color="r",
            linestyle="--",
            alpha=0.7,
            lw=1.5,
            label="Sell Threshold (-0.1%)",
        )

        # Mark signals on prediction track
        if len(buys) > 0:
            buy_indices = [df.index.get_loc(idx) for idx in buys.index]
            ax_pred.scatter(
                buy_indices,
                buys["predicted_return"].values,
                marker="^",
                color="darkgreen",
                s=100,
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )
        if len(sells) > 0:
            sell_indices = [df.index.get_loc(idx) for idx in sells.index]
            ax_pred.scatter(
                sell_indices,
                sells["predicted_return"].values,
                marker="v",
                color="darkred",
                s=100,
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )

        ax_pred.set_ylabel("Predicted Return", fontsize=12)
        if not has_pnl:
            ax_pred.set_xlabel("Time (Bar Index)", fontsize=12)
        ax_pred.legend(loc="upper left", fontsize=10)
        ax_pred.grid(True, alpha=0.3)

        # Format y-axis as percentage
        ax_pred.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y*100:.2f}%")
        )

        # --- Panel 4: PnL / Equity Curve (if available) ---
        if has_pnl:
            equity = df["equity"].values
            pnl = df["pnl"].values
            initial_capital = equity[0] - pnl[0] if len(equity) > 0 else 100000

            # Plot equity curve
            ax_pnl.plot(
                x, equity, label="Equity Curve", linewidth=2, color="blue", alpha=0.8
            )
            ax_pnl.axhline(
                y=initial_capital,
                color="k",
                linestyle="--",
                alpha=0.5,
                label=f"Initial Capital (${initial_capital:,.0f})",
            )

            # Shade profitable/unprofitable regions
            ax_pnl.fill_between(
                x,
                initial_capital,
                equity,
                where=(equity >= initial_capital),
                color="green",
                alpha=0.2,
                label="Profit",
            )
            ax_pnl.fill_between(
                x,
                initial_capital,
                equity,
                where=(equity < initial_capital),
                color="red",
                alpha=0.2,
                label="Loss",
            )

            # Calculate metrics
            final_pnl = pnl[-1]
            final_return = (
                (equity[-1] / initial_capital - 1) if initial_capital > 0 else 0
            )

            # Calculate Sharpe and Max Drawdown
            if "strategy_return" in df.columns:
                strategy_returns = df["strategy_return"].values
                sharpe = (
                    np.mean(strategy_returns)
                    / (np.std(strategy_returns) + 1e-10)
                    * np.sqrt(252 * 390)
                )
            else:
                sharpe = 0.0

            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / (running_max + 1e-10)
            max_dd = np.min(drawdown)

            # Add final PnL annotation
            ax_pnl.annotate(
                f"Final PnL: ${final_pnl:,.2f} ({final_return:+.2%})",
                xy=(len(equity) - 1, equity[-1]),
                xytext=(len(equity) * 0.7, equity[-1]),
                fontsize=12,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="lightgreen" if final_pnl > 0 else "lightcoral",
                    alpha=0.8,
                ),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            )

            ax_pnl.set_title(
                f"{ticker} Equity Curve | Sharpe: {sharpe:.3f} | Max DD: {max_dd:.2%}",
                fontsize=14,
                fontweight="bold",
            )
            ax_pnl.set_xlabel("Time (Bar Index)", fontsize=12)
            ax_pnl.set_ylabel("Equity ($)", fontsize=12)
            ax_pnl.legend(loc="upper left", fontsize=10)
            ax_pnl.grid(True, alpha=0.3)
            ax_pnl.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
            )

    else:
        # Simplified 2-panel chart without OHLCV
        fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
        ax_actual, ax_pred = axes
        x = np.arange(len(df))

        # --- Panel 1: Actual Returns ---
        if "actual_return" in df.columns:
            valid_actual = df["actual_return"].notna()
            ax_actual.plot(
                x[valid_actual],
                df.loc[valid_actual, "actual_return"],
                "k-",
                lw=1,
                alpha=0.7,
                label="Actual Return",
            )
            ax_actual.axhline(y=0, color="k", linestyle="-", lw=0.5)
            ax_actual.set_ylabel("Actual Return", fontsize=12)
            ax_actual.set_title(
                f"{ticker} - LSTM Predictions (No OHLCV Data)",
                fontsize=14,
                fontweight="bold",
            )
            ax_actual.legend(loc="upper left", fontsize=10)
            ax_actual.grid(True, alpha=0.3)
            ax_actual.yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: f"{y*100:.2f}%")
            )

        # --- Panel 2: Predicted Returns ---
        valid_pred = df["predicted_return"].notna()
        if valid_pred.any():
            ax_pred.plot(
                x[valid_pred],
                df.loc[valid_pred, "predicted_return"],
                "b-",
                lw=2,
                label="Predicted Return",
                alpha=0.8,
            )
            ax_pred.fill_between(
                x[valid_pred],
                0,
                df.loc[valid_pred, "predicted_return"],
                where=(df.loc[valid_pred, "predicted_return"] > 0),
                color="green",
                alpha=0.2,
            )
            ax_pred.fill_between(
                x[valid_pred],
                0,
                df.loc[valid_pred, "predicted_return"],
                where=(df.loc[valid_pred, "predicted_return"] < 0),
                color="red",
                alpha=0.2,
            )

        ax_pred.axhline(y=0, color="k", linestyle="-", lw=1, alpha=0.5)
        ax_pred.axhline(
            y=0.001, color="g", linestyle="--", alpha=0.7, lw=1.5, label="Buy Threshold"
        )
        ax_pred.axhline(
            y=-0.001,
            color="r",
            linestyle="--",
            alpha=0.7,
            lw=1.5,
            label="Sell Threshold",
        )

        # Mark signals
        buys = df[df["signal"] == 1]
        sells = df[df["signal"] == -1]

        if len(buys) > 0:
            buy_indices = [df.index.get_loc(idx) for idx in buys.index]
            ax_pred.scatter(
                buy_indices,
                buys["predicted_return"].values,
                marker="^",
                color="darkgreen",
                s=100,
                zorder=5,
                label=f"Buy ({len(buys)})",
                edgecolors="black",
                linewidths=0.5,
            )
        if len(sells) > 0:
            sell_indices = [df.index.get_loc(idx) for idx in sells.index]
            ax_pred.scatter(
                sell_indices,
                sells["predicted_return"].values,
                marker="v",
                color="darkred",
                s=100,
                zorder=5,
                label=f"Sell ({len(sells)})",
                edgecolors="black",
                linewidths=0.5,
            )

        ax_pred.set_ylabel("Predicted Return", fontsize=12)
        ax_pred.set_xlabel("Time (Bar Index)", fontsize=12)
        ax_pred.legend(loc="upper left", fontsize=10)
        ax_pred.grid(True, alpha=0.3)
        ax_pred.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y*100:.2f}%"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SELECTED] Saved plot to {save_path}")

    return fig


def generate_signal_summary(df, ticker):
    """Generate a text summary of trading signals and PnL."""
    summary = []
    summary.append(f"\n{'='*60}")
    summary.append(f"LSTM Trading Signal Summary for {ticker}")
    summary.append(f"{'='*60}")

    # Signal counts
    n_buy = (df["signal"] == 1).sum()
    n_sell = (df["signal"] == -1).sum()
    n_hold = (df["signal"] == 0).sum()
    total = len(df)

    summary.append(f"\nSignal Distribution:")
    summary.append(f"  Buy signals:  {n_buy:5d} ({n_buy/total*100:5.2f}%)")
    summary.append(f"  Sell signals: {n_sell:5d} ({n_sell/total*100:5.2f}%)")
    summary.append(f"  Hold signals: {n_hold:5d} ({n_hold/total*100:5.2f}%)")
    summary.append(f"  Total:        {total:5d}")

    # Prediction statistics
    if "predicted_return" in df.columns:
        pred_valid = df["predicted_return"].notna()
        if pred_valid.any():
            pred_mean = df.loc[pred_valid, "predicted_return"].mean()
            pred_std = df.loc[pred_valid, "predicted_return"].std()
            pred_min = df.loc[pred_valid, "predicted_return"].min()
            pred_max = df.loc[pred_valid, "predicted_return"].max()

            summary.append(f"\nPrediction Statistics:")
            summary.append(f"  Mean:   {pred_mean*100:7.4f}%")
            summary.append(f"  Std:    {pred_std*100:7.4f}%")
            summary.append(f"  Min:    {pred_min*100:7.4f}%")
            summary.append(f"  Max:    {pred_max*100:7.4f}%")

    # Actual return statistics (if available)
    if "actual_return" in df.columns:
        actual_valid = df["actual_return"].notna()
        if actual_valid.any():
            actual_mean = df.loc[actual_valid, "actual_return"].mean()
            actual_std = df.loc[actual_valid, "actual_return"].std()

            summary.append(f"\nActual Return Statistics:")
            summary.append(f"  Mean:   {actual_mean*100:7.4f}%")
            summary.append(f"  Std:    {actual_std*100:7.4f}%")

    # PnL statistics (if available)
    if "equity" in df.columns and "pnl" in df.columns:
        equity = df["equity"].values
        pnl = df["pnl"].values
        initial_capital = equity[0] - pnl[0] if len(equity) > 0 else 100000
        final_equity = equity[-1]
        final_pnl = pnl[-1]
        total_return = (
            (final_equity / initial_capital - 1) if initial_capital > 0 else 0
        )

        # Calculate Sharpe and Max Drawdown
        if "strategy_return" in df.columns:
            strategy_returns = df["strategy_return"].values
            sharpe = (
                np.mean(strategy_returns)
                / (np.std(strategy_returns) + 1e-10)
                * np.sqrt(252 * 390)
            )
        else:
            sharpe = 0.0

        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / (running_max + 1e-10)
        max_dd = np.min(drawdown)

        # Count trades
        if "signal" in df.columns:
            position = df["signal"].values
            position_changes = np.diff(position, prepend=0)
            n_trades = (np.abs(position_changes) > 0).sum()
        else:
            n_trades = 0

        summary.append(f"\nPnL Statistics:")
        summary.append(f"  Initial Capital:  ${initial_capital:12,.2f}")
        summary.append(f"  Final Equity:     ${final_equity:12,.2f}")
        summary.append(f"  Total PnL:        ${final_pnl:12,.2f}")
        summary.append(f"  Total Return:     {total_return:12.2%}")
        summary.append(f"  Sharpe Ratio:     {sharpe:12.3f}")
        summary.append(f"  Max Drawdown:     {max_dd:12.2%}")
        summary.append(f"  Number of Trades: {n_trades:12d}")

    summary.append(f"{'='*60}\n")

    return "\n".join(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize LSTM predictions with trading signals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker", required=True, help="Ticker symbol (e.g., SPY, AAPL)"
    )
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed used in training"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output plot path (default: results/plots/lstm_ohlcv_{ticker}.png)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Set default output path if not specified
    if args.output is None:
        output_dir = results_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"lstm_ohlcv_{args.ticker}_seed{args.seed}.png")

    print(f"\nLoading LSTM predictions for {args.ticker}...")

    try:
        df = load_and_align(args.ticker, results_dir, seed=args.seed)
        print(f"[SELECTED] Loaded {len(df)} predictions")

        # Generate and print summary
        summary = generate_signal_summary(df, args.ticker)
        print(summary)

        # Save summary to text file
        summary_path = Path(args.output).with_suffix(".txt")
        with open(summary_path, "w") as f:
            f.write(summary)
        print(f"[SELECTED] Saved summary to {summary_path}")

        # Create plot
        print(f"\nGenerating plot...")
        plot_ohlcv_with_signals(df, args.ticker, args.output)

        print(f"\n[SELECTED] Complete! Plot saved to {args.output}")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nMake sure you have run:")
        print(
            f"  python scripts/run_lstm_baseline.py --ticker {args.ticker} --mode test --seed {args.seed}"
        )
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
