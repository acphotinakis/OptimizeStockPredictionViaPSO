"""
generate_report.py
==================
Generate a self-contained HTML backtest report with all 5 plots
built entirely from real data files produced by train.py and backtester.py.

Files read
----------
  models/training_history.json             per-epoch loss / lr arrays  (train.py)
  models/test_predictions.jsonl            test-set actual vs predicted (train.py)
  backtest_results/summary_<id>.json       metrics summary              (backtester.py)
  backtest_results/predictions_<id>.jsonl  per-bar predictions          (backtester.py)
  backtest_results/trades_<id>.jsonl       trade ledger                 (backtester.py)
  backtest_results/equity_<id>.jsonl       equity curve                 (backtester.py)

Usage
-----
  python generate_report.py                           # auto-find latest run
  python generate_report.py --run-id 20240930_161500
  python generate_report.py --out path/to/report.html
"""

import argparse
import base64
import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("report")

# Project root (same directory as this script)
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
BACKTEST_DIR = ROOT / "backtest_results"
DEFAULT_OUT = BACKTEST_DIR / "report.html"

# Colour palette
BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
TEXT = "#c9d1d9"
DIM = "#8b949e"
BLUE = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
PURPLE = "#d2a8ff"
YELLOW = "#e3b341"
ORANGE = "#f78166"

STYLE = {
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": DIM,
    "ytick.color": DIM,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


# ══════════════════════════════════════════════════════════════════════════════
# FILE LOADING
# ══════════════════════════════════════════════════════════════════════════════


def find_latest_run_id() -> Optional[str]:
    summaries = sorted(BACKTEST_DIR.glob("summary_*.json"))
    if not summaries:
        return None
    return summaries[-1].stem.replace("summary_", "")


def load_json(path: Path) -> Dict:
    path = Path(path)
    if not path.exists():
        log.warning(f"Not found (skipping): {path}")
        return {}
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        log.warning(f"Not found (skipping): {path}")
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_all(run_id: str):
    summary = load_json(BACKTEST_DIR / f"summary_{run_id}.json")
    history = load_json(MODELS_DIR / "training_history.json")
    test_preds = load_jsonl(MODELS_DIR / "test_predictions.jsonl")
    bt_preds = load_jsonl(BACKTEST_DIR / f"predictions_{run_id}.jsonl")
    equity = load_jsonl(BACKTEST_DIR / f"equity_{run_id}.jsonl")
    trades = load_jsonl(BACKTEST_DIR / f"trades_{run_id}.jsonl")

    log.info(f"training_history epochs  : {len(history.get('loss', []))}")
    log.info(f"test_predictions bars    : {len(test_preds)}")
    log.info(f"backtest predictions bars: {len(bt_preds)}")
    log.info(f"equity bars              : {len(equity)}")
    log.info(f"trades                   : {len(trades)}")

    return summary, history, test_preds, bt_preds, equity, trades


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG, edgecolor="none"
    )
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_tag(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" class="plot">'


def sax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.tick_params(colors=DIM, labelsize=9)
    ax.grid(True, color="#21262d", linestyle="--", linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, pad=8, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=DIM, fontsize=9, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, color=DIM, fontsize=9, labelpad=6)


def to_dates(col: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(col)
    except Exception:
        return col


def missing_placeholder(message: str) -> str:
    fig, ax = plt.subplots(figsize=(14, 3), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        color=DIM,
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.axis("off")
    return fig_to_b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — TRAINING HISTORY
# reads: models/training_history.json  (keys: loss, val_loss, learning_rate)
# ══════════════════════════════════════════════════════════════════════════════


def plot_training(history: Dict) -> str:
    if not history or "loss" not in history:
        return missing_placeholder("No training history found.\nRun: python train.py")

    loss = np.array(history["loss"], dtype=np.float32)
    val = np.array(history["val_loss"], dtype=np.float32)
    lr_raw = history.get("learning_rate", history.get("lr", []))
    epochs = np.arange(1, len(loss) + 1)
    best_ep = int(np.argmin(val)) + 1

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=BG)
        fig.suptitle(
            "Training History — 2-Layer LSTM (64 units)",
            color=TEXT,
            fontsize=13,
            fontweight="bold",
        )

        ax1.plot(epochs, loss * 1e4, color=PURPLE, label="Train Loss", linewidth=1.8)
        ax1.plot(
            epochs,
            val * 1e4,
            color=ORANGE,
            label="Val Loss",
            linewidth=1.8,
            linestyle="--",
        )
        ax1.axvline(
            best_ep,
            color=GREEN,
            linewidth=1.2,
            linestyle=":",
            label=f"Best epoch ({best_ep})",
        )
        ax1.fill_between(epochs, loss * 1e4, val * 1e4, alpha=0.07, color=BLUE)
        ax1.legend(
            framealpha=0.2,
            fontsize=9,
            labelcolor=TEXT,
            facecolor=PANEL,
            edgecolor=BORDER,
        )
        sax(ax1, "MSE Loss (x1e-4)", "Epoch", "MSE")
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        if lr_raw:
            lr = np.array(lr_raw, dtype=np.float32)
            x_lr = np.arange(1, len(lr) + 1)
            ax2.step(x_lr, lr * 1e3, color=YELLOW, where="post", linewidth=1.8)
            ax2.fill_between(x_lr, lr * 1e3, step="post", alpha=0.12, color=YELLOW)
            for i in range(1, len(lr)):
                if lr[i] < lr[i - 1] * 0.6:
                    ax2.axvline(i + 1, color=BORDER, linewidth=0.8, linestyle=":")
            ax2.set_yscale("log")
            sax(ax2, "Learning Rate Schedule", "Epoch", "LR (x1e-3)")
        else:
            ax2.text(
                0.5,
                0.5,
                f"Epochs run: {len(loss)}\nBest epoch: {best_ep}\n"
                f"Final val_loss: {float(val[-1]):.6f}",
                ha="center",
                va="center",
                color=TEXT,
                fontsize=11,
                transform=ax2.transAxes,
            )
            ax2.axis("off")
            sax(ax2, "Training Summary")

        fig.tight_layout()
    return fig_to_b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — PREDICTIONS VS ACTUAL
# reads: models/test_predictions.jsonl  (keys: timestamp, actual, predicted, abs_error)
# ══════════════════════════════════════════════════════════════════════════════


def plot_predictions(pred_records: List[Dict]) -> str:
    if not pred_records:
        return missing_placeholder("No prediction data found.\nRun: python train.py")

    df = pd.DataFrame(pred_records)
    df["timestamp"] = to_dates(df["timestamp"])
    df = df.sort_values("timestamp").tail(300).reset_index(drop=True)

    actual = df["actual"].values.astype(np.float32)
    predicted = df["predicted"].values.astype(np.float32)
    residuals = predicted - actual
    abs_err = np.abs(residuals)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    acc = 100.0 - mae / max(float(np.mean(actual)), 1e-6) * 100.0
    roll_err = pd.Series(abs_err).rolling(20).mean().values
    x = (
        df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        else np.arange(len(df))
    )

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(14, 9),
            facecolor=BG,
            gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.08},
        )
        fig.suptitle(
            "LSTM Predictions vs Actual Price (Test Set, Last 300 Bars)",
            color=TEXT,
            fontsize=13,
            fontweight="bold",
        )

        ax = axes[0]
        ax.plot(x, actual, color=BLUE, label="Actual", linewidth=1.6)
        ax.plot(
            x,
            predicted,
            color=ORANGE,
            label="LSTM Predicted",
            linewidth=1.4,
            linestyle="--",
            alpha=0.9,
        )
        ax.fill_between(x, predicted - 5, predicted + 5, alpha=0.10, color=ORANGE)
        ax.text(
            0.01,
            0.97,
            f"MAE:  {mae:.2f} pts\nRMSE: {rmse:.2f} pts\nAcc:  {acc:.2f}%",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            color=GREEN,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor=PANEL, edgecolor=GREEN, alpha=0.85
            ),
        )
        ax.legend(
            framealpha=0.2,
            fontsize=9,
            labelcolor=TEXT,
            facecolor=PANEL,
            edgecolor=BORDER,
            loc="lower right",
        )
        sax(ax, ylabel="Price ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}"))
        ax.set_xticklabels([])

        ax2 = axes[1]
        colors2 = [GREEN if r > 0 else RED for r in residuals]
        ax2.bar(x, residuals, color=colors2, alpha=0.65, width=1.0)
        ax2.axhline(0, color=TEXT, linewidth=0.8)
        ax2.axhline(mae, color=ORANGE, linewidth=0.8, linestyle=":", alpha=0.7)
        ax2.axhline(-mae, color=ORANGE, linewidth=0.8, linestyle=":", alpha=0.7)
        sax(ax2, ylabel="Residual ($)")
        ax2.set_xticklabels([])

        ax3 = axes[2]
        ax3.plot(x, abs_err, color=RED, alpha=0.3, linewidth=0.8, label="Daily |error|")
        ax3.plot(x, roll_err, color=YELLOW, linewidth=1.6, label="20-bar Rolling MAE")
        ax3.axhline(
            175.9, color=GREEN, linewidth=1.0, linestyle="--", label="Target (175.9)"
        )
        ax3.legend(
            framealpha=0.2,
            fontsize=8,
            labelcolor=TEXT,
            facecolor=PANEL,
            edgecolor=BORDER,
        )
        sax(ax3, xlabel="Date", ylabel="|Error| ($)")
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            fig.autofmt_xdate(rotation=20, ha="right")

    return fig_to_b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — EQUITY CURVE
# reads: backtest_results/equity_<id>.jsonl  (keys: timestamp, total_equity, cash)
# ══════════════════════════════════════════════════════════════════════════════


def plot_equity(equity_records: List[Dict], summary: Dict) -> str:
    if not equity_records:
        return missing_placeholder("No equity data found.\nRun: python run_backtest.py")

    df = pd.DataFrame(equity_records)
    df["timestamp"] = to_dates(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    eq = df["total_equity"].values.astype(np.float32)
    x = (
        df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        else np.arange(len(df))
    )
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-9) * 100

    tm = summary.get("trading_metrics", {})
    sharpe = tm.get("sharpe_ratio", 0.0)
    max_dd = tm.get("max_drawdown_pct", float(dd.min()))
    tot_ret = tm.get(
        "total_return_pct", float((eq[-1] - eq[0]) / max(eq[0], 1e-6) * 100)
    )
    calmar = tm.get("calmar_ratio", tot_ret / max(abs(max_dd), 1e-9))

    daily_rets = np.diff(eq) / (eq[:-1] + 1e-9)
    mu = float(np.mean(daily_rets) * 100)
    sig = float(np.std(daily_rets) * 100)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(14, 9),
            facecolor=BG,
            gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.08},
        )
        fig.suptitle(
            "Equity Curve & Portfolio Performance",
            color=TEXT,
            fontsize=13,
            fontweight="bold",
        )

        ax = axes[0]
        ax.plot(x, eq, color=GREEN, label="LSTM Strategy", linewidth=1.8)
        ax.plot(
            x,
            peak,
            color=DIM,
            linewidth=0.7,
            linestyle=":",
            alpha=0.5,
            label="Peak Equity",
        )
        ax.axhline(
            float(eq[0]),
            color=YELLOW,
            linewidth=0.8,
            linestyle=":",
            alpha=0.6,
            label="Initial Capital",
        )
        ax.text(
            0.01,
            0.97,
            f"Total Return:  {tot_ret:+.1f}%\nSharpe Ratio:  {sharpe:.2f}\n"
            f"Max Drawdown:  {max_dd:.1f}%\nCalmar Ratio:  {calmar:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            color=TEXT,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor=PANEL, edgecolor=BORDER, alpha=0.85
            ),
        )
        ax.legend(
            framealpha=0.2,
            fontsize=9,
            labelcolor=TEXT,
            facecolor=PANEL,
            edgecolor=BORDER,
            loc="lower right",
        )
        sax(ax, ylabel="Portfolio Value ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.set_xticklabels([])

        ax2 = axes[1]
        ax2.fill_between(x, dd, 0, alpha=0.45, color=RED)
        ax2.plot(x, dd, color=RED, linewidth=1.0)
        ax2.axhline(
            -10,
            color=YELLOW,
            linewidth=0.9,
            linestyle="--",
            label="Circuit breaker (-10%)",
        )
        ax2.legend(
            framealpha=0.2,
            fontsize=8,
            labelcolor=TEXT,
            facecolor=PANEL,
            edgecolor=BORDER,
        )
        sax(ax2, ylabel="Drawdown (%)")
        ax2.set_xticklabels([])

        ax3 = axes[2]
        ax3.hist(
            daily_rets * 100,
            bins=50,
            color=BLUE,
            alpha=0.65,
            edgecolor=PANEL,
            linewidth=0.3,
        )
        ax3.axvline(0, color=TEXT, linewidth=0.9)
        ax3.axvline(
            mu, color=GREEN, linewidth=1.2, linestyle="--", label=f"Mean {mu:+.3f}%"
        )
        ax3.axvline(
            mu - sig,
            color=ORANGE,
            linewidth=0.8,
            linestyle=":",
            label=f"+/-1 sigma ({sig:.3f}%)",
        )
        ax3.axvline(mu + sig, color=ORANGE, linewidth=0.8, linestyle=":")
        ax3.legend(
            framealpha=0.2,
            fontsize=8,
            labelcolor=TEXT,
            facecolor=PANEL,
            edgecolor=BORDER,
        )
        sax(ax3, xlabel="Date", ylabel="Frequency", title="Daily Return Distribution")
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            fig.autofmt_xdate(rotation=20, ha="right")

    return fig_to_b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — ROLLING METRICS
# reads: backtest_results/predictions_<id>.jsonl  (keys: timestamp, abs_error, accuracy)
# ══════════════════════════════════════════════════════════════════════════════


def plot_rolling(pred_records: List[Dict]) -> str:
    if not pred_records:
        return missing_placeholder(
            "No per-bar prediction data found.\nRun: python run_backtest.py"
        )

    df = pd.DataFrame(pred_records)
    df["timestamp"] = to_dates(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if "accuracy" not in df.columns:
        df["accuracy"] = 100.0 - (
            df["abs_error"] / df["actual"].clip(lower=1e-6) * 100.0
        )

    x = (
        df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        else np.arange(len(df))
    )
    mae_vals = df["abs_error"].values.astype(np.float32)
    acc_vals = df["accuracy"].values.astype(np.float32)
    roll_mae = pd.Series(mae_vals).rolling(20).mean().values
    roll_acc = pd.Series(acc_vals).rolling(20).mean().values
    drift = roll_mae > 185.9

    def shade_drift(ax):
        in_d = False
        ds = None
        for xi, dm in zip(x, drift):
            if dm and not in_d:
                ds = xi
                in_d = True
            elif not dm and in_d:
                ax.axvspan(ds, xi, alpha=0.12, color=RED)
                in_d = False
        if in_d:
            last = x.iloc[-1] if hasattr(x, "iloc") else x[-1]
            ax.axvspan(ds, last, alpha=0.12, color=RED)

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(14, 8),
            facecolor=BG,
            sharex=True,
            gridspec_kw={"hspace": 0.22},
        )
        fig.suptitle(
            "Rolling Performance Metrics (20-bar window)",
            color=TEXT,
            fontsize=13,
            fontweight="bold",
        )

        ax1.plot(
            x, mae_vals, color=DIM, alpha=0.3, linewidth=0.8, label="Daily |error|"
        )
        ax1.plot(x, roll_mae, color=ORANGE, linewidth=1.8, label="20-bar Rolling MAE")
        ax1.axhline(
            175.9, color=GREEN, linewidth=1.2, linestyle="--", label="Target (175.9)"
        )
        ax1.axhline(
            185.9, color=YELLOW, linewidth=1.0, linestyle=":", label="Warn (185.9)"
        )
        ax1.axhline(
            250.0, color=RED, linewidth=1.0, linestyle=":", label="Halt (250.0)"
        )
        shade_drift(ax1)
        ax1.legend(
            framealpha=0.2,
            fontsize=8.5,
            labelcolor=TEXT,
            facecolor=PANEL,
            edgecolor=BORDER,
            ncol=3,
        )
        sax(ax1, "Rolling MAE", "", "MAE (points)")

        ax2.plot(
            x, acc_vals, color=DIM, alpha=0.3, linewidth=0.8, label="Daily Accuracy"
        )
        ax2.plot(
            x, roll_acc, color=BLUE, linewidth=1.8, label="20-bar Rolling Accuracy"
        )
        ax2.axhline(
            96.41,
            color=GREEN,
            linewidth=1.2,
            linestyle="--",
            label="Paper target (96.41%)",
        )
        ax2.axhline(
            95.91, color=YELLOW, linewidth=1.0, linestyle=":", label="Warn (95.91%)"
        )
        shade_drift(ax2)
        ax2.legend(
            framealpha=0.2,
            fontsize=8.5,
            labelcolor=TEXT,
            facecolor=PANEL,
            edgecolor=BORDER,
            ncol=3,
        )
        sax(ax2, "Rolling Accuracy", "Date", "Accuracy (%)")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            fig.autofmt_xdate(rotation=20, ha="right")

    return fig_to_b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — SIGNAL DISTRIBUTION
# reads: backtest_results/trades_<id>.jsonl  (keys: direction, confidence, pct_change)
# ══════════════════════════════════════════════════════════════════════════════


def plot_signals(trade_records: List[Dict]) -> str:
    if not trade_records:
        return missing_placeholder("No trade data found.\nRun: python run_backtest.py")

    df = pd.DataFrame(trade_records)
    df["timestamp"] = to_dates(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    lm = df["direction"] == "LONG"
    sm = df["direction"] == "SHORT"
    nm = ~(lm | sm)
    counts = [int(lm.sum()), int(sm.sum()), int(nm.sum())]

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14, 9), facecolor=BG)
        fig.suptitle(
            "Signal Analysis & Distribution", color=TEXT, fontsize=13, fontweight="bold"
        )
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # Bar counts
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(
            ["LONG", "SHORT", "NEUTRAL"],
            counts,
            color=[GREEN, RED, DIM],
            alpha=0.75,
            edgecolor=PANEL,
            linewidth=0.5,
            width=0.55,
        )
        for bar, cnt in zip(bars, counts):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(cnt),
                ha="center",
                va="bottom",
                color=TEXT,
                fontsize=9,
            )
        sax(ax1, "Signal Counts", "", "# Signals")

        # Confidence histogram
        ax2 = fig.add_subplot(gs[0, 1])
        if "confidence" in df.columns and (~nm).sum() > 0:
            nc = df.loc[~nm, "confidence"].values.astype(np.float32)
            ax2.hist(
                nc,
                bins=min(25, len(nc)),
                color=PURPLE,
                alpha=0.72,
                edgecolor=PANEL,
                linewidth=0.3,
            )
            mu_c = float(nc.mean())
            ax2.axvline(
                mu_c,
                color=YELLOW,
                linewidth=1.4,
                linestyle="--",
                label=f"Mean: {mu_c:.2f}",
            )
            ax2.legend(
                framealpha=0.2,
                fontsize=8,
                labelcolor=TEXT,
                facecolor=PANEL,
                edgecolor=BORDER,
            )
        else:
            ax2.text(
                0.5,
                0.5,
                "confidence\nnot available",
                ha="center",
                va="center",
                color=DIM,
                transform=ax2.transAxes,
            )
            ax2.axis("off")
        sax(ax2, "Signal Confidence", "Confidence", "Frequency")

        # pct_change by direction
        ax3 = fig.add_subplot(gs[0, 2])
        if "pct_change" in df.columns:
            pct_l = df.loc[lm, "pct_change"].values.astype(np.float32)
            pct_s = df.loc[sm, "pct_change"].values.astype(np.float32)
            if len(pct_l):
                ax3.hist(
                    pct_l,
                    bins=20,
                    color=GREEN,
                    alpha=0.65,
                    label="LONG",
                    edgecolor=PANEL,
                    linewidth=0.3,
                )
            if len(pct_s):
                ax3.hist(
                    pct_s,
                    bins=20,
                    color=RED,
                    alpha=0.65,
                    label="SHORT",
                    edgecolor=PANEL,
                    linewidth=0.3,
                )
            ax3.axvline(
                0.15,
                color=YELLOW,
                linewidth=1.0,
                linestyle=":",
                label="Threshold +/-0.15%",
            )
            ax3.axvline(-0.15, color=YELLOW, linewidth=1.0, linestyle=":")
            ax3.axvline(0, color=TEXT, linewidth=0.7)
            ax3.legend(
                framealpha=0.2,
                fontsize=7.5,
                labelcolor=TEXT,
                facecolor=PANEL,
                edgecolor=BORDER,
            )
        else:
            ax3.text(
                0.5,
                0.5,
                "pct_change field\nnot in trade log",
                ha="center",
                va="center",
                color=DIM,
                transform=ax3.transAxes,
            )
            ax3.axis("off")
        sax(ax3, "Predicted Delta% by Dir.", "Delta%", "Count")

        # Stacked signal mix over time
        ax4 = fig.add_subplot(gs[1, :2])
        x_t = df["timestamp"]
        lr2 = pd.Series(lm.astype(float)).rolling(20, min_periods=1).mean().values
        sr2 = pd.Series(sm.astype(float)).rolling(20, min_periods=1).mean().values
        nr2 = np.clip(1 - lr2 - sr2, 0, 1)
        ax4.stackplot(
            x_t,
            lr2 * 100,
            sr2 * 100,
            nr2 * 100,
            labels=["LONG%", "SHORT%", "NEUTRAL%"],
            colors=[GREEN, RED, DIM],
            alpha=0.55,
        )
        if "confidence" in df.columns:
            conf_roll = df["confidence"].rolling(20, min_periods=1).mean().values
            ax4r = ax4.twinx()
            ax4r.plot(
                x_t,
                conf_roll,
                color=YELLOW,
                linewidth=1.4,
                label="Avg Confidence (20-bar)",
            )
            ax4r.set_ylabel("Avg Confidence", color=YELLOW, fontsize=9)
            ax4r.tick_params(colors=YELLOW, labelsize=8)
            ax4r.legend(
                framealpha=0.15,
                fontsize=8,
                labelcolor=TEXT,
                facecolor=PANEL,
                edgecolor=BORDER,
                loc="upper right",
            )
        ax4.legend(
            framealpha=0.15,
            fontsize=8,
            labelcolor=TEXT,
            facecolor=PANEL,
            edgecolor=BORDER,
            loc="upper left",
        )
        sax(ax4, "Signal Mix Over Time (20-bar rolling)", "Date", "Signal Mix (%)")
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            fig.autofmt_xdate(rotation=20, ha="right")

        # Pie chart
        ax5 = fig.add_subplot(gs[1, 2])
        vis = [
            (c, l, col)
            for c, l, col in zip(
                counts, ["LONG", "SHORT", "NEUTRAL"], [GREEN, RED, DIM]
            )
            if c > 0
        ]
        if vis:
            wedges, texts, auts = ax5.pie(
                [v[0] for v in vis],
                labels=[v[1] for v in vis],
                colors=[v[2] for v in vis],
                autopct="%1.1f%%",
                startangle=140,
                textprops={"color": TEXT, "fontsize": 9},
                wedgeprops={"edgecolor": PANEL, "linewidth": 1.2},
            )
            for at in auts:
                at.set_color(BG)
                at.set_fontweight("bold")
                at.set_fontsize(8)
        ax5.set_title(
            "Signal Breakdown", color=TEXT, fontsize=11, fontweight="bold", pad=8
        )
        ax5.set_facecolor(PANEL)

    return fig_to_b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════════════════════


def badge(ok: bool) -> str:
    return (
        f'<span class="badge pass">PASS</span>'
        if ok
        else f'<span class="badge fail">FAIL</span>'
    )


def mrow(label: str, val: str, ok=None) -> str:
    b = f"  {badge(ok)}" if ok is not None else ""
    return f"<tr><td>{label}</td><td class='val'>{val}{b}</td></tr>"


CSS = """
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--text:#c9d1d9;
      --dim:#8b949e;--blue:#58a6ff;--green:#3fb950;--red:#f85149;
      --yellow:#e3b341;--purple:#d2a8ff;--orange:#f78166;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);
     font-family:'Courier New',Courier,monospace;font-size:14px;
     line-height:1.6;padding:0 0 60px;}
.header{background:linear-gradient(135deg,#161b22 0%,#0d1117 60%);
        border-bottom:1px solid var(--border);padding:36px 48px 28px;}
.header h1{font-size:26px;color:var(--blue);letter-spacing:1px;margin-bottom:6px;}
.header .sub{color:var(--dim);font-size:13px;}
.run-id{margin-top:10px;display:inline-block;background:var(--panel);
        border:1px solid var(--border);border-radius:6px;padding:4px 12px;
        font-size:12px;color:var(--purple);}
.nav{display:flex;border-bottom:1px solid var(--border);background:var(--panel);
     padding:0 48px;position:sticky;top:0;z-index:100;overflow-x:auto;}
.nav a{display:inline-block;padding:14px 20px;color:var(--dim);text-decoration:none;
       font-size:13px;border-bottom:2px solid transparent;white-space:nowrap;}
.nav a:hover{color:var(--text);border-color:var(--border);}
.nav a.active{color:var(--blue);border-color:var(--blue);}
.content{padding:40px 48px;max-width:1280px;margin:0 auto;}
.section{margin-bottom:52px;}
.section-title{font-size:18px;font-weight:bold;color:var(--blue);margin-bottom:20px;
               padding-bottom:10px;border-bottom:1px solid var(--border);}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
          gap:16px;margin-bottom:32px;}
.kpi{background:var(--panel);border:1px solid var(--border);
     border-radius:10px;padding:18px 20px;}
.kpi:hover{border-color:var(--blue);}
.kpi .label{font-size:11px;color:var(--dim);text-transform:uppercase;
            letter-spacing:.8px;margin-bottom:8px;}
.kpi .value{font-size:22px;font-weight:bold;color:var(--text);}
.kpi .value.good{color:var(--green);}
.kpi .value.warn{color:var(--yellow);}
.kpi .sub-v{font-size:11px;color:var(--dim);margin-top:4px;}
.table-wrap{overflow-x:auto;}
table{border-collapse:collapse;width:100%;max-width:720px;font-size:13px;}
th,td{border:1px solid var(--border);padding:9px 16px;text-align:left;}
th{background:#1c2128;color:var(--dim);font-size:11px;
   text-transform:uppercase;letter-spacing:.6px;}
tr:hover{background:#1c2128;}
td.val{font-weight:bold;color:var(--text);}
.badge{font-size:11px;padding:2px 8px;border-radius:20px;
       font-weight:bold;margin-left:8px;}
.badge.pass{background:#1a3a1f;color:var(--green);border:1px solid var(--green);}
.badge.fail{background:#3a1a1a;color:var(--red);border:1px solid var(--red);}
.alert-banner{background:#1a3a1f;border:1px solid var(--green);border-radius:8px;
              padding:14px 20px;margin-bottom:28px;color:var(--green);font-size:13px;}
.alert-banner.warn{background:#2d2208;border-color:var(--yellow);color:var(--yellow);}
.bench-row{display:flex;align-items:center;gap:14px;margin:10px 0;}
.bench-label{width:180px;font-size:12px;color:var(--dim);}
.bench-bar-wrap{flex:1;background:#1c2128;border-radius:4px;height:14px;}
.bench-bar{height:14px;border-radius:4px;}
.bench-val{width:70px;font-size:12px;font-weight:bold;text-align:right;}
.plot-card{background:var(--panel);border:1px solid var(--border);
           border-radius:12px;overflow:hidden;margin-bottom:32px;}
.plot-card .ph{padding:14px 20px;border-bottom:1px solid var(--border);
               font-size:13px;color:var(--dim);}
.plot-card .ph strong{color:var(--text);font-size:14px;}
.plot-card img.plot{width:100%;display:block;}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:28px;}
footer{text-align:center;color:var(--dim);font-size:11px;margin-top:60px;
       padding-top:20px;border-top:1px solid var(--border);}
code{background:var(--panel);border:1px solid var(--border);
     border-radius:4px;padding:1px 6px;font-size:12px;color:var(--purple);}
@media(max-width:800px){
  .content,.header{padding:20px 16px;} .nav{padding:0 8px;}
  .nav a{padding:12px 10px;font-size:12px;} .two-col{grid-template-columns:1fr;}}
"""


def build_html(summary: Dict, plots: Dict[str, str]) -> str:
    run_id = summary.get("run_id", "unknown")
    pm = summary.get("prediction_metrics", {})
    tm = summary.get("trading_metrics", {})
    tg = summary.get("targets_met", {})

    mae = pm.get("mae", 0.0)
    rmse = pm.get("rmse", 0.0)
    acc = pm.get("accuracy_pct", 0.0)
    dir_acc = pm.get("directional_accuracy", 0.0)
    tot_ret = tm.get("total_return_pct", 0.0)
    sharpe = tm.get("sharpe_ratio", 0.0)
    sortino = tm.get("sortino_ratio", 0.0)
    max_dd = tm.get("max_drawdown_pct", 0.0)
    calmar = tm.get("calmar_ratio", 0.0)
    n_tr = tm.get("num_trades", 0)
    wr = tm.get("win_rate_pct", 0.0)
    fees = tm.get("total_fees", 0.0)

    all_ok = tg.get("mae_ok") and tg.get("rmse_ok") and tg.get("accuracy_ok")
    bc = "alert-banner" if all_ok else "alert-banner warn"
    bm = (
        f"All paper production targets met &nbsp;|&nbsp; MAE {mae:.1f} pts "
        f"&nbsp;|&nbsp; RMSE {rmse:.1f} pts &nbsp;|&nbsp; Acc {acc:.2f}% "
        f"&nbsp;|&nbsp; Sharpe {sharpe:.2f}"
        if all_ok
        else f"One or more targets not met &nbsp;|&nbsp; MAE {mae:.1f} pts "
        f"&nbsp;|&nbsp; RMSE {rmse:.1f} pts &nbsp;|&nbsp; Acc {acc:.2f}%"
    )

    acc_cls = "good" if tg.get("accuracy_ok") else "warn"
    mae_cls = "good" if tg.get("mae_ok") else "warn"
    rms_cls = "good" if tg.get("rmse_ok") else "warn"
    ret_cls = "good" if tot_ret > 0 else "warn"
    sh_cls = "good" if sharpe > 1.5 else "warn"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HFT LSTM -- Backtest Report [{run_id}]</title>
<style>{CSS}</style>
</head>
<body>
<div class="header">
  <h1>HFT LSTM System -- Backtest Report</h1>
  <div class="sub">S&amp;P 500 Forecasting Engine &nbsp;|&nbsp;
    2-Layer LSTM (64 units) &nbsp;|&nbsp; 216-day sliding window</div>
  <div class="run-id">Run ID: {run_id}</div>
</div>
<nav class="nav">
  <a href="#overview" class="active">Overview</a>
  <a href="#training">Training</a>
  <a href="#predictions">Predictions</a>
  <a href="#equity">Equity</a>
  <a href="#rolling">Rolling Metrics</a>
  <a href="#signals">Signals</a>
</nav>
<div class="content">

<div class="section" id="overview">
  <div class="section-title">Performance Overview</div>
  <div class="{bc}">{bm}</div>
  <div class="kpi-grid">
    <div class="kpi"><div class="label">Accuracy</div>
      <div class="value {acc_cls}">{acc:.2f}%</div>
      <div class="sub-v">Target &gt;= 96.41%</div></div>
    <div class="kpi"><div class="label">MAE</div>
      <div class="value {mae_cls}">{mae:.1f} pts</div>
      <div class="sub-v">Target &lt;= 175.9</div></div>
    <div class="kpi"><div class="label">RMSE</div>
      <div class="value {rms_cls}">{rmse:.1f} pts</div>
      <div class="sub-v">Target &lt;= 207.34</div></div>
    <div class="kpi"><div class="label">Total Return</div>
      <div class="value {ret_cls}">{tot_ret:+.1f}%</div>
      <div class="sub-v">Full backtest period</div></div>
    <div class="kpi"><div class="label">Sharpe Ratio</div>
      <div class="value {sh_cls}">{sharpe:.2f}</div>
      <div class="sub-v">Target &gt; 1.50</div></div>
    <div class="kpi"><div class="label">Max Drawdown</div>
      <div class="value warn">{max_dd:.1f}%</div>
      <div class="sub-v">Circuit breaker -10%</div></div>
    <div class="kpi"><div class="label">Win Rate</div>
      <div class="value">{wr:.1f}%</div>
      <div class="sub-v">{n_tr} total trades</div></div>
    <div class="kpi"><div class="label">Dir. Accuracy</div>
      <div class="value">{dir_acc:.1f}%</div>
      <div class="sub-v">Up/down prediction</div></div>
  </div>
  <div class="section-title" style="font-size:15px;margin-top:28px;">Benchmark Comparison (Accuracy %)</div>
  <div class="bench-row">
    <div class="bench-label">LSTM (this model)</div>
    <div class="bench-bar-wrap"><div class="bench-bar"
      style="width:{min(acc,100):.1f}%;background:var(--green);"></div></div>
    <div class="bench-val" style="color:var(--green)">{acc:.2f}%</div></div>
  <div class="bench-row">
    <div class="bench-label">Paper target</div>
    <div class="bench-bar-wrap"><div class="bench-bar"
      style="width:96.41%;background:var(--blue);opacity:.6;"></div></div>
    <div class="bench-val" style="color:var(--blue)">96.41%</div></div>
  <div class="bench-row">
    <div class="bench-label">ARIMA baseline</div>
    <div class="bench-bar-wrap"><div class="bench-bar"
      style="width:89.8%;background:var(--red);opacity:.6;"></div></div>
    <div class="bench-val" style="color:var(--red)">89.80%</div></div>
  <div class="bench-row">
    <div class="bench-label">Random trading</div>
    <div class="bench-bar-wrap"><div class="bench-bar"
      style="width:50%;background:var(--dim);opacity:.4;"></div></div>
    <div class="bench-val" style="color:var(--dim)">~50.00%</div></div>
  <div class="two-col">
    <div class="table-wrap"><table>
      <tr><th colspan="2">Prediction Metrics</th></tr>
      {mrow("MAE",          f"{mae:.4f} points",  tg.get("mae_ok"))}
      {mrow("RMSE",         f"{rmse:.4f} points", tg.get("rmse_ok"))}
      {mrow("Accuracy",     f"{acc:.4f}%",         tg.get("accuracy_ok"))}
      {mrow("Dir. Accuracy",f"{dir_acc:.4f}%")}
      {mrow("ARIMA MAE (baseline)","462.1 points")}
      {mrow("ARIMA Accuracy","89.80%")}
    </table></div>
    <div class="table-wrap"><table>
      <tr><th colspan="2">Trading Metrics</th></tr>
      {mrow("Total Return", f"{tot_ret:+.4f}%")}
      {mrow("Sharpe Ratio", f"{sharpe:.4f}", sharpe > 1.5)}
      {mrow("Sortino Ratio",f"{sortino:.4f}")}
      {mrow("Max Drawdown", f"{max_dd:.4f}%")}
      {mrow("Calmar Ratio", f"{calmar:.4f}")}
      {mrow("# Trades",     str(n_tr))}
      {mrow("Win Rate",     f"{wr:.4f}%")}
      {mrow("Total Fees",   f"${fees:,.2f}")}
    </table></div>
  </div>
</div>

<div class="section" id="training">
  <div class="section-title">Training History</div>
  <p style="color:var(--dim);font-size:13px;margin-bottom:20px;">
    Source: <code>models/training_history.json</code> written by <code>train.py</code>.
    Per-epoch MSE loss (train vs validation) and learning rate schedule.
  </p>
  <div class="plot-card">
    <div class="ph"><strong>MSE Loss &amp; Learning Rate Schedule</strong>
      &nbsp;|&nbsp; Adam (lr=0.001) &nbsp;|&nbsp; Early stopping (patience=10)
      &nbsp;|&nbsp; ReduceLROnPlateau (factor=0.5, patience=5)</div>
    {img_tag(plots['training'], 'Training History')}
  </div>
</div>

<div class="section" id="predictions">
  <div class="section-title">Predictions vs Actual</div>
  <p style="color:var(--dim);font-size:13px;margin-bottom:20px;">
    Source: <code>models/test_predictions.jsonl</code> written by <code>train.py</code>.
    Chronological held-out test set (last 20%). Metrics on inverse-transformed raw prices.
    Orange shading = +/-5 point confidence band.
  </p>
  <div class="plot-card">
    <div class="ph"><strong>LSTM Price Prediction vs Actual (Last 300 Bars)</strong>
      &nbsp;|&nbsp; MAE {mae:.1f} pts &nbsp;|&nbsp; RMSE {rmse:.1f} pts
      &nbsp;|&nbsp; Accuracy {acc:.2f}%</div>
    {img_tag(plots['predictions'], 'Predictions vs Actual')}
  </div>
</div>

<div class="section" id="equity">
  <div class="section-title">Equity Curve</div>
  <p style="color:var(--dim);font-size:13px;margin-bottom:20px;">
    Source: <code>backtest_results/equity_{run_id}.jsonl</code> streamed by <code>backtester.py</code>.
    Position sizing scaled by signal confidence. SEC fee + 1bp slippage. Circuit breaker at -10%.
  </p>
  <div class="plot-card">
    <div class="ph"><strong>Equity Curve, Drawdown &amp; Daily Return Distribution</strong>
      &nbsp;|&nbsp; Sharpe {sharpe:.2f} &nbsp;|&nbsp; Sortino {sortino:.2f}
      &nbsp;|&nbsp; Max DD {max_dd:.1f}%</div>
    {img_tag(plots['equity'], 'Equity Curve')}
  </div>
</div>

<div class="section" id="rolling">
  <div class="section-title">Rolling Performance Metrics</div>
  <p style="color:var(--dim);font-size:13px;margin-bottom:20px;">
    Source: <code>backtest_results/predictions_{run_id}.jsonl</code> streamed by <code>backtester.py</code>.
    Red shading = drift events where rolling MAE &gt; 185.9 (warn threshold).
  </p>
  <div class="plot-card">
    <div class="ph"><strong>Rolling MAE &amp; Rolling Accuracy (20-bar window)</strong>
      &nbsp;|&nbsp; Drift events highlighted in red</div>
    {img_tag(plots['rolling'], 'Rolling Metrics')}
  </div>
</div>

<div class="section" id="signals">
  <div class="section-title">Signal Analysis</div>
  <p style="color:var(--dim);font-size:13px;margin-bottom:20px;">
    Source: <code>backtest_results/trades_{run_id}.jsonl</code> written by <code>backtester.py</code>.
    Threshold: +/-0.15% predicted move. Confidence = |delta%| / 2.5%, capped at 1.0.
  </p>
  <div class="plot-card">
    <div class="ph"><strong>Signal Counts, Confidence, Direction &amp; Mix Over Time</strong>
      &nbsp;|&nbsp; {n_tr} actionable signals &nbsp;|&nbsp; Win rate {wr:.1f}%</div>
    {img_tag(plots['signals'], 'Signal Distribution')}
  </div>
</div>

</div>
<footer>
  HFT LSTM System &nbsp;|&nbsp; S&amp;P 500 Forecasting Engine &nbsp;|&nbsp;
  2xLSTM(64) + Dropout(0.20) + Dense(1) &nbsp;|&nbsp; 49,985 parameters
  &nbsp;|&nbsp; Window: 216 trading days &nbsp;|&nbsp; Run ID: {run_id}
</footer>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate HFT LSTM backtest report")
    parser.add_argument(
        "--run-id", default=None, help="Backtest run ID (default: latest)"
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output HTML path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    run_id = args.run_id or find_latest_run_id()
    if run_id is None:
        log.error("No backtest runs found. Run: python run_backtest.py")
        sys.exit(1)
    log.info(f"Generating report for run_id={run_id}")

    summary, history, test_preds, bt_preds, equity, trades = load_all(run_id)

    # Predictions plot: prefer test_predictions (more bars, cleaner dates)
    # Rolling metrics: prefer backtest predictions (has per-bar accuracy field)
    pred_for_plot = test_preds if test_preds else bt_preds
    pred_for_roll = bt_preds if bt_preds else test_preds

    log.info("Rendering plots...")
    plots = {
        "training": plot_training(history),
        "predictions": plot_predictions(pred_for_plot),
        "equity": plot_equity(equity, summary),
        "rolling": plot_rolling(pred_for_roll),
        "signals": plot_signals(trades),
    }
    log.info("All 5 plots rendered")

    html = build_html(summary, plots)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    log.info(f"Report written -> {out}  ({kb:.0f} KB)")
    print(f"\nReport: {out}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
