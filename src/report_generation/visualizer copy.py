"""
visualizer.py
=============
Performance visualisation and reporting.

Generates:
  1. Training loss curves
  2. Predicted vs actual price overlay
  3. Equity curve with drawdown shading
  4. Rolling metrics (accuracy, MAE)
  5. Signal distribution histogram
  6. Walk-forward fold comparison
  7. Full HTML performance report

All plots saved to backtest_results/plots/ as PNG.
Memory optimisation: figures closed immediately after save (no accumulation).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Matplotlib — lazy import ──────────────────────────────────────────────────
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend (no display needed)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning(
        "matplotlib not installed — visualisation disabled. pip install matplotlib"
    )

PLOT_DIR = Path("backtest_results/plots")
STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#c9d1d9",
    "ytick.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "font.family": "monospace",
}
C_ACTUAL = "#58a6ff"
C_PRED = "#f78166"
C_EQUITY = "#3fb950"
C_DD = "#f85149"
C_TRAIN = "#d2a8ff"


def _setup_ax(ax):
    ax.grid(True, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, name: str, plot_dir: Path) -> Path:
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / name
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=STYLE["figure.facecolor"])
    plt.close(fig)
    logger.info(f"Plot saved -> {path}")
    return path


def _require_mpl(func):
    def wrapper(*a, **kw):
        if not _MPL_AVAILABLE:
            logger.warning(f"Skipping {func.__name__}: matplotlib not available.")
            return None
        return func(*a, **kw)

    wrapper.__name__ = func.__name__
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# Plot Functions
# ══════════════════════════════════════════════════════════════════════════════


@_require_mpl
def plot_training_history(
    history: Dict,
    plot_dir: Path = PLOT_DIR,
) -> Path:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle("Training History", fontsize=13, fontweight="bold")

        # Loss
        ax = axes[0]
        ax.plot(history.get("loss", []), label="Train Loss", color=C_TRAIN)
        ax.plot(history.get("val_loss", []), label="Val Loss", color=C_PRED)
        ax.set_title("MSE Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (normalised)")
        ax.legend(framealpha=0.2)
        _setup_ax(ax)

        # LR (if available)
        ax2 = axes[1]
        if "lr" in history:
            ax2.plot(history["lr"], color="#e3b341")
            ax2.set_title("Learning Rate")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("LR")
            ax2.set_yscale("log")
        else:
            ax2.text(
                0.5,
                0.5,
                "No LR history",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                color="#c9d1d9",
            )
        _setup_ax(ax2)

        plt.tight_layout()
        return _save(fig, "training_history.png", plot_dir)


@_require_mpl
def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    n_show: int = 200,
    plot_dir: Path = PLOT_DIR,
) -> Path:
    y_true = y_true.flatten()[-n_show:]
    y_pred = y_pred.flatten()[-n_show:]
    x = dates[-n_show:] if dates is not None else np.arange(len(y_true))

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.suptitle("LSTM Price Prediction vs Actual", fontsize=13, fontweight="bold")

        ax1.plot(x, y_true, label="Actual SPX", color=C_ACTUAL, alpha=0.9)
        ax1.plot(
            x, y_pred, label="LSTM Predicted", color=C_PRED, alpha=0.85, linestyle="--"
        )
        ax1.set_ylabel("Price ($)")
        ax1.legend(framealpha=0.2)
        _setup_ax(ax1)

        # Residuals
        residuals = y_pred - y_true
        ax2.bar(
            x,
            residuals,
            color=[C_PRED if r > 0 else C_DD for r in residuals],
            alpha=0.6,
            width=1,
        )
        ax2.axhline(0, color="#c9d1d9", linewidth=0.8)
        ax2.set_ylabel("Residual ($)")
        ax2.set_xlabel("Date" if dates is not None else "Bar")
        _setup_ax(ax2)

        plt.tight_layout()
        return _save(fig, "predictions_vs_actual.png", plot_dir)


@_require_mpl
def plot_equity_curve(
    equity_records: List[Dict],
    initial_capital: float = 100_000,
    plot_dir: Path = PLOT_DIR,
) -> Path:
    if not equity_records:
        logger.warning("No equity records to plot.")
        return None

    df = pd.DataFrame(equity_records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    eq = df["total_equity"].values.astype(np.float32)
    x = df.index if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(eq))
    peak = np.maximum.accumulate(eq)
    dd_pct = (eq - peak) / (peak + 1e-9) * 100

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.suptitle("Equity Curve & Drawdown", fontsize=13, fontweight="bold")

        ax1.plot(x, eq, color=C_EQUITY, label="Equity")
        ax1.plot(
            x,
            peak,
            color="#c9d1d9",
            linewidth=0.8,
            linestyle="--",
            label="Peak",
            alpha=0.5,
        )
        ax1.axhline(
            initial_capital,
            color="#e3b341",
            linewidth=0.8,
            linestyle=":",
            label="Initial",
        )
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax1.legend(framealpha=0.2)
        _setup_ax(ax1)

        ax2.fill_between(x, dd_pct, 0, alpha=0.5, color=C_DD)
        ax2.plot(x, dd_pct, color=C_DD, linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        _setup_ax(ax2)

        plt.tight_layout()
        return _save(fig, "equity_curve.png", plot_dir)


@_require_mpl
def plot_rolling_metrics(
    metrics_jsonl: Path,
    window: int = 20,
    plot_dir: Path = PLOT_DIR,
) -> Path:
    if not Path(metrics_jsonl).exists():
        return None

    records = [json.loads(l) for l in open(metrics_jsonl)]
    if not records:
        return None

    df = pd.DataFrame(records)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        fig.suptitle(
            f"Rolling Metrics (window={window})", fontsize=13, fontweight="bold"
        )

        if "abs_error" in df.columns:
            roll_mae = df["abs_error"].rolling(window).mean()
            axes[0].plot(
                roll_mae.values, color=C_PRED, label=f"Rolling MAE (w={window})"
            )
            axes[0].axhline(
                185.9,
                color=C_DD,
                linewidth=1.0,
                linestyle="--",
                label="Warn threshold (185.9)",
            )
            axes[0].axhline(
                250.0,
                color="#f85149",
                linewidth=1.0,
                linestyle="--",
                label="Halt threshold (250)",
            )
            axes[0].set_ylabel("MAE (points)")
            axes[0].legend(framealpha=0.2)
            _setup_ax(axes[0])

        if "accuracy" in df.columns:
            roll_acc = df["accuracy"].rolling(window).mean()
            axes[1].plot(
                roll_acc.values, color=C_ACTUAL, label=f"Rolling Accuracy (w={window})"
            )
            axes[1].axhline(
                96.41,
                color=C_EQUITY,
                linewidth=1.0,
                linestyle="--",
                label="Target (96.41%)",
            )
            axes[1].axhline(
                95.91,
                color="#e3b341",
                linewidth=1.0,
                linestyle="--",
                label="Warn (95.91%)",
            )
            axes[1].set_ylabel("Accuracy (%)")
            axes[1].set_xlabel("Bar")
            axes[1].legend(framealpha=0.2)
            _setup_ax(axes[1])

        plt.tight_layout()
        return _save(fig, "rolling_metrics.png", plot_dir)


@_require_mpl
def plot_signal_distribution(
    trades_jsonl: Path,
    plot_dir: Path = PLOT_DIR,
) -> Path:
    if not Path(trades_jsonl).exists():
        return None

    records = [json.loads(l) for l in open(trades_jsonl)]
    if not records:
        return None

    df = pd.DataFrame(records)

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Signal Analysis", fontsize=13, fontweight="bold")

        # Signal counts
        if "direction" in df.columns:
            counts = df["direction"].value_counts()
            ax1.bar(
                counts.index,
                counts.values,
                color=[C_EQUITY if d == "LONG" else C_DD for d in counts.index],
            )
            ax1.set_title("Signal Direction Counts")
            ax1.set_ylabel("Count")
            _setup_ax(ax1)

        # Confidence distribution
        if "confidence" in df.columns:
            ax2.hist(
                df["confidence"],
                bins=20,
                color=C_ACTUAL,
                alpha=0.75,
                edgecolor="#30363d",
            )
            ax2.set_title("Signal Confidence Distribution")
            ax2.set_xlabel("Confidence")
            ax2.set_ylabel("Frequency")
            _setup_ax(ax2)

        plt.tight_layout()
        return _save(fig, "signal_distribution.png", plot_dir)


@_require_mpl
def plot_walk_forward(
    wf_results: List[Dict],
    plot_dir: Path = PLOT_DIR,
) -> Path:
    if not wf_results:
        return None

    folds = [r.get("fold", i) for i, r in enumerate(wf_results)]
    accs = [r.get("accuracy", 0) for r in wf_results]
    maes = [r.get("mae", 0) for r in wf_results]

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle(
            "Walk-Forward Validation (Overfitting Detection)",
            fontsize=13,
            fontweight="bold",
        )

        ax1.plot(folds, accs, "o-", color=C_ACTUAL)
        ax1.axhline(95.91, color=C_DD, linestyle="--", linewidth=0.8, label="Target")
        ax1.set_title("Accuracy per Fold")
        ax1.set_xlabel("Fold")
        ax1.set_ylabel("Accuracy (%)")
        ax1.legend(framealpha=0.2)
        _setup_ax(ax1)

        ax2.plot(folds, maes, "o-", color=C_PRED)
        ax2.axhline(185.9, color=C_DD, linestyle="--", linewidth=0.8, label="Target")
        ax2.set_title("MAE per Fold")
        ax2.set_xlabel("Fold")
        ax2.set_ylabel("MAE ($)")
        ax2.legend(framealpha=0.2)
        _setup_ax(ax2)

        plt.tight_layout()
        return _save(fig, "walk_forward.png", plot_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Full HTML Report
# ══════════════════════════════════════════════════════════════════════════════


def generate_html_report(
    backtest_results: Dict,
    plot_dir: Path = PLOT_DIR,
    output_path: Path = Path("backtest_results/report.html"),
) -> Path:
    """
    Generate a self-contained HTML performance report embedding all PNG plots.
    """
    import base64

    def _embed(png_path: Path) -> str:
        if png_path and Path(png_path).exists():
            data = base64.b64encode(Path(png_path).read_bytes()).decode()
            return f'<img src="data:image/png;base64,{data}" style="width:100%;max-width:900px;margin:12px 0;">'
        return "<p><em>Plot not available</em></p>"

    pm = backtest_results.get("prediction_metrics", {})
    tm = backtest_results.get("trading_metrics", {})
    tg = backtest_results.get("targets_met", {})

    def _row(label, value, ok=None):
        color = ("#3fb950" if ok else "#f85149") if ok is not None else "#c9d1d9"
        badge = (
            f" <span style='color:{color};font-weight:bold'>{'✅' if ok else '❌'}</span>"
            if ok is not None
            else ""
        )
        return f"<tr><td>{label}</td><td>{value}{badge}</td></tr>"

    plots = {
        k: plot_dir / f"{k}.png"
        for k in [
            "training_history",
            "predictions_vs_actual",
            "equity_curve",
            "rolling_metrics",
            "signal_distribution",
        ]
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HFT LSTM Backtest Report</title>
<style>
  body {{ background:#0d1117; color:#c9d1d9; font-family:monospace; padding:24px; }}
  h1,h2 {{ color:#58a6ff; }}
  table {{ border-collapse:collapse; width:100%; max-width:700px; margin:16px 0; }}
  th,td {{ border:1px solid #30363d; padding:8px 14px; text-align:left; }}
  th {{ background:#161b22; color:#8b949e; font-size:0.85em; text-transform:uppercase; }}
  tr:hover {{ background:#161b22; }}
  .section {{ margin-top:32px; }}
</style>
</head>
<body>
<h1>🚀 HFT LSTM System — Backtest Report</h1>
<p>Run ID: <strong>{backtest_results.get('run_id','—')}</strong></p>

<div class="section">
<h2>📊 Prediction Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{_row("MAE",       f"{pm.get('mae',0):.2f} pts",    tg.get("mae_ok"))}
{_row("RMSE",      f"{pm.get('rmse',0):.2f} pts",   tg.get("rmse_ok"))}
{_row("Accuracy",  f"{pm.get('accuracy_pct',0):.2f}%", tg.get("accuracy_ok"))}
{_row("Dir. Accuracy", f"{pm.get('directional_accuracy',0):.2f}%")}
{_row("ARIMA Baseline", "89.80% accuracy, MAE 462.1")}
</table>
</div>

<div class="section">
<h2>💰 Trading Performance</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{_row("Total Return",   f"{tm.get('total_return_pct',0):.2f}%")}
{_row("Sharpe Ratio",   f"{tm.get('sharpe_ratio',0):.3f}  (target >1.5)")}
{_row("Sortino Ratio",  f"{tm.get('sortino_ratio',0):.3f}")}
{_row("Max Drawdown",   f"{tm.get('max_drawdown_pct',0):.2f}%")}
{_row("Calmar Ratio",   f"{tm.get('calmar_ratio',0):.3f}")}
{_row("# Trades",       f"{tm.get('num_trades',0)}")}
{_row("Win Rate",       f"{tm.get('win_rate_pct',0):.2f}%")}
{_row("Total Fees",     f"${tm.get('total_fees',0):.2f}")}
</table>
</div>

<div class="section">
<h2>📈 Training History</h2>
{_embed(plots["training_history"])}
</div>

<div class="section">
<h2>🎯 Predictions vs Actual</h2>
{_embed(plots["predictions_vs_actual"])}
</div>

<div class="section">
<h2>💼 Equity Curve</h2>
{_embed(plots["equity_curve"])}
</div>

<div class="section">
<h2>🔄 Rolling Metrics</h2>
{_embed(plots["rolling_metrics"])}
</div>

<div class="section">
<h2>📡 Signal Distribution</h2>
{_embed(plots["signal_distribution"])}
</div>

</body>
</html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"HTML report saved -> {output_path}")
    return output_path
