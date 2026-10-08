"""
src/evaluation/metrics.py

All statistical prediction metrics and trading/financial metrics
referenced in experiment_plan.md and backtesting_framework.md.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ANNUALISE_1MIN = np.sqrt(252 * 390)  # 1-minute bars --> annual

logger = logging.getLogger(__name__)

# ======================================================================
# Statistical metrics
# ======================================================================


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_pred - y_true)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + eps))) * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


# def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     """Fraction of predictions with the correct sign."""
#     correct = np.sign(y_pred) == np.sign(y_true)
#     return float(correct.mean())


def directional_accuracy(y_true, y_pred, threshold=1e-8, exclude_zeros=True):
    """Directional accuracy with correct zero handling.

    Args:
        y_true: Actual returns.
        y_pred: Predicted returns.
        threshold: Minimum absolute value to classify as directional. The
            default 1e-8 filters subnormal floating-point noise while keeping
            any meaningful log-return magnitude.
        exclude_zeros: If True, drop samples where ``|y_true|`` falls at or
            below ``threshold`` before scoring (recommended).

    Returns:
        Fraction of correct direction predictions.
    """
    if exclude_zeros:
        # Only evaluate where actual has clear direction
        mask = np.abs(y_true) > threshold

        if mask.sum() == 0:
            logger.warning("No directional samples (all |y| <= threshold)")
            return float("nan")

        y_true_filt = y_true[mask]
        y_pred_filt = y_pred[mask]

        # Safe to use sign() now (no zeros)
        correct = np.sign(y_true_filt) == np.sign(y_pred_filt)
        return float(correct.mean())
    else:
        # Ternary: treats (0,0) as correct but (0,±1) as wrong
        yt = np.where(np.abs(y_true) < threshold, 0, np.sign(y_true))
        yp = np.where(np.abs(y_pred) < threshold, 0, np.sign(y_pred))
        return float((yt == yp).mean())


def f1_ternary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 1e-4,
) -> float:
    """Micro-averaged F1 for ternary classification (up / flat / down).

    Args:
        y_true: Actual log returns.
        y_pred: Predicted log returns.
        threshold: Returns within ±threshold are classified 'flat'.

    Returns:
        Micro-F1 score in [0, 1].
    """
    from sklearn.metrics import f1_score

    def classify(arr: np.ndarray) -> np.ndarray:
        c = np.zeros(len(arr), dtype=int)  # flat = 0
        c[arr > threshold] = 1  # up
        c[arr < -threshold] = -1  # down
        return c

    return float(f1_score(classify(y_true), classify(y_pred), average="micro"))


def auc_ternary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 1e-4,
) -> float:
    """Micro-averaged one-vs-rest AUC for ternary classification."""
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize

        def classify(arr):
            c = np.zeros(len(arr), dtype=int)
            c[arr > threshold] = 1
            c[arr < -threshold] = -1
            return c

        y_true_c = classify(y_true)
        classes = [-1, 0, 1]
        y_bin = label_binarize(y_true_c, classes=classes)
        # Use predicted probabilities from soft output (treated as rank scores)
        score_matrix = np.column_stack(
            [
                -y_pred,  # score for "down"
                -np.abs(y_pred),  # score for "flat"  (low magnitude)
                y_pred,  # score for "up"
            ]
        )
        return float(roc_auc_score(y_bin, score_matrix, average="micro", multi_class="ovr"))
    except Exception:
        return float("nan")


# ======================================================================
# Trading / financial metrics
# ======================================================================


def sharpe_ratio(bar_returns: np.ndarray) -> float:
    """Annualised Sharpe ratio (risk-free rate = 0).

    Args:
        bar_returns: Per-bar strategy returns (after costs).
    """
    std = bar_returns.std()
    if std < 1e-10:
        return 0.0
    return float(bar_returns.mean() / std * ANNUALISE_1MIN)


def sortino_ratio(bar_returns: np.ndarray) -> float:
    """Annualised Sortino ratio (penalises only downside deviation)."""
    downside = bar_returns[bar_returns < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = float(np.sqrt(np.mean(downside**2)))
    if downside_std < 1e-10:
        return float("inf")
    return float(bar_returns.mean() / downside_std * ANNUALISE_1MIN)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of an equity curve."""
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / (peak + 1e-10)
    return float(dd.max())


def cagr(equity_curve: np.ndarray, n_bars_per_year: int = 252 * 390) -> float:
    """Compound Annual Growth Rate from an equity curve.

    Args:
        equity_curve: Absolute portfolio value over time.
        n_bars_per_year: Number of 1-minute bars in a trading year.

    Returns:
        Annualised growth rate. Returns 0.0 if the curve is empty or starts at
        non-positive equity, and -1.0 if final equity is non-positive (full loss).
    """
    n_bars = len(equity_curve)
    if n_bars == 0 or equity_curve[0] <= 0:
        return 0.0
    if equity_curve[-1] <= 0:
        return -1.0
    n_years = n_bars / n_bars_per_year
    return float((equity_curve[-1] / equity_curve[0]) ** (1.0 / n_years) - 1.0)


def calmar_ratio(equity_curve: np.ndarray) -> float:
    """CAGR / Maximum Drawdown."""
    mdd = max_drawdown(equity_curve)
    if mdd < 1e-10:
        return float("inf")
    return float(cagr(equity_curve) / mdd)


def profit_factor(bar_returns: np.ndarray) -> float:
    """Gross profit / gross loss."""
    gains = bar_returns[bar_returns > 0].sum()
    losses = np.abs(bar_returns[bar_returns < 0].sum())
    if losses < 1e-10:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def win_rate(bar_returns: np.ndarray) -> float:
    """Fraction of positive-return bars (excluding zeros)."""
    nonzero = bar_returns[bar_returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).mean())


def information_ratio(
    bar_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> float:
    """Annualised Information Ratio vs. a benchmark."""
    active = bar_returns - benchmark_returns
    std = active.std()
    if std < 1e-10:
        return 0.0
    return float(active.mean() / std * ANNUALISE_1MIN)


# ======================================================================
# Metrics + Logging Utilities
# ======================================================================


# ----------------------------------------------------------------------
# Metric Orders (controls display)
# ----------------------------------------------------------------------

STATS_METRIC_ORDER = [
    "rmse",
    "mae",
    "mape",
    "r2",
    "directional_accuracy",
    "f1_ternary",
    "auc_ternary",
]

TRADING_METRIC_ORDER = [
    "sharpe",
    "sortino",
    "max_drawdown",
    "cagr",
    "calmar",
    "profit_factor",
    "win_rate",
    "information_ratio",
    "n_bars",
]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def _nan_to_none(d):
    """Replace NaN floats with None for JSON-safe serialisation.

    RFC 8259 disallows ``NaN`` in JSON, so callers that ``json.dump`` the
    metrics dict require finite floats or ``None``. Recurses into nested
    dicts and lists so summary blocks built by downstream callers are also
    sanitised. Non-float values pass through untouched.

    Args:
        d: Metrics dictionary, list, or scalar with possibly NaN float values.

    Returns:
        Same shape as the input with NaN floats replaced by ``None``.
    """
    if isinstance(d, dict):
        return {k: _nan_to_none(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_nan_to_none(v) for v in d]
    if isinstance(d, float) and np.isnan(d):
        return None
    return d


def compute_and_log_all_statistical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "Stats",
) -> Dict[str, float]:
    """Compute statistical metrics and optionally log them."""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    logger.info(f"y_true shape {y_true.shape} || y_pred shape {y_pred.shape}")
    result = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "f1_ternary": f1_ternary(y_true, y_pred),
        "auc_ternary": auc_ternary(y_true, y_pred),
    }

    _log_metrics(label, result, STATS_METRIC_ORDER, metric_type="stats")

    return _nan_to_none(result)


def compute_and_log_all_trading_metrics(
    equity_curve: np.ndarray,
    bar_returns: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    label: str = "Trading",
) -> Dict[str, float]:
    """Compute trading metrics and optionally log them."""
    result = {
        "sharpe": sharpe_ratio(bar_returns),
        "sortino": sortino_ratio(bar_returns),
        "max_drawdown": max_drawdown(equity_curve),
        "cagr": cagr(equity_curve),
        "calmar": calmar_ratio(equity_curve),
        "profit_factor": profit_factor(bar_returns),
        "win_rate": win_rate(bar_returns),
        "n_bars": len(bar_returns),
    }

    if benchmark_returns is not None:
        result["information_ratio"] = information_ratio(bar_returns, benchmark_returns)

    _log_metrics(label, result, TRADING_METRIC_ORDER, metric_type="trading")

    return _nan_to_none(result)


# ----------------------------------------------------------------------
# Internal Logging Helpers
# ----------------------------------------------------------------------


def _log_metrics(
    label: str,
    metrics: Dict[str, float],
    order: list[str],
    metric_type: str,
) -> None:
    """Generic metric logger with consistent formatting."""
    parts = []

    for k in order:
        if k not in metrics:
            continue

        v = metrics[k]

        if metric_type == "stats":
            if k in {"rmse", "mae", "mape"}:
                parts.append(f"{k.upper()}={v:.10f}")
            else:
                parts.append(f"{k.upper()}={v:.10f}")

        elif metric_type == "trading":
            if k == "n_bars":
                parts.append(f"{k.upper()}={int(v)}")
            elif k in {"max_drawdown", "cagr", "win_rate"}:
                parts.append(f"{k.upper()}={v:.10%}")
            else:
                parts.append(f"{k.upper()}={v:.10f}")

    logger.info(f"{label} - " + "  ".join(parts))


def _log_results(
    stat_metrics: dict,
    result,
    theta: float,
) -> None:
    logger.info("=" * 55)
    logger.info("STATISTICAL METRICS")
    logger.info("  RMSE : %.10f", stat_metrics["rmse"])
    logger.info("  DA   : %.10f", stat_metrics["directional_accuracy"])
    logger.info("  F1   : %.10f", stat_metrics["f1_ternary"])
    logger.info("  R²   : %.10f", stat_metrics["r2"])
    logger.info("TRADING METRICS (θ=%.5f)", theta)
    logger.info("  Sharpe        : %.3f", result.sharpe)
    logger.info("  Sortino       : %.3f", result.sortino)
    logger.info("  Max Drawdown  : %.2f%%", result.mdd * 100)
    logger.info("  CAGR          : %.2f%%", result.cagr_ * 100)
    logger.info("  Calmar        : %.3f", result.calmar)
    logger.info("  Profit Factor : %.3f", result.profit_factor_)
    logger.info("  Win Rate      : %.2f%%", result.win_rate_ * 100)
    logger.info("  Trades        : %d", result.n_trades)
    logger.info("  Turnover      : %.4f", result.turnover)
    logger.info("=" * 55)
