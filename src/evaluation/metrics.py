"""
src/evaluation/metrics.py

All statistical prediction metrics and trading/financial metrics
referenced in experiment_plan.md and backtesting_framework.md.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

ANNUALISE_1MIN = np.sqrt(252 * 390)  # 1-minute bars → annual


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


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with the correct sign."""
    correct = np.sign(y_pred) == np.sign(y_true)
    return float(correct.mean())


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
        return float(
            roc_auc_score(y_bin, score_matrix, average="micro", multi_class="ovr")
        )
    except Exception:
        return float("nan")


def all_statistical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all statistical metrics and return as a dict."""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "f1_ternary": f1_ternary(y_true, y_pred),
        "auc_ternary": auc_ternary(y_true, y_pred),
    }


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
    """
    n_bars = len(equity_curve)
    if n_bars == 0 or equity_curve[0] <= 0:
        return 0.0
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


def all_trading_metrics(
    equity_curve: np.ndarray,
    bar_returns: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute all trading metrics and return as a dict."""
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
    return result
