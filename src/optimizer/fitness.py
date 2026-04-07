"""
src/optimizer/fitness.py

Composite fitness function for the PSO optimizer.

F(x) = α₁·norm_RMSE + α₂·(1 − norm_Sharpe) + α₃·norm_MDD

Each component is normalised online to [0, 1] using the running
min/max observed across all particle evaluations in the current run.
Lower is better.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Default weights (must sum to 1)
DEFAULT_WEIGHTS = dict(rmse=0.4, sharpe=0.4, mdd=0.2)

# Annualisation factor for 1-minute bars
ANNUALISE = np.sqrt(252 * 390)

# Signal threshold (1 bp)
SIGNAL_THRESHOLD = 1e-4


def generate_signals(
    y_pred: np.ndarray, threshold: float = SIGNAL_THRESHOLD
) -> np.ndarray:
    """Convert predicted log returns to ternary trade signals {-1, 0, +1}."""
    sig = np.zeros(len(y_pred), dtype=np.float32)
    sig[y_pred > threshold] = 1.0
    sig[y_pred < -threshold] = -1.0
    return sig


def sharpe_from_signals(
    signals: np.ndarray,
    y_true: np.ndarray,
    transaction_cost: float = 0.001,
) -> float:
    """Compute annualised Sharpe ratio from signals and actual returns.

    Args:
        signals: [N] ternary array.
        y_true:  [N] actual log returns.
        transaction_cost: One-way cost fraction.

    Returns:
        Annualised Sharpe ratio (higher is better).
    """
    # Bar-level strategy return
    bar_ret = signals * y_true
    # Subtract transaction cost when the signal changes
    signal_change = np.abs(np.diff(signals, prepend=0.0)) > 0
    bar_ret -= signal_change.astype(float) * transaction_cost

    std = bar_ret.std()
    if std < 1e-10:
        return 0.0
    return float(bar_ret.mean() / std * ANNUALISE)


def max_drawdown_from_signals(signals: np.ndarray, y_true: np.ndarray) -> float:
    """Compute maximum drawdown of the strategy equity curve.

    Args:
        signals: [N] ternary array.
        y_true:  [N] actual log returns.

    Returns:
        Maximum drawdown in [0, 1] (lower is better).
    """
    bar_ret = signals * y_true
    equity = np.cumprod(1.0 + bar_ret)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-10)
    return float(dd.max())


class CompositeFitness:
    """Online-normalised composite fitness calculator.

    Maintains running min/max for each component across all evaluations
    within a single PSO run so that all terms are scaled to [0, 1].

    Args:
        weights: Dict with keys 'rmse', 'sharpe', 'mdd' that sum to 1.
        signal_threshold: Threshold to classify predictions as up/down.
        transaction_cost: One-way cost used in Sharpe calculation.
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        signal_threshold: float = SIGNAL_THRESHOLD,
        transaction_cost: float = 0.001,
    ) -> None:
        self.weights = weights or DEFAULT_WEIGHTS
        self.threshold = signal_threshold
        self.tc = transaction_cost

        # Running bounds for online normalisation
        self._rmse_min = float("inf")
        self._rmse_max = float("-inf")
        self._sharpe_min = float("inf")
        self._sharpe_max = float("-inf")
        self._mdd_min = float("inf")
        self._mdd_max = float("-inf")

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluate composite fitness for one particle.

        Args:
            y_true: [N] validation actual log returns.
            y_pred: [N] model predictions.

        Returns:
            Scalar fitness value (lower is better).
        """
        # Raw components
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        signals = generate_signals(y_pred, self.threshold)
        sharpe = sharpe_from_signals(signals, y_true, self.tc)
        mdd = max_drawdown_from_signals(signals, y_true)

        # Update running bounds
        self._rmse_min = min(self._rmse_min, rmse)
        self._rmse_max = max(self._rmse_max, rmse)
        self._sharpe_min = min(self._sharpe_min, sharpe)
        self._sharpe_max = max(self._sharpe_max, sharpe)
        self._mdd_min = min(self._mdd_min, mdd)
        self._mdd_max = max(self._mdd_max, mdd)

        # Normalise each component to [0, 1]
        norm_rmse = self._normalise(rmse, self._rmse_min, self._rmse_max)
        norm_sharpe = self._normalise(sharpe, self._sharpe_min, self._sharpe_max)
        norm_mdd = self._normalise(mdd, self._mdd_min, self._mdd_max)

        # Sharpe: higher is better --> invert for minimisation
        fitness = (
            self.weights["rmse"] * norm_rmse
            + self.weights["sharpe"] * (1.0 - norm_sharpe)
            + self.weights["mdd"] * norm_mdd
        )
        return float(fitness)

    @staticmethod
    def _normalise(value: float, vmin: float, vmax: float) -> float:
        rng = vmax - vmin
        if rng < 1e-10:
            return 0.0
        return float(np.clip((value - vmin) / rng, 0.0, 1.0))

    def reset(self) -> None:
        """Reset running bounds (call between PSO runs)."""
        self._rmse_min = float("inf")
        self._rmse_max = float("-inf")
        self._sharpe_min = float("inf")
        self._sharpe_max = float("-inf")
        self._mdd_min = float("inf")
        self._mdd_max = float("-inf")
