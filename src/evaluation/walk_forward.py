"""
src/evaluation/walk_forward.py

Expanding-window walk-forward validation.
For each fold:
  1. Retrain model on all data up to fold boundary.
  2. Evaluate on the next out-of-sample window.
Aggregates metrics across folds (mean ± std).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import all_statistical_metrics, all_trading_metrics

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Expanding-window walk-forward cross-validator.

    Args:
        fold_size_bars: Number of bars per test fold (default: 1 month ≈ 21×390).
        min_train_bars: Minimum training bars before first fold.
        retrain_fn: Callable(X_train, y_train, X_val, y_val) → fitted model.
        predict_fn:  Callable(model, X) → y_pred array.
    """

    def __init__(
        self,
        fold_size_bars: int = 21 * 390,  # ~1 month of 1-min bars
        min_train_bars: int = 3 * 252 * 390,
        retrain_fn: Optional[Callable] = None,
        predict_fn: Optional[Callable] = None,
    ) -> None:
        self.fold_size = fold_size_bars
        self.min_train = min_train_bars
        self.retrain_fn = retrain_fn
        self.predict_fn = predict_fn

    # ------------------------------------------------------------------

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        opens: Optional[np.ndarray] = None,
        closes: Optional[np.ndarray] = None,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> Dict[str, Any]:
        """Run all folds and aggregate results.

        Args:
            X: [N, T, F] full feature tensor (pre-scaled with lookback applied).
            y: [N] full return target.
            opens, closes: [N] prices for backtest (optional).
            timestamps: [N] DatetimeIndex (optional).

        Returns:
            Dict with per-fold results and aggregate mean/std.
        """
        N = len(X)
        fold_starts = list(range(self.min_train, N - self.fold_size, self.fold_size))
        logger.info(
            "Walk-forward: %d folds, fold_size=%d bars",
            len(fold_starts),
            self.fold_size,
        )

        fold_results: List[Dict] = []

        for fold_idx, start in enumerate(fold_starts):
            end = min(start + self.fold_size, N)
            X_train, y_train = X[:start], y[:start]
            X_test, y_test = X[start:end], y[start:end]

            logger.info(
                "Fold %d/%d: train=%d  test=%d",
                fold_idx + 1,
                len(fold_starts),
                len(X_train),
                len(X_test),
            )

            # Retrain
            model = self.retrain_fn(X_train, y_train, X_test, y_test)
            y_pred = self.predict_fn(model, X_test)

            stat = all_statistical_metrics(y_test, y_pred)
            fold_result: Dict[str, Any] = {"fold": fold_idx + 1, **stat}

            # Trading metrics if prices are provided
            if opens is not None and closes is not None and timestamps is not None:
                from .backtester import Backtester

                bt = Backtester()
                bt_result = bt.run(
                    y_pred.copy(),
                    opens[start:end],
                    closes[start:end],
                    timestamps[start:end],
                )
                fold_result.update(
                    {
                        "sharpe": bt_result.sharpe,
                        "mdd": bt_result.mdd,
                        "cagr": bt_result.cagr_,
                        "calmar": bt_result.calmar,
                        "profit_factor": bt_result.profit_factor_,
                        "win_rate": bt_result.win_rate_,
                        "n_trades": bt_result.n_trades,
                    }
                )

            fold_results.append(fold_result)

        return self._aggregate(fold_results)

    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(fold_results: List[Dict]) -> Dict[str, Any]:
        """Compute mean and std for each numeric metric across folds."""
        if not fold_results:
            return {}

        numeric_keys = [
            k
            for k in fold_results[0]
            if k != "fold" and isinstance(fold_results[0][k], (int, float))
        ]
        summary: Dict[str, Any] = {"folds": fold_results, "n_folds": len(fold_results)}
        for key in numeric_keys:
            vals = np.array([f[key] for f in fold_results if key in f])
            summary[f"{key}_mean"] = float(vals.mean())
            summary[f"{key}_std"] = float(vals.std())
        return summary
