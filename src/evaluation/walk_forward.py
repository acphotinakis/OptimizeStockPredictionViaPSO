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

from .metrics import compute_and_log_all_statistical_metrics

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Expanding-window walk-forward cross-validator.

    Args:
        fold_size_bars: Number of bars per test fold (default: 1 month ≈ 21×390).
        min_train_bars: Minimum training bars before first fold.
        retrain_fn: Callable(X_train, y_train, X_val, y_val) --> fitted model.
        predict_fn:  Callable(model, X) --> y_pred array.
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

        fold_results: List[Dict[str, Any]] = []

        for fold_idx, start in enumerate(fold_starts):
            end = min(start + self.fold_size, N)

            X_train, y_train = X[:start], y[:start]
            X_test, y_test = X[start:end], y[start:end]

            logger.info(
                "Fold %d/%d: train=%d test=%d",
                fold_idx + 1,
                len(fold_starts),
                len(X_train),
                len(X_test),
            )

            model = self.retrain_fn(X_train, y_train, X_test, y_test)
            y_pred = self.predict_fn(model, X_test).reshape(-1)

            metrics = compute_and_log_all_statistical_metrics(y_test, y_pred)

            fold_results.append(
                {
                    "fold": fold_idx + 1,
                    "start": int(start),
                    "end": int(end),
                    "y_true": y_test,
                    "y_pred": y_pred,
                    **metrics,
                }
            )

        return self._aggregate(fold_results), fold_results

    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(fold_results: List[Dict]) -> Dict[str, Any]:
        if not fold_results:
            return {}

        numeric_keys = [
            k
            for k in fold_results[0]
            if k not in ("fold", "start", "end", "y_true", "y_pred")
            and isinstance(fold_results[0][k], (int, float))
        ]

        summary: Dict[str, Any] = {
            "folds": fold_results,
            "n_folds": len(fold_results),
        }

        for key in numeric_keys:
            vals = np.array([f[key] for f in fold_results])
            summary[f"{key}_mean"] = float(vals.mean())
            summary[f"{key}_std"] = float(vals.std())

        return summary
