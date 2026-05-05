from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, include_mse: bool = False) -> Dict[str, float]:
    metrics = {
        "MAPE": mape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }
    if include_mse:
        metrics["MSE"] = float(mean_squared_error(y_true, y_pred))
    metrics["R2"] = float(r2_score(y_true, y_pred))
    return metrics
