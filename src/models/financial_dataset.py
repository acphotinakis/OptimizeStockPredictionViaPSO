"""Lightweight ``torch.utils.data.Dataset`` wrapper for windowed sequences."""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FinancialDataset(Dataset):
    """Wraps ``(X, y)`` numpy arrays as a PyTorch ``Dataset``.

    Args:
        X: Feature sequences with shape ``(N, T, F)``.
        y: Target vector with shape ``(N,)``; reshaped to ``(N, 1)``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
