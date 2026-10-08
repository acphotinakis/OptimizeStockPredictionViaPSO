"""
src/utils/seed.py
Global seed management for full reproducibility across NumPy, PyTorch, and Python.
"""

import os
import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds for all relevant libraries.

    Must be called BEFORE any model instantiation or data generation.

    Args:
        seed: Integer seed value. Default 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng(seed: int) -> np.random.Generator:
    """Return a seeded NumPy Generator for per-particle determinism.

    Args:
        seed: Unique integer seed (e.g. global_seed * 1000 + particle_idx).
    """
    return np.random.default_rng(seed)
