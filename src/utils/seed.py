import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Fixes seeds for random, numpy, and torch to ensure reproducible experiments.

    Args:
        seed: The integer seed value (typically pulled from cfg.seed).
    """
    logger.info(f"Setting global seed for reproducibility: {seed}")

    random.seed(seed)
    np.random.seed(seed)

    # Torch seed control
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.debug("Random, NumPy, and PyTorch seeds have been synchronized.")
