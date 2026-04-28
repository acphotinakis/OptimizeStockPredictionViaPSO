from .backtest import CanonicalBacktest
from .backtest_results import (
    BacktestResults,
    save_backtest_results,
    load_backtest_results,
)
from .canonical_split import compute_canonical_split, verify_split_integrity
from .model_loader import load_model, ModelAdapter

# from .walk_forward_pso import (
#     ExpandingWindowWalkForward,
#     validate_walk_forward_compliance,
# )

__all__ = [
    # Splitting
    "compute_canonical_split",
    "verify_split_integrity",
    # Evaluation
    # "ExpandingWindowWalkForward",
    # "validate_walk_forward_compliance",
    "CanonicalBacktest",
    # Model Loading
    "load_model",
    "ModelAdapter",
    # Results
    "BacktestResults",
    "save_backtest_results",
    "load_backtest_results",
]

__version__ = "PRODUCTION_2.1"
