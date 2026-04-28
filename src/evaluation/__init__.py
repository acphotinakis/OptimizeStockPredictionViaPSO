"""
Canonical Evaluation Module

Implements the authoritative evaluation framework.

This module supersedes all prior validation and backtesting specifications.

Public API:
- compute_canonical_split: 70/10/20 temporal split
- verify_split_integrity: Split validation
- ExpandingWindowWalkForward: Production walk-forward with per-fold PSO
- validate_walk_forward_compliance: TRD compliance validation
- CanonicalBacktest: Backtesting with transaction costs
- load_model: Unified model loading
- BacktestResults: Result storage and persistence
- Plotting: Visualization utilities

Version: PRODUCTION 2.1
Source: BACKTEST_DESIGN.md
"""

from .backtest import CanonicalBacktest
from .backtest_results import BacktestResults, save_backtest_results, load_backtest_results
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
