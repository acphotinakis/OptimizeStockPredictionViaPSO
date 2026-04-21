"""
Canonical Evaluation Module

Implements the authoritative evaluation framework defined in FINAL_PLAN.md.

This module supersedes all prior validation and backtesting specifications.

Public API:
- compute_canonical_split: 70/10/20 temporal split
- verify_split_integrity: Split validation
- FrozenPipelineState: Immutable pipeline state container
- PipelineStateFitter: One-time pipeline fitting
- transform_with_frozen_state: Apply frozen transformations
- CanonicalWalkForward: Walk-forward evaluation with frozen models
- CanonicalBacktest: Backtesting with transaction costs

Version: CANONICAL 1.0
Source: FINAL_PLAN.md
"""

from .backtest import CanonicalBacktest
from .canonical_split import compute_canonical_split, verify_split_integrity
from .frozen_pipeline import (
    FrozenPipelineState,
    PipelineStateFitter,
    transform_with_frozen_state,
)
from .walk_forward import CanonicalWalkForward

__all__ = [
    # Splitting
    "compute_canonical_split",
    "verify_split_integrity",
    # Pipeline
    "FrozenPipelineState",
    "PipelineStateFitter",
    "transform_with_frozen_state",
    # Evaluation
    "CanonicalWalkForward",
    "CanonicalBacktest",
]

__version__ = "CANONICAL_1.0"
