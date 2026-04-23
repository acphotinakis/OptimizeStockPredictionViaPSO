"""
Canonical Evaluation Module

Implements the authoritative evaluation framework.

This module supersedes all prior validation and backtesting specifications.

Public API:
- compute_canonical_split: 70/10/20 temporal split
- verify_split_integrity: Split validation
- FrozenPipelineState: Immutable pipeline state container
- PipelineStateFitter: One-time pipeline fitting
- transform_with_frozen_state: Apply frozen transformations
- ExpandingWindowWalkForward: Production walk-forward with per-fold PSO
- validate_walk_forward_compliance: TRD compliance validation
- CanonicalBacktest: Backtesting with transaction costs

Version: PRODUCTION 2.0
Source: WALK_FORWARD_PLAN.md
"""

from .backtest import CanonicalBacktest
from .canonical_split import compute_canonical_split, verify_split_integrity
from .frozen_pipeline import (
    FrozenPipelineState,
    PipelineStateFitter,
    transform_with_frozen_state,
)
from .walk_forward_pso import (
    ExpandingWindowWalkForward,
    validate_walk_forward_compliance,
)

__all__ = [
    # Splitting
    "compute_canonical_split",
    "verify_split_integrity",
    # Pipeline
    "FrozenPipelineState",
    "PipelineStateFitter",
    "transform_with_frozen_state",
    # Evaluation
    "ExpandingWindowWalkForward",
    "validate_walk_forward_compliance",
    "CanonicalBacktest",
]

__version__ = "PRODUCTION_2.0"
