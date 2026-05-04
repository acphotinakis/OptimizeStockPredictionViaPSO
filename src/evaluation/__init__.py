"""Evaluation: split protocol, statistical and trading metrics, plotting."""

from .canonical_split import compute_canonical_split, verify_split_integrity

__all__ = [
    "compute_canonical_split",
    "verify_split_integrity",
]
