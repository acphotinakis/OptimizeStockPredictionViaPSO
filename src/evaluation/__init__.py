from .canonical_split import compute_canonical_split, verify_split_integrity
from .model_loader import load_model, ModelAdapter


__all__ = [
    # Splitting
    "compute_canonical_split",
    "verify_split_integrity",
    # Model Loading
    "load_model",
    "ModelAdapter",
]

__version__ = "PRODUCTION_2.1"
