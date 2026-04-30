"""
Unified Feature Engineering Module

TRD-compliant feature engineering system for financial time-series prediction.

Public API:
- FeaturePipeline: Main pipeline class integrating all stages
- build_windows: Temporal windowing for LSTM sequences
- compute_trd_technical_features: TRD-aligned technical indicators
- apply_wavelet_denoising: Wavelet denoising with leakage prevention
- FeatureSelector: 4-stage feature selection

Version: 1.0.0 UNIFIED
"""

from .feature_creators_funcs import (
    compute_statistical_features,
    compute_volume_features,
    compute_trd_technical_features,
    compute_price_features,
    # compute_target,
)
from .wavelet import apply_wavelet_denoising, denoise_pipeline
from .selector import FeatureSelector

__all__ = [
    # Feature computation
    # "compute_target",
    "compute_trd_technical_features",
    "compute_price_features",
    "compute_statistical_features",
    "compute_volume_features",
    # Feature transformation
    "apply_wavelet_denoising",
    "denoise_pipeline",
    "FeatureSelector",
]

__version__ = "1.0.0"
