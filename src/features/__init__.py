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

from .pipeline import FeaturePipeline
from .technical import compute_trd_technical_features, compute_price_features
from .wavelet import apply_wavelet_denoising, denoise_pipeline
from .selector import FeatureSelector
from .normalization import transform_features, save_transformer_state
from .cross_ticker import compute_cross_ticker_features
from .statistical import compute_statistical_features
from .volume import compute_volume_features

__all__ = [
    # Primary API
    "FeaturePipeline",
    # Feature computation
    "compute_trd_technical_features",
    "compute_price_features",
    "compute_cross_ticker_features",
    "compute_statistical_features",
    "compute_volume_features",
    # Feature transformation
    "apply_wavelet_denoising",
    "denoise_pipeline",
    "FeatureSelector",
    "transform_features",
    "save_transformer_state",
]

__version__ = "1.0.0"
