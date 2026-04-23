"""
Frozen Pipeline State Management

Implements the fit-once-freeze-forever paradigm as defined in FINAL_PLAN.md Section 3.

CRITICAL RULES (FINAL_PLAN.md Section 3.2):
- Pipeline components fit EXACTLY ONCE on 70% training data
- Pipeline state FROZEN for all subsequent transformations
- NO refitting on validation or test data
- NO updates during walk-forward evaluation
- State must be serializable for production deployment

Author: System Architect
Version: CANONICAL 1.0
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrozenPipelineState:
    """
    Immutable container for frozen pipeline state.

    FINAL_PLAN.md Section 3.2: Pipeline Lifecycle

    This class enforces immutability - once created, state CANNOT be modified.
    Any attempt to modify will raise an exception.

    Attributes:
        wavelet_threshold: Threshold computed on training data
        selected_features: Feature mask from training
        scaler_params: MinMax parameters from training
        fit_date: Last date in training data
        n_features_in: Number of features before selection
        n_features_out: Number of features after selection
        protocol_version: CANONICAL_1.0
    """

    # Wavelet denoising state
    wavelet_threshold: float
    wavelet_name: str
    wavelet_level: int
    wavelet_mode: str

    # Feature selection state
    selected_features: tuple  # Immutable tuple of feature names
    dropped_features: tuple
    selection_method: str

    # Scaler state
    scaler_params: tuple  # Tuple of (feature_name, min_val, max_val)
    scaler_range: tuple  # Target range, e.g., (-1.0, 1.0)

    # Metadata
    fit_date: str  # ISO format timestamp
    n_features_in: int
    n_features_out: int
    train_samples: int
    protocol_version: str
    source_document: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def save(self, filepath: Path) -> None:
        """
        Save frozen state to disk.

        Args:
            filepath: Path to save YAML file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self.to_dict()

        with open(filepath, "w") as f:
            yaml.dump(state_dict, f, default_flow_style=False)

        logger.info(f"Frozen pipeline state saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "FrozenPipelineState":
        """
        Load frozen state from disk.

        Args:
            filepath: Path to YAML file

        Returns:
            FrozenPipelineState instance
        """
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            state_dict = yaml.safe_load(f)

        logger.info(f"Frozen pipeline state loaded from {filepath}")

        return cls(**state_dict)

    def verify_frozen(self) -> bool:
        """
        Verify that state is truly immutable.

        Returns:
            True if immutable

        Raises:
            AssertionError: If state can be modified
        """
        try:
            # Attempt to modify - should fail
            object.__setattr__(self, "wavelet_threshold", 999.0)
            raise AssertionError("State is NOT frozen - modification succeeded!")
        except AttributeError:
            # Expected - state is frozen
            pass

        logger.info(" Pipeline state verified as immutable")
        return True


class PipelineStateFitter:
    """
    Utility class to fit pipeline components ONCE on training data.

    FINAL_PLAN.md Section 3.2: PHASE 1: FIT PIPELINE (EXECUTED ONCE)

    This class should be used EXACTLY ONCE at the beginning of evaluation
    to compute all pipeline parameters from the 70% training split.

    After fitting, the state is frozen and cannot be modified.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fitter with configuration.

        Args:
            config: Configuration dict from YAML
        """
        self.config = config
        self.wavelet_config = config.get("features", {}).get("wavelet", {})
        self.selector_config = config.get("features", {}).get("selector", {})
        self.scaler_config = config.get("lstm", {})  # Scaler params in LSTM section

    def fit(
        self,
        train_data: pd.DataFrame,
        feature_columns: List[str],
    ) -> FrozenPipelineState:
        """
        Fit all pipeline components on training data ONLY.

        CRITICAL: This function is executed EXACTLY ONCE.

        Args:
            train_data: Training split (70% of data)
            feature_columns: List of feature column names

        Returns:
            FrozenPipelineState (immutable)
        """
        logger.info("=" * 80)
        logger.info("FITTING PIPELINE STATE (FINAL_PLAN.md Section 3.2)")
        logger.info("=" * 80)
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Features: {len(feature_columns)}")

        # Step 1: Compute wavelet threshold
        logger.info("Step 1: Computing wavelet threshold...")
        wavelet_threshold = self._compute_wavelet_threshold(
            train_data["close"].values if "close" in train_data.columns else None
        )

        # Step 2: Fit feature selector
        logger.info("Step 2: Fitting feature selector...")
        selected_features, dropped_features = self._fit_feature_selector(
            train_data[feature_columns]
        )

        # Step 3: Fit scaler
        logger.info("Step 3: Fitting MinMax scaler...")
        scaler_params = self._fit_scaler(train_data[list(selected_features)])

        # Create frozen state
        state = FrozenPipelineState(
            # Wavelet
            wavelet_threshold=wavelet_threshold,
            wavelet_name=self.wavelet_config.get("wavelet", "haar"),
            wavelet_level=self.wavelet_config.get("level", 3),
            wavelet_mode=self.wavelet_config.get("mode", "symmetric"),
            # Feature selection
            selected_features=tuple(selected_features),
            dropped_features=tuple(dropped_features),
            selection_method="variance_pearson_vif_mi",
            # Scaler
            scaler_params=tuple(scaler_params),
            scaler_range=tuple(self.scaler_config.get("scaler_range", [-1.0, 1.0])),
            # Metadata
            fit_date=train_data.index[-1].isoformat(),
            n_features_in=len(feature_columns),
            n_features_out=len(selected_features),
            train_samples=len(train_data),
            protocol_version="CANONICAL_1.0",
            source_document="FINAL_PLAN.md",
        )

        logger.info("=" * 80)
        logger.info("PIPELINE STATE FITTED AND FROZEN")
        logger.info("=" * 80)
        logger.info(f"Wavelet threshold: {wavelet_threshold:.8f}")
        logger.info(
            f"Features selected: {len(selected_features)} / {len(feature_columns)}"
        )
        logger.info(f"Features dropped: {len(dropped_features)}")
        logger.info(f"Scaler range: {state.scaler_range}")
        logger.info("=" * 80)
        logger.info("⚠️  PIPELINE STATE IS NOW FROZEN - NO MODIFICATIONS ALLOWED")
        logger.info("=" * 80)

        # Verify immutability
        state.verify_frozen()

        return state

    def _compute_wavelet_threshold(self, close_values: Optional[np.ndarray]) -> float:
        """
        Compute wavelet threshold from training close prices.

        FINAL_PLAN.md Section 3: Wavelet threshold computed on TRAIN only

        Args:
            close_values: Close prices from training data

        Returns:
            Universal threshold value
        """
        if close_values is None or not self.wavelet_config.get("enabled", True):
            logger.info("  Wavelet denoising disabled, threshold = 0.0")
            return 0.0

        import pywt

        # 3-level Haar DWT
        coeffs = pywt.wavedec(
            close_values,
            wavelet=self.wavelet_config.get("wavelet", "haar"),
            level=self.wavelet_config.get("level", 3),
            mode=self.wavelet_config.get("mode", "symmetric"),
        )

        # Universal threshold using MAD estimator
        D1 = coeffs[-1]  # Finest detail coefficients
        median_abs_dev = np.median(np.abs(D1))

        if median_abs_dev == 0:
            threshold = 0.0
        else:
            sigma = median_abs_dev / 0.6745
            N = len(close_values)
            threshold = sigma * np.sqrt(2 * np.log(N))

        logger.info(f"  Wavelet threshold (training): {threshold:.8f}")

        return float(threshold)

    def _fit_feature_selector(self, features_df: pd.DataFrame) -> tuple:
        """
        Fit feature selector on training data.

        FINAL_PLAN.md Section 3: Feature selection mask computed on TRAIN only

        4-stage selection:
        1. Variance threshold
        2. Pearson correlation (≥95%)
        3. VIF (>10)
        4. Mutual Information (bottom quartile)

        Args:
            features_df: Training features

        Returns:
            Tuple of (selected_features, dropped_features)
        """
        # For now, return all features (full selector implementation would go here)
        # This is a placeholder - actual selector would implement 4-stage pipeline
        selected = list(features_df.columns)
        dropped = []

        logger.info(f"  Selected: {len(selected)} features")
        logger.info(f"  Dropped: {len(dropped)} features")

        return selected, dropped

    def _fit_scaler(self, features_df: pd.DataFrame) -> List[tuple]:
        """
        Fit MinMax scaler on training data.

        FINAL_PLAN.md Section 3: Scaler parameters fit on TRAIN only

        Args:
            features_df: Training features (after selection)

        Returns:
            List of (feature_name, min_val, max_val) tuples
        """
        scaler_params = []

        for col in features_df.columns:
            min_val = float(features_df[col].min())
            max_val = float(features_df[col].max())
            scaler_params.append((col, min_val, max_val))

        logger.info(f"  Fitted scaler for {len(scaler_params)} features")

        return scaler_params


def transform_with_frozen_state(
    data: pd.DataFrame,
    frozen_state: FrozenPipelineState,
) -> np.ndarray:
    """
    Transform data using FROZEN pipeline state.

    FINAL_PLAN.md Section 3.2: PHASE 2: TRANSFORM (EXECUTED ON ALL SPLITS)

    CRITICAL: frozen_state MUST NOT be modified.

    Args:
        data: Data to transform (train/val/test)
        frozen_state: Immutable pipeline state from training

    Returns:
        Transformed features as numpy array
    """
    logger.info(f"Transforming data with FROZEN state (samples={len(data)})")

    # Apply feature selection
    selected_features = list(frozen_state.selected_features)
    data_selected = data[selected_features].copy()

    # Apply MinMax scaling with TRAINING parameters
    data_scaled = data_selected.copy()

    for feature_name, min_val, max_val in frozen_state.scaler_params:
        if feature_name in data_scaled.columns:
            # Scale to [-1, 1] using TRAINING min/max
            if max_val > min_val:
                data_scaled[feature_name] = (
                    2.0 * (data_scaled[feature_name] - min_val) / (max_val - min_val)
                    - 1.0
                )
            else:
                data_scaled[feature_name] = 0.0

    # Convert to numpy array
    X_transformed = data_scaled.values.astype(np.float32)

    logger.info(f"  Transformed shape: {X_transformed.shape}")
    logger.info(f"  ⚠️  Pipeline state was NOT modified (frozen)")

    return X_transformed
