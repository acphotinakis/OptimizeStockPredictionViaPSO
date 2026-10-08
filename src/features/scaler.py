import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FrozenMinMaxScaler:
    """
    Immutable MinMax scaler for features or targets.

    Formula: x_norm = 2 * (x - x_min) / (x_max - x_min) - 1

    For feature_range=(-1, 1):
        - Maps [x_min, x_max] --> [-1, 1]

    For feature_range=(0, 1):
        - Maps [x_min, x_max] --> [0, 1]

    CRITICAL INVARIANTS:
    - fit() can only be called once
    - Parameters frozen after fit
    - transform() uses frozen parameters
    - Separate scalers for features and targets
    """

    def __init__(self, feature_range: Tuple[float, float] = (-1, 1)):
        """
        Initialize scaler.

        Args:
            feature_range: Target range (default (-1, 1) per TRD)
        """
        if len(feature_range) != 2:
            raise ValueError("feature_range must be tuple of (min, max)")

        self.feature_range = feature_range
        self._fitted = False
        self._params: Optional[Dict[str, Dict[str, float]]] = None
        self._feature_names: Optional[List[str]] = None

    @property
    def is_fitted(self) -> bool:
        """Return whether scaler has been fitted."""
        return self._fitted

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "FrozenMinMaxScaler":
        """
        Fit scaler on training data.

        CRITICAL: This can only be called ONCE.

        Args:
            X: Training data (N, F) or (N, 1) for targets
            feature_names: Optional feature names (for logging)

        Returns:
            self

        Raises:
            RuntimeError: If already fitted
            ValueError: If X contains NaN or Inf
        """
        if self._fitted:
            raise RuntimeError("Scaler already fitted! Create new instance to refit.")

        if np.isnan(X).any():
            raise ValueError("X contains NaN values - clean data before scaling")

        if np.isinf(X).any():
            raise ValueError("X contains Inf values - clean data before scaling")

        # Handle 1D and 2D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        N, F = X.shape

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(F)]

        if len(feature_names) != F:
            raise ValueError(f"feature_names length ({len(feature_names)}) != columns ({F})")

        # Compute min/max per feature
        self._params = {}

        for i, fname in enumerate(feature_names):
            x_col = X[:, i]
            x_min = float(x_col.min())
            x_max = float(x_col.max())

            self._params[fname] = {
                "min": x_min,
                "max": x_max,
                "range": x_max - x_min,
            }

            logger.info(
                f"Scaler fit [{fname}]: min={x_min:.8f}, max={x_max:.8f}, range={x_max - x_min:.8f}"
            )

        self._feature_names = list(feature_names)
        self._fitted = True

        logger.info(f"Scaler fitted: {F} features, range={self.feature_range}, samples={N}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using frozen parameters.

        Args:
            X: Data to transform (N, F) or (N, 1)

        Returns:
            Transformed data (same shape as X)

        Raises:
            RuntimeError: If not fitted
            ValueError: If shape mismatch
        """
        if not self._fitted:
            raise RuntimeError("Scaler not fitted! Call fit() first.")

        if np.isnan(X).any():
            raise ValueError("X contains NaN values - clean before transform")

        # Handle 1D and 2D
        original_shape = X.shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        N, F = X.shape

        if F != len(self._feature_names):
            raise ValueError(
                f"Feature count mismatch: fitted on {len(self._feature_names)}, got {F}"
            )

        X_scaled = np.zeros_like(X, dtype=np.float32)

        range_min, range_max = self.feature_range
        range_span = range_max - range_min

        for i, fname in enumerate(self._feature_names):
            params = self._params[fname]
            x_min = params["min"]
            x_range = params["range"]

            if x_range == 0:
                # Constant column - map to midpoint of target range
                X_scaled[:, i] = (range_min + range_max) / 2
            else:
                # Standard MinMax formula
                X_scaled[:, i] = range_min + (X[:, i] - x_min) / x_range * range_span

        # Clip to range (handle slight numerical errors)
        X_scaled = np.clip(X_scaled, range_min, range_max)

        # Restore original shape
        if len(original_shape) == 1:
            X_scaled = X_scaled.ravel()

        return X_scaled

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform from normalized to original scale.

        Formula: x = x_min + (x_norm - range_min) / range_span * (x_max - x_min)

        Args:
            X_scaled: Scaled data (N, F) or (N, 1)

        Returns:
            Original scale data (same shape)

        Raises:
            RuntimeError: If not fitted
        """
        if not self._fitted:
            raise RuntimeError("Scaler not fitted! Call fit() first.")

        # Handle 1D and 2D
        original_shape = X_scaled.shape
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)

        N, F = X_scaled.shape

        if F != len(self._feature_names):
            raise ValueError(
                f"Feature count mismatch: fitted on {len(self._feature_names)}, got {F}"
            )

        X_orig = np.zeros_like(X_scaled, dtype=np.float32)

        range_min, range_max = self.feature_range
        range_span = range_max - range_min

        for i, fname in enumerate(self._feature_names):
            params = self._params[fname]
            x_min = params["min"]
            x_range = params["range"]

            if x_range == 0:
                # Constant column - return original constant
                X_orig[:, i] = x_min
            else:
                # Inverse formula
                X_orig[:, i] = x_min + (X_scaled[:, i] - range_min) / range_span * x_range

        # Restore original shape
        if len(original_shape) == 1:
            X_orig = X_orig.ravel()

        return X_orig

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit and transform in one step (training data only).

        Args:
            X: Training data
            feature_names: Optional feature names

        Returns:
            Transformed data
        """
        self.fit(X, feature_names)
        return self.transform(X)

    def get_params(self) -> Dict:
        """
        Get scaler parameters for serialization.

        Returns:
            Dictionary with parameters

        Raises:
            RuntimeError: If not fitted
        """
        if not self._fitted:
            raise RuntimeError("Scaler not fitted!")

        return {
            "feature_range": self.feature_range,
            "feature_names": self._feature_names,
            "params": self._params,
            "fitted": self._fitted,
        }

    @classmethod
    def from_params(cls, params: Dict) -> "FrozenMinMaxScaler":
        """
        Reconstruct scaler from saved parameters.

        Args:
            params: Dictionary from get_params()

        Returns:
            Reconstructed scaler
        """
        scaler = cls(feature_range=tuple(params["feature_range"]))
        scaler._fitted = params["fitted"]
        scaler._feature_names = params["feature_names"]
        scaler._params = params["params"]

        logger.info(
            f"Scaler reconstructed: {len(scaler._feature_names)} features, "
            f"range={scaler.feature_range}"
        )

        return scaler

    def verify_transform(
        self, X: np.ndarray, X_scaled: np.ndarray, tolerance: float = 1e-6
    ) -> bool:
        """
        Verify that transform/inverse_transform are consistent.

        Args:
            X: Original data
            X_scaled: Scaled data
            tolerance: Numerical tolerance

        Returns:
            True if verification passes

        Raises:
            AssertionError: If verification fails
        """
        X_reconstructed = self.inverse_transform(X_scaled)

        # Handle NaN in original (from indicator warm-up)
        valid_mask = ~np.isnan(X)

        if valid_mask.sum() == 0:
            logger.warning("No valid samples to verify")
            return True

        diff = np.abs(X[valid_mask] - X_reconstructed[valid_mask])
        max_diff = diff.max()

        if max_diff > tolerance:
            raise AssertionError(
                f"Transform verification failed: max_diff={max_diff:.2e} > tolerance={tolerance:.2e}"
            )

        logger.info(f"Transform verification passed: max_diff={max_diff:.2e}")

        return True


class FrozenStandardScaler:
    """Immutable z-score scaler.

    Formula: ``x_norm = (x - mean) / std`` (per column, computed on the train
    slice).

    Targets ``MSE``-trained models that need a near-symmetric, zero-centered
    distribution. Unlike :class:`FrozenMinMaxScaler`, this does not clip:
    val/test values outside the train range scale linearly. Use it when a
    constant-mean predictor is too easy to learn (which is the typical
    failure mode for noisy-target models trained with MinMax-to-[-1,1]).

    The "fit-once-freeze-forever" invariant is identical to ``FrozenMinMaxScaler``.
    """

    def __init__(self) -> None:
        self._fitted = False
        self._params: Optional[Dict[str, Dict[str, float]]] = None
        self._feature_names: Optional[List[str]] = None

    @property
    def is_fitted(self) -> bool:
        """Return whether scaler has been fitted."""
        return self._fitted

    def fit(
        self, X: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> "FrozenStandardScaler":
        """Fit on training data; can only be called once.

        Args:
            X: Training data ``(N, F)`` or ``(N,)`` for a single target column.
            feature_names: Optional column names (used as keys in the
                serialised parameter dict).

        Returns:
            self.

        Raises:
            RuntimeError: If already fitted.
            ValueError: If X contains NaN or Inf.
        """
        if self._fitted:
            raise RuntimeError("Scaler already fitted! Create a new instance to refit.")
        if np.isnan(X).any():
            raise ValueError("X contains NaN values - clean data before scaling")
        if np.isinf(X).any():
            raise ValueError("X contains Inf values - clean data before scaling")

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        N, F = X.shape

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(F)]
        if len(feature_names) != F:
            raise ValueError(f"feature_names length ({len(feature_names)}) != columns ({F})")

        self._params = {}
        for i, fname in enumerate(feature_names):
            x_col = X[:, i]
            mu = float(x_col.mean())
            sd = float(x_col.std(ddof=0))
            self._params[fname] = {"mean": mu, "std": sd}
            logger.info(f"StandardScaler fit [{fname}]: mean={mu:.8f}, std={sd:.8f}")

        self._feature_names = list(feature_names)
        self._fitted = True
        logger.info(f"StandardScaler fitted: {F} features, samples={N}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardise using frozen parameters."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted! Call fit() first.")
        if np.isnan(X).any():
            raise ValueError("X contains NaN values - clean before transform")

        original_shape = X.shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        N, F = X.shape

        if F != len(self._feature_names):
            raise ValueError(
                f"Feature count mismatch: fitted on {len(self._feature_names)}, got {F}"
            )

        out = np.zeros_like(X, dtype=np.float32)
        for i, fname in enumerate(self._feature_names):
            mu = self._params[fname]["mean"]
            sd = self._params[fname]["std"]
            if sd < 1e-12:
                # Constant column: map to 0.0 (the standardised mean).
                out[:, i] = 0.0
            else:
                out[:, i] = (X[:, i] - mu) / sd

        if len(original_shape) == 1:
            out = out.ravel()
        return out

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverse z-score: ``x = x_norm * std + mean``."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted! Call fit() first.")

        original_shape = X_scaled.shape
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)
        N, F = X_scaled.shape

        if F != len(self._feature_names):
            raise ValueError(
                f"Feature count mismatch: fitted on {len(self._feature_names)}, got {F}"
            )

        out = np.zeros_like(X_scaled, dtype=np.float32)
        for i, fname in enumerate(self._feature_names):
            mu = self._params[fname]["mean"]
            sd = self._params[fname]["std"]
            if sd < 1e-12:
                out[:, i] = mu
            else:
                out[:, i] = X_scaled[:, i] * sd + mu

        if len(original_shape) == 1:
            out = out.ravel()
        return out

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        self.fit(X, feature_names)
        return self.transform(X)

    def get_params(self) -> Dict:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted!")
        return {
            "scaler_type": "FrozenStandardScaler",
            "feature_names": self._feature_names,
            "params": self._params,
            "fitted": self._fitted,
        }

    @classmethod
    def from_params(cls, params: Dict) -> "FrozenStandardScaler":
        scaler = cls()
        scaler._fitted = params["fitted"]
        scaler._feature_names = params["feature_names"]
        scaler._params = params["params"]
        logger.info(f"StandardScaler reconstructed: {len(scaler._feature_names)} features")
        return scaler
