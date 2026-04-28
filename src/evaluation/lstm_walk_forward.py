"""
Production-Grade Expanding-Window Walk-Forward Validation with PSO

Implements expanding-window walk-forward validation as specified in WALK_FORWARD_PLAN.md.

CRITICAL ARCHITECTURE:
- Expanding training window (not rolling)
- Per-fold independent scaling (feature + target)
- Per-fold PSO optimization
- Per-fold model training (fresh initialization)
- Inverse transform before metrics
- TRD-compliant (no leakage)

TRD Compliance:
- TRD1 §8: All data leakage rules enforced
- TRD2 §7.4: PSO per-fold with internal 90/10 split
- TRD1 §9.1: Full reproducibility

Author: Production System
Version: 1.0
Source: WALK_FORWARD_PLAN.md Phase 4
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.scaler import FrozenMinMaxScaler
from src.data.windowing import build_lstm_windows
from src.models.pso_lstm_model import PSOLSTMModel
from src.models.pso_lstm_trainer import PSOLSTMTrainer
from src.optimizer import IPSO

logger = logging.getLogger(__name__)


class ExpandingWindowWalkForward:
    """
    Expanding-window walk-forward validator with per-fold PSO and scaling.

    ARCHITECTURE (WALK_FORWARD_PLAN.md Phase 4.2):

    For each fold i:
        1. Train window: [0 : initial_size + i * step_size] (EXPANDING)
        2. Val window: [train_end : train_end + val_size]
        3. Fit feature scaler on train[i]
        4. Fit target scaler on train[i]
        5. Transform train[i] and val[i]
        6. Run PSO on train[i]
        7. Train LSTM with PSO params
        8. Predict on val[i]
        9. Inverse transform predictions
        10. Compute metrics on original scale

    NO shared state between folds.
    """

    def __init__(
        self,
        initial_train_pct: float = 0.60,
        fold_step_pct: float = 0.05,
        val_pct: float = 0.05,
        lookback: int = 20,
        config: Any = None,
        run_pso: bool = True,
        max_folds: int = 0,
    ):
        """
        Initialize expanding-window walk-forward validator.

        Args:
            initial_train_pct: Initial training window size (default: 60%)
            fold_step_pct: Step size for window expansion (default: 5%)
            val_pct: Validation size per fold (default: 5%)
            lookback: LSTM lookback window (default: 20)
            config: Configuration object
            run_pso: Whether to run PSO per fold (default: True)
            max_folds: Maximum folds to run (0 = unlimited)
        """
        self.initial_train_pct = initial_train_pct
        self.fold_step_pct = fold_step_pct
        self.val_pct = val_pct
        self.lookback = lookback
        self.config = config
        self.run_pso = run_pso
        self.max_folds = max_folds

        logger.info("=" * 80)
        logger.info("EXPANDING WINDOW WALK-FORWARD VALIDATOR (TRD-COMPLIANT)")
        logger.info("=" * 80)
        logger.info(f"Initial train: {initial_train_pct:.0%} of data")
        logger.info(f"Fold step: {fold_step_pct:.0%} of data")
        logger.info(f"Val size: {val_pct:.0%} per fold")
        logger.info(f"Lookback: {lookback} days")
        logger.info(f"PSO enabled: {run_pso}")
        logger.info(f"Max folds: {max_folds if max_folds > 0 else 'unlimited'}")
        logger.info("=" * 80)

    def validate(
        self,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> Dict:
        """
        Execute expanding-window walk-forward validation.

        CRITICAL: X_raw and y_raw must be UNSCALED (global preprocessing only).

        Args:
            X_raw: Preprocessed features (NO SCALING)
            y_raw: Target returns (NO SCALING)
            timestamps: Optional datetime index

        Returns:
            Complete validation results with per-fold metrics
        """
        logger.info("=" * 80)
        logger.info("STARTING WALK-FORWARD VALIDATION")
        logger.info("=" * 80)
        logger.info(f"Total samples: {len(X_raw)}")
        logger.info(f"Features: {X_raw.shape[1] if X_raw.ndim > 1 else 'N/A'}")
        logger.info("=" * 80)

        # Compute fold boundaries
        fold_boundaries = self._compute_fold_boundaries(len(X_raw))

        logger.info(f"Computed {len(fold_boundaries)} folds")
        for i, (ts, te, vs, ve) in enumerate(fold_boundaries[:5]):
            logger.info(
                f"  Fold {i+1}: Train[{ts}:{te}] ({te-ts}), Val[{vs}:{ve}] ({ve-vs})"
            )
        if len(fold_boundaries) > 5:
            logger.info(f"  ... ({len(fold_boundaries) - 5} more folds)")

        # Execute fold loop
        fold_results = []

        for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(
            fold_boundaries
        ):
            logger.info("\n" + "=" * 80)
            logger.info(f"PROCESSING FOLD {fold_idx + 1}/{len(fold_boundaries)}")
            logger.info("=" * 80)

            fold_result = self._process_fold(
                fold_idx=fold_idx,
                X_raw=X_raw,
                y_raw=y_raw,
                timestamps=timestamps,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            )

            fold_results.append(fold_result)

            logger.info(
                f" Fold {fold_idx + 1} complete: RMSE={fold_result['metrics']['rmse']:.6f}"
            )

        # Aggregate results
        aggregated_metrics = self._aggregate_fold_results(fold_results)

        logger.info("\n" + "=" * 80)
        logger.info("WALK-FORWARD VALIDATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total folds: {len(fold_results)}")
        logger.info(
            f"Aggregate RMSE: {aggregated_metrics['rmse_mean']:.6f} ± {aggregated_metrics['rmse_std']:.6f}"
        )
        logger.info(
            f"Aggregate R²: {aggregated_metrics['r2_mean']:.4f} ± {aggregated_metrics['r2_std']:.4f}"
        )
        logger.info("=" * 80)

        return {
            "fold_results": fold_results,
            "aggregated_metrics": aggregated_metrics,
            "fold_boundaries": fold_boundaries,
            "config": {
                "initial_train_pct": self.initial_train_pct,
                "fold_step_pct": self.fold_step_pct,
                "val_pct": self.val_pct,
                "lookback": self.lookback,
                "pso_enabled": self.run_pso,
            },
        }

    def _process_fold(
        self,
        fold_idx: int,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex],
        train_start: int,
        train_end: int,
        val_start: int,
        val_end: int,
    ) -> Dict:
        """
        Process a single fold with COMPLETE ISOLATION.

        CRITICAL: NO shared state with other folds.

        Steps:
        A. Fit feature scaler on THIS fold's train
        B. Fit target scaler on THIS fold's train
        C. Transform train & val
        D. Build sequences
        E. Run PSO (if enabled)
        F. Train LSTM
        G. Predict
        H. Inverse transform
        I. Compute metrics

        Args:
            fold_idx: Fold index
            X_raw: Full feature array (unscaled)
            y_raw: Full target array (unscaled)
            timestamps: Optional datetime index
            train_start, train_end: Training slice indices
            val_start, val_end: Validation slice indices

        Returns:
            Fold results with metrics, predictions, scalers, PSO params
        """
        logger.info(
            f"Train: [{train_start}:{train_end}] ({train_end - train_start} samples)"
        )
        logger.info(f"Val:   [{val_start}:{val_end}] ({val_end - val_start} samples)")

        if timestamps is not None:
            logger.info(
                f"Dates: {timestamps[train_start]} → "
                f"{timestamps[train_end-1]} (train), "
                f"{timestamps[val_start]} → {timestamps[val_end-1]} (val)"
            )

        # Extract fold data (UNSCALED)
        X_train_raw = X_raw[train_start:train_end]
        y_train_raw = y_raw[train_start:train_end]
        X_val_raw = X_raw[val_start:val_end]
        y_val_raw = y_raw[val_start:val_end]

        logger.info(
            f"Extracted raw data: X_train={X_train_raw.shape}, y_train={y_train_raw.shape}"
        )

        # ================================================================
        # A. FIT FEATURE SCALER (TRAIN ONLY)
        # ================================================================
        feature_scaler = FrozenMinMaxScaler(feature_range=(-1, 1))
        feature_scaler.fit(X_train_raw)

        logger.info(f" Feature scaler fitted on TRAIN ONLY (fold {fold_idx + 1})")

        # ================================================================
        # B. FIT TARGET SCALER (TRAIN ONLY)
        # ================================================================
        target_scaler = FrozenMinMaxScaler(feature_range=(-1, 1))
        target_scaler.fit(y_train_raw.reshape(-1, 1))

        logger.info(f" Target scaler fitted on TRAIN ONLY (fold {fold_idx + 1})")

        # ================================================================
        # C. TRANSFORM WITH FITTED SCALERS
        # ================================================================
        X_train_scaled = feature_scaler.transform(X_train_raw)
        y_train_scaled = target_scaler.transform(y_train_raw.reshape(-1, 1)).flatten()
        X_val_scaled = feature_scaler.transform(X_val_raw)
        y_val_scaled = target_scaler.transform(y_val_raw.reshape(-1, 1)).flatten()

        logger.info(" Data scaled with fold-specific scalers")
        logger.info(
            f"  X_train_scaled: {X_train_scaled.shape}, range=[{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]"
        )
        logger.info(
            f"  y_train_scaled: {y_train_scaled.shape}, range=[{y_train_scaled.min():.2f}, {y_train_scaled.max():.2f}]"
        )

        # ================================================================
        # D. BUILD LSTM SEQUENCES (HANDLE BOUNDARIES)
        # ================================================================
        X_train_seq, y_train_seq = build_lstm_windows(
            X_train_scaled, y_train_scaled, self.lookback
        )
        X_val_seq, y_val_seq = build_lstm_windows(
            X_val_scaled, y_val_scaled, self.lookback
        )

        logger.info(f" Sequences built:")
        logger.info(f"  X_train_seq: {X_train_seq.shape}")
        logger.info(f"  X_val_seq: {X_val_seq.shape}")

        # Validate no boundary violations
        assert X_train_seq.shape[0] == len(y_train_seq), "Train sequence mismatch"
        assert X_val_seq.shape[0] == len(y_val_seq), "Val sequence mismatch"

        # ================================================================
        # E. RUN PSO OPTIMIZATION (IF ENABLED)
        # ================================================================
        if self.run_pso and self.config is not None:
            pso_result = self._run_pso_for_fold(X_train_seq, y_train_seq, fold_idx)
            best_params = pso_result["best_params"]
            pso_fitness = pso_result["best_fitness"]

            logger.info(f" PSO optimization complete:")
            logger.info(f"  Best params: {best_params}")
            logger.info(f"  Best fitness: {pso_fitness:.6f}")
        else:
            # Use fixed baseline parameters
            best_params = self._get_baseline_params()
            pso_fitness = None

            logger.info(" Using baseline (fixed) hyperparameters (PSO disabled)")

        # ================================================================
        # F. TRAIN LSTM WITH PSO/BASELINE PARAMS
        # ================================================================
        model = self._train_lstm_with_params(
            X_train_seq, y_train_seq, best_params, fold_idx
        )

        logger.info(" LSTM trained")

        # ================================================================
        # G. PREDICT ON VALIDATION SET
        # ================================================================
        y_pred_scaled = model.predict(X_val_seq, verbose=0).flatten()

        logger.info(f" Predictions generated: {len(y_pred_scaled)} samples")

        # ================================================================
        # H. INVERSE TRANSFORM PREDICTIONS (CRITICAL)
        # ================================================================
        y_pred_original = target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()
        y_val_original = target_scaler.inverse_transform(
            y_val_seq.reshape(-1, 1)
        ).flatten()

        logger.info(" Predictions inverse-transformed to ORIGINAL scale")

        # ================================================================
        # I. COMPUTE METRICS ON ORIGINAL SCALE
        # ================================================================
        metrics = self._compute_fold_metrics(y_pred_original, y_val_original)

        logger.info(f" Fold metrics computed:")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  MAE:  {metrics['mae']:.6f}")
        logger.info(f"  R²:   {metrics['r2']:.4f}")
        logger.info(f"  DA:   {metrics['directional_accuracy']:.2%}")

        return {
            "fold_idx": fold_idx,
            "train_size": train_end - train_start,
            "val_size": val_end - val_start,
            "train_indices": (train_start, train_end),
            "val_indices": (val_start, val_end),
            "pso_params": best_params if self.run_pso else None,
            "pso_fitness": pso_fitness,
            "metrics": metrics,
            "predictions": {
                "y_pred": y_pred_original.tolist(),
                "y_true": y_val_original.tolist(),
                "timestamps": (
                    timestamps[val_start + self.lookback : val_end].tolist()
                    if timestamps is not None
                    else None
                ),
            },
            "scalers": {
                "feature_scaler_params": feature_scaler.get_params(),
                "target_scaler_params": target_scaler.get_params(),
            },
        }

    def _compute_fold_boundaries(
        self, n_samples: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Compute fold boundaries for expanding-window validation.

        Returns:
            List of (train_start, train_end, val_start, val_end) tuples
        """
        initial_train_size = int(n_samples * self.initial_train_pct)
        fold_step_size = int(n_samples * self.fold_step_pct)
        val_size = int(n_samples * self.val_pct)

        boundaries = []
        train_start = 0
        fold_idx = 0

        while True:
            # Expanding window
            train_end = initial_train_size + fold_idx * fold_step_size
            val_start = train_end
            val_end = val_start + val_size

            # Stop if validation extends beyond data
            if val_end > n_samples:
                break

            # Stop if max_folds reached
            if self.max_folds > 0 and fold_idx >= self.max_folds:
                break

            boundaries.append((train_start, train_end, val_start, val_end))
            fold_idx += 1

        return boundaries

    def _run_pso_for_fold(
        self,
        X_train_seq: np.ndarray,
        y_train_seq: np.ndarray,
        fold_idx: int,
    ) -> Dict:
        """
        Run PSO optimization for this fold.

        PSO uses internal 90/10 split of fold's training data.

        TRD2 §7.4: PSO validation split is internal to fold training data.

        Args:
            X_train_seq: Fold training sequences (scaled)
            y_train_seq: Fold training targets (scaled)
            fold_idx: Fold index (used for seed variation)

        Returns:
            Dictionary with best_params and best_fitness
        """
        logger.info(f"Running PSO for fold {fold_idx + 1}...")

        # Internal PSO split (90/10)
        pso_split_idx = int(len(X_train_seq) * 0.9)

        X_pso_train = X_train_seq[:pso_split_idx]
        y_pso_train = y_train_seq[:pso_split_idx]
        X_pso_val = X_train_seq[pso_split_idx:]
        y_pso_val = y_train_seq[pso_split_idx:]

        logger.info(
            f"  PSO internal split: train={len(X_pso_train)}, val={len(X_pso_val)}"
        )

        # Get PSO config
        pso_config = self.config.pso
        seed = pso_config.random_seed + fold_idx  # Vary seed per fold

        # Define search space
        search_space = {
            "epochs": {
                "min": pso_config.search_space.epochs.min,
                "max": pso_config.search_space.epochs.max,
            },
            "units_1": {
                "min": pso_config.search_space.lstm_units_1.min,
                "max": pso_config.search_space.lstm_units_1.max,
            },
            "units_2": {
                "min": pso_config.search_space.lstm_units_2.min,
                "max": pso_config.search_space.lstm_units_2.max,
            },
            "learning_rate": {
                "min": pso_config.search_space.learning_rate.min,
                "max": pso_config.search_space.learning_rate.max,
                "scale": pso_config.search_space.learning_rate.scale,
            },
            "dropout": {
                "min": pso_config.search_space.dropout_rate.min,
                "max": pso_config.search_space.dropout_rate.max,
            },
            "batch_size": {
                "choices": pso_config.search_space.batch_size.choices,
            },
        }

        # Define fitness function
        def fitness_fn(params):
            # Build model config
            model_config = {
                "input_shape": (self.lookback, X_train_seq.shape[2]),
                "lstm_units_1": params["units_1"],
                "lstm_units_2": params["units_2"],
                "dropout_rate": params["dropout"],
                "activation": self.config.lstm.activation,
                "output_units": self.config.lstm.output_units,
                "output_activation": self.config.lstm.output_activation,
                "learning_rate": params["learning_rate"],
                "loss": self.config.lstm.loss,
            }

            # Create and train model
            model = PSOLSTMModel(seed=seed)
            trainer = PSOLSTMTrainer(model_config, seed=seed)

            model, _ = trainer.train(
                X_pso_train,
                y_pso_train,
                X_pso_val,
                y_pso_val,
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                patience=self.config.lstm.early_stopping.patience,
                shuffle=False,
            )

            # Compute validation MSE
            y_pred = model.predict(X_pso_val, verbose=0)
            mse = np.mean((y_pso_val - y_pred.flatten()) ** 2)

            return float(mse)

        # Initialize PSO
        optimizer = IPSOOptimizer(
            n_particles=pso_config.n_particles,
            n_iterations=pso_config.n_iterations,
            search_space=search_space,
            fitness_func=fitness_fn,
            inertia_min=pso_config.inertia_min,
            inertia_max=pso_config.inertia_max,
            c1=pso_config.c1,
            c2=pso_config.c2,
            v_clamp_fraction=pso_config.v_clamp_fraction,
            seed=seed,
        )

        # Run optimization
        best_params, best_fitness = optimizer.optimize()

        logger.info(f"  PSO complete: fitness={best_fitness:.6f}")

        return {
            "best_params": best_params,
            "best_fitness": float(best_fitness),
            "pso_train_size": len(X_pso_train),
            "pso_val_size": len(X_pso_val),
            "seed": seed,
        }

    def _train_lstm_with_params(
        self,
        X_train_seq: np.ndarray,
        y_train_seq: np.ndarray,
        params: Dict,
        fold_idx: int,
    ) -> Any:
        """
        Train LSTM with given parameters on full fold training data.

        CRITICAL: Fresh model initialization (no weight carryover).

        Args:
            X_train_seq: Training sequences (scaled)
            y_train_seq: Training targets (scaled)
            params: Hyperparameters (from PSO or baseline)
            fold_idx: Fold index (for seed variation)

        Returns:
            Trained LSTMModel
        """
        seed = self.config.pso.random_seed + fold_idx if self.config else 42 + fold_idx

        model_config = {
            "input_shape": (self.lookback, X_train_seq.shape[2]),
            "lstm_units_1": params["units_1"],
            "lstm_units_2": params["units_2"],
            "dropout_rate": params["dropout"],
            "activation": self.config.lstm.activation if self.config else "relu",
            "output_units": 1,
            "output_activation": "linear",
            "learning_rate": params["learning_rate"],
            "loss": "mse",
        }

        model = PSOLSTMModel(seed=seed)
        trainer = PSOLSTMTrainer(model_config, seed=seed)

        # Train on full fold training data (no validation in final fit)
        model, _ = trainer.train(
            X_train_seq,
            y_train_seq,
            None,  # No validation in final training
            None,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            patience=None,  # No early stopping in final fit
            shuffle=False,
        )

        return model

    def _get_baseline_params(self) -> Dict:
        """
        Get baseline (fixed) hyperparameters when PSO is disabled.

        Returns:
            Dictionary of baseline hyperparameters
        """
        if self.config and hasattr(self.config, "lstm_baseline"):
            return {
                "units_1": self.config.lstm_baseline.lstm_units_1,
                "units_2": self.config.lstm_baseline.lstm_units_2,
                "dropout": self.config.lstm_baseline.dropout_rate,
                "learning_rate": self.config.lstm_baseline.learning_rate,
                "epochs": self.config.lstm_baseline.epochs,
                "batch_size": self.config.lstm_baseline.batch_size,
            }
        else:
            # Fallback defaults
            return {
                "units_1": 128,
                "units_2": 64,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "epochs": 100,
                "batch_size": 32,
            }

    def _compute_fold_metrics(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute fold-specific metrics on ORIGINAL scale.

        CRITICAL: This must be called AFTER inverse transform.

        Args:
            y_pred: Predictions (original scale)
            y_true: Actuals (original scale)

        Returns:
            Dictionary of metrics
        """
        # Basic regression metrics
        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(mse))

        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

        # MAPE (avoid division by zero)
        nonzero_mask = np.abs(y_true) > 1e-10
        if nonzero_mask.any():
            mape = float(
                np.mean(
                    np.abs(
                        (y_true[nonzero_mask] - y_pred[nonzero_mask])
                        / y_true[nonzero_mask]
                    )
                )
                * 100
            )
        else:
            mape = float("inf")

        # FIXED (2026-04-28): Use corrected directional accuracy
        from src.evaluation.metrics import directional_accuracy as da_corrected
        da = da_corrected(y_true, y_pred, threshold=0.0, exclude_zeros=True)

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "directional_accuracy": da,
            "n_samples": len(y_true),
        }

    def _aggregate_fold_results(self, fold_results: List[Dict]) -> Dict:
        """
        Aggregate metrics across all folds.

        Computes mean ± std for all metrics.

        Args:
            fold_results: List of fold result dictionaries

        Returns:
            Aggregated metrics dictionary
        """
        metric_names = ["mse", "mae", "rmse", "r2", "mape", "directional_accuracy"]

        aggregated = {}

        for metric in metric_names:
            values = [
                fold["metrics"][metric]
                for fold in fold_results
                if not np.isinf(fold["metrics"][metric])
            ]

            if values:
                aggregated[f"{metric}_mean"] = float(np.mean(values))
                aggregated[f"{metric}_std"] = float(np.std(values))
                aggregated[f"{metric}_min"] = float(np.min(values))
                aggregated[f"{metric}_max"] = float(np.max(values))
                aggregated[f"{metric}_median"] = float(np.median(values))
            else:
                aggregated[f"{metric}_mean"] = float("nan")
                aggregated[f"{metric}_std"] = float("nan")
                aggregated[f"{metric}_min"] = float("nan")
                aggregated[f"{metric}_max"] = float("nan")
                aggregated[f"{metric}_median"] = float("nan")

        aggregated["n_folds"] = len(fold_results)
        aggregated["total_predictions"] = sum(
            f["metrics"]["n_samples"] for f in fold_results
        )

        logger.info("\n" + "=" * 80)
        logger.info("AGGREGATED METRICS (ACROSS FOLDS)")
        logger.info("=" * 80)
        logger.info(f"Number of folds: {aggregated['n_folds']}")
        logger.info(f"Total predictions: {aggregated['total_predictions']}")
        logger.info("-" * 80)
        logger.info(
            f"RMSE:  {aggregated['rmse_mean']:.6f} ± {aggregated['rmse_std']:.6f}"
        )
        logger.info(
            f"MAE:   {aggregated['mae_mean']:.6f} ± {aggregated['mae_std']:.6f}"
        )
        logger.info(f"R²:    {aggregated['r2_mean']:.4f} ± {aggregated['r2_std']:.4f}")
        logger.info(
            f"MAPE:  {aggregated['mape_mean']:.2f}% ± {aggregated['mape_std']:.2f}%"
        )
        logger.info(
            f"DA:    {aggregated['directional_accuracy_mean']:.2%} ± {aggregated['directional_accuracy_std']:.2%}"
        )
        logger.info("=" * 80)

        return aggregated


def validate_walk_forward_compliance(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    fold_boundaries: List[Tuple[int, int, int, int]],
) -> None:
    """
    Validate TRD compliance for walk-forward validation.

    TRD1 §8: Data Leakage Prevention Rules

    Args:
        X_raw: Full feature array
        y_raw: Full target array
        fold_boundaries: List of fold boundaries

    Raises:
        AssertionError: If any TRD rule violated
    """
    logger.info("Validating TRD compliance for walk-forward validation...")

    # Check temporal ordering (expanding window)
    for i in range(len(fold_boundaries) - 1):
        _, train_end_i, _, _ = fold_boundaries[i]
        _, train_end_j, _, _ = fold_boundaries[i + 1]

        assert (
            train_end_j > train_end_i
        ), f"TRD VIOLATION: Training window not expanding (fold {i} → {i+1})"

    logger.info(" Expanding window validated")

    # Check no overlap between folds
    for i, (ts, te, vs, ve) in enumerate(fold_boundaries):
        assert vs == te, f"TRD VIOLATION: Gap between train and val in fold {i}"
        assert ve > vs, f"TRD VIOLATION: Empty validation set in fold {i}"
        assert te > ts, f"TRD VIOLATION: Empty training set in fold {i}"

    logger.info(" Fold boundaries validated")

    # Check sufficient data per fold
    for i, (ts, te, vs, ve) in enumerate(fold_boundaries):
        assert (te - ts) >= 20, f"TRD VIOLATION: Insufficient train data in fold {i}"
        assert (ve - vs) >= 1, f"TRD VIOLATION: Insufficient val data in fold {i}"

    logger.info(" Sufficient data per fold")
    logger.info(" TRD compliance validated")
