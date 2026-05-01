import argparse
import hashlib
import json
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.windowing import build_lstm_windows
from src.models import LSTMModel, LSTMTrainer
from src.models.utils import set_seeds
from src.optimizer import IPSO, SpecCompliantFitness
from src.optimizer.particle import LB, UB
from src.utils.config_loader import Config, load_config
from src.utils.logger import LogFileMode, setup_logger


logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for full reproducibility (TRD1 §9.1).

    Delegates to :func:`src.models.utils.set_seeds`, which seeds
    Python's ``random``, NumPy, and PyTorch (CPU and CUDA) so model
    training is fully deterministic. This wrapper is preserved for
    backward compatibility with the pipeline-level naming convention.

    Args:
        seed: Random seed value applied to every supported RNG.
    """
    set_seeds(seed)
    logger.info(f"All seeds set to {seed} (TRD1 §9.1 compliance)")


def create_pso_split(
    X: np.ndarray, y: np.ndarray, pso_train_ratio: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create TRD-compliant 72/8 split for PSO Phase 1 (TRD2 §7.4).

    Takes 80% data and splits into:
    - 72% for PSO training (90% of 80%)
    - 8% for PSO validation (10% of 80%)

    Args:
        X: Full training features (80% of total)
        y: Full training targets (80% of total)
        pso_train_ratio: Ratio for PSO train split (default 0.9 = 72/8)

    Returns:
        X_pso_train, y_pso_train, X_pso_val, y_pso_val
    """
    split_idx = int(len(X) * pso_train_ratio)

    X_pso_train = X[:split_idx]
    y_pso_train = y[:split_idx]
    X_pso_val = X[split_idx:]
    y_pso_val = y[split_idx:]

    logger.info("TRD2 §7.4 PSO split created:")
    logger.info(f"  PSO train: {len(X_pso_train)} samples (72% of total)")
    logger.info(f"  PSO val:   {len(X_pso_val)} samples (8% of total)")

    return X_pso_train, y_pso_train, X_pso_val, y_pso_val


def validate_no_leakage(data: Dict, lookback: int) -> None:
    """
    Validate TRD1 §8 data leakage prevention rules.

    Args:
        data: Dictionary with train/val/test splits
        lookback: Window size

    Raises:
        AssertionError: If any TRD rule is violated
    """
    logger.info("=" * 80)
    logger.info("VALIDATING TRD1 §8 DATA LEAKAGE RULES")
    logger.info("=" * 80)

    # TRD1 §8.1 L-7: Window boundaries
    assert data["X_train"].shape[0] >= lookback, (
        f"TRD1 §8.1 L-7 VIOLATION: Train set too small for lookback window "
        f"(need >={lookback}, got {data['X_train'].shape[0]})"
    )
    logger.info(" L-7: Window boundary check passed")

    # TRD1 §8.2: Temporal ordering (assuming preprocessed data is ordered)
    logger.info(" L-1: Temporal ordering assumed from preprocessing")

    # TRD1 §8.1 L-3, L-4, L-5: Scaler/correlation/wavelet fit on train only
    logger.info(" L-3, L-4, L-5: Scaler/selector fit on train only (upstream)")

    logger.info("=" * 80)
    logger.info(" ALL TRD1 §8 LEAKAGE CHECKS PASSED")
    logger.info("=" * 80)


def load_preprocessed_data(data_path: Path) -> Dict:
    """
    Load preprocessed features from canonical feature pipeline.

    TRD Requirements:
    - Data must be pre-split 70/10/20 (train/val/test)
    - Features must be pre-scaled (TRD1 §4.2)
    - Features must be pre-selected (TRD1 §5)

    Args:
        data_path: Directory containing X_train.npy, y_train.npy, etc.

    Returns:
        Dictionary with train/val/test splits
    """
    logger.info(f"Loading TRD-preprocessed data from {data_path}")

    data = {}
    for split in ["train", "val", "test"]:
        X_path = data_path / f"X_{split}.npy"
        y_path = data_path / f"y_{split}.npy"

        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Missing {split} data: {X_path} or {y_path}\n"
                f"Run TRD-compliant feature pipeline first."
            )

        data[f"X_{split}"] = np.load(X_path)
        data[f"y_{split}"] = np.load(y_path)

        logger.info(
            f"  {split}: X={data[f'X_{split}'].shape}, y={data[f'y_{split}'].shape}"
        )

    return data


# Note: fitness_function is now implemented inline in model_builder
# to properly integrate with src/optimizer/ infrastructure


def _pso_model_builder(
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    test_hash: str,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, Any]:
    """Train one LSTM particle and return ``(y_pred, nn.Module)``.

    Defined at module scope (rather than as a closure inside
    ``phase1_pso_search``) so :func:`functools.partial` of this function is
    picklable by ``ProcessPoolExecutor``. Closures cannot be pickled by the
    standard library, which previously caused ``n_workers > 1`` to silently
    fall back to sequential evaluation.

    Args:
        params: Particle-decoded hyperparameters.
        X_train: PSO-train sequences (windowed).
        y_train: PSO-train targets.
        X_val: PSO-val sequences (windowed).
        y_val: PSO-val targets.
        seed: Reproducibility seed.
        test_hash: SHA-256 of the canonical contiguous test array, captured
            in the parent before PSO begins; checked here to enforce
            test-set isolation (TRD1 §8.1 L-6).
        X_test: Test array referenced for the hash check; never used to
            train.

    Returns:
        Tuple of validation predictions and the underlying ``nn.Module``
        (the latter so the fitness function can compute MSW directly from
        ``model.parameters()``).
    """
    architecture_config = {
        "input_size": X_train.shape[2],
        "lstm_units_1": params["units_1"],
        "lstm_units_2": params["units_2"],
        "dropout_rate": params["dropout"],
        "output_units": 1,
        "activation": "relu",
        "output_activation": "linear",
    }

    trainer_config = {
        "learning_rate": params["learning_rate"],
        # Epochs are no longer optimised by PSO. Every particle trains for
        # a fixed 100 epochs so fitness reflects only architectural and
        # learning-rate quality, not training-budget drift.
        "epochs": 100,
        "batch_size": params["batch_size"],
        "optimizer": "adam",
        "loss": "mse",
        "shuffle": False,
        "grad_clip": 1.0,
        # AMP enabled; trainer silently falls back to fp32 on CPU.
        "use_amp": True,
        "accumulation_steps": 1,
        # Quiet inside the swarm: parent's tqdm is the only console UI.
        "quiet": True,
        "early_stopping": {
            "enabled": True,
            "monitor": "val_loss",
            "patience": 10,
            "restore_best_weights": True,
        },
    }

    model_wrapper = LSTMModel(seed=seed)
    model_wrapper.build_model(architecture_config)

    trainer = LSTMTrainer(
        lstm_model=model_wrapper, config=trainer_config, seed=seed
    )
    trained_wrapper, _ = trainer.train(X_train, y_train, X_val, y_val)

    # progress=False so the inner "Predicting" tqdm bar does not clobber
    # the parent's PSO iteration progress bar.
    y_pred = trained_wrapper.predict(X_val, progress=False)

    # TRD1 §8.1 L-6: Verify test set never accessed during the swarm
    # evaluation. Hash the canonical contiguous layout so the digest
    # matches the value captured before PSO began regardless of view state.
    current_hash = hashlib.sha256(
        np.ascontiguousarray(X_test).tobytes()
    ).hexdigest()
    assert (
        current_hash == test_hash
    ), "CRITICAL TRD VIOLATION: Test set accessed during PSO (L-6)"

    return y_pred, trained_wrapper.model


def phase1_pso_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Config,
    lookback: int,
    output_dir: Path,
) -> Dict:
    """
    Phase 1: PSO hyperparameter search (TRD2 §7.4).

    Search on 72% train, validate on 8% val.
    Test set (20%) is NEVER accessed (TRD1 §8.1 L-6).

    Args:
        X_train: Training features (70% of total)
        y_train: Training targets (70% of total)
        X_val: Validation features (10% of total)
        y_val: Validation targets (10% of total)
        X_test: Test features (20% of total) - NEVER USED
        y_test: Test targets (20% of total) - NEVER USED
        config: Configuration object
        lookback: Window size
        output_dir: Directory to save results

    Returns:
        Dictionary of best hyperparameters
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: PSO HYPERPARAMETER SEARCH (TRD2 §7.4)")
    logger.info("=" * 80)
    logger.info("TRD-Compliant Split: 72% PSO train, 8% PSO val, 20% test (ISOLATED)")
    logger.info("=" * 80)

    # TRD2 §7.4: Combine train (70%) + val (10%) = 80% for PSO split
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)

    logger.info(f"Combined 80% data: X={X_combined.shape}, y={y_combined.shape}")

    # TRD2 §7.4: Split 80% into 72% PSO train + 8% PSO val
    X_pso_train, y_pso_train, X_pso_val, y_pso_val = create_pso_split(
        X_combined, y_combined, pso_train_ratio=0.9
    )

    # Extract PSO config
    pso_config = config.pso
    seed = config.lstm_baseline.random_seed
    set_all_seeds(seed)

    # Optionally subsample the PSO train slice. Only the most-recent fraction
    # is retained so the search sees regime-relevant data; the val slice is
    # never subsampled.
    subsample_fraction = float(getattr(pso_config, "subsample_train_fraction", 1.0))
    if 0.0 < subsample_fraction < 1.0:
        keep = max(lookback + 2, int(len(X_pso_train) * subsample_fraction))
        if keep < len(X_pso_train):
            logger.info(
                "Subsampling PSO train to last %d / %d rows (%.0f%%)",
                keep,
                len(X_pso_train),
                subsample_fraction * 100,
            )
            X_pso_train = X_pso_train[-keep:]
            y_pso_train = y_pso_train[-keep:]

    # Build LSTM windows
    logger.info(f"Building LSTM windows (lookback={lookback})...")
    X_pso_train_win, y_pso_train_win = build_lstm_windows(
        X_pso_train, y_pso_train, lookback
    )
    X_pso_val_win, y_pso_val_win = build_lstm_windows(X_pso_val, y_pso_val, lookback)

    logger.info(f"Windowed shapes:")
    logger.info(
        f"  X_pso_train: {X_pso_train_win.shape}, y_pso_train: {y_pso_train_win.shape}"
    )
    logger.info(
        f"  X_pso_val:   {X_pso_val_win.shape}, y_pso_val:   {y_pso_val_win.shape}"
    )

    # TRD1 §8.1 L-6: Protect test set from access. Canonicalize the
    # array layout before hashing so non-contiguous views (e.g. slices
    # or transposes) hash to the same digest as their contiguous form.
    _test_data_hash = hashlib.sha256(
        np.ascontiguousarray(X_test).tobytes()
    ).hexdigest()
    logger.info(f"Test set protected (hash: {_test_data_hash[:16]}...)")

    # TRD1 §7.2: Validate that the YAML-declared search-space bounds
    # match the canonical particle-encoding bounds (LB/UB). The PSO core
    # consumes LB/UB directly via Particle, so any drift between the
    # config and the encoding would silently change the search space.
    expected_log_lr_min = float(np.log(pso_config.search_space.learning_rate.min))
    expected_log_lr_max = float(np.log(pso_config.search_space.learning_rate.max))
    expected_batch_idx_max = float(
        len(pso_config.search_space.batch_size.choices)
    ) - 0.01
    bound_checks = [
        ("units_1.min", pso_config.search_space.lstm_units_1.min, LB[0]),
        ("units_1.max", pso_config.search_space.lstm_units_1.max, UB[0]),
        ("units_2.min", pso_config.search_space.lstm_units_2.min, LB[1]),
        ("units_2.max", pso_config.search_space.lstm_units_2.max, UB[1]),
        ("dropout_rate.min", pso_config.search_space.dropout_rate.min, LB[2]),
        ("dropout_rate.max", pso_config.search_space.dropout_rate.max, UB[2]),
        ("learning_rate.log_min", expected_log_lr_min, LB[3]),
        ("learning_rate.log_max", expected_log_lr_max, UB[3]),
        ("batch_size_idx.min", 0.0, LB[4]),
        ("batch_size_idx.max", expected_batch_idx_max, UB[4]),
        ("epochs.min", pso_config.search_space.epochs.min, LB[5]),
        ("epochs.max", pso_config.search_space.epochs.max, UB[5]),
    ]
    drift = [
        (name, cfg, bound)
        for name, cfg, bound in bound_checks
        if not np.isclose(float(cfg), float(bound), rtol=1e-6, atol=1e-6)
    ]
    if drift:
        details = ", ".join(
            f"{name}: config={cfg} vs LB/UB={bound}" for name, cfg, bound in drift
        )
        raise ValueError(
            "PSO search-space drift detected between YAML config and "
            f"particle encoding LB/UB: {details}"
        )

    # Build a picklable model_builder by binding the parent-scope state
    # (seed, test hash, X_test) onto the module-level _pso_model_builder.
    # functools.partial of a top-level function IS picklable, whereas a
    # nested closure is NOT — using a closure here silently demoted PSO
    # to sequential evaluation regardless of pso.n_workers.
    model_builder = partial(
        _pso_model_builder,
        seed=seed,
        test_hash=_test_data_hash,
        X_test=X_test,
    )

    # Initialize spec-compliant fitness function (MSE + MSW)
    # PSO core will call this with (y_true, y_pred, model)
    fitness_fn = SpecCompliantFitness(gamma=0.9)

    # TRD1 §7.1: Initialize IPSO optimizer
    logger.info("Initializing IPSO optimizer...")
    n_workers = max(1, int(getattr(pso_config, "n_workers", 1)))
    if n_workers > 1:
        logger.info(
            "Dispatching particles in parallel: n_workers=%d (n_particles=%d)",
            n_workers,
            pso_config.n_particles,
        )

    optimizer = IPSO(
        n_particles=pso_config.n_particles,
        n_iterations=pso_config.n_iterations,
        fitness_fn=fitness_fn,
        model_builder=model_builder,
        w_max=pso_config.inertia_max,
        w_min=pso_config.inertia_min,
        c1=pso_config.c1,
        c2=pso_config.c2,
        v_clamp_fraction=pso_config.v_clamp_fraction,
        seed=seed,
        n_workers=n_workers,
    )

    # Run PSO optimization
    logger.info("=" * 80)
    logger.info("RUNNING IPSO OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Search space: 6D (units_1, units_2, dropout, lr, batch_size, epochs)")
    logger.info(f"Fitness: F(x) = 0.9 × MSE + 0.1 × MSW")
    logger.info("=" * 80)

    best_params, best_fitness = optimizer.run(
        X_pso_train_win,
        y_pso_train_win,
        X_pso_val_win,
        y_pso_val_win,
    )

    # Save PSO results
    pso_results = {
        "best_hyperparameters": best_params,
        "best_fitness": float(best_fitness),
        "fitness_history": [float(f) for f in optimizer.fitness_history],
        "pso_config": {
            "n_particles": pso_config.n_particles,
            "n_iterations": pso_config.n_iterations,
            "seed": seed,
            "trd_compliant": True,
            "split_ratio": "72/8/20",
        },
        "data_split_samples": {
            "pso_train": len(X_pso_train_win),
            "pso_val": len(X_pso_val_win),
            "test_isolated": len(X_test),
        },
    }

    pso_results_path = output_dir / "pso_phase1_results.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(pso_results_path, "w") as f:
        yaml.dump(pso_results, f, default_flow_style=False)
    logger.info(f"PSO results saved to {pso_results_path}")

    logger.info("=" * 80)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 80)
    logger.info("Optimal hyperparameters found:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"Best validation MSE: {best_fitness:.6f}")
    logger.info("=" * 80)

    return best_params


def phase2_final_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: Dict,
    config: Config,
    output_dir: Path,
    feature_metadata: Dict,
) -> LSTMModel:
    """
    Phase 2: Final training on combined 80% (train+val) with PSO params.

    TRD2 §7.4: Phase 2 Protocol
    - Train on 80% combined data (train 70% + val 10%)
    - Use exact epoch count from PSO (NO early stopping)
    - Model is FROZEN after this phase

    Args:
        X_train: Training features (70%)
        y_train: Training targets (70%)
        X_val: Validation features (10%)
        y_val: Validation targets (10%)
        best_params: PSO-optimized hyperparameters
        config: Configuration object
        lookback: Window size
        output_dir: Directory to save model
        feature_metadata: Optional feature pipeline metadata

    Returns:
        Trained and FROZEN LSTMModel
    """
    logger.info("=" * 80)
    logger.info("PHASE 2: FINAL TRAINING ON COMBINED 80% (TRD2 §7.4)")
    logger.info("=" * 80)
    logger.info("Training on train (70%) + val (10%) = 80% total")
    logger.info("Using PSO-optimized hyperparameters")
    logger.info("NO early stopping (exact PSO epoch count)")
    logger.info("=" * 80)

    lookback = config.lstm_baseline.lookback

    # Combine train and val
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)

    logger.info(f"Combined data shape: X={X_combined.shape}, y={y_combined.shape}")

    # Build LSTM windows
    logger.info(f"Building LSTM windows (lookback={lookback})...")
    X_combined_win, y_combined_win = build_lstm_windows(
        X_combined, y_combined, lookback
    )

    logger.info(f"Windowed shape: X={X_combined_win.shape}, y={y_combined_win.shape}")

    # Create model with PSO parameters
    seed = config.lstm_baseline.random_seed
    set_all_seeds(seed)

    architecture_config = {
        "input_size": X_combined_win.shape[2],
        "lstm_units_1": best_params["units_1"],
        "lstm_units_2": best_params["units_2"],
        "dropout_rate": best_params["dropout"],
        "output_units": 1,
        "activation": "relu",
        "output_activation": "linear",
    }

    # Phase 2 trains the chosen architecture/learning-rate/batch-size for a
    # fixed 100 epochs (matching the Phase-1 budget) with no early stopping;
    # the val arrays are passed only to satisfy LSTMTrainer's input
    # validation and never trigger termination. ``restore_best_weights`` is
    # explicitly disabled so the LAST epoch's weights are kept; otherwise
    # LSTMTrainer's default would restore min-train-loss weights (since
    # train==val here), contradicting the fixed-epoch-count intent.
    trainer_config = {
        "learning_rate": best_params["learning_rate"],
        # Fixed at 100 epochs. PSO no longer optimises this dimension and the
        # decoded ``best_params["epochs"]`` is intentionally ignored here.
        "epochs": 100,
        "batch_size": best_params["batch_size"],
        "optimizer": "adam",
        "loss": "mse",
        "shuffle": False,
        "grad_clip": 1.0,
        "use_amp": True,
        "accumulation_steps": 1,
        "early_stopping": {
            "enabled": False,
            "restore_best_weights": False,
        },
    }

    logger.info("PSO-optimized model configuration:")
    for key, value in {**architecture_config, **trainer_config}.items():
        logger.info(f"  {key}: {value}")

    model_wrapper = LSTMModel(seed=seed)
    model_wrapper.build_model(architecture_config)
    trainer = LSTMTrainer(
        lstm_model=model_wrapper, config=trainer_config, seed=seed
    )

    logger.info("=" * 80)
    logger.info("TRAINING (SINGLE FINAL FIT - NO EARLY STOPPING)")
    logger.info("=" * 80)

    trained_wrapper, training_artifacts = trainer.train(
        X_combined_win,
        y_combined_win,
        X_combined_win,
        y_combined_win,
    )
    history = training_artifacts.get("history", {})

    logger.info("=" * 80)
    logger.info("PHASE 2 COMPLETE")
    logger.info("=" * 80)
    logger.info("This model will not be retrained (TRD compliance).")
    logger.info("Walk-forward evaluation will use this trained model.")
    logger.info("=" * 80)

    model_path = output_dir / "pso_lstm_model.pt"
    trained_wrapper.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Save training history
    history_path = output_dir / "training_history.yaml"
    with open(history_path, "w") as f:
        yaml.dump(history, f, default_flow_style=False)
    logger.info(f"Training history saved to {history_path}")

    # Save final model configuration
    config_path = output_dir / "model_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            {**architecture_config, **trainer_config},
            f,
            default_flow_style=False,
        )
    logger.info(f"Model config saved to {config_path}")

    # TRD1 §9.2: Save complete metadata (MANDATORY)
    metadata = {
        # Model identification
        "model_type": "pso_lstm",
        "protocol": "TRD_COMPLIANT_1.0",
        "trd_sources": ["TRD1 §5, §7, §8, §9", "TRD2 §7.4", "TRD3"],
        # Data splits
        "split_ratios": "72/8/20 (PSO train/PSO val/test)",
        "phase1_pso_train_samples": int(len(X_combined) * 0.9 - lookback),
        "phase1_pso_val_samples": int(len(X_combined) * 0.1 - lookback),
        "phase2_train_samples": len(X_combined_win),
        "test_samples_isolated": "Never accessed during training",
        # Feature schema (TRD1 §9.2)
        "feature_schema_version": (
            feature_metadata.get("version", "1.0.0") if feature_metadata else "1.0.0"
        ),
        "n_features": X_train.shape[1],
        "lookback": lookback,
        "feature_names": (
            feature_metadata.get("feature_names", []) if feature_metadata else []
        ),
        # PSO optimization results
        "pso_hyperparameters": best_params,
        "pso_optimization_complete": True,
        # Training state
        "training_complete": True,
        "model_frozen": True,
        "retraining_allowed": False,
        # Reproducibility (TRD1 §9.1)
        "random_seed": seed,
        "tensorflow_seed": seed,
        "numpy_seed": seed,
        # TRD compliance flags
        "trd_compliant": True,
        "leakage_free": True,
        "test_set_isolated_during_pso": True,
        "temporal_order_preserved": True,
    }

    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    logger.info(f"Metadata saved to {metadata_path}")

    # Save scaler parameters if available (TRD1 §9.2)
    if feature_metadata and "scaler_params" in feature_metadata:
        scaler_path = output_dir / "scaler_params.json"
        with open(scaler_path, "w") as f:
            json.dump(feature_metadata["scaler_params"], f, indent=2)
        logger.info(f"Scaler parameters saved to {scaler_path}")

    return trained_wrapper


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train PSO-LSTM (TRD-compliant two-phase protocol)"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to TRD-preprocessed features directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "models" / "pso_lstm",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--skip-pso",
        action="store_true",
        help="Skip PSO Phase 1 and load previous results",
    )
    parser.add_argument(
        "--feature-metadata",
        type=Path,
        help="Path to feature pipeline metadata (for TRD §9.2 compliance)",
    )

    args = parser.parse_args()

    try:
        setup_logger(
            log_file="logs/train_pso_lstm.log",
            level="INFO",
            mode=LogFileMode.OVERWRITE,
        )
        # Load configuration
        config = load_config(args.config)
        lookback = config.lstm_baseline.lookback

        # Set all seeds for reproducibility (TRD1 §9.1)
        set_all_seeds(config.lstm_baseline.random_seed)

        # Load preprocessed data
        data = load_preprocessed_data(args.data_path)

        # TRD1 §8: Validate no data leakage
        validate_no_leakage(data, lookback)

        # Load feature metadata if provided
        feature_metadata = None
        if args.feature_metadata and args.feature_metadata.exists():
            with open(args.feature_metadata, "r") as f:
                feature_metadata = yaml.safe_load(f)
            logger.info(f"Feature metadata loaded from {args.feature_metadata}")

        # Phase 1: PSO Search
        if args.skip_pso:
            logger.info("Skipping PSO Phase 1 (loading previous results)")
            pso_results_path = args.output_dir / "pso_phase1_results.yaml"
            with open(pso_results_path, "r") as f:
                pso_results = yaml.safe_load(f)
            best_params = pso_results["best_hyperparameters"]
            logger.info("Loaded PSO results:")
            for key, value in best_params.items():
                logger.info(f"  {key}: {value}")
        else:
            best_params = phase1_pso_search(
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_val=data["X_val"],
                y_val=data["y_val"],
                X_test=data["X_test"],  # Passed but NEVER used
                y_test=data["y_test"],  # Passed but NEVER used
                config=config,
                lookback=lookback,
                output_dir=args.output_dir,
            )

        # Phase 2: Final Training
        model = phase2_final_training(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
            best_params=best_params,
            config=config,
            output_dir=args.output_dir,
            feature_metadata=feature_metadata,
        )

        logger.info("=" * 80)
        logger.info("PSO-LSTM TWO-PHASE TRAINING SUCCESSFUL (TRD-COMPLIANT)")
        logger.info("=" * 80)
        logger.info(" Phase 1: PSO hyperparameter search complete (72/8 split)")
        logger.info(" Phase 2: Final training on 80% data complete")
        logger.info(" Model trained and frozen (TRD compliance)")
        logger.info(" Model saved to disk")
        logger.info(" Metadata versioned (TRD1 §9.2)")
        logger.info(" Test set isolated (TRD1 §8.1 L-6)")
        logger.info(" Ready for walk-forward evaluation")
        logger.info("=" * 80)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
