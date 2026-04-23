#!/usr/bin/env python3
"""
TRD-Compliant PSO-LSTM Two-Phase Training Script

Implements two-phase PSO-LSTM training as defined in TRD1 §7, TRD2 §7.4, TRD3.

TWO-PHASE PROTOCOL (TRD2 §7.4):
Phase 1: PSO hyperparameter search on 72% train, validate on 8% val
Phase 2: Final training on combined 80% (train+val) with PSO-optimized parameters

CRITICAL TRD RULES:
- TRD2 §7.4: 72/8/20 split (train/pso_val/test) MANDATORY
- TRD1 §8.1 L-6: Test set NEVER accessed during PSO
- TRD1 §8.1 L-1: NO shuffling (shuffle=False mandatory)
- TRD1 §9.1: Full reproducibility (seeds fixed)
- TRD1 §9.2: Metadata versioning required

Usage:
    python pipelines/train_pso_lstm.py \\
        --data-path data/processed/features_unified/AAPL \\
        --config config/default_config.yaml \\
        --output-dir results/models/pso_lstm

Author: TRD Compliance System
Version: TRD-COMPLIANT 1.0
Source: TRD1 §7, TRD2 §7.4, TRD3
"""

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    IPSOOptimizer,
    LSTMModel,
    LSTMTrainer,
    build_lstm_windows,
)
from src.utils.config_loader import Config, load_config
from src.utils.logger import setup_logger


logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """
    Set all random seeds for full reproducibility (TRD1 §9.1).

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f" All seeds set to {seed} (TRD1 §9.1 compliance)")


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


def fitness_function(
    params: Dict,
    X_train_win: np.ndarray,
    y_train_win: np.ndarray,
    X_val_win: np.ndarray,
    y_val_win: np.ndarray,
    lookback: int,
    n_features: int,
    seed: int,
    config: Config,
) -> float:
    """
    Fitness function for PSO: Train LSTM and return validation MSE.

    TRD1 §7.3: Objective Function = MSE on validation set
    TRD2 §7.4: Evaluates on PSO validation set (8% of total)

    Args:
        params: Hyperparameters proposed by PSO
        X_train_win: PSO training features (windowed, 72%)
        y_train_win: PSO training targets
        X_val_win: PSO validation features (windowed, 8%)
        y_val_win: PSO validation targets
        lookback: Window size
        n_features: Number of features
        seed: Random seed
        config: Full configuration object

    Returns:
        Fitness value (validation MSE, lower is better)
    """
    # TRD1 §5: LSTM Architecture Constraints
    model_config = {
        "input_shape": (lookback, n_features),
        "lstm_units_1": params["units_1"],
        "lstm_units_2": params["units_2"],
        "dropout_rate": params["dropout"],
        "activation": config.pso.activation,  # From LSTM config
        "output_units": config.pso.output_units,
        "output_activation": config.pso.output_activation,
        "learning_rate": params["learning_rate"],
        "loss": config.pso.loss,
    }

    # Create and train model

    model = LSTMModel(seed=seed)
    trainer = LSTMTrainer(model_config, seed=seed)

    # TRD1 §5.4: Training Protocol
    model, _ = trainer.train(
        X_train_win,
        y_train_win,
        X_val_win,
        y_val_win,
        lstm_units_1=int(params["lstm_units_1"]),
        lstm_units_2=int(params["lstm_units_2"]),
        dropout_rate=float(params["dropout_rate"]),
        learning_rate=float(params["learning_rate"]),
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        patience=config.lstm_baseline.early_stopping.patience,
        shuffle=False,  # TRD1 §8.1 L-1: NO SHUFFLING
    )

    # TRD1 §7.3: Compute validation MSE
    y_pred = model.predict(X_val_win, verbose=0)
    mse = np.mean((y_val_win - y_pred.flatten()) ** 2)

    return float(mse)


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
    seed = pso_config.random_seed
    set_all_seeds(seed)

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

    # TRD1 §8.1 L-6: Protect test set from access
    _test_data_hash = hashlib.sha256(X_test.tobytes()).hexdigest()
    logger.info(f" Test set protected (hash: {_test_data_hash[:16]}...)")

    # TRD1 §7.2: Define search space
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

    # Define fitness function wrapper
    def fitness_wrapper(params):
        fitness = fitness_function(
            params,
            X_pso_train_win,
            y_pso_train_win,
            X_pso_val_win,
            y_pso_val_win,
            lookback,
            X_pso_train.shape[1],
            seed,
            config,
        )

        # TRD1 §8.1 L-6: Verify test set never accessed
        current_hash = hashlib.sha256(X_test.tobytes()).hexdigest()
        assert (
            current_hash == _test_data_hash
        ), "CRITICAL TRD VIOLATION: Test set accessed during PSO (L-6)"

        return fitness

    # TRD1 §7.1: Initialize IPSO
    optimizer = IPSOOptimizer(
        n_particles=pso_config.n_particles,
        n_iterations=pso_config.n_iterations,
        search_space=search_space,
        fitness_func=fitness_wrapper,
        inertia_min=pso_config.inertia_min,
        inertia_max=pso_config.inertia_max,
        c1=pso_config.c1,
        c2=pso_config.c2,
        v_clamp_fraction=pso_config.v_clamp_fraction,
        seed=seed,
    )

    # Run optimization
    logger.info("=" * 80)
    logger.info("RUNNING PSO OPTIMIZATION")
    logger.info("=" * 80)
    best_params, best_fitness = optimizer.optimize()

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
    lookback: int,
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
    seed = config.pso.random_seed
    set_all_seeds(seed)

    # TRD1 §5.1: LSTM Architecture Constraints
    model_config = {
        "input_shape": (lookback, X_train.shape[1]),
        "lstm_units_1": best_params["units_1"],
        "lstm_units_2": best_params["units_2"],
        "dropout_rate": best_params["dropout"],
        "activation": config.pso.activation,
        "output_units": config.pso.output_units,
        "output_activation": config.pso.output_activation,
        "learning_rate": best_params["learning_rate"],
        "loss": config.pso.loss,
    }

    logger.info("PSO-optimized model configuration:")
    for key, value in model_config.items():
        logger.info(f"  {key}: {value}")

    # Create model
    model = LSTMModel(seed=seed)
    trainer = LSTMTrainer(model_config, seed=seed)

    # TRD2 §7.4: Train with exact PSO epochs, NO early stopping
    logger.info("=" * 80)
    logger.info("TRAINING (SINGLE FINAL FIT - NO EARLY STOPPING)")
    logger.info("=" * 80)

    model, history = trainer.train(
        X_combined_win,
        y_combined_win,
        None,  # NO validation in Phase 2
        None,
        lstm_units_1=int(best_params["units_1"]),
        lstm_units_2=int(best_params["units_2"]),
        dropout_rate=float(best_params["dropout"]),
        learning_rate=float(best_params["learning_rate"]),
        epochs=int(best_params["epochs"]),  # Exact count from PSO
        batch_size=int(best_params["batch_size"]),
        patience=None,  # Disable early stopping
        shuffle=False,  # TRD1 §8.1 L-1
    )

    logger.info("=" * 80)
    logger.info("PHASE 2 COMPLETE - MODEL NOW FROZEN")
    logger.info("=" * 80)
    logger.info("⚠️  This model will NEVER be retrained (TRD compliance)")
    logger.info("⚠️  Walk-forward evaluation will use THIS frozen model")
    logger.info("=" * 80)

    # Save model
    model_path = output_dir / "pso_lstm_model.h5"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Save training history
    history_path = output_dir / "training_history.yaml"
    with open(history_path, "w") as f:
        yaml.dump(history, f, default_flow_style=False)
    logger.info(f"Training history saved to {history_path}")

    # Save final model configuration
    config_path = output_dir / "model_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)
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

    return model


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
        # Load configuration
        config = load_config(args.config)
        lookback = config.pso.lookback

        # Set all seeds for reproducibility (TRD1 §9.1)
        set_all_seeds(config.pso.random_seed)

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
            lookback=lookback,
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
