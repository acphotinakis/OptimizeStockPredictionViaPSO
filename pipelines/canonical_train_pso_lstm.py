#!/usr/bin/env python3
"""
Canonical PSO-LSTM Two-Phase Training Script

Implements two-phase PSO-LSTM training as defined in FINAL_PLAN.md Section 4.2.

TWO-PHASE PROTOCOL:
Phase 1: PSO hyperparameter search on 70% train, validate on 10% val
Phase 2: Final training on combined 80% (train+val) with PSO-optimized parameters

CRITICAL RULES:
- Phase 1: IPSO optimization to find best hyperparameters
- Phase 2: Single final training on 80% data with PSO params
- NO shuffling (shuffle=False mandatory)
- Model is FROZEN after Phase 2
- NO retraining during walk-forward

Usage:
    python pipelines/canonical_train_pso_lstm.py \\
        --data-path data/processed/features_unified/AAPL \\
        --config config/canonical_config.yaml \\
        --output-dir results/canonical/models/pso_lstm

Author: System Architect
Version: CANONICAL 1.0
Source: FINAL_PLAN.md Section 4.2
"""

import argparse
import logging
import sys
from pathlib import Path

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
    set_seeds,
)
from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_preprocessed_data(data_path: Path) -> dict:
    """
    Load preprocessed features from canonical feature pipeline.
    
    Args:
        data_path: Directory containing X_train.npy, y_train.npy, etc.
    
    Returns:
        Dictionary with train/val/test splits
    """
    logger.info(f"Loading preprocessed data from {data_path}")
    
    data = {}
    for split in ["train", "val", "test"]:
        X_path = data_path / f"X_{split}.npy"
        y_path = data_path / f"y_{split}.npy"
        
        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Missing {split} data: {X_path} or {y_path}\n"
                f"Run canonical feature pipeline first."
            )
        
        data[f"X_{split}"] = np.load(X_path)
        data[f"y_{split}"] = np.load(y_path)
        
        logger.info(
            f"  {split}: X={data[f'X_{split}'].shape}, y={data[f'y_{split}'].shape}"
        )
    
    return data


def fitness_function(
    params: dict,
    X_train_win: np.ndarray,
    y_train_win: np.ndarray,
    X_val_win: np.ndarray,
    y_val_win: np.ndarray,
    lookback: int,
    n_features: int,
    seed: int,
) -> float:
    """
    Fitness function for PSO: Train LSTM and return validation MSE.
    
    FINAL_PLAN.md Section 4.2: PSO Fitness Function
    
    Args:
        params: Hyperparameters proposed by PSO
        X_train_win: Training features (windowed)
        y_train_win: Training targets
        X_val_win: Validation features (windowed)
        y_val_win: Validation targets
        lookback: Window size
        n_features: Number of features
        seed: Random seed
    
    Returns:
        Fitness value (validation MSE)
    """
    # Create model config
    model_config = {
        "input_shape": (lookback, n_features),
        "lstm_units_1": params["units_1"],
        "lstm_units_2": params["units_2"],
        "dropout_rate": params["dropout"],
        "activation": "relu",
        "output_units": 1,
        "output_activation": "linear",
        "learning_rate": params["learning_rate"],
        "loss": "mse",
    }
    
    # Create and train model
    model = LSTMModel(model_config, seed=seed)
    trainer = LSTMTrainer(model_config, seed=seed)
    
    model, _ = trainer.train(
        X_train_win,
        y_train_win,
        X_val_win,
        y_val_win,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        patience=10,
        shuffle=False,
        verbose=0,
    )
    
    # Compute validation MSE
    y_pred = model.predict(X_val_win, verbose=0)
    mse = np.mean((y_val_win - y_pred.flatten()) ** 2)
    
    return float(mse)


def phase1_pso_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    lookback: int,
    output_dir: Path,
) -> dict:
    """
    Phase 1: PSO hyperparameter search.
    
    FINAL_PLAN.md Section 4.2: PSO Phase 1
    
    Search on 70% train, validate on 10% val.
    
    Args:
        X_train: Training features (70%)
        y_train: Training targets (70%)
        X_val: Validation features (10%)
        y_val: Validation targets (10%)
        config: Configuration dict
        lookback: Window size
        output_dir: Directory to save results
    
    Returns:
        Dictionary of best hyperparameters
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: PSO HYPERPARAMETER SEARCH")
    logger.info("=" * 80)
    logger.info("Searching on 70% train, validating on 10% val")
    logger.info("=" * 80)
    
    # Extract PSO config
    pso_config = config.pso_lstm.pso
    seed = pso_config.random_seed
    
    # Build LSTM windows
    logger.info(f"Building LSTM windows (lookback={lookback})...")
    X_train_win, y_train_win = build_lstm_windows(X_train, y_train, lookback)
    X_val_win, y_val_win = build_lstm_windows(X_val, y_val, lookback)
    
    logger.info(f"Windowed shapes:")
    logger.info(f"  X_train: {X_train_win.shape}, y_train: {y_train_win.shape}")
    logger.info(f"  X_val:   {X_val_win.shape}, y_val:   {y_val_win.shape}")
    
    # Define search space
    search_space = {
        "epochs": {
            "min": pso_config.search_space.epochs.min,
            "max": pso_config.search_space.epochs.max,
        },
        "units_1": {
            "min": pso_config.search_space.units_1.min,
            "max": pso_config.search_space.units_1.max,
        },
        "units_2": {
            "min": pso_config.search_space.units_2.min,
            "max": pso_config.search_space.units_2.max,
        },
        "learning_rate": {
            "min": pso_config.search_space.learning_rate.min,
            "max": pso_config.search_space.learning_rate.max,
            "scale": pso_config.search_space.learning_rate.scale,
        },
        "dropout": {
            "min": pso_config.search_space.dropout.min,
            "max": pso_config.search_space.dropout.max,
        },
        "batch_size": {
            "choices": pso_config.search_space.batch_size.choices,
        },
    }
    
    # Define fitness function wrapper
    def fitness_wrapper(params):
        return fitness_function(
            params,
            X_train_win,
            y_train_win,
            X_val_win,
            y_val_win,
            lookback,
            X_train.shape[1],
            seed,
        )
    
    # Initialize IPSO
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
    best_params, best_fitness = optimizer.optimize()
    
    # Save PSO results
    pso_results = {
        "best_hyperparameters": best_params,
        "best_fitness": float(best_fitness),
        "fitness_history": optimizer.fitness_history,
        "pso_config": {
            "n_particles": pso_config.n_particles,
            "n_iterations": pso_config.n_iterations,
            "seed": seed,
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
    best_params: dict,
    config: dict,
    lookback: int,
    output_dir: Path,
) -> LSTMModel:
    """
    Phase 2: Final training on combined 80% (train+val) with PSO params.
    
    FINAL_PLAN.md Section 4.2: PSO Phase 2
    
    Train on 80% combined data using exact epoch count from PSO.
    NO early stopping in final fit.
    
    Args:
        X_train: Training features (70%)
        y_train: Training targets (70%)
        X_val: Validation features (10%)
        y_val: Validation targets (10%)
        best_params: PSO-optimized hyperparameters
        config: Configuration dict
        lookback: Window size
        output_dir: Directory to save model
    
    Returns:
        Trained and FROZEN LSTMModel
    """
    logger.info("=" * 80)
    logger.info("PHASE 2: FINAL TRAINING ON COMBINED 80%")
    logger.info("=" * 80)
    logger.info("Training on COMBINED train (70%) + val (10%) = 80% total")
    logger.info("Using PSO-optimized hyperparameters")
    logger.info("NO early stopping (using exact PSO epoch count)")
    logger.info("=" * 80)
    
    # Combine train and val
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    
    logger.info(f"Combined data shape: X={X_combined.shape}, y={y_combined.shape}")
    
    # Build LSTM windows
    logger.info(f"Building LSTM windows (lookback={lookback})...")
    X_combined_win, y_combined_win = build_lstm_windows(X_combined, y_combined, lookback)
    
    logger.info(f"Windowed shape: X={X_combined_win.shape}, y={y_combined_win.shape}")
    
    # Create model with PSO parameters
    seed = config.pso_lstm.random_seed
    set_seeds(seed)
    
    model_config = {
        "input_shape": (lookback, X_train.shape[1]),
        "lstm_units_1": best_params["units_1"],
        "lstm_units_2": best_params["units_2"],
        "dropout_rate": best_params["dropout"],
        "activation": config.pso_lstm.activation,
        "output_units": config.pso_lstm.output_units,
        "output_activation": config.pso_lstm.output_activation,
        "learning_rate": best_params["learning_rate"],
        "loss": config.pso_lstm.loss,
    }
    
    logger.info("PSO-optimized model configuration:")
    for key, value in model_config.items():
        logger.info(f"  {key}: {value}")
    
    # Create model
    model = LSTMModel(model_config, seed=seed)
    trainer = LSTMTrainer(model_config, seed=seed)
    
    # Train with exact PSO epochs (NO early stopping)
    logger.info("=" * 80)
    logger.info("TRAINING (SINGLE FINAL FIT - NO EARLY STOPPING)")
    logger.info("=" * 80)
    
    model, history = trainer.train(
        X_combined_win,
        y_combined_win,
        X_combined_win,  # Use training data for validation too (monitoring only)
        y_combined_win,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        patience=None,  # NO early stopping
        shuffle=False,  # MUST be False
        verbose=1,
    )
    
    logger.info("=" * 80)
    logger.info("PHASE 2 COMPLETE - MODEL NOW FROZEN")
    logger.info("=" * 80)
    logger.info("⚠️  This model will NEVER be retrained")
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
    
    # Save metadata
    metadata = {
        "model_type": "pso_lstm",
        "protocol": "CANONICAL_1.0",
        "source": "FINAL_PLAN.md Section 4.2",
        "phase1_samples": len(X_train),
        "phase2_samples": len(X_combined_win),
        "features": X_train.shape[1],
        "lookback": lookback,
        "pso_hyperparameters": best_params,
        "training_complete": True,
        "frozen": True,
        "retraining_allowed": False,
    }
    
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    logger.info(f"Metadata saved to {metadata_path}")
    
    return model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train PSO-LSTM (two-phase protocol)"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to preprocessed features directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "canonical_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "canonical" / "models" / "pso_lstm",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--skip-pso",
        action="store_true",
        help="Skip PSO Phase 1 and load previous results",
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        lookback = config.features.windowing.lookback
        
        # Load preprocessed data
        data = load_preprocessed_data(args.data_path)
        
        # Phase 1: PSO Search
        if args.skip_pso:
            logger.info("Skipping PSO Phase 1 (loading previous results)")
            pso_results_path = args.output_dir / "pso_phase1_results.yaml"
            with open(pso_results_path, "r") as f:
                pso_results = yaml.safe_load(f)
            best_params = pso_results["best_hyperparameters"]
        else:
            best_params = phase1_pso_search(
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_val=data["X_val"],
                y_val=data["y_val"],
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
        )
        
        logger.info("=" * 80)
        logger.info("PSO-LSTM TWO-PHASE TRAINING SUCCESSFUL")
        logger.info("=" * 80)
        logger.info("✓ Phase 1: PSO hyperparameter search complete")
        logger.info("✓ Phase 2: Final training on 80% data complete")
        logger.info("✓ Model trained and frozen")
        logger.info("✓ Model saved to disk")
        logger.info("✓ Ready for walk-forward evaluation")
        logger.info("=" * 80)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
