from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml
import uuid

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plots.lstm_baseline_plots import generate_all_plots
from src.models import LSTMModel, LSTMTrainer, set_seeds
from src.data.windowing import build_lstm_windows
from src.utils.config_loader import Config, load_config
from src.utils.logger import LogFileMode, setup_logger

logger = logging.getLogger(__name__)

MODEL_TYPE = "lstm_baseline"


def load_feature_data(data_path: Path) -> dict:
    """
    Load preprocessed features from canonical feature pipeline with full diagnostics.

    Args:
        data_path: Directory containing X_train.npy, y_train.npy, etc.
        config: Optional config dict for dataset/version metadata.

    Returns:
        Dictionary with train/val/test splits.
    """
    logger.info(f"Resolving dataset path: {data_path.resolve()}")

    splits = ["train", "val", "test"]
    data = {}

    total_samples = 0
    feature_dim = None

    logger.info(f"Expected splits: {splits}")

    for split in splits:
        X_path = data_path / f"X_{split}.npy"
        y_path = data_path / f"y_{split}.npy"

        # ----------------------------
        # File existence + metadata
        # ----------------------------
        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Missing {split} data: {X_path} or {y_path}\n"
                f"Run canonical feature pipeline first."
            )

        x_size_mb = X_path.stat().st_size / 1e6
        y_size_mb = y_path.stat().st_size / 1e6

        logger.info(f"[{split}] Loading files:")
        logger.info(f"  X: {X_path} ({x_size_mb:.2f} MB)")
        logger.info(f"  y: {y_path} ({y_size_mb:.2f} MB)")

        # ----------------------------
        # Load data
        # ----------------------------
        X = np.load(X_path)
        y = np.load(y_path)

        data[f"X_{split}"] = X
        data[f"y_{split}"] = y

    # ----------------------------
    # Global dataset summary
    # ----------------------------
    logger.info("==== Dataset Summary ====")
    logger.info(f"Total samples across splits: {total_samples}")
    logger.info(f"Feature dimension: {feature_dim}")
    logger.info(f"Splits loaded successfully: {list(data.keys())}")

    return data


def load_test_data(data_path: Path, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Load test features and build LSTM windows."""
    X_path = data_path / "X_test.npy"
    y_path = data_path / "y_test.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing test data: {X_path} or {y_path}")

    X_test = np.load(X_path)
    y_test = np.load(y_path)

    logger.info(f"[test] Raw shapes -> X: {X_test.shape}, y: {y_test.shape}")

    X_test_win, y_test_win = build_lstm_windows(X_test, y_test, lookback)
    logger.info(
        f"[test] Windowed shapes -> X: {X_test_win.shape}, y: {y_test_win.shape}"
    )

    if np.isnan(X_test_win).any() or np.isnan(y_test_win).any():
        raise ValueError("Test data contains NaN values")

    return X_test_win, y_test_win


def load_trained_model(
    model_dir: Path, config: Config, device: Optional[str] = None
) -> LSTMModel:
    """Reconstruct LSTM architecture and load frozen weights."""
    model_path = model_dir / "baseline_lstm_model.pt"
    config_path = model_dir / "model_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path, "r") as f:
        model_config = json.load(f)

    seed = config.lstm_baseline.random_seed
    set_seeds(seed)

    lstm_wrapper = LSTMModel(seed=seed, device=device)
    lstm_wrapper.build_model(model_config)
    lstm_wrapper.load(str(model_path))

    # Explicit freeze (defense in depth)
    if lstm_wrapper.model is not None:
        lstm_wrapper.model.eval()
        for param in lstm_wrapper.model.parameters():
            param.requires_grad = False
        logger.info("Model explicitly frozen for inference")

    return lstm_wrapper


def save_results(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    model_dir: Path,
    prefix: str = "test",
) -> None:
    """Persist predictions, metrics, and provenance metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{prefix}_predictions.npy", y_pred)
    np.save(output_dir / f"{prefix}_ground_truth.npy", y_true)

    metrics_path = output_dir / f"{prefix}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    results = {
        "predictions": y_pred.tolist(),
        "ground_truth": y_true.tolist(),
        "residuals": (y_true - y_pred).tolist(),
        "metrics": metrics,
        "provenance": {
            "model_dir": str(model_dir.resolve()),
            "model_type": MODEL_TYPE,
            "protocol": "CANONICAL_1.0",
        },
    }
    with open(output_dir / f"{prefix}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {output_dir}")


def build_experiment_dir(ticker: str, timeframe: str, run_id: str) -> Path:
    return (
        PROJECT_ROOT
        / "results"
        / "experiments"
        / f"{ticker}_{timeframe}_{MODEL_TYPE}_{run_id}"
    )


def build_experiment_dirs(base: Path) -> dict:
    return {
        "root": base,
        "train": base / "train",
        "val": base / "val",
        "test": base / "test",
        "model": base / "model",
        "logs": base / "logs",
        "plots": base / "plots",
    }


def run_training_for_ticker(
    ticker: str, data_path: Path, timeframe: str, run_id: str, config: Config
):
    logger.info("=" * 80)
    logger.info(f"Training ticker: {ticker}")
    logger.info("=" * 80)
    data = load_feature_data(data_path)

    experiment_dir = build_experiment_dir(
        ticker=ticker,
        timeframe=timeframe,
        run_id=run_id,
    )

    output_dirs = build_experiment_dirs(experiment_dir)

    log_dir = output_dirs["logs"]
    logger.info(f"Log Directory --> {log_dir}")
    setup_logger(
        log_file=f"{log_dir}/{ticker}_train.log",
        level="INFO",
        mode=LogFileMode.OVERWRITE,
    )

    train_dir = output_dirs["train"]
    logger.info(f"Train Directory --> {train_dir}")
    train_baseline_lstm(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        config=config,
        output_dir=train_dir,
    )


def train_baseline_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
    output_dir: Path,
) -> None:
    """
    Train Baseline LSTM with fixed hyperparameters.

    FINAL_PLAN.md Section 4.1: Baseline LSTM Training Phase

    CRITICAL: This function executes EXACTLY ONCE. The model is then FROZEN.

    Args:
        X_train: Training features (N, F)
        y_train: Training targets (N,)
        X_val: Validation features (M, F)
        y_val: Validation targets (M,)
        config: Configuration dict
        output_dir: Directory to save model

    Returns:
        Trained and FROZEN LSTMModel
    """
    logger.info("=" * 80)
    logger.info("BASELINE LSTM TRAINING (FINAL_PLAN.md Section 4.1)")
    logger.info("=" * 80)
    logger.info("CRITICAL: Model will be trained EXACTLY ONCE and then FROZEN")
    logger.info("=" * 80)

    # Extract config
    lstm_config = config.lstm_baseline
    lookback = config.lstm_baseline.lookback
    seed = config.lstm_baseline.random_seed

    # Set seeds for reproducibility
    set_seeds(seed)
    logger.info(f"Random seed: {seed}")

    # Build LSTM windows
    logger.info(f"Building LSTM windows (lookback={lookback})...")
    X_train_win, y_train_win = build_lstm_windows(X_train, y_train, lookback)
    X_val_win, y_val_win = build_lstm_windows(X_val, y_val, lookback)

    logger.info(f"Windowed shapes:")
    logger.info(f"  X_train: {X_train_win.shape}, y_train: {y_train_win.shape}")
    logger.info(f"  X_val:   {X_val_win.shape}, y_val:   {y_val_win.shape}")

    # Model configuration dict (for metadata persistence)
    model_config = {
        "input_size": X_train_win.shape[2],
        "lstm_units_1": lstm_config.lstm_units_1,
        "lstm_units_2": lstm_config.lstm_units_2,
        "dropout_rate": lstm_config.dropout_rate,
        "output_units": lstm_config.output_units,
        "activation": lstm_config.activation,
        "output_activation": lstm_config.output_activation,
    }

    # Create model
    lstm_model_wrapper = LSTMModel(seed=seed)
    lstm_network, lstm_model = lstm_model_wrapper.build_model(model_config)

    logger.info("========== LSTM MODEL BUILT ==========")
    logger.info(lstm_model)
    logger.info("======================================")

    # Build unified trainer config dict — ALL hyperparameters from YAML
    trainer_config = {
        # Training
        "optimizer": lstm_config.optimizer,
        "learning_rate": lstm_config.learning_rate,
        "loss": lstm_config.loss,
        "epochs": lstm_config.epochs,
        "batch_size": lstm_config.batch_size,
        "shuffle": lstm_config.shuffle,
        # Advanced training
        "grad_clip": lstm_config.grad_clip,
        "use_amp": lstm_config.use_amp,
        "accumulation_steps": lstm_config.accumulation_steps,
        # Early stopping
        "early_stopping": {
            "enabled": lstm_config.early_stopping.enabled,
            "monitor": lstm_config.early_stopping.monitor,
            "patience": lstm_config.early_stopping.patience,
            "restore_best_weights": lstm_config.early_stopping.restore_best_weights,
        },
    }

    # create trainer and run
    trainer = LSTMTrainer(lstm_model=lstm_model, config=trainer_config, seed=seed)

    logger.info("========== LSTM TRAINER BUILT ==========")
    logger.info(trainer)
    logger.info("======================================")

    # Train with early stopping
    logger.info("=" * 80)
    logger.info("TRAINING (SINGLE FIT - NO RETRAINING)")
    logger.info("=" * 80)

    model, history = trainer.train(
        X_train_win,
        y_train_win,
        X_val_win,
        y_val_win,
    )

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE - MODEL NOW FROZEN")
    logger.info("=" * 80)

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "baseline_lstm_model.pt"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    logger.info(f"Training history saved to {history_path}")

    # Save model configuration
    config_path = output_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=4)
    logger.info(f"Model config saved to {config_path}")

    # Save metadata
    metadata = {
        "model_type": MODEL_TYPE,
        "protocol": "CANONICAL_1.0",
        "source": "FINAL_PLAN.md Section 4.1",
        "training_samples": len(X_train_win),
        "validation_samples": len(X_val_win),
        "features": X_train.shape[1],
        "lookback": lookback,
        "hyperparameters": {
            "lstm_units_1": lstm_config.lstm_units_1,
            "lstm_units_2": lstm_config.lstm_units_2,
            "dropout_rate": lstm_config.dropout_rate,
            "learning_rate": lstm_config.learning_rate,
            "batch_size": lstm_config.batch_size,
            "epochs": lstm_config.epochs,
        },
        "training_complete": True,
        "frozen": True,
        "retraining_allowed": False,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved to {metadata_path}")

    # return model


def test_baseline_lstm(
    ticker: str,
    model_dir: Path,
    data_path: Path,
    config: Config,
    output_dir: Path,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    End-to-end test pipeline using LSTMModel.evaluate() as the canonical path.
    """
    logger.info("=" * 80)
    logger.info(f"BASELINE LSTM INFERENCE: {ticker}")
    logger.info("=" * 80)

    lookback = config.lstm_baseline.lookback
    X_test, y_test = load_test_data(data_path, lookback)

    model = load_trained_model(model_dir, config, device=device)

    logger.info("Running canonical evaluation via model.evaluate()...")
    metrics, y_pred = model.evaluate(X_test, y_test)

    save_results(output_dir, y_test, y_pred, metrics, model_dir)

    logger.info("=" * 80)
    logger.info("TESTING COMPLETE")
    logger.info("=" * 80)

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config/default_config.yaml"
    )

    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--timeframe", type=str, required=True)

    parser.add_argument("--data-path", type=Path, required=True)

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--run-id", type=str, default=None)

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")

    args = parser.parse_args()

    logger.info(json.dumps(vars(args), indent=4, default=str))

    config = load_config(args.config)

    run_id = args.run_id or str(uuid.uuid4())

    experiment_dir = build_experiment_dir(
        ticker=args.ticker,
        timeframe=args.timeframe,
        run_id=run_id,
    )

    dirs = build_experiment_dirs(experiment_dir)

    setup_logger(
        log_file=f"{dirs['logs']}/{args.ticker}.log",
        level="INFO",
        mode=LogFileMode.OVERWRITE,
    )

    logger.info(f"RUN ID: {run_id}")
    logger.info(f"EXPERIMENT: {experiment_dir}")

    # ==========================================================
    # STEP 1: TRAIN
    # ==========================================================
    if not args.skip_train:
        logger.info("STEP 1: TRAINING")

        run_training_for_ticker(
            ticker=args.ticker,
            data_path=args.data_path,
            timeframe=args.timeframe,
            run_id=run_id,
            config=config,
        )

    # ==========================================================
    # STEP 2: TEST
    # ==========================================================
    if not args.skip_test:
        logger.info("STEP 2: TESTING")

        model_dir = dirs["train"]
        test_metrics = test_baseline_lstm(
            ticker=args.ticker,
            model_dir=model_dir,
            data_path=args.data_path,
            config=config,
            output_dir=dirs["test"],
            device=args.device,
        )

        logger.info(f"TEST METRICS: {test_metrics}")

    # ==========================================================
    # STEP 3: PLOT
    # ==========================================================
    if not args.skip_plot:
        logger.info("STEP 3: PLOTTING")

        generate_all_plots(
            experiment_dir,
            data_path=dirs["train"] / "training_history.json",
            save_dir=dirs["plots"],
        )

    logger.info("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
