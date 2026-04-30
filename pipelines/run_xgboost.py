import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional
import uuid

import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plots.xgboost_plots import plot_full_xgboost_training_report
from src.models.xgboost_model import XGBoostModel
from src.models.xgboost_trainer import XGBoostTrainer
from src.data.windowing import build_xgboost_lag_features
from src.utils.config_loader import Config, load_config
from src.utils.logger import LogFileMode, setup_logger
from src.models import set_seeds

# Configure logging

logger = logging.getLogger(__name__)

MODEL_TYPE = "xgboost"


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

        total_samples += len(X)
        feature_dim = X.shape[1]

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

    X_test_win, y_test_win = build_xgboost_lag_features(
        X_test, y_test, lookback=lookback
    )
    logger.info(
        f"[test] Windowed shapes -> X: {X_test_win.shape}, y: {y_test_win.shape}"
    )

    if np.isnan(X_test_win).any() or np.isnan(y_test_win).any():
        raise ValueError("Test data contains NaN values")

    return X_test_win, y_test_win


def save_results(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    model_dir: Path,
) -> None:
    """Persist predictions, metrics, and provenance metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "test_predictions.npy", y_pred)
    np.save(output_dir / "test_ground_truth.npy", y_true)

    metrics_path = output_dir / "test_metrics.json"
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
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {output_dir}")


def load_trained_model(model_dir: Path, config: Config) -> XGBoostModel:
    model_path = model_dir / "xgboost_model.json"
    config_path = model_dir / "model_config.json"
    feature_importance_json_path = model_dir / "feature_importance.json"
    feature_names_path = model_dir / "feature_names.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

    with open(feature_importance_json_path, "r") as f:
        feature_importance_json = json.load(f)

    set_seeds(config.xgboost.random_state)

    model = XGBoostModel.load(
        path=model_path,
        feature_names=feature_names,
        feature_importance=feature_importance_json,
    )

    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
    output_dir: Path,
) -> None:
    """
    Train XGBoost with fixed hyperparameters.

    FINAL_PLAN.md Section 4.3: XGBoost Training Phase

    CRITICAL: This function executes EXACTLY ONCE. The model is then FROZEN.

    Args:
        X_train: Training features (N, F) tabular
        y_train: Training targets (N,)
        X_val: Validation features (M, F) tabular
        y_val: Validation targets (M,)
        config: Configuration dict
        output_dir: Directory to save model

    Returns:
        Trained and FROZEN XGBoostModel
    """
    logger.info("=" * 80)
    logger.info("XGBOOST TRAINING (FINAL_PLAN.md Section 4.3)")
    logger.info("=" * 80)
    logger.info("CRITICAL: Model will be trained EXACTLY ONCE and then FROZEN")
    logger.info("=" * 80)

    # Extract config
    xgb_config = config.xgboost
    lookback = xgb_config.lookback
    seed = xgb_config.random_state

    logger.info(f"Random seed: {seed}")
    logger.info(f"Lookback: {lookback} (lag-based features)")

    # Build lag-based features
    logger.info("=" * 80)
    logger.info("BUILDING LAG-BASED FEATURES")
    logger.info("=" * 80)
    logger.info("Using LAG-BASED representation (NOT flattened sequences)")
    logger.info(f"Each sample: [X[t], X[t-1], ..., X[t-{lookback}]]")

    X_train_lag, y_train_lag = build_xgboost_lag_features(
        X_train, y_train, lookback=lookback
    )
    X_val_lag, y_val_lag = build_xgboost_lag_features(X_val, y_val, lookback=lookback)

    logger.info(f"Lag-based feature shapes:")
    logger.info(f"  X_train: {X_train_lag.shape}, y_train: {y_train_lag.shape}")
    logger.info(f"  X_val:   {X_val_lag.shape}, y_val:   {y_val_lag.shape}")
    logger.info(f"  Features per sample: {X_train_lag.shape[1]}")

    xgb_params = {
        "objective": xgb_config.objective,
        "max_depth": xgb_config.max_depth,
        "learning_rate": xgb_config.learning_rate,
        "n_estimators": xgb_config.n_estimators,
        "subsample": xgb_config.subsample,
        "colsample_bytree": xgb_config.colsample_bytree,
        "min_child_weight": xgb_config.min_child_weight,
        "gamma": xgb_config.gamma,
        "reg_alpha": xgb_config.reg_alpha,
        "reg_lambda": xgb_config.reg_lambda,
        "tree_method": xgb_config.tree_method,
        "max_bin": xgb_config.max_bin,
        "n_jobs": xgb_config.n_jobs,
        "importance_type": xgb_config.importance_type,
        "verbosity": xgb_config.verbosity,
        "random_state": xgb_config.random_state,
    }

    early_stopping_rounds = xgb_config.early_stopping_rounds
    feature_type = xgb_config.feature_type

    trainer = XGBoostTrainer(
        xgboost_config=xgb_params,
        early_stopping_rounds=early_stopping_rounds,
    )

    feature_names = [f"f{i}" for i in range(X_train_lag.shape[1])]

    feature_names_path = output_dir / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=4)

    logger.info(feature_names)

    model, history = trainer.train(
        X_train_lag, y_train_lag, X_val_lag, y_val_lag, feature_names
    )

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgboost_model.json"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for key, value in history.items():
            if isinstance(value, dict):
                history_serializable[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                history_serializable[key] = value
        json.dump(history_serializable, f, indent=4)
    logger.info(f"Training history saved to {history_path}")

    # Save model configuration
    config_dict = {
        "xgb_params": xgb_params,
        "lookback": lookback,
        "feature_type": feature_type,
    }
    config_path = output_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    logger.info(f"Model config saved to {config_path}")

    # Save feature importance
    if model.feature_importance:
        importance_path = output_dir / "feature_importance.json"
        sorted_importance = {
            str(k): float(v)
            for k, v in sorted(model.feature_importance.items(), key=lambda x: -x[1])
        }
        with open(importance_path, "w") as f:
            json.dump(sorted_importance, f, indent=4)
        logger.info(f"Feature importance saved to {importance_path}")

    # Save metadata
    metadata = {
        "model_type": MODEL_TYPE,
        "protocol": "CANONICAL_2.0",
        "source": "FINAL_PLAN.md Section 4.3",
        "training_samples": int(len(X_train_lag)),
        "validation_samples": int(len(X_val_lag)),
        "features_original": int(X_train.shape[1]),
        "features_lag_based": int(X_train_lag.shape[1]),
        "lookback": int(lookback),
        "hyperparameters": xgb_params,
        "training_complete": True,
        "frozen": True,
        "retraining_allowed": False,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            metadata,
            f,
            indent=4,
            default=lambda x: (
                float(x)
                if isinstance(x, (np.float32, np.float64))
                else int(x) if isinstance(x, (np.int32, np.int64)) else str(x)
            ),
        )

    logger.info(f"Metadata saved to {metadata_path}")


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
        log_file=f"{log_dir}/{ticker}__train.log",
        level="INFO",
        mode=LogFileMode.OVERWRITE,
    )

    train_dir = output_dirs["train"]
    logger.info(f"Train Directory --> {train_dir}")
    train_xgboost(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        config=config,
        output_dir=train_dir,
    )


def test_xgboost(
    ticker: str,
    model_dir: Path,
    data_path: Path,
    config: Config,
    output_dir: Path,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    End-to-end test pipeline using XGBoost.evaluate() as the canonical path.
    """
    logger.info("=" * 80)
    logger.info(f"BASELINE XGBoost INFERENCE: {ticker}")
    logger.info("=" * 80)

    lookback = config.xgboost.lookback
    X_test, y_test = load_test_data(data_path, lookback)

    model = load_trained_model(model_dir, config)

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
        test_metrics = test_xgboost(
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

        plot_full_xgboost_training_report(
            experiment_dir,
        )

    logger.info("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
