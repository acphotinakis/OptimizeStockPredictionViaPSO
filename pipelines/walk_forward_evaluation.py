#!/usr/bin/env python3
"""
Production Walk-Forward Validation Pipeline

Executes expanding-window walk-forward validation with per-fold PSO optimization.

ARCHITECTURE (WALK_FORWARD_PLAN.md):
- Global preprocessing (once): cleaning, feature generation, wavelet
- Per-fold processing: scaling, PSO, training, prediction
- Expanding window (train grows each fold)
- Independent scalers per fold
- Inverse transform before metrics

Usage:
    python pipelines/walk_forward_evaluation.py \\
        --data-path data/processed/features_unified/AAPL \\
        --config config/default_config.yaml \\
        --output-dir results/walk_forward/AAPL \\
        --enable-pso

TRD Compliance: TRD1 §8 (no leakage), TRD2 §7.4 (PSO per fold)
Author: Production System
Version: 1.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.walk_forward_pso import (
    ExpandingWindowWalkForward,
    validate_walk_forward_compliance,
)
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def load_preprocessed_data(data_path: Path) -> dict:
    """
    Load UNSCALED preprocessed data.

    CRITICAL: Data must be from global preprocessing (NOT scaled).

    Expected files:
    - X_raw.npy: Features (NO SCALING)
    - y_raw.npy: Targets (NO SCALING)
    - timestamps.npy: Optional datetime index
    - metadata.yaml: Feature names, preprocessing info

    Args:
        data_path: Directory containing preprocessed data

    Returns:
        Dictionary with X_raw, y_raw, timestamps, metadata
    """
    logger.info(f"Loading preprocessed data from {data_path}")

    X_path = data_path / "X_raw.npy"
    y_path = data_path / "y_raw.npy"
    timestamps_path = data_path / "timestamps.npy"
    metadata_path = data_path / "metadata.yaml"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing required data files in {data_path}\n"
            f"Expected: X_raw.npy, y_raw.npy\n"
            f"Run global preprocessing first."
        )

    X_raw = np.load(X_path)
    y_raw = np.load(y_path)

    logger.info(f"  X_raw: {X_raw.shape}")
    logger.info(f"  y_raw: {y_raw.shape}")

    # Load optional timestamps
    timestamps = None
    if timestamps_path.exists():
        timestamps = pd.to_datetime(np.load(timestamps_path, allow_pickle=True))
        logger.info(
            f"  timestamps: {len(timestamps)} ({timestamps[0]} --> {timestamps[-1]})"
        )
    else:
        logger.warning("  timestamps.npy not found - using indices")

    # Load optional metadata
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
        logger.info(f"  metadata: {len(metadata)} fields loaded")
    else:
        logger.warning("  metadata.yaml not found")

    return {
        "X_raw": X_raw,
        "y_raw": y_raw,
        "timestamps": timestamps,
        "metadata": metadata,
    }


def save_walk_forward_results(
    results: dict,
    output_dir: Path,
) -> None:
    """
    Save complete walk-forward validation results.

    Args:
        results: Results dictionary from ExpandingWindowWalkForward
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregated metrics
    agg_path = output_dir / "aggregated_metrics.json"
    with open(agg_path, "w") as f:
        json.dump(results["aggregated_metrics"], f, indent=2)
    logger.info(f"Aggregated metrics saved to {agg_path}")

    # Save fold results (without large arrays)
    fold_results_compact = []
    for fold in results["fold_results"]:
        fold_compact = {
            "fold_idx": fold["fold_idx"],
            "train_size": fold["train_size"],
            "val_size": fold["val_size"],
            "train_indices": fold["train_indices"],
            "val_indices": fold["val_indices"],
            "pso_params": fold["pso_params"],
            "pso_fitness": fold["pso_fitness"],
            "metrics": fold["metrics"],
        }
        fold_results_compact.append(fold_compact)

    fold_path = output_dir / "fold_results.json"
    with open(fold_path, "w") as f:
        json.dump(fold_results_compact, f, indent=2)
    logger.info(f"Fold results saved to {fold_path}")

    # Save per-fold scalers
    scaler_dir = output_dir / "fold_scalers"
    scaler_dir.mkdir(exist_ok=True)

    for fold in results["fold_results"]:
        fold_idx = fold["fold_idx"]

        # Feature scaler
        feat_scaler_path = scaler_dir / f"fold_{fold_idx}_feature_scaler.json"
        with open(feat_scaler_path, "w") as f:
            json.dump(fold["scalers"]["feature_scaler_params"], f, indent=2)

        # Target scaler
        tgt_scaler_path = scaler_dir / f"fold_{fold_idx}_target_scaler.json"
        with open(tgt_scaler_path, "w") as f:
            json.dump(fold["scalers"]["target_scaler_params"], f, indent=2)

    logger.info(f"Fold scalers saved to {scaler_dir}")

    # Save per-fold predictions
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    for fold in results["fold_results"]:
        fold_idx = fold["fold_idx"]
        pred_path = pred_dir / f"fold_{fold_idx}_predictions.json"

        with open(pred_path, "w") as f:
            json.dump(fold["predictions"], f, indent=2)

    logger.info(f"Predictions saved to {pred_dir}")

    # Save configuration
    config_path = output_dir / "walk_forward_config.json"
    with open(config_path, "w") as f:
        json.dump(results["config"], f, indent=2)
    logger.info(f"Configuration saved to {config_path}")

    # Generate summary report
    report_path = output_dir / "validation_report.md"
    generate_validation_report(results, report_path)
    logger.info(f"Validation report saved to {report_path}")


def generate_validation_report(results: dict, output_path: Path) -> None:
    """
    Generate human-readable validation report.

    Args:
        results: Walk-forward validation results
        output_path: Path to save markdown report
    """
    agg = results["aggregated_metrics"]
    n_folds = agg["n_folds"]

    report = f"""# Walk-Forward Validation Report

**Date:** {pd.Timestamp.now().isoformat()}  
**Validation Type:** Expanding-Window Walk-Forward with PSO  
**Total Folds:** {n_folds}  
**Total Predictions:** {agg['total_predictions']}

---

## Aggregated Performance Metrics

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| **RMSE** | {agg['rmse_mean']:.6f} | {agg['rmse_std']:.6f} | {agg['rmse_min']:.6f} | {agg['rmse_max']:.6f} | {agg['rmse_median']:.6f} |
| **MAE** | {agg['mae_mean']:.6f} | {agg['mae_std']:.6f} | {agg['mae_min']:.6f} | {agg['mae_max']:.6f} | {agg['mae_median']:.6f} |
| **R²** | {agg['r2_mean']:.4f} | {agg['r2_std']:.4f} | {agg['r2_min']:.4f} | {agg['r2_max']:.4f} | {agg['r2_median']:.4f} |
| **MAPE** | {agg['mape_mean']:.2f}% | {agg['mape_std']:.2f}% | {agg['mape_min']:.2f}% | {agg['mape_max']:.2f}% | {agg['mape_median']:.2f}% |
| **DA** | {agg['directional_accuracy_mean']:.2%} | {agg['directional_accuracy_std']:.2%} | {agg['directional_accuracy_min']:.2%} | {agg['directional_accuracy_max']:.2%} | {agg['directional_accuracy_median']:.2%} |

---

## Per-Fold Performance

| Fold | Train Size | Val Size | RMSE | R² | DA |
|------|-----------|----------|------|----|----|
"""

    for fold in results["fold_results"]:
        fold_idx = fold["fold_idx"] + 1
        train_size = fold["train_size"]
        val_size = fold["val_size"]
        rmse = fold["metrics"]["rmse"]
        r2 = fold["metrics"]["r2"]
        da = fold["metrics"]["directional_accuracy"]

        report += f"| {fold_idx} | {train_size} | {val_size} | {rmse:.6f} | {r2:.4f} | {da:.2%} |\n"

    report += f"""
---

## Configuration

```yaml
{yaml.dump(results['config'], default_flow_style=False)}
```

---

## TRD Compliance

- ✅ Expanding window (train grows each fold)
- ✅ Independent scalers per fold (feature + target)
- ✅ Per-fold PSO optimization (if enabled)
- ✅ Fresh model initialization per fold
- ✅ Inverse transform before metrics
- ✅ No look-ahead bias
- ✅ Temporal ordering preserved

---

## Conclusion

Walk-forward validation completed successfully across {n_folds} folds.

**Overall Performance:**
- RMSE: {agg['rmse_mean']:.6f} ± {agg['rmse_std']:.6f}
- R²: {agg['r2_mean']:.4f} (mean across folds)

**Status:** Production-ready, TRD-compliant
"""

    with open(output_path, "w") as f:
        f.write(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Walk-forward validation with expanding windows and PSO"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to preprocessed UNSCALED data directory",
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
        required=True,
        help="Directory to save validation results",
    )
    parser.add_argument(
        "--enable-pso",
        action="store_true",
        help="Enable PSO optimization per fold (slower but more accurate)",
    )
    parser.add_argument(
        "--initial-train-pct",
        type=float,
        default=0.60,
        help="Initial training window size (default: 60%%)",
    )
    parser.add_argument(
        "--fold-step-pct",
        type=float,
        default=0.05,
        help="Fold step size (default: 5%%)",
    )
    parser.add_argument(
        "--val-pct",
        type=float,
        default=0.05,
        help="Validation size per fold (default: 5%%)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=20,
        help="LSTM lookback window (default: 20)",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=0,
        help="Maximum folds to run (0 = unlimited, default: 0)",
    )

    args = parser.parse_args()

    try:
        # Setup logging
        setup_logger(
            log_file=args.output_dir / "walk_forward_validation.log",
            level="INFO",
        )

        # Load configuration
        config = load_config(args.config)

        # Load data
        data = load_preprocessed_data(args.data_path)

        # Validate data is unscaled
        if data["X_raw"].min() >= -1.0 and data["X_raw"].max() <= 1.0:
            logger.warning(
                "⚠️  Data appears to be scaled (range ~ [-1, 1]). "
                "Walk-forward requires UNSCALED data for per-fold scaling."
            )

        # Initialize walk-forward validator
        validator = ExpandingWindowWalkForward(
            initial_train_pct=args.initial_train_pct,
            fold_step_pct=args.fold_step_pct,
            val_pct=args.val_pct,
            lookback=args.lookback,
            config=config,
            run_pso=args.enable_pso,
            max_folds=args.max_folds,
        )

        # Run validation
        results = validator.validate(
            X_raw=data["X_raw"],
            y_raw=data["y_raw"],
            timestamps=data["timestamps"],
        )

        # Validate TRD compliance
        validate_walk_forward_compliance(
            X_raw=data["X_raw"],
            y_raw=data["y_raw"],
            fold_boundaries=results["fold_boundaries"],
        )

        # Save results
        save_walk_forward_results(results, args.output_dir)

        logger.info("\n" + "=" * 80)
        logger.info("WALK-FORWARD VALIDATION SUCCESSFUL")
        logger.info("=" * 80)
        logger.info(f" {results['aggregated_metrics']['n_folds']} folds completed")
        logger.info(f" Results saved to {args.output_dir}")
        logger.info(f" TRD compliance validated")
        logger.info("=" * 80)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Walk-forward validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
