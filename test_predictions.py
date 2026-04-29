import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.scaler import FrozenMinMaxScaler
from src.evaluation.metrics import compute_and_log_all_statistical_metrics
from src.utils.logger import setup_logger, LogFileMode

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Test LSTM predictions by comparing with ground truth on original scale."
    )

    # Path options
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directory containing test_predictions.npy and test_ground_truth.npy",
    )
    parser.add_argument(
        "--frozen-state",
        type=Path,
        required=True,
        help="Path to the frozen_state.json containing target_scaler params",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save original scale metrics and plots (defaults to test-dir/original_scale)",
    )
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")

    args = parser.parse_args()

    # Setup logger
    setup_logger(level="INFO")

    if not args.test_dir.exists():
        logger.error(f"Test directory not found: {args.test_dir}")
        sys.exit(1)

    if not args.frozen_state.exists():
        logger.error(f"Frozen state file not found: {args.frozen_state}")
        sys.exit(1)

    output_dir = args.output_dir or args.test_dir / "original_scale"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logger.info(f"Loading predictions from {args.test_dir}")
    try:
        y_pred_scaled = np.load(args.test_dir / "test_predictions.npy")
        y_true_scaled = np.load(args.test_dir / "test_ground_truth.npy")
        metrics_original = compute_and_log_all_statistical_metrics(
            y_true_scaled, y_pred_scaled, label="Original Scale"
        )
    except FileNotFoundError as e:
        logger.error(f"Missing data file: {e}")
        sys.exit(1)

    logger.info(
        f"Loaded scaled data: y_pred={y_pred_scaled.shape}, y_true={y_true_scaled.shape}"
    )

    # 2. Load target scaler from frozen state
    logger.info(f"Loading target scaler from {args.frozen_state}")
    with open(args.frozen_state, "r") as f:
        frozen_state = json.load(f)

    if "target_scaler" not in frozen_state:
        logger.error(f"'target_scaler' not found in {args.frozen_state}")
        sys.exit(1)

    target_scaler = FrozenMinMaxScaler.from_params(frozen_state["target_scaler"])

    # 3. Inverse transform
    logger.info("Applying inverse transformation to original scale...")
    # Reshape if necessary (scaler expects (N, 1) or (N, F))
    y_pred_original = target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()
    y_true_original = target_scaler.inverse_transform(
        y_true_scaled.reshape(-1, 1)
    ).ravel()

    # 4. Compute metrics on original scale
    logger.info("Computing metrics on original scale:")
    metrics_original = compute_and_log_all_statistical_metrics(
        y_true_original, y_pred_original, label="Original Scale"
    )

    # Save metrics
    metrics_path = output_dir / "test_metrics_original.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_original, f, indent=4)
    logger.info(f"Saved original scale metrics to {metrics_path}")

    # 5. Plotting
    if args.plot:
        logger.info("Generating comparison plots...")

        # Prediction vs Ground Truth
        plt.figure(figsize=(12, 6))
        plt.plot(y_true_original, label="Ground Truth", alpha=0.7)
        plt.plot(y_pred_original, label="Predictions", alpha=0.7)
        plt.title("Original Scale: Predictions vs Ground Truth")
        plt.xlabel("Sample Index")
        plt.ylabel("Value (Original Scale)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            output_dir / "predictions_comparison.png", dpi=300, bbox_inches="tight"
        )

        # Residuals
        plt.figure(figsize=(10, 6))
        residuals = y_true_original - y_pred_original
        plt.hist(residuals, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        plt.title("Original Scale: Residual Distribution")
        plt.xlabel("Error (True - Pred)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            output_dir / "residuals_distribution.png", dpi=300, bbox_inches="tight"
        )

        # Scatter Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true_original, y_pred_original, alpha=0.5, s=10)
        # Add 45-degree line
        min_val = min(y_true_original.min(), y_pred_original.min())
        max_val = max(y_true_original.max(), y_pred_original.max())
        plt.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction"
        )
        plt.title("Original Scale: Predicted vs Actual Scatter")
        plt.xlabel("Actual Value")
        plt.ylabel("Predicted Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "scatter_fit.png", dpi=300, bbox_inches="tight")

        logger.info(f"Plots saved to {output_dir}")

    logger.info("Test complete.")


if __name__ == "__main__":
    main()
