from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import json

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any


# =========================================================
# LOADERS
# =========================================================


def load_test_artifacts(test_dir: Path):
    """
    Loads standardized test outputs.

    Expected:
        test_predictions.npy
        test_ground_truth.npy
        test_metrics.json
    """
    y_pred = np.load(test_dir / "test_predictions.npy")
    y_true = np.load(test_dir / "test_ground_truth.npy")

    with open(test_dir / "test_metrics.json", "r") as f:
        metrics = json.load(f)

    return y_true, y_pred, metrics


def load_training_history(run_dir: Path):
    """
    Loads training history JSON.
    """
    with open(run_dir / "train" / "training_history.json", "r") as f:
        return json.load(f)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# =========================================================
# PLOTTING: TEST RESULTS
# =========================================================


def plot_test_predictions(
    run_dir: Path,
    save: bool = True,
):
    """
    Plot:
    - Ground truth vs predictions (full series)
    """

    test_dir = run_dir / "test"

    y_true, y_pred, _ = load_test_artifacts(test_dir)

    plt.figure()
    plt.plot(y_true, label="Ground Truth")
    plt.plot(y_pred, label="Predictions")
    plt.title("Test Predictions vs Ground Truth")
    plt.legend()

    if save:
        out = run_dir / "plots" / "test_predictions.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")

    plt.close()


def plot_residuals(
    run_dir: Path,
    save: bool = True,
):
    """
    Residual distribution + bias visualization.
    """

    test_dir = run_dir / "test"

    y_true, y_pred, _ = load_test_artifacts(test_dir)

    residuals = y_true - y_pred

    plt.figure()
    plt.hist(residuals, bins=50)
    plt.title("Residual Distribution")

    if save:
        out = run_dir / "plots" / "residual_distribution.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")

    plt.close()


def plot_scatter_fit(
    run_dir: Path,
    save: bool = True,
):
    """
    Predicted vs Actual scatter plot.
    """

    test_dir = run_dir / "test"

    y_true, y_pred, _ = load_test_artifacts(test_dir)

    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    plt.title("Predicted vs Actual")

    if save:
        out = run_dir / "plots" / "scatter_fit.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")

    plt.close()


# =========================================================
# PLOTTING: TRAINING CURVES
# =========================================================


def plot_full_training_report(data_path: Path, save_dir: Path) -> None:
    """
    Generates full training visualization suite and saves all plots to disk.

    Args:
        run: Training artifact containing "history" and "final_metrics"
        save_dir: Directory to store all plots
    """

    run = load_json(data_path)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = run["history"]
    final = run.get("final_metrics", {})

    _plot_loss(history, save_dir)
    _plot_grad_norm(history, save_dir)
    _plot_learning_rate(history, save_dir)
    _plot_epoch_time(history, save_dir)
    _plot_variance_ratio(history, save_dir)
    _plot_val_loss_slope(history, save_dir)
    _plot_metrics_over_time(history, save_dir)

    _save_summary(final, save_dir)


# ============================================================
# 1. LOSS
# ============================================================


def _plot_loss(history: Dict[str, Any], save_dir: Path) -> None:
    plt.figure()
    plt.plot(history["train_loss"])
    plt.plot(history["val_loss"])
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.grid(True)

    plt.savefig(save_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 2. GRADIENT NORM
# ============================================================


def _plot_grad_norm(history: Dict[str, Any], save_dir: Path) -> None:
    if "grad_norm" not in history:
        return

    plt.figure()
    plt.plot(history["grad_norm"])
    plt.title("Gradient Norm")
    plt.xlabel("Epoch")
    plt.ylabel("Norm")
    plt.grid(True)

    plt.savefig(save_dir / "grad_norm.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 3. LEARNING RATE
# ============================================================


def _plot_learning_rate(history: Dict[str, Any], save_dir: Path) -> None:
    if "lr" not in history:
        return

    plt.figure()
    plt.plot(history["lr"])
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.grid(True)

    plt.savefig(save_dir / "learning_rate.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 4. EPOCH TIME
# ============================================================


def _plot_epoch_time(history: Dict[str, Any], save_dir: Path) -> None:
    if "epoch_time" not in history:
        return

    plt.figure()
    plt.plot(history["epoch_time"])
    plt.title("Epoch Time")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.grid(True)

    plt.savefig(save_dir / "epoch_time.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 5. VARIANCE RATIO
# ============================================================


def _plot_variance_ratio(history: Dict[str, Any], save_dir: Path) -> None:
    if "variance_ratio" not in history:
        return

    plt.figure()
    plt.plot(history["variance_ratio"])
    plt.title("Variance Ratio")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)

    plt.savefig(save_dir / "variance_ratio.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 6. VAL LOSS SLOPE
# ============================================================


def _plot_val_loss_slope(history: Dict[str, Any], save_dir: Path) -> None:
    if "val_loss_slope" not in history:
        return

    plt.figure()
    plt.plot(history["val_loss_slope"])
    plt.axhline(0, linewidth=1)
    plt.title("Validation Loss Slope")
    plt.xlabel("Epoch")
    plt.ylabel("Slope")
    plt.grid(True)

    plt.savefig(save_dir / "val_loss_slope.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 7. METRICS OVER TIME
# ============================================================


def _plot_metrics_over_time(history: Dict[str, Any], save_dir: Path) -> None:
    if "train_metrics" not in history or "val_metrics" not in history:
        return

    keys = ["rmse", "mae", "mape", "r2", "directional_accuracy", "auc_ternary"]

    train_metrics = history["train_metrics"]
    val_metrics = history["val_metrics"]

    for key in keys:
        if key not in train_metrics[0]:
            continue

        train_series = [m[key] for m in train_metrics]
        val_series = [m[key] for m in val_metrics]

        plt.figure()
        plt.plot(train_series)
        plt.plot(val_series)

        plt.title(f"{key.upper()} Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.legend(["train", "val"])
        plt.grid(True)

        plt.savefig(save_dir / f"{key}_curve.png", dpi=150, bbox_inches="tight")
        plt.close()


# ============================================================
# 8. FINAL METRICS SUMMARY (TEXT FILE)
# ============================================================


def _save_summary(final: Dict[str, Any], save_dir: Path) -> None:
    if not final:
        return

    path = save_dir / "final_metrics.txt"

    with open(path, "w") as f:
        f.write("FINAL METRICS\n")
        f.write("====================\n\n")
        for k, v in final.items():
            f.write(f"{k}: {v}\n")


# =========================================================
# MASTER ENTRYPOINT
# =========================================================


def generate_all_plots(run_dir: Path, data_path: Path, save_dir: Path):
    """
    One-call plotting for full experiment artifact.
    """

    plot_test_predictions(run_dir)
    plot_residuals(run_dir)
    plot_scatter_fit(run_dir)

    plot_full_training_report(
        data_path=data_path,
        save_dir=save_dir,
    )
