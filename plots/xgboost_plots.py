import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any


# ============================================================
# IO HELPERS
# ============================================================


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_npy(path: Path):
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# MASTER ENTRYPOINT
# ============================================================


def plot_full_xgboost_training_report(experiment_path: Path) -> None:
    """
    Generates full training visualization suite for XGBoost experiment folder.
    """

    experiment_path = Path(experiment_path)

    train_dir = experiment_path / "train"
    test_dir = experiment_path / "test"
    plot_dir = experiment_path / "plots"

    ensure_dir(plot_dir)

    # -------------------------
    # LOAD ARTIFACTS
    # -------------------------
    feature_importance = load_json(train_dir / "feature_importance.json")
    training_history = load_json(train_dir / "training_history.json")

    test_metrics = load_json(test_dir / "test_metrics.json")

    y_true = load_npy(test_dir / "test_ground_truth.npy")
    y_pred = load_npy(test_dir / "test_predictions.npy")

    # -------------------------
    # PLOTS
    # -------------------------

    _plot_loss(training_history, plot_dir)
    _plot_feature_importance(feature_importance, plot_dir)
    _plot_predictions(y_true, y_pred, plot_dir)
    _plot_test_metrics(test_metrics, plot_dir)

    print(f"Plots saved to: {plot_dir}")


# ============================================================
# 1. LOSS / TRAINING CURVES
# ============================================================


def _plot_loss(data: Dict[str, Any], save_dir: Path) -> None:
    if not data:
        return

    plt.figure()

    # Safe extraction for XGBoost eval history format
    if "validation_0" in data and "rmse" in data["validation_0"]:
        plt.plot(data["validation_0"]["rmse"], label="train (validation_0)")
    if "validation_1" in data and "rmse" in data["validation_1"]:
        plt.plot(data["validation_1"]["rmse"], label="test (validation_1)")

    plt.title("Validation RMSE Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.legend()

    plt.savefig(save_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 2. FEATURE IMPORTANCE
# ============================================================


def _plot_feature_importance(data: Dict[str, Any], save_dir: Path) -> None:
    if not data:
        return

    # filter out zero-importance features
    filtered = {k: v for k, v in data.items() if v != 0.0}

    if not filtered:
        return

    # sort by importance (descending) ONLY FIRST 20
    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:20]

    features = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    plt.figure()
    plt.barh(features, values)

    plt.title("Feature Importance (XGBoost)")
    plt.xlabel("Importance Score")

    plt.gca().invert_yaxis()  # highest importance at top

    plt.savefig(save_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 3. PREDICTIONS VS GROUND TRUTH
# ============================================================


def _plot_predictions(y_true, y_pred, save_dir: Path) -> None:
    if y_true is None or y_pred is None:
        return

    plt.figure()

    plt.plot(y_true, label="Ground Truth")
    plt.plot(y_pred, label="Predictions", alpha=0.8)

    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Time Step")
    plt.ylabel("Value")

    plt.legend()

    plt.savefig(save_dir / "predictions_vs_truth.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 4. TEST METRICS VISUALIZATION
# ============================================================

REQUIRED_TEST_METRICS_KEYS = {
    "rmse",
    "mae",
    "mape",
    "r2",
    "directional_accuracy",
    "f1_ternary",
    "auc_ternary",
}


def _plot_test_metrics(metrics: Dict[str, Any], save_dir: Path) -> None:
    keys = [
        "rmse",
        "mae",
        "mape",
        "r2",
        "directional_accuracy",
        "f1_ternary",
        "auc_ternary",
    ]

    values = [metrics[k] for k in keys]

    plt.figure()

    # diverging baseline
    plt.axhline(0, linewidth=1)

    plt.bar(keys, values)

    plt.title("Test Metrics (Diverging View)")
    plt.ylabel("Value (centered at 0)")
    plt.xticks(rotation=45)

    plt.savefig(save_dir / "test_metrics_diverging.png", dpi=150, bbox_inches="tight")
    plt.close()
