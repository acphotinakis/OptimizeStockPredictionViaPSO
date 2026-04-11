from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: str | Path,
    title: str = "Training History",
) -> None:
    """
    Plots training/validation loss and all evaluation metrics over epochs.

    Args:
        history: Dictionary returned by LSTM.fit().
        output_path: File path to save PNG.
        title: Plot title.
    """
    # -------------------------
    # Load history if file path
    # -------------------------
    if isinstance(history, (str, Path)):
        history_path = Path(history)
        with history_path.open("r") as f:
            history = json.load(f)

    if not isinstance(history, dict):
        raise TypeError("history must be a dict or a path to a JSON file")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    metrics_to_plot = [
        ("train_loss", "Train Loss"),
        ("val_loss", "Validation Loss"),
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("mape", "MAPE"),
        ("r2", "R²"),
        ("directional_accuracy", "Directional Accuracy"),
        ("f1_ternary", "F1 Ternary"),
    ]

    # --- Layout: single figure, multiple subplots ---
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 18), sharex=True)
    fig.suptitle(title, fontsize=14)

    for ax, (key, label) in zip(axes, metrics_to_plot):
        if key not in history:
            continue

        ax.plot(epochs, history[key])
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Epoch")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
