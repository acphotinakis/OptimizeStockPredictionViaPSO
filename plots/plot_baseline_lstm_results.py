#!/usr/bin/env python3
"""
Canonical Model Result Plotter (Ticker-Based)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


# =========================================================
# IO HELPERS
# =========================================================


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# =========================================================
# LSTM PLOTTING
# =========================================================


def plot_lstm_history(history: dict, out_path: Path, title: str):
    train = history.get("train_loss", [])
    val = history.get("val_loss", [])

    if not train and not val:
        return

    ensure_dir(out_path.parent)

    plt.figure(figsize=(10, 6))

    if train:
        plt.plot(train, label="Train Loss")
    if val:
        plt.plot(val, label="Validation Loss")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_metadata(metadata: dict, output_path: Path, title: str):
    if not metadata:
        return

    ensure_dir(output_path.parent)

    scalars = {
        "training_samples": metadata.get("training_samples", 0),
        "validation_samples": metadata.get("validation_samples", 0),
        "features": metadata.get("features", 0),
        "lookback": metadata.get("lookback", 0),
    }

    hyperparams = metadata.get("hyperparameters", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(list(scalars.keys()), list(scalars.values()))
    axes[0].set_title("Dataset & Model Scale")
    axes[0].tick_params(axis="x", rotation=30)

    if hyperparams:
        axes[1].bar(list(hyperparams.keys()), list(hyperparams.values()))
    axes[1].set_title("Hyperparameters")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle(title)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


# =========================================================
# XGBOOST PLOTTING (FIXED)
# =========================================================


def plot_xgboost_history(history: dict, output_path: Path):
    """
    FIXED: now accepts dict directly (not file path)
    """

    if not history:
        return

    train_rmse = history.get("validation_0", {}).get("rmse", [])
    val_rmse = history.get("validation_1", {}).get("rmse", [])

    if not train_rmse and not val_rmse:
        return

    ensure_dir(output_path.parent)

    rounds = list(range(len(train_rmse)))

    plt.figure(figsize=(10, 6))

    if train_rmse:
        plt.plot(rounds, train_rmse, label="Train RMSE")
    if val_rmse:
        plt.plot(rounds, val_rmse, label="Validation RMSE")

    plt.xlabel("Boosting Round")
    plt.ylabel("RMSE")
    plt.title("XGBoost Training History")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# =========================================================
# FEATURE IMPORTANCE
# =========================================================


def plot_feature_importance(fi: dict, out_path: Path, top_k: int = 20):
    if not fi:
        return

    ensure_dir(out_path.parent)

    sorted_items = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:top_k]

    names = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    plt.figure(figsize=(10, 6))
    plt.barh(names[::-1], values[::-1])

    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================================================
# MAIN
# =========================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--results-dir", default="results/canonical/models")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()

    ticker = args.ticker
    base = Path(args.results_dir)
    out_base = Path(args.output_dir) / ticker

    ensure_dir(out_base)

    # =====================================================
    # LSTM
    # =====================================================
    lstm_dir = base / "baseline_lstm" / ticker
    lstm_history = load_json(lstm_dir / "training_history.json")
    lstm_metadata = load_json(lstm_dir / "metadata.json")

    plot_lstm_history(
        lstm_history,
        out_base / "lstm_loss.png",
        f"LSTM Training Loss - {ticker}",
    )

    plot_metadata(
        lstm_metadata,
        out_base / "lstm_metadata.png",
        f"LSTM Metadata - {ticker}",
    )

    # =====================================================
    # XGBOOST
    # =====================================================
    xgb_dir = base / "xgboost" / ticker

    xgb_history = load_json(xgb_dir / "training_history.json")
    xgb_fi = load_json(xgb_dir / "feature_importance.json")

    plot_xgboost_history(
        xgb_history,
        out_base / "xgboost_training.png",
    )

    plot_feature_importance(
        xgb_fi,
        out_base / "xgb_feature_importance.png",
    )

    print(f"[OK] Plots saved to: {out_base}")


if __name__ == "__main__":
    main()
