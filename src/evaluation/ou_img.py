from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_img_dir(img_dir: str) -> str:
    Path(img_dir).mkdir(parents=True, exist_ok=True)
    return img_dir


def plot_forecast(y_true: np.ndarray, y_pred: np.ndarray, title: str, outpath: Optional[str] = None, show: bool = True) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Truth")
    plt.plot(y_pred, label="Prediction")
    plt.title(title)
    plt.xlabel("Test time step")
    plt.ylabel("Close price")
    plt.legend()
    plt.tight_layout()
    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=160)
    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_curves(history_by_method: Dict[str, Dict[str, List[float]]], title: str, outpath: str) -> None:
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for method, hist in history_by_method.items():
        loss = hist.get("loss", [])
        if loss:
            plt.plot(range(1, len(loss) + 1), loss, label=method.upper())
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_search_convergence(search_by_method: Dict[str, List[Dict[str, float]]], title: str, outpath: str) -> None:
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for method, history in search_by_method.items():
        if not history:
            continue
        hist_df = pd.DataFrame(history)
        if "iter" in hist_df.columns and "best_fitness" in hist_df.columns:
            plt.plot(hist_df["iter"], hist_df["best_fitness"], label=method.upper())
    plt.title(title)
    plt.xlabel("Search iteration")
    plt.ylabel("Best RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
