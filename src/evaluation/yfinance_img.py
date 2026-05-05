from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from ..models.yfinance_core import ForecastArtifacts


def plot_forecast_comparison(artifacts_by_method: Dict[str, ForecastArtifacts], title: str, outpath: str) -> None:
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    first = next(iter(artifacts_by_method.values()))
    plt.plot(first.y_true, label="Truth")
    for method, art in artifacts_by_method.items():
        plt.plot(art.y_pred, label=method.upper())
    plt.title(title)
    plt.xlabel("Test time step")
    plt.ylabel("Closing price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
