from __future__ import annotations

from typing import Dict


def format_metrics(metrics: Dict[str, float]) -> str:
    return "  ".join([f"{k}={v:.6f}" for k, v in metrics.items()])


def safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("^", "")
