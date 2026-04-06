"""
src/utils/config_loader.py
YAML configuration loader with dot-access support.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


class Config:
    """Nested dot-access wrapper around a YAML-loaded dictionary."""

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"Config({self.__dict__})"


def load_config(path: str | Path) -> Config:
    """Load a YAML file and return a Config object.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Config object with dot-access attributes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(data)
