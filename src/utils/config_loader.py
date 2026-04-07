"""
src/utils/config_loader.py
YAML configuration loader with robust error handling and dot-access support.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml


class Config:
    """Nested dot-access wrapper around a YAML-loaded dictionary."""

    def __init__(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise TypeError(f"Config expects dict, got {type(data).__name__}")

        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Config keys must be strings, got {type(key).__name__}"
                )
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert Config back to dictionary."""
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out

    def __repr__(self) -> str:
        return f"Config({self.__dict__})"


def load_config(path: str | Path) -> Config:
    """Load a YAML file and return a validated Config object.

    Raises:
        FileNotFoundError: If config file does not exist
        PermissionError: If file cannot be read
        ValueError: If YAML is empty or invalid structure
        yaml.YAMLError: If YAML parsing fails
        TypeError: If parsed config is not a dict
    """
    path = Path(path)

    # ---------------------------
    # File existence + access
    # ---------------------------
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except PermissionError:
        raise PermissionError(f"Permission denied when reading config: {path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {path}: {e}")

    # ---------------------------
    # Validation
    # ---------------------------
    if data is None:
        raise ValueError(f"Config file is empty: {path}")

    if not isinstance(data, dict):
        raise TypeError(
            f"Top-level YAML structure must be a dict, got {type(data).__name__}"
        )

    # Optional: enforce non-empty config
    if not data:
        raise ValueError(f"Config file contains no keys: {path}")

    return Config(data)
