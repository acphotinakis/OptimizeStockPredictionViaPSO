# src/utils/config_loader.py
import dataclasses
from pathlib import Path
import yaml
import logging
from dataclasses import fields

from src.utils.config_schema import *

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default_config.yaml")


def _safe_instantiate(cls, raw: dict):
    """
    Instantiate a dataclass from a dict, filling missing fields with defaults
    or None if optional.
    """
    init_kwargs = {}
    for f in fields(cls):
        if f.name in raw:
            value = raw[f.name]
            # Recursively handle nested dataclasses
            if hasattr(f.type, "__dataclass_fields__") and isinstance(value, dict):
                value = _safe_instantiate(f.type, value)
            init_kwargs[f.name] = value
        else:
            # Use default if available, else None
            if f.default is not dataclasses.MISSING:
                init_kwargs[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore
                init_kwargs[f.name] = f.default_factory()  # type: ignore
            else:
                init_kwargs[f.name] = None
    return cls(**init_kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Load a Config object from a YAML file. Uses defaults if path missing."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.info("Config path %s not found. Using default config.", path)
        path = DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Default config not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Instantiate all nested dataclasses safely
    return Config(
        pso=_safe_instantiate(PSOConfig, raw.get("pso", {})),
        lstm=_safe_instantiate(LSTMConfig, raw.get("lstm", {})),
        lstm_baseline=_safe_instantiate(
            LSTMBaselineConfig, raw.get("lstm_baseline", {})
        ),
        xgboost=_safe_instantiate(XGBoostConfig, raw.get("xgboost", {})),
        fitness=_safe_instantiate(FitnessConfig, raw.get("fitness", {})),
        data=_safe_instantiate(DataConfig, raw.get("data", {})),
        features=_safe_instantiate(FeaturesConfig, raw.get("features", {})),
        backtesting=_safe_instantiate(BacktestingConfig, raw.get("backtesting", {})),
        logging=_safe_instantiate(LoggingConfig, raw.get("logging", {})),
    )
