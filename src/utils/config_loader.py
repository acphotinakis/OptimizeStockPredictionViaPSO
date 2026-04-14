# src/utils/config_loader.py
import dataclasses
from pathlib import Path
import yaml
import logging
from dataclasses import fields

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from constants import DEFAULT_CONFIG_PATH


logger = logging.getLogger(__name__)


# =========================
# PSO
# =========================
@dataclass
class PSOConfig:
    n_particles: int
    n_iterations: int
    w_min: float
    w_max: float
    c1: float
    c2: float
    v_clamp_fraction: float
    seed: int
    checkpoint_dir: str
    n_workers: int


# =========================
# LSTM Search Space
# =========================
@dataclass
class RangeInt:
    min: int
    max: int
    step: Optional[int] = None


@dataclass
class RangeFloat:
    min: float
    max: float
    scale: Optional[str] = None


@dataclass
class LookbackConfig:
    choices: List[int]


@dataclass
class LSTMConfig:
    num_layers: RangeInt
    hidden_units: RangeInt
    dropout: RangeFloat
    learning_rate: RangeFloat
    lookback: LookbackConfig
    use_amp: bool
    max_epochs: int
    batch_size: int
    early_stopping_patience: int
    grad_clip: float
    accumulation_steps: int


# =========================
# LSTM Baseline
# =========================
@dataclass
class LSTMBaselineConfig:
    input_size: int
    num_layers: int
    hidden_units: int
    dropout: float
    learning_rate: float
    lookback: int

    max_epochs: int
    patience: int
    batch_size: int
    use_checkpointing: bool

    grad_clip: float
    use_amp: bool
    accumulation_steps: int

    wfv_fold_size: int
    wfv_folds: int

    output_size: int

    # initial_capital: float
    # position_fraction: float
    # transaction_cost: float
    # slippage: float
    # stop_loss: float
    # daily_loss_limit: float


# =========================
# XGBoost
# =========================
@dataclass
class XGBoostConfig:
    objective: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: float
    gamma: float
    reg_alpha: float
    reg_lambda: float
    early_stopping_rounds: int
    use_optuna: bool
    optuna_trials: int
    tree_method: str
    max_bin: int
    lookback: int
    importance_type: str


# =========================
# Fitness
# =========================
@dataclass
class FitnessConfig:
    rmse_weight: float
    sharpe_weight: float
    drawdown_weight: float
    signal_threshold: float
    transaction_cost: float


# =========================
# Data
# =========================
@dataclass
class DataConfig:
    start_date: str
    end_date: str
    train_end: str
    val_end: str
    test_end: str

    train_years: int
    val_years: int
    test_years: int

    freq: str
    session_start: str
    session_end: str
    timezone: str

    benchmark_ticker: str
    max_missing_fraction: float
    max_ffill_bars: int


# =========================
# Features
# =========================
@dataclass
class XGBParams:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float


@dataclass
class SelectorConfig:
    method: str
    importance_threshold: float
    xgb_params: XGBParams


@dataclass
class FeaturesConfig:
    selector: SelectorConfig


# =========================
# Backtesting
# =========================
@dataclass
class BacktestingConfig:
    transaction_cost: float
    slippage: float
    position_size: float
    stop_loss: float
    take_profit: float
    initial_capital: float
    position_fraction: float
    daily_loss_limit: float


# =========================
# Logging
# =========================
@dataclass
class LoggingConfig:
    level: str
    format: str
    file: str


# =========================
# Symbol Universe
# =========================
@dataclass
class PeerSelectionConfig:
    max_peers: int
    method: str
    min_correlation: float
    lookback_days: int


@dataclass
class SymbolUniverseConfig:
    market_context: List[str]
    sector_etfs: Dict[str, Dict[str, Any]]
    market_internals: List[str]
    peer_selection: PeerSelectionConfig
    prediction_targets: List[str]
    context_only: List[str]


# =========================
# ROOT CONFIG
# =========================
@dataclass
class Config:
    pso: PSOConfig
    lstm: LSTMConfig
    lstm_baseline: LSTMBaselineConfig
    xgboost: XGBoostConfig
    fitness: FitnessConfig
    data: DataConfig
    features: FeaturesConfig
    backtesting: BacktestingConfig
    logging: LoggingConfig
    symbol_universe: Optional[SymbolUniverseConfig] = None


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
