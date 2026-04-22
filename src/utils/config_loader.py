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
# Logging
# =========================
@dataclass
class LoggingConfig:
    level: str
    format: str
    file: str


# =========================
# Output
# =========================
@dataclass
class OutputConfig:
    results_dir: str
    save_predictions: bool
    save_backtests: bool
    save_metrics: bool
    save_frozen_state: bool


# =========================
# PROTOCOL ENFORCEMENT (CANONICAL ONLY)
# =========================
@dataclass
class EnforcementConfig:
    allow_retraining: bool
    allow_pipeline_refitting: bool
    allow_test_access_training: bool
    allow_shuffling: bool
    require_frozen_state: bool
    require_chronological_splits: bool


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
# Features (Unified)
# =========================
@dataclass
class WaveletConfig:
    enabled: bool
    wavelet: str
    level: int
    mode: str


@dataclass
class SelectorConfig:
    variance_threshold: float
    correlation_threshold: float
    vif_threshold: float
    mi_quantile_threshold: float


@dataclass
class CrossTickerConfig:
    enabled: bool
    spy_features: bool
    peer_features: bool
    peer_count: int
    peer_corr_threshold: float
    peer_corr_window: int


@dataclass
class ScalarConfig:
    enabled: bool
    feature_range: List[float]


@dataclass
class WindowConfig:
    lookback: int
    prediction_horizon: int


@dataclass
class FeaturesConfig:
    wavelet: WaveletConfig
    selector: SelectorConfig
    cross_ticker: CrossTickerConfig
    scalar: ScalarConfig
    windowing: WindowConfig


# =========================
# Evaluation
# =========================


@dataclass
class SplitsConfig:
    train_pct: float
    val_pct: float
    test_pct: float


@dataclass
class WalkForwardValidationConfig:
    window_size: int
    step_size: int
    window_type: str
    retraining: bool


@dataclass
class EvaluationPipelineConfig:
    fit_once: bool
    freeze_forever: bool
    refit_per_step: bool


@dataclass
class EvaluationConfig:
    splits: SplitsConfig
    walk_forward: WalkForwardValidationConfig
    pipeline: EvaluationPipelineConfig


# =========================
# LSTM (Unified Canonical)
# =========================
@dataclass
class EarlyStoppingConfig:
    enabled: bool
    monitor: str
    patience: int
    restore_best_weights: bool


@dataclass
class LSTMBaselineConfig:
    # Model metadata
    name: str
    framework: str

    # Architecture (2-layer TRD-compliant)
    lstm_units_1: int
    lstm_units_2: int
    dropout_rate: float
    activation: str

    # Output layer
    output_units: int
    output_activation: str

    # Training parameters
    optimizer: str
    learning_rate: float
    loss: str
    epochs: int
    batch_size: int

    # Early stopping
    early_stopping: EarlyStoppingConfig

    # Data handling
    shuffle: bool
    validation_split: float

    # Input specification
    lookback: int
    prediction_horizon: int

    # Reproducibility
    random_seed: int
    deterministic: bool

    # Advanced training
    grad_clip: float
    use_amp: bool
    accumulation_steps: int


# =========================
# XGBoost
# =========================
@dataclass
class XGBoostConfig:
    name: str  # Model identifier
    objective: str  # "reg:squarederror"
    max_depth: int  # Tree depth
    learning_rate: float  # Boosting learning rate
    n_estimators: int  # Number of boosting rounds
    subsample: float  # Row sampling fraction
    colsample_bytree: float  # Feature sampling fraction
    min_child_weight: float  # Minimum leaf weight
    gamma: float  # Minimum split loss reduction
    reg_alpha: float  # L1 regularization
    reg_lambda: float  # L2 regularization

    early_stopping_rounds: int  # Early stopping patience

    feature_type: str  # Feature type
    lookback: int  # Number of lags for lag-based features

    tree_method: str  # Tree construction algorithm
    max_bin: int  # Histogram bins
    n_jobs: int  # Parallel threads

    random_seed: int  # Reproducibility seed

    importance_type: str  # Feature importance metric (diagnostic)


# =========================
# PSO (Unified)
# =========================
@dataclass
class PSOSearchSpaceParam:
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[int] = None
    scale: Optional[str] = None
    choices: Optional[List[int]] = None


@dataclass
class LSTMUnitSearchSpaceParam:
    min: int
    max: int
    step: int


@dataclass
class PSOLSTMDropOutConfig:
    min: float
    max: float


@dataclass
class PSOLSTMLearningRateConfig:
    min: float
    max: float
    scale: str


@dataclass
class PSOBathSizeConfig:
    choices: List[int]


@dataclass
class PSOLSTMEpochsConfig:
    min: float
    max: float


@dataclass
class PSOSearchSpace:
    lstm_units_1: LSTMUnitSearchSpaceParam
    lstm_units_2: LSTMUnitSearchSpaceParam
    dropout_rate: PSOLSTMDropOutConfig
    learning_rate: PSOLSTMLearningRateConfig
    batch_size: PSOBathSizeConfig
    epochs: PSOLSTMEpochsConfig


# =========================
# Fitness
# =========================
@dataclass
class FitnessConfig:
    mse_weight: float
    msw_weight: float
    rmse_weight: float
    sharpe_weight: float
    drawdown_weight: float
    signal_threshold: float
    transaction_cost: float


@dataclass
class PSOConfig:
    enabled: bool
    n_particles: int
    n_iterations: int
    inertia_min: float
    inertia_max: float
    c1: float
    c2: float
    v_clamp_fraction: float
    seed: int
    search_space: PSOSearchSpace

    activation: str
    output_units: int
    output_activation: str
    optimizer: str
    loss: str
    shuffle: bool
    deterministic: bool

    checkpoint_dir: str

    fitness: FitnessConfig

    n_workers: int

    random_seed: int


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
    round_trip_cost: float
    signal_generation: str
    position_sizing: str


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
# ROOT CONFIG (Unified)
# =========================
@dataclass
class Config:
    logging: LoggingConfig
    output: OutputConfig
    enforcement: EnforcementConfig
    data: DataConfig
    features: FeaturesConfig
    evaluation: EvaluationConfig
    lstm_baseline: LSTMBaselineConfig
    xgboost: XGBoostConfig
    pso: PSOConfig
    backtesting: BacktestingConfig
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
    """
    Load a Config object from a YAML file. Uses defaults if path missing.
    """

    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.info("Config path %s not found. Using default config.", path)
        path = DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Default config not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    config = Config(
        logging=_safe_instantiate(LoggingConfig, raw.get("logging", {})),
        output=_safe_instantiate(OutputConfig, raw.get("output", {})),
        enforcement=_safe_instantiate(EnforcementConfig, raw.get("enforcement", {})),
        data=_safe_instantiate(DataConfig, raw.get("data", {})),
        features=_safe_instantiate(FeaturesConfig, raw.get("features", {})),
        evaluation=_safe_instantiate(EvaluationConfig, raw.get("evaluation", {})),
        lstm_baseline=_safe_instantiate(
            LSTMBaselineConfig,
            raw.get("lstm_baseline", {}),
        ),
        xgboost=_safe_instantiate(XGBoostConfig, raw.get("xgboost", {})),
        pso=_safe_instantiate(PSOConfig, raw.get("pso", {})),
        backtesting=_safe_instantiate(BacktestingConfig, raw.get("backtesting", {})),
        symbol_universe=(
            _safe_instantiate(SymbolUniverseConfig, raw.get("symbol_universe", {}))
            if "symbol_universe" in raw
            else None
        ),
    )

    logger.info(f"Loaded config from {path}")
    logger.info(
        f"  LSTM Baseline: {config.lstm_baseline.name} (framework={config.lstm_baseline.framework})"
    )
    logger.info(f"  PSO enabled: {config.pso.enabled}")
    logger.info(f"  Wavelet enabled: {config.features.wavelet.enabled}")
    logger.info(f"  Cross-ticker enabled: {config.features.cross_ticker.enabled}")

    return config
