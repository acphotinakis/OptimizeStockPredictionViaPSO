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
class PSOSearchSpace:
    lstm_units_1: Optional[PSOSearchSpaceParam] = None
    lstm_units_2: Optional[PSOSearchSpaceParam] = None
    dropout_rate: Optional[PSOSearchSpaceParam] = None
    learning_rate: Optional[PSOSearchSpaceParam] = None
    batch_size: Optional[PSOSearchSpaceParam] = None
    epochs: Optional[PSOSearchSpaceParam] = None


@dataclass
class PSOConfig:
    enabled: bool
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
    search_space: Optional[PSOSearchSpace] = None


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
class LSTMConfig:
    # Model metadata
    name: str
    framework: str
    
    # Architecture (2-layer TRD-compliant)
    lstm_units_1: int
    lstm_units_2: int
    activation: str
    dropout_rate: float
    
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
    expected_features: int
    
    # Preprocessing
    scaler: str
    scaler_range: List[float]
    
    # Reproducibility
    random_seed: int
    deterministic: bool
    
    # Advanced training
    grad_clip: float
    use_amp: bool
    accumulation_steps: int


# =========================
# LSTM Baseline (Deprecated - kept for backward compatibility)
# =========================
@dataclass
class LSTMBaselineConfig:
    """Deprecated: Use LSTMConfig instead. Kept for backward compatibility."""
    input_size: Optional[int] = None
    num_layers: Optional[int] = None
    hidden_units: Optional[int] = None
    dropout: Optional[float] = None
    learning_rate: Optional[float] = None
    lookback: Optional[int] = None
    max_epochs: Optional[int] = None
    patience: Optional[int] = None
    batch_size: Optional[int] = None
    use_checkpointing: Optional[bool] = None
    grad_clip: Optional[float] = None
    use_amp: Optional[bool] = None
    accumulation_steps: Optional[int] = None
    wfv_fold_size: Optional[int] = None
    wfv_folds: Optional[int] = None
    output_size: Optional[int] = None


# =========================
# =========================
# XGBoost (Unified TRD-Compliant)
# =========================
@dataclass
class XGBoostConfig:
    """
    Canonical XGBoost configuration (TRD-aligned).
    
    Architecture:
    - Gradient boosted trees for regression
    - Next-period return prediction
    - Early stopping on validation RMSE
    
    Constraints:
    - Consumes features from unified pipeline
    - No internal feature engineering
    - No normalization in model layer
    - Deterministic training
    """
    name: str                      # Model identifier
    objective: str                 # "reg:squarederror"
    n_estimators: int              # Number of boosting rounds
    max_depth: int                 # Tree depth
    learning_rate: float           # Boosting learning rate
    subsample: float               # Row sampling fraction
    colsample_bytree: float        # Feature sampling fraction
    min_child_weight: float        # Minimum leaf weight
    gamma: float                   # Minimum split loss reduction
    reg_alpha: float               # L1 regularization
    reg_lambda: float              # L2 regularization
    early_stopping_rounds: int     # Early stopping patience
    importance_type: str           # Feature importance metric (diagnostic)
    tree_method: str               # Tree construction algorithm
    max_bin: int                   # Histogram bins
    lookback: int                  # Number of lags for lag-based features
    random_seed: int               # Reproducibility seed


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
class FeaturesConfig:
    wavelet: WaveletConfig
    selector: SelectorConfig
    cross_ticker: CrossTickerConfig


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
# ROOT CONFIG (Unified)
# =========================
@dataclass
class Config:
    pso: PSOConfig
    lstm: LSTMConfig
    xgboost: XGBoostConfig
    fitness: FitnessConfig
    data: DataConfig
    features: FeaturesConfig
    backtesting: BacktestingConfig
    logging: LoggingConfig
    lstm_baseline: Optional[LSTMBaselineConfig] = None  # Deprecated, kept for compatibility
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
    
    Updated to support unified configuration structure with:
    - Unified LSTM config (replaces lstm + lstm_baseline)
    - Expanded PSO config with search_space
    - Unified features config (wavelet + selector + cross_ticker)
    
    Args:
        path: Path to YAML config file
    
    Returns:
        Config object with all nested dataclasses
    
    Raises:
        FileNotFoundError: If config file not found
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.info("Config path %s not found. Using default config.", path)
        path = DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Default config not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Instantiate all nested dataclasses safely
    config = Config(
        pso=_safe_instantiate(PSOConfig, raw.get("pso", {})),
        lstm=_safe_instantiate(LSTMConfig, raw.get("lstm", {})),
        xgboost=_safe_instantiate(XGBoostConfig, raw.get("xgboost", {})),
        fitness=_safe_instantiate(FitnessConfig, raw.get("fitness", {})),
        data=_safe_instantiate(DataConfig, raw.get("data", {})),
        features=_safe_instantiate(FeaturesConfig, raw.get("features", {})),
        backtesting=_safe_instantiate(BacktestingConfig, raw.get("backtesting", {})),
        logging=_safe_instantiate(LoggingConfig, raw.get("logging", {})),
        lstm_baseline=_safe_instantiate(
            LSTMBaselineConfig, raw.get("lstm_baseline", {})
        ) if "lstm_baseline" in raw else None,
    )
    
    logger.info(f"Loaded config from {path}")
    logger.info(f"  LSTM: {config.lstm.name} (framework={config.lstm.framework})")
    logger.info(f"  PSO: {'enabled' if config.pso.enabled else 'disabled'}")
    logger.info(f"  Wavelet: {'enabled' if config.features.wavelet.enabled else 'disabled'}")
    logger.info(f"  Cross-ticker: {'enabled' if config.features.cross_ticker.enabled else 'disabled'}")
    
    return config
