# src/utils/config_schema.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


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

    max_epochs: int
    batch_size: int
    early_stopping_patience: int
    grad_clip: float


# =========================
# LSTM Baseline
# =========================
@dataclass
class LSTMBaselineConfig:
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
