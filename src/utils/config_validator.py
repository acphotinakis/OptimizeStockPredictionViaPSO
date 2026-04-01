import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def validate_config(cfg: DictConfig) -> bool:
    """
    Validates that all required configuration keys exist and have valid values.
    
    This prevents runtime errors due to missing or malformed configuration.
    
    Args:
        cfg: The Hydra configuration object.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    logger.info("Validating configuration structure...")
    
    errors = []
    
    # A. Data Configuration
    required_data_keys = [
        "data.tickers",
        "data.timeframe",
        "data.start_date",
        "data.end_date",
        "data.splits.train_years",
        "data.splits.val_years",
        "data.splits.test_years",
        "data.validation.max_missing_pct",
        "data.cleaning.short_gap_threshold",
        "data.cleaning.long_gap_threshold",
    ]
    
    for key in required_data_keys:
        if not OmegaConf.select(cfg, key):
            errors.append(f"Missing required config key: {key}")
    
    # B. Features Configuration
    required_feature_keys = [
        "features.indicators.rsi.window",
        "features.indicators.macd.fast",
        "features.indicators.macd.slow",
        "features.indicators.macd.signal",
        "features.indicators.bollinger_bands.window",
        "features.indicators.bollinger_bands.std_dev",
        "features.indicators.ema.windows",
        "features.indicators.sma.windows",
        "features.statistics.rolling_mean",
        "features.statistics.rolling_std",
        "features.denoising.enabled",
        "features.selection.pearson_threshold",
    ]
    
    for key in required_feature_keys:
        if OmegaConf.select(cfg, key) is None:
            errors.append(f"Missing required config key: {key}")
    
    # C. Model Configuration
    required_model_keys = [
        "model.architecture.hidden_size",
        "model.architecture.num_layers",
        "model.architecture.dropout",
        "model.training.optimizer",
        "model.training.learning_rate",
        "model.training.batch_size",
        "model.training.epochs",
        "model.training.early_stopping_patience",
    ]
    
    for key in required_model_keys:
        if OmegaConf.select(cfg, key) is None:
            errors.append(f"Missing required config key: {key}")
    
    # D. Optimization Configuration
    required_opt_keys = [
        "optimization.swarm.particles",
        "optimization.swarm.iterations",
        "optimization.swarm.cognitive_coeff",
        "optimization.swarm.social_coeff",
        "optimization.swarm.inertia_weight",
    ]
    
    for key in required_opt_keys:
        if OmegaConf.select(cfg, key) is None:
            errors.append(f"Missing required config key: {key}")
    
    # E. Path Configuration
    required_path_keys = [
        "paths.log_dir",
        "paths.model_save_path",
        "paths.results_dir",
        "paths.data_storage.raw",
        "paths.data_storage.interim",
        "paths.data_storage.processed",
    ]
    
    for key in required_path_keys:
        if not OmegaConf.select(cfg, key):
            errors.append(f"Missing required config key: {key}")
    
    # F. Value Validation
    if OmegaConf.select(cfg, "data.tickers"):
        if not isinstance(cfg.data.tickers, list) or len(cfg.data.tickers) == 0:
            errors.append("data.tickers must be a non-empty list")
    
    if OmegaConf.select(cfg, "data.validation.max_missing_pct"):
        if not (0 <= cfg.data.validation.max_missing_pct <= 1):
            errors.append("data.validation.max_missing_pct must be between 0 and 1")
    
    if OmegaConf.select(cfg, "model.training.learning_rate"):
        if cfg.model.training.learning_rate <= 0:
            errors.append("model.training.learning_rate must be positive")
    
    if OmegaConf.select(cfg, "model.training.batch_size"):
        if cfg.model.training.batch_size <= 0:
            errors.append("model.training.batch_size must be positive")
    
    if OmegaConf.select(cfg, "features.selection.pearson_threshold"):
        if not (0 <= cfg.features.selection.pearson_threshold <= 1):
            errors.append("features.selection.pearson_threshold must be between 0 and 1")
    
    # G. Seed Validation
    if not OmegaConf.select(cfg, "seed"):
        errors.append("Missing required config key: seed")
    
    # Report Results
    if errors:
        logger.error(f"Configuration validation FAILED with {len(errors)} errors:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("Configuration validation PASSED. All required keys present and valid.")
    return True


def validate_environment() -> bool:
    """
    Validates that required environment variables are set.
    
    Returns:
        bool: True if all required env vars exist, False otherwise.
    """
    import os
    
    logger.info("Validating environment variables...")
    
    required_vars = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.error("Please ensure .env file exists with ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return False
    
    logger.info("Environment validation PASSED.")
    return True
