import sys
import logging
import typer
from pathlib import Path
from hydra import initialize, compose

# Import our custom modules
from src.data.data_ingestion import AlpacaIngestor
from src.data.data_validation import run_validation_suite
from src.data.data_cleaning import run_cleaning_pipeline
from src.features.build_features import calculate_indicators
from src.features.wavelet_denoising import apply_denoising_pipeline
from src.features.selection import run_selection_pipeline
from src.features.scaling import fit_scaler, transform_data, save_scaler, validate_inverse_transform
from src.features.target import create_target_variable, validate_target
from src.data.split import time_series_split
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.paths import ProjectPaths
from src.utils.config_validator import validate_config, validate_environment

# Initialize CLI app
app = typer.Typer(help="PSO-LSTM Stock Tuner Execution Pipeline")


def load_config():
    """Initializes Hydra and loads the global config."""
    with initialize(version_base=None, config_path="configs"):
        return compose(config_name="config")


@app.command()
def ingest():
    """Phase 1: Fetch raw OHLCV data from Alpaca."""
    cfg = load_config()
    logger = setup_logger("ingest", cfg.paths.log_dir)
    
    if not validate_config(cfg):
        logger.error("Configuration validation failed. Aborting.")
        raise SystemExit(1)
    
    if not validate_environment():
        logger.error("Environment validation failed. Aborting.")
        raise SystemExit(1)
    
    set_seed(cfg.seed)
    ProjectPaths(cfg).ensure_directories()

    logger.info("--- Starting Data Ingestion Phase ---")
    ingestor = AlpacaIngestor(cfg)
    ingestor.run(cfg.data.tickers)
    logger.info("--- Ingestion Complete ---")


@app.command()
def process():
    """Phase 2: Validate, Clean, and Engineer Features."""
    cfg = load_config()
    logger = setup_logger("process", cfg.paths.log_dir)
    
    if not validate_config(cfg):
        logger.error("Configuration validation failed. Aborting.")
        raise SystemExit(1)
    
    set_seed(cfg.seed)
    paths = ProjectPaths(cfg)

    import pandas as pd

    logger.info("--- Starting Data Processing Phase ---")
    for ticker in cfg.data.tickers:
        logger.info(f"Processing ticker: {ticker}")
        raw_file = paths.get_raw_file_path(ticker)

        if not raw_file.exists():
            logger.warning(f"Raw data missing for {ticker}. Skipping.")
            continue

        df = pd.read_parquet(raw_file)

        # 1. Validation & Cleaning
        val_report = run_validation_suite(df, cfg)
        if not val_report["is_fit_for_training"]:
            logger.error(f"Data validation failed for {ticker}: {val_report['failed_reasons']}")
            continue
        df_clean = run_cleaning_pipeline(df, cfg)

        # 2. Denoising FIRST (before calculating indicators to reduce noise impact)
        df_denoised = apply_denoising_pipeline(df_clean, cfg.features.denoising)

        # 3. Feature Engineering on denoised data
        df_feats = calculate_indicators(df_denoised, cfg.features)

        # 4. Create target variable (mid-price return for next time step)
        target_method = getattr(cfg.features, "target_method", "mid_price_return")
        df_with_target = create_target_variable(df_feats, method=target_method, horizon=1)
        
        if not validate_target(df_with_target):
            logger.error(f"Target validation failed for {ticker}. Skipping.")
            continue

        # 5. Split data BEFORE feature selection to prevent leakage
        train_df, val_df, test_df = time_series_split(df_with_target, cfg)
        
        # 6. Feature Selection ONLY on training data
        train_selected = run_selection_pipeline(train_df, cfg.features.selection, selected_features=None)
        selected_feature_names = [col for col in train_selected.columns if col != "target"]
        
        # Apply same features to val/test sets
        val_selected = run_selection_pipeline(val_df, cfg.features.selection, selected_features=selected_feature_names)
        test_selected = run_selection_pipeline(test_df, cfg.features.selection, selected_features=selected_feature_names)
        
        # 7. Scaling - fit ONLY on training data
        logger.info("Fitting scaler on training data only...")
        train_features = train_selected.drop(columns=["target"]) if "target" in train_selected.columns else train_selected
        scaler = fit_scaler(train_features)
        
        # Validate inverse transform
        if not validate_inverse_transform(train_features, scaler):
            logger.error(f"Inverse transform validation failed for {ticker}")
            continue
        
        # Transform all splits
        train_scaled = transform_data(train_features, scaler)
        val_features = val_selected.drop(columns=["target"]) if "target" in val_selected.columns else val_selected
        test_features = test_selected.drop(columns=["target"]) if "target" in test_selected.columns else test_selected
        val_scaled = transform_data(val_features, scaler)
        test_scaled = transform_data(test_features, scaler)
        
        # Re-attach targets
        if "target" in train_selected.columns:
            train_scaled["target"] = train_selected["target"]
            val_scaled["target"] = val_selected["target"]
            test_scaled["target"] = test_selected["target"]

        # Save processed data and scaler
        save_path_train = paths.processed_data / f"{ticker}_train.parquet"
        save_path_val = paths.processed_data / f"{ticker}_val.parquet"
        save_path_test = paths.processed_data / f"{ticker}_test.parquet"
        
        train_scaled.to_parquet(save_path_train)
        val_scaled.to_parquet(save_path_val)
        test_scaled.to_parquet(save_path_test)
        
        scaler_path = Path("models/scalers") / f"{ticker}_scaler.joblib"
        save_scaler(scaler, str(scaler_path))
        
        logger.info(f"Saved processed splits for {ticker}")
        logger.info(f"  Train: {save_path_train} ({len(train_scaled)} samples)")
        logger.info(f"  Val:   {save_path_val} ({len(val_scaled)} samples)")
        logger.info(f"  Test:  {save_path_test} ({len(test_scaled)} samples)")


@app.command()
def optimize():
    """Phase 3: Run the PSO Swarm to tune the LSTM."""
    cfg = load_config()
    logger = setup_logger("optimize", cfg.paths.log_dir)
    
    if not validate_config(cfg):
        logger.error("Configuration validation failed. Aborting.")
        raise SystemExit(1)
    
    set_seed(cfg.seed)

    # Imports scoped to avoid loading PyTorch overhead during simple data ingestion
    from src.optimization.pso import ImprovedPSO
    import torch

    logger.info("--- Starting PSO-LSTM Optimization Phase ---")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device: {device}")

    # Note: In a full execution, you would load your DataLoaders here
    # using the `src.data.dataset` and `src.data.split` modules.
    train_loader, val_loader = None, None  # Placeholder for data loaders

    pso = ImprovedPSO(
        cfg, cfg.optimization.search_space, train_loader, val_loader, device
    )
    best_config = pso.search()

    logger.info(f"--- Optimization Complete ---")
    logger.info(f"Best Hyperparameters: {best_config}")


if __name__ == "__main__":
    app()
