#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import logging
import numpy as np
import psutil
import os
import json
import sys
from pathlib import Path
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from src.utils.config_schema import Config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# from pipelines.train_xgboost import run_train
# from pipelines.validate_xgboost import run_val
# from pipelines.test_xgboost import run_test
from src.utils import set_all_seeds, setup_logger
from src.data.splitter import build_windows
from src.utils.config_loader import load_config
from src.models.xgboost.xgboost_model import XGBoostModel, XGBoostTuner

logger = logging.getLogger(__name__)


# ======================================================================
# Utilities
# ======================================================================


def log_memory_usage(label: str):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"[{label}] Memory: {mem_info.rss / 1024**3:.2f} GB")


def extract_hyperparameters(cfg: Config) -> Dict[str, Any]:
    """Extract XGBoost hyperparameters from config with defaults."""
    from src.models.xgboost.xgboost_model import _DEFAULT_PARAMS

    hp = getattr(cfg, "xgboost", {})
    return {key: hp.get(key, default) for key, default in _DEFAULT_PARAMS.items()}


def load_artefacts(results_dir: Path, ticker: str, mode: str, seed: int):
    """Load booster and metadata written by training pipeline."""
    tag = f"{ticker}_{mode}_seed{seed}"

    model_path = results_dir / f"xgb_model_{tag}.ubj"
    params_path = results_dir / f"xgb_params_{tag}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Booster not found: {model_path}\n" f"Run training first"
        )

    with open(params_path) as f:
        meta = json.load(f)

    hparams = meta["hyperparameters"]

    import inspect

    sig = inspect.signature(XGBoostModel.__init__)
    valid_keys = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    model = XGBoostModel(**{k: v for k, v in hparams.items() if k in valid_keys})
    model.load(str(model_path))

    return model, meta, tag


def load_model(
    results_dir: Path, ticker: str, mode: str, seed: int
) -> Tuple[XGBoostModel, dict]:
    """Load trained model and metadata."""
    tag = f"{ticker}_{mode}_seed{seed}"
    model_path = results_dir / f"xgb_model_{tag}.ubj"
    params_path = results_dir / f"xgb_params_{tag}.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Booster not found: {model_path}")

    with open(params_path) as f:
        meta = json.load(f)

    hparams = meta["hyperparameters"]

    import inspect

    sig = inspect.signature(XGBoostModel.__init__)
    valid_keys = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    model = XGBoostModel(**{k: v for k, v in hparams.items() if k in valid_keys})
    model.load(str(model_path))

    return model, meta


def load_optimal_threshold(
    results_dir: Path, ticker: str, mode: str, seed: int
) -> float:
    """Load the threshold selected during validation."""
    tag = f"{ticker}_{mode}_seed{seed}"
    path = results_dir / f"xgb_val_threshold_{tag}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        theta = float(data.get("optimal_threshold", 1e-4))
        logger.info(f"  Optimal threshold (from validation): {theta:.5f}")
        return theta
    logger.info("  WARNING: threshold file not found. Falling back to θ=1e-4.")
    return 1e-4


def load_prices(
    ticker: str, features_dir: str, n_test: int
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load price data for backtesting."""
    raw_path = Path("data/raw") / f"{ticker}.parquet"

    if not raw_path.exists():
        raise ValueError("Raw data doesn't exist")

    from src.data import AlpacaIngestor, DataCleaner, DataSplitter

    raw = AlpacaIngestor._load_bars(raw_path)
    df = DataCleaner().clean(raw)
    _, _, df_test = DataSplitter().split(df)

    opens = np.asarray(df_test["open"].to_numpy(), dtype=np.float32)
    closes = np.asarray(df_test["close"].to_numpy(), dtype=np.float32)

    ts = df_test.index
    if not isinstance(ts, pd.DatetimeIndex):
        ts = pd.DatetimeIndex(ts)

    if len(opens) > n_test:
        opens = opens[-n_test:]
        closes = closes[-n_test:]
        ts = ts[-n_test:]

    return opens, closes, ts
