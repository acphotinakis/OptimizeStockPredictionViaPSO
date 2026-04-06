import inspect
import json
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.xgboost_model import XGBoostModel
from src.utils.config_loader import Config

logger = logging.getLogger(__name__)


def extract_hyperparameters(cfg: Config) -> Dict[str, Any]:
    """Extract XGBoost hyperparameters from config with defaults."""
    hp = getattr(cfg, "xgboost", {})

    defaults = {
        "objective": "multi:softprob",
        # "num_class": 3,
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 1e-5,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 50,
        "tree_method": "hist",
        "max_bin": 128,
    }

    return {key: hp.get(key, default) for key, default in defaults.items()}


def load_artefacts(results_dir: Path, ticker: str, mode: str, seed: int):
    """Load booster and metadata written by script 06."""
    tag = f"{ticker}_{mode}_seed{seed}"

    model_path = results_dir / f"xgb_model_{tag}.ubj"
    params_path = results_dir / f"xgb_params_{tag}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Booster not found: {model_path}\n"
            f"Run script 06 first:\n"
            f"  python scripts/06_train_xgboost.py --ticker {ticker} --mode {mode}"
        )

    with open(params_path) as f:
        meta = json.load(f)

    hparams = meta["hyperparameters"]
    valid_hparams = filter_valid_kwargs(XGBoostModel, hparams)
    model = XGBoostModel(**valid_hparams)
    model.load(str(model_path))
    return model, meta, tag


def filter_valid_kwargs(cls: type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to only include valid parameters for a class constructor."""
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def load_model(
    results_dir: Path, ticker: str, mode: str, seed: int
) -> Tuple[XGBoostModel, dict]:
    tag = f"{ticker}_{mode}_seed{seed}"
    model_path = results_dir / f"xgb_model_{tag}.ubj"
    params_path = results_dir / f"xgb_params_{tag}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Booster not found: {model_path}\n"
            f"Run script 06 first: python scripts/06_train_xgboost.py "
            f"--ticker {ticker} --mode {mode}"
        )

    with open(params_path) as f:
        meta = json.load(f)

    hparams = meta["hyperparameters"]

    # -------------------------------
    # Inspect-based safe filtering
    # -------------------------------
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
    """Load the threshold selected by script 07; fall back to 1e-4 if missing."""
    tag = f"{ticker}_{mode}_seed{seed}"
    path = results_dir / f"xgb_val_threshold_{tag}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        theta = float(data.get("optimal_threshold", 1e-4))
        print(f"  Optimal threshold (from script 07): {theta:.5f}")
        return theta
    print(
        "  WARNING: threshold file not found (run script 07 first). Falling back to θ=1e-4."
    )
    return 1e-4


def load_prices(
    ticker: str, features_dir: str, n_test: int
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    raw_path = Path("data/raw") / f"{ticker}.parquet"

    if not raw_path.exists():
        logger.warning(
            "Raw price file not found: %s. Using synthetic prices.", raw_path
        )

        y_test = np.load(Path(features_dir) / ticker / "y_test.npy")

        base_price = 100.0
        closes = base_price * np.exp(np.cumsum(y_test))
        closes = np.asarray(closes, dtype=np.float32)

        rng = np.random.default_rng(42)
        opens = np.roll(closes, 1)
        opens[0] = base_price
        opens = opens * (1 + rng.normal(0, 0.0001, len(opens)))
        opens = np.asarray(opens, dtype=np.float32)

        ts = pd.date_range(
            "2023-01-03 14:30", periods=len(closes), freq="1min", tz="UTC"
        )
        ts = pd.DatetimeIndex(ts)  # enforce exact type

        return opens, closes, ts

    from src.data import AlpacaIngestor, DataCleaner, DataSplitter

    raw = AlpacaIngestor.load_bars(raw_path)
    df = DataCleaner().clean(raw)
    _, _, df_test = DataSplitter(train_end="2022-01-03", val_end="2023-01-03").split(df)

    # ---------------------------
    # FORCE STRICT TYPES
    # ---------------------------
    opens = np.asarray(df_test["open"].to_numpy(), dtype=np.float32)
    closes = np.asarray(df_test["close"].to_numpy(), dtype=np.float32)

    ts = df_test.index
    if not isinstance(ts, pd.DatetimeIndex):
        ts = pd.DatetimeIndex(ts)

    # Align lengths
    if len(opens) > n_test:
        opens = opens[-n_test:]
        closes = closes[-n_test:]
        ts = ts[-n_test:]

    return opens, closes, ts


# ======================================================================
# Cross-model comparison table helpers
# ======================================================================


def load_existing_results(results_dir: Path, ticker: str, seed: int) -> dict:
    """Load script-04 statistical metrics if available."""
    path = results_dir / f"metrics_{ticker}_seed{seed}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"ticker": ticker, "seed": seed, "models": {}}


def load_existing_backtest(results_dir: Path, ticker: str, seed: int) -> dict:
    """Load script-05 trading metrics if available."""
    path = results_dir / f"backtest_metrics_{ticker}_seed{seed}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}
