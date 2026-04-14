import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import json
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import psutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment.usage_enums import Phase
from src.models.xgboost.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup(log_file: str, seed: int, config_path: str):
    """Bootstrap logging, seeds, and config in one call."""
    from src.utils import set_all_seeds, setup_logger
    from src.utils.config_loader import load_config

    Path("logs").mkdir(exist_ok=True)
    setup_logger(log_file=log_file, level="INFO")
    set_all_seeds(seed)
    return load_config(config_path)


def make_tag(ticker: str, mode: str, seed: int) -> str:
    return f"{ticker}_{mode}_seed{seed}"


# ---------------------------------------------------------------------------
# Data Saving
# ---------------------------------------------------------------------------


def save_json(path: Path, obj: dict) -> None:
    """JSON-serialise numpy scalars/arrays transparently."""
    import datetime

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, datetime.datetime)):
            return o.isoformat()
        return str(o)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_split(ticker_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load X_{split}.npy and y_{split}.npy for a ticker."""
    X = np.load(ticker_dir / f"X_{split}.npy")
    y = np.load(ticker_dir / f"y_{split}.npy")
    return X, y


# ---------------------------------------------------------------------------
# Artefact I/O (XGBoost)
# ---------------------------------------------------------------------------


def load_xgb_model(results_dir: Path, ticker: str, mode: str, seed: int):
    """Load a saved XGBoostModel and its metadata."""
    import inspect
    from src.models.xgboost.xgboost_model import XGBoostModel

    tag = make_tag(ticker, mode, seed)
    model_path = results_dir / f"xgb_model_{tag}.ubj"
    params_path = results_dir / f"xgb_params_{tag}.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")

    meta = json.loads(params_path.read_text())
    valid = {
        p for p in inspect.signature(XGBoostModel.__init__).parameters if p != "self"
    }
    model = XGBoostModel(
        **{k: v for k, v in meta["hyperparameters"].items() if k in valid}
    )
    model.load(str(model_path))
    return model, meta


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
    results_dir: Path, ticker: str, mode: str, seed: int, model_name: str, phase: Phase
) -> float:
    """Load the threshold selected during validation."""
    tag = f"{ticker}_{mode}_seed{seed}"
    path = results_dir / f"{model_name}_{phase}_threshold_{tag}.json"
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


# ---------------------------------------------------------------------------
# Log Messages
# ---------------------------------------------------------------------------


def _log_table_stats(alpaca_ingestor, raw_dir, tickers):
    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = [
        "Ticker",
        "Rows",
        "Start",
        "End",
        "Mean Close",
        "Std Close",
        "Mean Vol",
        "Std Vol",
        "Missing %",
    ]

    stats_summary = []

    for ticker in tickers:
        path = raw_dir / f"{ticker}.parquet"
        if not path.exists():
            logger.warning("Missing raw data for %s", ticker)
            continue

        df = alpaca_ingestor.load_bars(path)

        # Ensure datetime index
        df.index = pd.to_datetime(df.index)

        rows = len(df)
        start = df.index.min()
        end = df.index.max()

        mean_close = df["close"].mean()
        std_close = df["close"].std()

        mean_vol = df["volume"].mean()
        std_vol = df["volume"].std()

        missing = df.isna().mean().mean()

        stats_summary.append(
            {"ticker": ticker, "mean_close": mean_close, "std_close": std_close}
        )

        table.add_row(
            [
                ticker,
                rows,
                str(start),
                str(end),
                f"{mean_close:.2f}",
                f"{std_close:.2f}",
                f"{mean_vol:.2f}",
                f"{std_vol:.2f}",
                f"{missing:.4%}",
            ]
        )

    logger.info("\n%s", table)

    # ---- Cross-ticker comparison ----
    df_stats = pd.DataFrame(stats_summary)

    logger.info("\n=== Cross-Ticker Dispersion ===")
    logger.info(
        "Mean Close (min/max): %.2f / %.2f",
        df_stats["mean_close"].min(),
        df_stats["mean_close"].max(),
    )

    logger.info(
        "Std Close (min/max): %.2f / %.2f",
        df_stats["std_close"].min(),
        df_stats["std_close"].max(),
    )
