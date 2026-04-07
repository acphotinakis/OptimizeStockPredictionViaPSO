from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.models.lstm.lstm_model import LSTMModel, LSTMTrainer


# ======================================================================
# Helpers
# ======================================================================


def load_windows(features_dir: Path, ticker: str):
    """Load pre-built numpy arrays from script 02."""
    prefix = features_dir / ticker

    file_map = {
        "X_train": prefix / "X_train.npy",
        "X_val": prefix / "X_val.npy",
        "X_test": prefix / "X_test.npy",
        "y_train": prefix / "y_train.npy",
        "y_val": prefix / "y_val.npy",
        "y_test": prefix / "y_test.npy",
    }

    arrays = {}

    for key, path in file_map.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run script 02 first:\n"
                f"  python scripts/02_build_features.py --target {ticker}"
            )
        arrays[key] = np.load(path)

    return (
        arrays["X_train"],
        arrays["y_train"],
        arrays["X_val"],
        arrays["y_val"],
        arrays["X_test"],
        arrays["y_test"],
    )


_DEFAULT_PARAMS = {
    # Fixed hyperparameters (no PSO optimization)
    "num_layers": 2,
    "hidden_units": 128,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "lookback": 30,
    "max_epochs": 100,
    "patience": 10,
    "batch_size": 256,
    # Training settings
    "grad_clip": 1.0,
    "use_amp": True,
    "accumulation_steps": 1,
    # Walk-forward validation
    "wfv_fold_size": 252,
    "wfv_folds": 10,
    # Backtesting
    "initial_capital": 100000.0,
    "position_fraction": 0.02,
    "transaction_cost": 0.001,
    "slippage": 0.0005,
    "stop_loss": 0.02,
    "daily_loss_limit": 0.05,
}


class LSTMBaseline:
    """LSTM with manually chosen hyperparameters (no PSO optimization).

    Serves as the 'standard LSTM' baseline from the experiment plan.
    Default values reflect common choices in the literature.

    Args:
        input_size: Feature dimension F.
        num_layers: 2 (Ji et al. default; Zeng et al. best result).
        hidden_units: 128.
        dropout: 0.2.
        learning_rate: 0.001 (Adam default).
        lookback: 30 minutes (Lanbouri & Achchab optimal short-term window).
        max_epochs: 100.
        patience: 10.
        batch_size: 256.
        device: Torch device.
    """

    def __init__(
        self,
        input_size: int,
        device: Optional[str] = None,
        **overrides,
    ) -> None:
        params = {**_DEFAULT_PARAMS, **overrides}
        self.lookback = params["lookback"]

        model = LSTMModel(
            input_size=input_size,
            num_layers=params["num_layers"],
            hidden_units=params["hidden_units"],
            dropout=params["dropout"],
            use_checkpointing=params.get("use_checkpointing", False),
        )
        self._trainer = LSTMTrainer(
            model=model,
            lr=params["learning_rate"],
            max_epochs=params["max_epochs"],
            patience=params["patience"],
            batch_size=params["batch_size"],
            device=device,
            grad_clip=params["grad_clip"],
            use_amp=params["use_amp"],
            accumulation_steps=params["accumulation_steps"],
        )

    def fit(self, X_train, y_train, X_val, y_val):
        return self._trainer.fit(X_train, y_train, X_val, y_val)

    def predict(self, X) -> np.ndarray:
        return self._trainer.predict(X)
