"""Data loader with strict alignment validation for feature diagnostics."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pyarrow.parquet as pq


class FeatureDataset:
    """Strictly aligned feature matrix with metadata and OHLCV reconstruction."""

    def __init__(
        self, ticker: str, data_dir: Path, processed_dir: Path = Path("data/processed")
    ):
        self.ticker = ticker
        self.data_dir = data_dir / ticker
        self.processed_dir = processed_dir
        self._metadata: Optional[Dict] = None
        self._scaler = None

        # Core aligned data
        self.X_train: Optional[pd.DataFrame] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_val: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        # Raw OHLCV
        self.ohlcv: Optional[pd.DataFrame] = None

    def load(self) -> "FeatureDataset":
        """Load all artifacts with strict alignment validation."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Ticker directory not found: {self.data_dir}")

        # Load metadata first (contains feature names)
        with open(self.data_dir / "metadata.pkl", "rb") as f:
            self._metadata = pickle.load(f)

        feature_names = self._metadata["feature_names"]

        # Load feature matrices and indices
        splits = ["train", "val", "test"]
        for split in splits:
            X = np.load(self.data_dir / f"X_{split}.npy")
            y = np.load(self.data_dir / f"y_{split}.npy")
            idx = np.load(self.data_dir / f"{split}_index.npy")

            # Validate alignment
            assert (
                len(X) == len(y) == len(idx)
            ), f"{split}: Length mismatch X={len(X)}, y={len(y)}, idx={len(idx)}"

            # Create aligned DataFrames/Series
            df_X = pd.DataFrame(X, index=idx, columns=feature_names)
            series_y = pd.Series(y, index=idx, name="next_log_return")

            setattr(self, f"X_{split}", df_X)
            setattr(self, f"y_{split}", series_y)

        # Load OHLCV data
        parquet_path = self.processed_dir / f"{self.ticker}.parquet"
        if parquet_path.exists():
            self.ohlcv = pd.read_parquet(parquet_path)
            if self.ohlcv.index.tz is not None:
                self.ohlcv.index = self.ohlcv.index.tz_localize(None)

        return self

    def get_split(self, split: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Get X, y for specified split."""
        X = getattr(self, f"X_{split}")
        y = getattr(self, f"y_{split}")
        if X is None or y is None:
            raise ValueError(f"Split {split} not loaded")
        return X, y

    def get_feature_stats(self, split: str = "train") -> pd.DataFrame:
        """Compute comprehensive feature statistics."""
        X, _ = self.get_split(split)
        stats = X.describe().T
        stats["skew"] = X.skew()
        stats["kurt"] = X.kurtosis()
        stats["nan_pct"] = X.isna().mean() * 100
        stats["inf_pct"] = np.isinf(X).mean() * 100
        return stats

    @property
    def feature_names(self) -> List[str]:
        if self._metadata is None:
            return []
        return self._metadata["feature_names"]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def metadata(self) -> Dict:
        return self._metadata or {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_ticker_data(ticker: str, data_dir: str = "data/features") -> FeatureDataset:
    """Cached loader for ticker datasets."""
    dataset = FeatureDataset(ticker, Path(data_dir))
    return dataset.load()


def get_available_tickers(data_dir: Path = Path("data/features")) -> List[str]:
    """List all tickers with processed features."""
    if not data_dir.exists():
        return []
    return sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
