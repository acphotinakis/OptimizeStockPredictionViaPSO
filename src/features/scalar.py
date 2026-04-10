"""features/scalar.py — RobustScaler for features + MinMaxScaler for target."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler


class PipelineScaler:
    """Fit on training data only; transform train / val / test.

    Features: RobustScaler (median + IQR — stable for financial data).
    Target:   MinMaxScaler → [-1, 1] (preserves interpretability).
    """

    def __init__(self) -> None:
        self.feature_scaler = RobustScaler()
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_cols: list[str] | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "log_return",
        feature_cols: list[str] | None = None,
    ) -> "PipelineScaler":
        self.feature_cols = feature_cols or [
            c for c in train_df.columns if c != target_col
        ]
        self.feature_scaler.fit(train_df[self.feature_cols])
        self.target_scaler.fit(train_df[[target_col]])
        return self

    def transform(
        self, df: pd.DataFrame, target_col: str = "log_return"
    ) -> pd.DataFrame:
        out = df.copy()
        out[self.feature_cols] = self.feature_scaler.transform(df[self.feature_cols])
        out[[target_col]] = self.target_scaler.transform(df[[target_col]])
        return out

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(values.reshape(-1, 1)).flatten()

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(
            {
                "feature": self.feature_scaler,
                "target": self.target_scaler,
                "cols": self.feature_cols,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "PipelineScaler":
        import joblib

        obj = cls()
        data = joblib.load(path)
        obj.feature_scaler, obj.target_scaler, obj.feature_cols = (
            data["feature"],
            data["target"],
            data["cols"],
        )
        return obj
