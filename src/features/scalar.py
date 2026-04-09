from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler


class PipelineScaler:
    """
    Design decisions:

    1. [-1,1] over [0,1]:
       - tanh activation saturates at extremes. [-1,1] centered input
         better matches tanh range.
       - ReLU-based LSTMs are less sensitive, but [-1,1] remains convention.

    2. RobustScaler for features (not MinMax):
       - MinMax is highly sensitive to outliers even after Winsorization.
       - RobustScaler uses median and IQR — more stable for financial data.
       - EXCEPTION: use MinMax for the TARGET variable to preserve
         interpretability of predictions.

    3. Fit on TRAIN only, transform train/val/test.
       - This is what ALL four papers fail to explicitly guarantee.
    """

    def __init__(self):
        self.feature_scaler = RobustScaler()
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_cols = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "log_return",
        feature_cols: list = None,
    ):
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

    def save(self, path: str):
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
    def load(cls, path: str):
        import joblib

        obj = cls()
        data = joblib.load(path)
        obj.feature_scaler = data["feature"]
        obj.target_scaler = data["target"]
        obj.feature_cols = data["cols"]
        return obj
