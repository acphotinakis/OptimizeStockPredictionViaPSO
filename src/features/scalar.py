"""features/scalar.py — RobustScaler for features only. Target (log_return) is NOT scaled.

Log returns are already near-zero mean with small variance; scaling them with MinMaxScaler
clips out-of-sample extremes to [-1,1] and distorts val/test metrics. Features use
RobustScaler (median + IQR) which is robust to the fat tails common in financial data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class PipelineScaler:
    """Fit on training data only; transform train / val / test.

    Features: RobustScaler (median + IQR — stable for financial data).
    Target:   NOT scaled. Log returns are left in their natural units so that
              val/test returns outside the training range are never clipped.
    """

    def __init__(self) -> None:
        self.feature_scaler = RobustScaler()
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
        return self

    def transform(
        self, df: pd.DataFrame, target_col: str = "log_return"
    ) -> pd.DataFrame:
        out = df.copy()
        out[self.feature_cols] = self.feature_scaler.transform(df[self.feature_cols])
        # Target column is passed through unchanged
        return out

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        """No-op: target is not scaled, so inverse transform is identity."""
        return values.copy()

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(
            {
                "feature": self.feature_scaler,
                "cols": self.feature_cols,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "PipelineScaler":
        import joblib

        obj = cls()
        data = joblib.load(path)
        obj.feature_scaler = data["feature"]
        obj.feature_cols = data["cols"]
        return obj
