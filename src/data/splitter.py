"""
src/data/splitter.py

Chronological (no-shuffle) train / validation / test split for time-series data.
Does NOT scale data. Use PipelineScaler separately to avoid leakage.
"""

from __future__ import annotations

import logging
from typing import Tuple
import numpy as np
import pandas as pd
from prettytable import PrettyTable

logger = logging.getLogger(__name__)


class DataSplitter:
    """Chronological split by fixed date boundaries (preferred) or percentages.

    Fixed date splits are strongly preferred — percentage splits shift when new
    data is appended, causing silent train/test contamination.  Pass ``train_end``
    and ``val_end`` to use date-based splitting; fall back to percentages only
    when no dates are provided.

    Args:
        train_end: ISO date string for the end of the training period (inclusive).
                   E.g. ``"2024-04-05"``.  If supplied, ``val_end`` is also required.
        val_end:   ISO date string for the end of the validation period (inclusive).
        train_pct: Fraction of data for training (default: 0.6).  Used only when
                   ``train_end`` / ``val_end`` are not provided.
        val_pct:   Fraction of data for validation (default: 0.2).
        test_pct:  Fraction of data for testing (default: 0.2).
    """

    def __init__(
        self,
        train_end: str | None = None,
        val_end: str | None = None,
        train_pct: float = 0.6,
        val_pct: float = 0.2,
        test_pct: float = 0.2,
    ) -> None:
        self._use_dates = train_end is not None and val_end is not None
        if self._use_dates:
            self.train_end = pd.Timestamp(train_end)
            self.val_end = pd.Timestamp(val_end)
        else:
            assert (
                abs(train_pct + val_pct + test_pct - 1.0) < 1e-6
            ), "Percentages must sum to 1"
            self.train_pct = train_pct
            self.val_pct = val_pct
            self.test_pct = test_pct

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame chronologically into (train, val, test)."""
        if self._use_dates:
            logger.info(
                f"Splitting by dates: train_end={self.train_end}, val_end={self.val_end}"
            )
            return self._split_by_dates(df)
        logger.info(
            f"Splitting by percentages: train_pct={self.train_pct}, val_pct={self.val_pct}, test_pct={self.test_pct}"
        )
        return self._split_by_pct(df)

    def _split_by_dates(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fixed date-boundary split with explicit America/New_York normalization.
        """

        df = df.copy()

        # --- FORCE timezone to America/New_York ---
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")

        idx = df.index

        # --- Ensure split boundaries are also in America/New_York ---
        if self.train_end.tz is None:
            train_end = self.train_end.tz_localize("America/New_York")
        else:
            train_end = self.train_end.tz_convert("America/New_York")

        if self.val_end.tz is None:
            val_end = self.val_end.tz_localize("America/New_York")
        else:
            val_end = self.val_end.tz_convert("America/New_York")

        # --- Perform split ---
        df_train = df.loc[idx <= train_end]
        df_val = df.loc[(idx > train_end) & (idx <= val_end)]
        df_test = df.loc[idx > val_end]

        logger.info(
            "Date split — Train: %d (%s–%s)  Val: %d (%s–%s)  Test: %d (%s–%s)",
            len(df_train),
            df_train.index.min() if len(df_train) else "N/A",
            df_train.index.max() if len(df_train) else "N/A",
            len(df_val),
            df_val.index.min() if len(df_val) else "N/A",
            df_val.index.max() if len(df_val) else "N/A",
            len(df_test),
            df_test.index.min() if len(df_test) else "N/A",
            df_test.index.max() if len(df_test) else "N/A",
        )

        return df_train, df_val, df_test

    def _split_by_pct(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Percentage-based split — only use when fixed dates are unavailable."""
        N = len(df)
        train_end_idx = int(N * self.train_pct)
        val_end_idx = train_end_idx + int(N * self.val_pct)

        df_train = df.iloc[:train_end_idx]
        df_val = df.iloc[train_end_idx:val_end_idx]
        df_test = df.iloc[val_end_idx:]

        logger.info(
            "Pct split — Train: %d  Val: %d  Test: %d",
            len(df_train),
            len(df_val),
            len(df_test),
        )
        return df_train, df_val, df_test

    def inspect_split(self, name: str, dfs: dict, n_head: int = 3, n_tail: int = 3):
        """
        Prints:
        - number of tickers
        - per-ticker shape
        - columns
        - head (first 3 rows)
        - tail (last 3 rows)
        """
        header = f"\n{'=' * 80}\n{name} | tickers={len(dfs)}\n{'=' * 80}"
        (logger.info if logger else print)(header)

        for ticker, df in dfs.items():
            msg = f"\n[{ticker}] shape={df.shape}"
            (logger.info if logger else print)(msg)

            # columns
            cols = list(df.columns)
            (logger.info if logger else print)(f"Columns ({len(cols)}): {cols}")

            # head
            head = df.head(n_head)
            tail = df.tail(n_tail)

            head_table = PrettyTable()
            head_table.title = f"{ticker} | FIRST {n_head} ROWS"
            head_table.field_names = ["index"] + cols

            for idx, row in head.iterrows():
                head_table.add_row([idx] + list(row.values))

            tail_table = PrettyTable()
            tail_table.title = f"{ticker} | LAST {n_tail} ROWS"
            tail_table.field_names = ["index"] + cols

            for idx, row in tail.iterrows():
                tail_table.add_row([idx] + list(row.values))

            output = f"{head_table}\n\n{tail_table}\n"
            (logger.info if logger else print)(output)


# """
# src/data/splitter.py

# Chronological (no-shuffle) train / validation / test split for time-series data.
# Does NOT scale data. Use PipelineScaler separately to avoid leakage.
# """

# from __future__ import annotations

# import logging
# from typing import Tuple
# import numpy as np
# import pandas as pd
# from prettytable import PrettyTable

# logger = logging.getLogger(__name__)


# class DataSplitter:
#     """Chronological split by fixed date boundaries (preferred) or percentages.

#     Fixed date splits are strongly preferred — percentage splits shift when new
#     data is appended, causing silent train/test contamination.  Pass ``train_end``
#     and ``val_end`` to use date-based splitting; fall back to percentages only
#     when no dates are provided.

#     Args:
#         train_end: ISO date string for the end of the training period (inclusive).
#                    E.g. ``"2024-04-05"``.  If supplied, ``val_end`` is also required.
#         val_end:   ISO date string for the end of the validation period (inclusive).
#         train_pct: Fraction of data for training (default: 0.6).  Used only when
#                    ``train_end`` / ``val_end`` are not provided.
#         val_pct:   Fraction of data for validation (default: 0.2).
#         test_pct:  Fraction of data for testing (default: 0.2).
#     """

#     def __init__(
#         self,
#         train_end: str | None = None,
#         val_end: str | None = None,
#         train_pct: float = 0.6,
#         val_pct: float = 0.2,
#         test_pct: float = 0.2,
#     ) -> None:
#         self._use_dates = train_end is not None and val_end is not None
#         if self._use_dates:
#             self.train_end = pd.Timestamp(train_end)
#             self.val_end = pd.Timestamp(val_end)
#         else:
#             assert (
#                 abs(train_pct + val_pct + test_pct - 1.0) < 1e-6
#             ), "Percentages must sum to 1"
#             self.train_pct = train_pct
#             self.val_pct = val_pct
#             self.test_pct = test_pct

#     def split(
#         self, df: pd.DataFrame
#     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#         """Split DataFrame chronologically into (train, val, test)."""
#         if self._use_dates:
#             logger.info(
#                 f"Splitting by dates: train_end={self.train_end}, val_end={self.val_end}"
#             )
#             return self._split_by_dates(df)
#         logger.info(
#             f"Splitting by percentages: train_pct={self.train_pct}, val_pct={self.val_pct}, test_pct={self.test_pct}"
#         )
#         return self._split_by_pct(df)

#     def _split_by_dates(
#         self, df: pd.DataFrame
#     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#         """Fixed date-boundary split — reproducible even when data is updated."""
#         idx = df.index
#         if hasattr(idx, "tz") and idx.tz is not None:
#             train_end = (
#                 self.train_end.tz_localize(idx.tz)
#                 if self.train_end.tz is None
#                 else self.train_end.tz_convert(idx.tz)
#             )
#             val_end = (
#                 self.val_end.tz_localize(idx.tz)
#                 if self.val_end.tz is None
#                 else self.val_end.tz_convert(idx.tz)
#             )
#         else:
#             train_end = self.train_end.tz_localize(None)
#             val_end = self.val_end.tz_localize(None)

#         df_train = df.loc[idx <= train_end]
#         df_val = df.loc[(idx > train_end) & (idx <= val_end)]
#         df_test = df.loc[idx > val_end]

#         logger.info(
#             "Date split — Train: %d (%s–%s)  Val: %d (%s–%s)  Test: %d (%s–%s)",
#             len(df_train),
#             df_train.index.min() if len(df_train) else "N/A",
#             df_train.index.max() if len(df_train) else "N/A",
#             len(df_val),
#             df_val.index.min() if len(df_val) else "N/A",
#             df_val.index.max() if len(df_val) else "N/A",
#             len(df_test),
#             df_test.index.min() if len(df_test) else "N/A",
#             df_test.index.max() if len(df_test) else "N/A",
#         )
#         return df_train, df_val, df_test

#     def _split_by_pct(
#         self, df: pd.DataFrame
#     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#         """Percentage-based split — only use when fixed dates are unavailable."""
#         N = len(df)
#         train_end_idx = int(N * self.train_pct)
#         val_end_idx = train_end_idx + int(N * self.val_pct)

#         df_train = df.iloc[:train_end_idx]
#         df_val = df.iloc[train_end_idx:val_end_idx]
#         df_test = df.iloc[val_end_idx:]

#         logger.info(
#             "Pct split — Train: %d  Val: %d  Test: %d",
#             len(df_train),
#             len(df_val),
#             len(df_test),
#         )
#         return df_train, df_val, df_test

#     def inspect_split(self, name: str, dfs: dict, n_head: int = 3, n_tail: int = 3):
#         """
#         Prints:
#         - number of tickers
#         - per-ticker shape
#         - columns
#         - head (first 3 rows)
#         - tail (last 3 rows)
#         """
#         header = f"\n{'=' * 80}\n{name} | tickers={len(dfs)}\n{'=' * 80}"
#         (logger.info if logger else print)(header)

#         for ticker, df in dfs.items():
#             msg = f"\n[{ticker}] shape={df.shape}"
#             (logger.info if logger else print)(msg)

#             # columns
#             cols = list(df.columns)
#             (logger.info if logger else print)(f"Columns ({len(cols)}): {cols}")

#             # head
#             head = df.head(n_head)
#             tail = df.tail(n_tail)

#             head_table = PrettyTable()
#             head_table.title = f"{ticker} | FIRST {n_head} ROWS"
#             head_table.field_names = ["index"] + cols

#             for idx, row in head.iterrows():
#                 head_table.add_row([idx] + list(row.values))

#             tail_table = PrettyTable()
#             tail_table.title = f"{ticker} | LAST {n_tail} ROWS"
#             tail_table.field_names = ["index"] + cols

#             for idx, row in tail.iterrows():
#                 tail_table.add_row([idx] + list(row.values))

#             output = f"{head_table}\n\n{tail_table}\n"
#             (logger.info if logger else print)(output)
