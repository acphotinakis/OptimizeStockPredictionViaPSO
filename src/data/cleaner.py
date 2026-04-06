"""
src/data/cleaner.py

Session filtering, missing value imputation, OHLC consistency checks,
zero-volume removal, and return outlier clipping for 1-minute OHLCV data.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SESSION_START = "09:30"
SESSION_END = "15:59"
MAX_GAP_FILL_BARS = 5  # Forward-fill at most this many missing bars
OUTLIER_ZSCORE_THRESHOLD = 5  # Clip returns beyond ±5σ
OUTLIER_ROLLING_WINDOW = 60  # Rolling window (bars) for z-score computation


class DataCleaner:
    """Cleans a single-ticker 1-minute OHLCV DataFrame.

    Steps applied in order:
        1. Session filtering (09:30 – 15:59 ET)
        2. Zero-volume bar removal
        3. OHLC consistency check
        4. Reindex to full session minute grid
        5. Forward-fill short gaps (≤ 5 bars)
        6. Drop long gaps (> 5 bars) and flag post-gap bars
        7. Return outlier clipping (±5σ rolling)
        8. Add derived columns: log_return, session_minute, session_start flag
    """

    def __init__(
        self,
        session_start: str = SESSION_START,
        session_end: str = SESSION_END,
        max_gap_fill: int = MAX_GAP_FILL_BARS,
        outlier_z: float = OUTLIER_ZSCORE_THRESHOLD,
        outlier_window: int = OUTLIER_ROLLING_WINDOW,
    ) -> None:
        self.session_start = pd.Timestamp(session_start).time()
        self.session_end = pd.Timestamp(session_end).time()
        self.max_gap_fill = max_gap_fill
        self.outlier_z = outlier_z
        self.outlier_window = outlier_window

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full cleaning pipeline.

        Args:
            df: Raw OHLCV DataFrame with a UTC DatetimeIndex.

        Returns:
            Cleaned DataFrame with additional derived columns.
        """
        df = df.copy()
        df = self._ensure_utc(df)
        df = self._session_filter(df)
        df = self._remove_zero_volume(df)
        df = self._ohlc_consistency(df)
        df = self._reindex_and_fill(df)
        df = self._clip_outliers(df)
        df = self._add_derived(df)
        logger.debug("Cleaned DataFrame: %d rows, %d columns", len(df), len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df

    def _session_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only bars within NYSE regular trading hours (ET)."""
        df_et = df.copy()
        df_et.index = df.index.tz_convert("America/New_York")
        mask = (df_et.index.time >= self.session_start) & (
            df_et.index.time <= self.session_end
        )
        df_filtered = df.loc[mask]
        dropped = len(df) - len(df_filtered)
        if dropped:
            logger.debug("Session filter: dropped %d pre/post-market bars", dropped)
        return df_filtered

    @staticmethod
    def _remove_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[df["volume"] > 0]
        logger.debug("Zero-volume removal: dropped %d bars", before - len(df))
        return df

    @staticmethod
    def _ohlc_consistency(df: pd.DataFrame) -> pd.DataFrame:
        """Remove bars where OHLC ordering is violated."""
        mask = (
            (df["low"] <= df["open"])
            & (df["open"] <= df["high"])
            & (df["low"] <= df["close"])
            & (df["close"] <= df["high"])
            & (df[["open", "high", "low", "close"]] > 0).all(axis=1)
        )
        dropped = (~mask).sum()
        if dropped:
            logger.debug("OHLC consistency: dropped %d invalid bars", dropped)
        return df[mask]

    def _reindex_and_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex to a complete 1-min session grid; forward-fill short gaps."""
        # Build complete minute grid for every trading day in the data
        days = df.index.normalize().unique()
        pieces = []
        for day in days:
            day_et = day.tz_convert("America/New_York")
            start = day_et.replace(
                hour=int(SESSION_START.split(":")[0]),
                minute=int(SESSION_START.split(":")[1]),
                second=0,
            )
            end = day_et.replace(
                hour=int(SESSION_END.split(":")[0]),
                minute=int(SESSION_END.split(":")[1]),
                second=0,
            )
            idx = pd.date_range(start, end, freq="1min")
            pieces.append(idx)

        if not pieces:
            return df

        full_index = pieces[0].append(pieces[1:]).tz_convert("UTC")
        df_re = df.reindex(full_index)

        # Identify gap lengths
        is_missing = df_re["close"].isna().astype(int)
        run_id = (is_missing != is_missing.shift()).cumsum()
        gap_lengths = is_missing.groupby(run_id).transform("sum")
        df_re["gap_flag"] = is_missing.astype(bool)

        # Forward-fill short gaps only
        short_gap_mask = (is_missing == 1) & (gap_lengths <= self.max_gap_fill)
        df_re[short_gap_mask] = df_re[short_gap_mask].fillna(method="ffill")

        # Drop rows still missing (long gaps) — keep gap_flag as metadata
        df_re = df_re[df_re["close"].notna()]

        # Flag bars that immediately follow a long gap
        df_re["post_long_gap"] = False
        long_gap_ends = df_re.index[
            df_re["gap_flag"] & (gap_lengths > self.max_gap_fill)
        ]
        for gap_end in long_gap_ends:
            try:
                next_idx = df_re.index[df_re.index.get_loc(gap_end) + 1]
                df_re.at[next_idx, "post_long_gap"] = True
            except (IndexError, KeyError):
                pass

        return df_re

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and clip extreme returns (> ±outlier_z rolling σ)."""
        r = np.log(df["close"] / df["close"].shift(1))
        rolling_mean = r.rolling(self.outlier_window, min_periods=10).mean()
        rolling_std = r.rolling(self.outlier_window, min_periods=10).std()
        z = (r - rolling_mean) / (rolling_std + 1e-10)
        outlier_mask = z.abs() > self.outlier_z

        df["outlier_flag"] = outlier_mask

        # Reconstruct close prices from clipped returns
        for idx in df.index[outlier_mask]:
            loc = df.index.get_loc(idx)
            if loc == 0:
                continue
            prev_close = df["close"].iloc[loc - 1]
            sign = np.sign(r.iloc[loc])
            clipped_r = sign * self.outlier_z * rolling_std.iloc[loc]
            df.at[idx, "close"] = prev_close * np.exp(clipped_r)

        clipped = outlier_mask.sum()
        if clipped:
            logger.debug("Outlier clipping: %d bars clipped", clipped)
        return df

    @staticmethod
    def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
        """Add log_return, session_minute, session_start flag."""
        df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)

        et_index = df.index.tz_convert("America/New_York")
        session_open_time = pd.Timestamp(SESSION_START).time()
        session_open_minutes = session_open_time.hour * 60 + session_open_time.minute
        df["session_minute"] = (
            et_index.hour * 60 + et_index.minute - session_open_minutes
        ).clip(lower=0)
        df["session_start"] = (df["session_minute"] == 0).astype(bool)
        return df
