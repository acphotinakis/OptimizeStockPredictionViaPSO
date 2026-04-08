"""
src/data/cleaner.py

Session filtering, missing value imputation, OHLC consistency checks,
zero-volume removal, and return outlier clipping for 1-minute OHLCV data.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from datetime import datetime

import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

from constants import SESSION_START, SESSION_END, MAX_GAP_FILL_BARS, OUTLIER_ROLLING_WINDOW, OUTLIER_ZSCORE_THRESHOLD


class DataCleaner:
    """Cleans a single-ticker 1-minute OHLCV DataFrame.

    Steps applied in order:
        1. Session filtering (09:30 – 16:00 ET)
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
        df = df.copy()

        # Handle empty DataFrame early
        if len(df) == 0:
            logger.warning("Empty DataFrame provided to clean(); returning as-is")
            return df

        df = self._ensure_utc(df)
        df = self._remove_duplicates(df)
        df = self._session_filter(df)
        df = self._remove_zero_volume(df)
        df = self._ohlc_consistency(df)
        df = self._reindex_and_fill(df)
        df = self._clip_outliers(df)
        df = self._add_derived(df)
        logger.debug("Cleaned DataFrame: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        return df

    def _ensure_utc(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df

    def _session_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only bars within NYSE regular trading hours (ET)."""
        # Build mask without copying full DataFrame
        et_index = df.index.tz_convert("America/New_York")

        mask = (et_index.time >= self.session_start) & (
            et_index.time <= self.session_end
        )

        before = len(df)
        df = df.loc[mask]

        if before - len(df) != 0:
            logger.debug(
                "Session filter: dropped %d pre/post-market bars", before - len(df)
            )
        return df

    def _remove_zero_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[df["volume"] > 0]
        logger.debug("Zero-volume removal: dropped %d bars", before - len(df))
        return df

    def _ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
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
        """
        Optimized reindex and gap-filling for OHLCV DataFrame.

        Steps:
            1. Build NYSE trading minute index for the date range
            2. Reindex OHLCV DataFrame to full trading minutes
            3. Identify gaps and classify as short (<= max_gap_fill) or long
            4. Forward-fill short gaps
            5. Drop long gaps
            6. Flag first bar after long gaps as post_long_gap
        """
        if df.empty:
            logger.info("DataFrame is empty, skipping reindex.")
            return df

        import pandas_market_calendars as mcal

        # -----------------------------------------------------------------
        # 1. Trading calendar
        # -----------------------------------------------------------------
        start_date = df.index.min().date()
        end_date = df.index.max().date()
        logger.info("Reindexing from %s to %s", start_date, end_date)

        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        trading_minutes = mcal.date_range(schedule, frequency="1min")

        logger.info("Generated NYSE trading minutes: %d bars", len(trading_minutes))

        # -----------------------------------------------------------------
        # 2. Reindex
        # -----------------------------------------------------------------
        df_re = df.reindex(trading_minutes)
        logger.info("After reindex: %d bars (original %d)", len(df_re), len(df))

        # -----------------------------------------------------------------
        # 3. Identify missing gaps
        # -----------------------------------------------------------------
        is_missing = df_re["close"].isna()
        df_re["gap_flag"] = is_missing

        # Run-length encoding using shift & cumsum
        run_id = (is_missing != is_missing.shift()).cumsum()
        gap_lengths = is_missing.groupby(run_id).transform("sum")
        df_re["gap_length"] = gap_lengths.where(is_missing, 0)

        num_short_gaps = ((is_missing) & (gap_lengths <= self.max_gap_fill)).sum()
        num_long_gaps = ((is_missing) & (gap_lengths > self.max_gap_fill)).sum()
        logger.info(
            "Detected %d short-gap bars and %d long-gap bars",
            num_short_gaps,
            num_long_gaps,
        )

        # -----------------------------------------------------------------
        # 4. Forward-fill short gaps
        # -----------------------------------------------------------------
        short_gap_mask = is_missing & (gap_lengths <= self.max_gap_fill)
        cols = ["open", "high", "low", "close", "volume"]

        df_re.loc[short_gap_mask, cols] = df_re[cols].ffill()
        logger.info("Forward-filled %d short-gap bars", short_gap_mask.sum())

        # -----------------------------------------------------------------
        # 5. Drop long gaps
        # -----------------------------------------------------------------
        long_gap_mask = is_missing & (gap_lengths > self.max_gap_fill)
        long_gap_indices = df_re.index[long_gap_mask]

        df_re = df_re[df_re["close"].notna()]
        logger.info("Dropped %d long-gap bars", long_gap_mask.sum())

        # -----------------------------------------------------------------
        # 6. Flag first bar after long gap as post_long_gap
        # -----------------------------------------------------------------
        df_re["post_long_gap"] = False
        if len(long_gap_indices) > 0:
            # Get locations of bars immediately after long gaps
            post_gap_indices = df_re.index.searchsorted(long_gap_indices) + 1
            post_gap_indices = post_gap_indices[post_gap_indices < len(df_re)]
            df_re.iloc[post_gap_indices, df_re.columns.get_loc("post_long_gap")] = True
        logger.info("Flagged %d bars as post_long_gap", df_re["post_long_gap"].sum())

        # -----------------------------------------------------------------
        # 7. Final check
        # -----------------------------------------------------------------
        if df_re.isna().sum().sum() > 0:
            logger.warning(
                "NaNs remain after reindex/fill: %d total", df_re.isna().sum().sum()
            )
            df_re = df_re.dropna()

        logger.info("Final DataFrame length after reindex/fill: %d", len(df_re))

        return df_re

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logger.info("DataFrame empty, skipping outlier clipping.")
            return df

        logger.info(
            "Starting outlier clipping | window=%d | z=%.2f",
            self.outlier_window,
            self.outlier_z,
        )

        # ------------------------------------------------------------------
        # 1. Log returns (pandas Series)
        # ------------------------------------------------------------------
        r = np.log(df["close"] / df["close"].shift(1))
        logger.info("Computed %d valid log returns", r.notna().sum())

        # ------------------------------------------------------------------
        # 2. Rolling stats (pandas, simpler + stable)
        # ------------------------------------------------------------------
        rolling_mean = r.rolling(self.outlier_window, min_periods=10).mean()
        rolling_std = r.rolling(self.outlier_window, min_periods=10).std()

        logger.info("Computed rolling mean/std")
        logger.info(
            "Rolling stats → mean(avg): %.6f | std(avg): %.6f | mean(min/max): [%.6f, %.6f] | std(min/max): [%.6f, %.6f]",
            np.nanmean(rolling_mean),
            np.nanmean(rolling_std),
            np.nanmin(rolling_mean),
            np.nanmax(rolling_mean),
            np.nanmin(rolling_std),
            np.nanmax(rolling_std),
        )
        # ------------------------------------------------------------------
        # 3. Z-score
        # ------------------------------------------------------------------
        z = (r - rolling_mean) / (rolling_std + 1e-10)

        # ------------------------------------------------------------------
        # 4. Detect outliers
        # ------------------------------------------------------------------
        outlier_mask = z.abs() > self.outlier_z
        num_outliers = int(outlier_mask.sum())
        df["outlier_flag"] = outlier_mask

        logger.info("Detected %d outliers", num_outliers)

        if num_outliers == 0:
            return df

        # ------------------------------------------------------------------
        # 5. Clip safely
        # ------------------------------------------------------------------
        outlier_locs = np.where(outlier_mask.values)[0]
        outlier_locs = outlier_locs[outlier_locs > 0]  # avoid first index

        if len(outlier_locs) == 0:
            logger.info("No valid outliers after removing first index")
            return df

        prev_close = df["close"].values[outlier_locs - 1]
        signs = np.sign(r.values[outlier_locs])
        std_vals = rolling_std.values[outlier_locs]

        clipped_r = signs * self.outlier_z * std_vals

        df.loc[df.index[outlier_locs], "close"] = prev_close * np.exp(clipped_r)

        logger.info("Clipped %d outliers", len(outlier_locs))

        return df

    @staticmethod
    def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
        """Add log_return, session_minute, session_start flag."""
        df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)

        et_index = df.index.tz_convert("America/New_York")
        session_open_time = pd.Timestamp(SESSION_START).time()
        session_open_minutes = session_open_time.hour * 60 + session_open_time.minute

        # Convert to Series to allow clip
        session_minutes = pd.Series(
            et_index.hour * 60 + et_index.minute - session_open_minutes, index=df.index
        )
        df["session_minute"] = session_minutes.clip(lower=0)

        df["session_start"] = (df["session_minute"] == 0).astype(bool)
        return df
