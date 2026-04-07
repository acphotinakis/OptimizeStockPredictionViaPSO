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
from datetime import datetime

logger = logging.getLogger(__name__)

SESSION_START = "09:30"
SESSION_END = "16:00"
MAX_GAP_FILL_BARS = 5  # Forward-fill at most this many missing bars
OUTLIER_ZSCORE_THRESHOLD = 5  # Clip returns beyond ±5σ
OUTLIER_ROLLING_WINDOW = 60  # Rolling window (bars) for z-score computation


    def generate_cleaning_report(stats):
    from datetime import timezone
    
    # Handle empty stats
    if stats["rows_initial"] == 0:
        pct_rows_removed = 0.0
    else:
        pct_rows_removed = (stats["rows_initial"] - stats["rows_final"]) / stats["rows_initial"]
    
    if stats["expected_bars"] == 0:
        coverage_ratio = 0.0
    else:
        coverage_ratio = stats["rows_final"] / stats["expected_bars"]

    report = {
        "metadata": {
            "start_date": str(stats["metadata"]["start_date"]),
            "end_date": str(stats["metadata"]["end_date"]),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "integrity": {
            "rows_removed_total": stats["rows_initial"] - stats["rows_final"],
            "pct_rows_removed": pct_rows_removed,
        },
        "input_metrics": {
            "row_count": stats["rows_initial"],
        },
        "output_metrics": {
            "row_count": stats["rows_final"],
        },
        "cleaning_steps": {
            "duplicates_removed": stats["duplicates_removed"],
            "session_filtered": stats["session_filtered"],
            "zero_volume_removed": stats["zero_volume_removed"],
            "ohlc_invalid": stats["ohlc_invalid"],
        },
        "gaps": stats["missing_data"],
        "outliers": stats["outliers"],
        "returns": {
            "before": stats["returns_before"],
            "after": stats["returns_after"],
        },
        "density": {
            "expected_bars": stats["expected_bars"],
            "actual_bars": stats["rows_final"],
            "coverage_ratio": coverage_ratio,
        },
    }

    return report


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

    def _init_stats(self):
        self.stats = {
            "metadata": {
                "start_date": None,
                "end_date": None,
            },
            "rows_initial": 0,
            "duplicates_removed": 0,
            "session_filtered": 0,
            "zero_volume_removed": 0,
            "ohlc_invalid": 0,
            "missing_data": {
                "total_missing": 0,
                "num_gaps": 0,
                "max_gap": 0,
                "avg_gap": 0,
                "short_filled": 0,
                "long_dropped": 0,
            },
            "outliers": {
                "count": 0,
                "max_z": 0.0,
                "mean_z": 0.0,
            },
            "returns_before": {},
            "returns_after": {},
            "expected_bars": 0,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        self._init_stats()

        self.stats["rows_initial"] = len(df)
        
        # Handle empty DataFrame early
        if len(df) == 0:
            logger.warning("Empty DataFrame provided to clean(); returning as-is")
            self.stats["rows_final"] = 0
            return df

        df = self._ensure_utc(df)
        df = self._remove_duplicates(df)
        df = self._session_filter(df)
        df = self._remove_zero_volume(df)
        df = self._ohlc_consistency(df)

        # capture returns BEFORE cleaning distortion
        self._capture_return_stats(df, key="returns_before")

        df = self._reindex_and_fill(df)
        df = self._clip_outliers(df)

        # capture AFTER
        self._capture_return_stats(df, key="returns_after")

        df = self._add_derived(df)

        self.stats["rows_final"] = len(df)

        logger.debug("Cleaned DataFrame: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _capture_return_stats(self, df: pd.DataFrame, key: str):
        r = np.log(df["close"]).diff().dropna()

        if len(r) == 0:
            return

        self.stats[key] = {
            "mean": float(r.mean()),
            "std": float(r.std()),
            "skew": float(r.skew()),
            "kurtosis": float(r.kurtosis()),
        }

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        self.stats["duplicates_removed"] = before - len(df)
        return df

    def _ensure_utc(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        self.stats["metadata"]["start_date"] = df.index.min().date()
        self.stats["metadata"]["end_date"] = df.index.max().date()
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

        self.stats["session_filtered"] = before - len(df)

        if before - len(df) != 0:
            logger.debug(
                "Session filter: dropped %d pre/post-market bars", before - len(df)
            )
        return df

    def _remove_zero_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[df["volume"] > 0]
        logger.debug("Zero-volume removal: dropped %d bars", before - len(df))
        self.stats["zero_volume_removed"] = before - len(df)
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
        self.stats["ohlc_invalid"] = (~mask).sum()
        return df[mask]

    def _reindex_and_fill(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Hybrid missing data handler with:
        - Exchange-accurate trading calendar
        - Deterministic gap segmentation
        - Controlled forward-fill (short gaps only)
        - Explicit long gap removal
        - Gap-aware feature generation

        Rules:
            1. Reindex to NYSE trading minutes
            2. Identify contiguous missing gaps
            3. If gap_length <= short_gap → forward-fill
            4. If gap_length > short_gap → drop بالكامل
            5. Add structural features for modeling

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            start_date: str (YYYY-MM-DD)
            end_date: str (YYYY-MM-DD)
            short_gap: max gap length to forward-fill

        Returns:
            Cleaned DataFrame
        """

        if df.empty:
            return df

        import pandas_market_calendars as mcal

        # -------------------------------------------------
        # 1. Build exchange-accurate trading index
        # -------------------------------------------------
        nyse = mcal.get_calendar("NYSE")

        start_date = df.index.min().date()
        end_date = df.index.max().date()

        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        trading_minutes = mcal.date_range(schedule, frequency="1min")

        df = df.copy()
        df_re = df.reindex(trading_minutes)

        # -------------------------------------------------
        # 2. Identify missing gaps
        # -------------------------------------------------
        is_missing = df_re["close"].isna().astype(int)

        run_id = (is_missing != is_missing.shift()).cumsum()
        gap_lengths = is_missing.groupby(run_id).transform("sum")

        df_re["gap_flag"] = is_missing.astype(bool)
        df_re["gap_length"] = gap_lengths.where(is_missing == 1, 0)

        gap_sizes = gap_lengths[is_missing == 1]

        self.stats["missing_data"]["total_missing"] = int(is_missing.sum())
        self.stats["missing_data"]["num_gaps"] = int((gap_sizes > 0).sum())
        self.stats["missing_data"]["max_gap"] = (
            int(gap_sizes.max()) if len(gap_sizes) else 0
        )
        self.stats["missing_data"]["avg_gap"] = (
            float(gap_sizes.mean()) if len(gap_sizes) else 0
        )
        # -------------------------------------------------
        # 3. Short gap fill (deterministic mask)
        # -------------------------------------------------
        short_gap_mask = (is_missing == 1) & (gap_lengths <= MAX_GAP_FILL_BARS)

        cols = ["open", "high", "low", "close", "volume"]
        df_re.loc[short_gap_mask, cols] = df_re[cols].ffill()

        # -------------------------------------------------
        # 4. Long gap removal
        # -------------------------------------------------
        long_gap_mask = (is_missing == 1) & (gap_lengths > MAX_GAP_FILL_BARS)

        # Capture indices BEFORE dropping
        long_gap_indices = df_re.index[long_gap_mask]

        df_re = df_re[df_re["close"].notna()]

        self.stats["missing_data"]["short_filled"] = int(short_gap_mask.sum())
        self.stats["missing_data"]["long_dropped"] = int(long_gap_mask.sum())

        # -------------------------------------------------
        # 5. Post-long-gap feature
        # -------------------------------------------------
        df_re["post_long_gap"] = False

        for gap_end in long_gap_indices:
            if gap_end in df_re.index:
                loc = df_re.index.get_loc(gap_end)
                if loc + 1 < len(df_re):
                    df_re.iloc[loc + 1, df_re.columns.get_loc("post_long_gap")] = True

        self.stats["expected_bars"] = len(trading_minutes)

        # -------------------------------------------------
        # 6. Final safety (no NaNs allowed)
        # -------------------------------------------------
        df_re = df_re.dropna()

        return df_re

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and clip extreme returns (> ±outlier_z rolling σ)."""
        r = pd.Series(np.log(df["close"] / df["close"].shift(1)), index=df.index)
        rolling_mean = r.rolling(self.outlier_window, min_periods=10).mean()
        rolling_std = r.rolling(self.outlier_window, min_periods=10).std()
        z = (r - rolling_mean) / (rolling_std + 1e-10)
        outlier_mask = z.abs() > self.outlier_z

        df["outlier_flag"] = outlier_mask

        # Reconstruct close prices from clipped returns (vectorized)
        outlier_locs = df.index.get_indexer_for(df.index[outlier_mask])
        outlier_locs = outlier_locs[outlier_locs > 0]  # Skip first bar
        
        if len(outlier_locs) > 0:
            outlier_idx = df.index[outlier_locs]
            prev_idx = df.index[outlier_locs - 1]
            prev_close = df.loc[prev_idx, "close"].values
            signs = np.sign(r.iloc[outlier_locs].values)
            clipped_r = signs * self.outlier_z * rolling_std.iloc[outlier_locs].values
            df.loc[outlier_idx, "close"] = prev_close * np.exp(clipped_r)

        clipped = outlier_mask.sum()
        if clipped:
            logger.debug("Outlier clipping: %d bars clipped", clipped)

        self.stats["outliers"]["count"] = int(outlier_mask.sum())
        self.stats["outliers"]["max_z"] = float(z.abs().max())
        self.stats["outliers"]["mean_z"] = float(z.abs().mean())
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

    def reset_stats(self):
        self._init_stats()
