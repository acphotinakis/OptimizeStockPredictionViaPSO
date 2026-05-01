from __future__ import annotations

from dataclasses import dataclass, field
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import Timedelta

from src.data.utils import _parse_timeframe

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

from constants import MAX_GAP_FILL_BARS
from src.data.debug_logs import _log_ohlcv_validation_report


class DataCleaner:
    """
    Cleans a single-ticker 1-minute OHLCV DataFrame.

    TRD Stage 2 Cleaner (STRICT IMPLEMENTATION)

    INPUT ASSUMPTIONS:
    - Data is already:
        * UTC normalized
        * Sorted by time
        * Schema validated (Stage 1)
    - NO calendar reindexing exists at this stage
    """

    def __init__(
        self,
        ticker: str,
        timeframe: str,
        max_gap_fill: int = MAX_GAP_FILL_BARS,
    ) -> None:
        self.max_gap_fill = max_gap_fill
        self.validation_errors: Dict[str, Dict[str, Any]] = {}
        self.ticker = ticker
        self.timeframe = timeframe

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        df = df.copy()

        # Handle empty DataFrame early
        if len(df) == 0:
            logger.info("Empty DataFrame provided to clean(); returning as-is")
            return df, {}

        if df.empty:
            return df, {}

        # -----------------------------
        # 1. Ensure ordering invariant
        # -----------------------------
        df = df.sort_index()
        # Enforce strict chronological monotonicity (ascending)
        if not df.index.is_monotonic_increasing:
            logger.warning("Index not monotonic increasing, sorting...")
            df = df.sort_index()
        assert df.index.is_monotonic_increasing, "Index must be sorted"

        # -----------------------------
        # 2. Remove duplicate timestamps
        # -----------------------------
        before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        after = len(df)
        if before - after > 0:
            logger.info("Removed %d duplicate timestamp bars", before - after)
        else:
            logger.info("No duplicate timestamps found")

        # -----------------------------
        # 3. Strict OHLCV validation
        # -----------------------------
        df, ohlcv_report = self._validate_ohlcv(df)
        self.validation_errors["ohlcv_report"] = ohlcv_report
        _log_ohlcv_validation_report(ohlcv_report, ticker=self.ticker)
        # -----------------------------
        # 4. Gap classification (OBSERVATION-BASED)
        # -----------------------------
        gap_info = self._compute_observation_gaps(df)
        self.validation_errors["gap_info"] = gap_info

        # -----------------------------
        # 5. Remove long gaps (> max_gap_fill) BEFORE forward-fill so the
        #    forward-fill operates only on short, fillable gaps. Removing long
        #    gaps after fill (when they remain NaN) was the original off-by-one
        #    bug that left g["end"] alive and caused _final_validation to fail.
        # -----------------------------
        df, _remove_long_gaps_report = self._remove_long_gaps_by_segments(
            df, gap_info["gap_segments"]
        )
        self.validation_errors["_remove_long_gaps_report"] = _remove_long_gaps_report
        logger.info(f"First 10 rows LONG GAP REMOVAL")
        logger.info(f"\n{df[:10]}")

        # Refresh gap_info against the post-removal frame so any timestamp-
        # indexed segment data passed to downstream steps reflects current
        # rows (the original gap_info contains stale references to rows that
        # have just been removed).
        gap_info = self._compute_observation_gaps(df)
        self.validation_errors["gap_info"] = gap_info

        # -----------------------------
        # 6. Apply bounded causal forward fill on the surviving short gaps
        # -----------------------------
        df, _bounded_forward_fill_report = self._bounded_forward_fill(df, gap_info)
        self.validation_errors["_bounded_forward_fill_report"] = (
            _bounded_forward_fill_report
        )
        logger.info(f"First 10 rows FORWARD FILL")
        logger.info(f"\n{df[:10]}")

        # -----------------------------
        # 7. Final invariant enforcement
        # -----------------------------
        df, _final_validation_report = self._final_validation(df)
        self.validation_errors["_final_validation_report"] = _final_validation_report

        logger.info("Cleaned DataFrame: %d rows", len(df))
        return df, self.validation_errors

    def _remove_long_gaps_by_segments(self, df: pd.DataFrame, gaps: list[dict]):
        report = {
            "rows_before": len(df),
            "max_gap_fill": self.max_gap_fill,
            "long_gap_count": 0,
            "removed_indices_sample": [],
            "removed_fraction": 0.0,
        }

        mask = pd.Series(False, index=df.index)

        long_gaps = [g for g in gaps if g["missing_observations"] > self.max_gap_fill]
        report["long_gap_count"] = len(long_gaps)

        # build removal mask (inclusive of both endpoints — both g["start"] and
        # g["end"] are still-NaN rows belonging to the long gap segment)
        for g in long_gaps:
            gap_mask = (df.index >= g["start"]) & (df.index <= g["end"])
            mask |= gap_mask

        removed = int(mask.sum())

        if removed > 0:
            report["removed_indices_sample"] = list(df.index[mask][:10])

        df = df[~mask].copy()

        report["rows_after"] = len(df)
        report["rows_removed"] = removed
        report["removed_fraction"] = (
            removed / report["rows_before"] if report["rows_before"] > 0 else 0.0
        )

        logger.info(
            "Long gap removal | removed_rows=%d | remaining=%d",
            removed,
            len(df),
        )

        return df, report

    # =========================================================
    # STEP 3: OHLCV VALIDATION
    # =========================================================

    def _validate_ohlcv(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        logger.info(
            "OHLCV validation started | rows=%d | cols=%d",
            len(df),
            len(df.columns),
        )

        required = ["open", "high", "low", "close", "volume"]

        report: Dict = {
            "initial_rows": len(df),
            "missing_columns": [],
            "invalid_count": 0,
            "valid_count": 0,
            "dropped_count": 0,
        }

        # ------------------------------------------------------------------
        # Column validation
        # ------------------------------------------------------------------
        missing = list(set(required) - set(df.columns))
        report["missing_columns"] = missing

        if missing:
            logger.error("OHLCV validation failed | missing_columns=%s", missing)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("OHLCV validation | required columns present=%s", required)

        # ------------------------------------------------------------------
        # Row validation mask
        # ------------------------------------------------------------------
        mask = (
            (df["high"] >= df["low"])
            & (df["open"] > 0)
            & (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
            & (df["volume"] >= 0)
            & (df["close"] >= df["low"])
            & (df["close"] <= df["high"])
        )

        invalid_mask = ~mask

        invalid_count = int(invalid_mask.sum())
        valid_count = int(mask.sum())

        report["invalid_count"] = invalid_count
        report["valid_count"] = valid_count
        report["drop_rate"] = invalid_count / len(df) if len(df) > 0 else 0.0

        # ------------------------------------------------------------------
        # Breakdown stats
        # ------------------------------------------------------------------
        if invalid_count > 0:
            report["breakdown"] = {
                "high_lt_low": int((df["high"] < df["low"]).sum()),
                "open_le_0": int((df["open"] <= 0).sum()),
                "high_le_0": int((df["high"] <= 0).sum()),
                "low_le_0": int((df["low"] <= 0).sum()),
                "close_le_0": int((df["close"] <= 0).sum()),
                "volume_lt_0": int((df["volume"] < 0).sum()),
                "close_out_of_bounds": int(
                    ((df["close"] < df["low"]) | (df["close"] > df["high"])).sum()
                ),
            }

            logger.info(
                "Dropping invalid OHLCV rows | dropped=%d | remaining=%d",
                invalid_count,
                valid_count,
            )

        # ------------------------------------------------------------------
        # Apply filter
        # ------------------------------------------------------------------
        df = df[mask].copy()

        report["dropped_count"] = invalid_count
        report["final_rows"] = len(df)

        logger.info(
            "OHLCV validation complete | rows_before=%d | rows_after=%d | dropped=%d",
            report["initial_rows"],
            report["final_rows"],
            report["dropped_count"],
        )

        return df, report

    # def _validate_ohlcv(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    #     logger.info(
    #         "OHLCV validation started | rows=%d | cols=%d",
    #         len(df),
    #         len(df.columns),
    #     )

    #     required = ["open", "high", "low", "close", "volume"]

    #     report: Dict = {
    #         "initial_rows": len(df),
    #         "missing_columns": [],
    #         "invalid_count": 0,
    #         "valid_count": 0,
    #         "dropped_count": 0,
    #         "nan_rows": 0,
    #     }

    #     # ------------------------------------------------------------------
    #     # Column validation
    #     # ------------------------------------------------------------------
    #     missing = list(set(required) - set(df.columns))
    #     report["missing_columns"] = missing

    #     if missing:
    #         logger.error("OHLCV validation failed | missing_columns=%s", missing)
    #         raise ValueError(f"Missing required columns: {missing}")

    #     logger.info("OHLCV validation | required columns present=%s", required)

    #     # ------------------------------------------------------------------
    #     # CRITICAL FIX: Separate NaN rows from invalid data rows
    #     # ------------------------------------------------------------------

    #     # Check which rows have any NaN values (these are gap/missing rows, not invalid)
    #     has_nan = df[required].isna().any(axis=1)
    #     nan_count = int(has_nan.sum())
    #     report["nan_rows"] = nan_count

    #     if nan_count > 0:
    #         logger.info("Found %d rows with NaN values (gap/missing rows)", nan_count)

    #     # Only validate rows that have complete data (no NaN)
    #     df_valid = df[~has_nan].copy()

    #     if len(df_valid) == 0:
    #         logger.warning("No complete rows found for OHLCV validation")
    #         report["final_rows"] = len(df)
    #         return df, report

    #     # ------------------------------------------------------------------
    #     # Row validation mask (only on non-NaN rows)
    #     # ------------------------------------------------------------------
    #     mask = (
    #         (df_valid["high"] >= df_valid["low"])
    #         & (df_valid["open"] > 0)
    #         & (df_valid["high"] > 0)
    #         & (df_valid["low"] > 0)
    #         & (df_valid["close"] > 0)
    #         & (df_valid["volume"] >= 0)
    #         & (df_valid["close"] >= df_valid["low"])
    #         & (df_valid["close"] <= df_valid["high"])
    #     )

    #     invalid_mask = ~mask
    #     invalid_count = int(invalid_mask.sum())
    #     valid_count = int(mask.sum())

    #     report["invalid_count"] = invalid_count
    #     report["valid_count"] = valid_count
    #     report["drop_rate"] = (
    #         invalid_count / len(df_valid) if len(df_valid) > 0 else 0.0
    #     )

    #     # ------------------------------------------------------------------
    #     # Breakdown stats (only for actually invalid rows)
    #     # ------------------------------------------------------------------
    #     if invalid_count > 0:
    #         report["breakdown"] = {
    #             "high_lt_low": int((df_valid["high"] < df_valid["low"]).sum()),
    #             "open_le_0": int((df_valid["open"] <= 0).sum()),
    #             "high_le_0": int((df_valid["high"] <= 0).sum()),
    #             "low_le_0": int((df_valid["low"] <= 0).sum()),
    #             "close_le_0": int((df_valid["close"] <= 0).sum()),
    #             "volume_lt_0": int((df_valid["volume"] < 0).sum()),
    #             "close_out_of_bounds": int(
    #                 (
    #                     (df_valid["close"] < df_valid["low"])
    #                     | (df_valid["close"] > df_valid["high"])
    #                 ).sum()
    #             ),
    #         }

    #         logger.info(
    #             "Dropping invalid OHLCV rows | dropped=%d | remaining=%d",
    #             invalid_count,
    #             valid_count,
    #         )

    #     # ------------------------------------------------------------------
    #     # Apply filter: keep valid rows + NaN rows (for gap handling later)
    #     # ------------------------------------------------------------------
    #     df_valid_filtered = df_valid[mask].copy()

    #     # Recombine: valid data rows + NaN rows (for gap processing)
    #     df_nan = df[has_nan].copy()
    #     df = pd.concat([df_valid_filtered, df_nan]).sort_index()

    #     report["dropped_count"] = invalid_count
    #     report["final_rows"] = len(df)

    #     logger.info(
    #         "OHLCV validation complete | rows_before=%d | rows_after=%d | dropped=%d | nan_rows=%d",
    #         report["initial_rows"],
    #         report["final_rows"],
    #         report["dropped_count"],
    #         report["nan_rows"],
    #     )

    #     return df, report

    # =========================================================
    # STEP 4: GAP CLASSIFICATION (OBSERVATION-BASED)
    # =========================================================

    def _compute_observation_gaps(self, df: pd.DataFrame) -> dict:
        logger.info("STEP 4 | Gap classification started | rows=%d", len(df))

        is_missing = df["close"].isna()

        gaps = []
        in_gap = False
        start = None
        length = 0

        for i, missing in enumerate(is_missing.values):
            if missing:
                if not in_gap:
                    in_gap = True
                    start = df.index[i]
                    length = 1
                else:
                    length += 1
            else:
                if in_gap:
                    gaps.append(
                        {
                            "start": start,
                            "end": df.index[i - 1],
                            "missing_observations": length,
                        }
                    )
                    in_gap = False
                    length = 0

        if in_gap:
            gaps.append(
                {
                    "start": start,
                    "end": df.index[-1],
                    "missing_observations": length,
                }
            )

        return {
            "gap_segments": gaps,
            "max_gap": max((g["missing_observations"] for g in gaps), default=0),
            "min_gap": min((g["missing_observations"] for g in gaps), default=0),
            "total_gaps": len(gaps),
        }

    # =========================================================
    # STEP 5: BOUNDED CAUSAL FORWARD FILL
    # =========================================================

    def _bounded_forward_fill(
        self,
        df: pd.DataFrame,
        gap_info: dict,
    ) -> Tuple[pd.DataFrame, Dict]:

        logger.info(
            "STEP 5 | Bounded forward fill started | max_gap_fill=%d | rows=%d",
            self.max_gap_fill,
            len(df),
        )

        cols = ["open", "high", "low", "close", "volume"]

        result = df.copy(deep=True)

        report = {
            "max_gap_fill": self.max_gap_fill,
            "rows_before": len(df),
            "rows_after": len(df),
            "gap_info_present": gap_info is not None,
            "columns": {},
            "total_filled": 0,
            "total_skipped": 0,
            "total_leading_nan": 0,
            "total_gap_exceeded": 0,
        }

        # Safer NaN counting
        report["nan_before"] = int(pd.isna(result[cols]).sum().sum())

        for col in cols:
            logger.info("Forward-fill processing column=%s", col)

            # Always operate on a copy (avoid numpy view issues)
            values = result[col].to_numpy(copy=True)

            last_valid = None
            gap_count = 0

            filled = 0
            skipped = 0
            leading_nan = 0
            gap_exceeded = 0

            # Total NaNs in this column BEFORE filling
            total_nan_col = int(pd.isna(values).sum())

            for i in range(len(values)):
                if not pd.isna(values[i]):
                    last_valid = values[i]
                    gap_count = 0
                else:
                    if last_valid is None:
                        # No previous value --> cannot fill
                        leading_nan += 1
                        skipped += 1
                    elif gap_count < self.max_gap_fill:
                        values[i] = last_valid
                        gap_count += 1
                        filled += 1
                    else:
                        # Gap exceeded
                        gap_exceeded += 1
                        skipped += 1

            result[col] = values

            fill_ratio = filled / total_nan_col if total_nan_col > 0 else -1

            report["columns"][col] = {
                "filled": filled,
                "skipped": skipped,
                "leading_nan": leading_nan,
                "gap_exceeded": gap_exceeded,
                "fill_ratio": fill_ratio,
                "nan_before": total_nan_col,
            }

            report["total_filled"] += filled
            report["total_skipped"] += skipped
            report["total_leading_nan"] += leading_nan
            report["total_gap_exceeded"] += gap_exceeded

            logger.info(
                "Forward-fill column complete | column=%s | filled=%d | skipped=%d | leading_nan=%d | gap_exceeded=%d",
                col,
                filled,
                skipped,
                leading_nan,
                gap_exceeded,
            )

        report["nan_after"] = int(pd.isna(result[cols]).sum().sum())

        # Row-level validity (all OHLCV present)
        report["rows_fully_valid_after"] = int(result[cols].notna().all(axis=1).sum())

        logger.info("STEP 5 complete | bounded forward fill finished")

        return result, report

    # =========================================================
    # STEP 7: FINAL INVARIANT ENFORCEMENT
    # =========================================================

    def _final_validation(self, df: pd.DataFrame):
        logger.info("STEP 7 | Final invariant validation started | rows=%d", len(df))

        cols = ["open", "high", "low", "close", "volume"]

        report = {
            "rows": len(df),
            "nan_count": 0,
            "close_invalid": 0,
            "volume_invalid": 0,
            "is_valid": True,
        }

        # ------------------------------------------------------------
        # NaN validation
        # ------------------------------------------------------------
        null_count = int(df[cols].isna().sum().sum())
        report["nan_count"] = null_count

        if null_count > 0:
            logger.error("Final validation failed | NaNs_remaining=%d", null_count)
            report["is_valid"] = False
            raise ValueError(f"NaNs present in OHLCV after cleaning: {null_count}")

        logger.info("No NaNs in OHLCV confirmed")

        # ------------------------------------------------------------
        # Value sanity checks
        # ------------------------------------------------------------
        close_invalid = int((df["close"] <= 0).sum())
        volume_invalid = int((df["volume"] < 0).sum())

        report["close_invalid"] = close_invalid
        report["volume_invalid"] = volume_invalid

        logger.info(
            "Value validation | close_invalid=%d | volume_invalid=%d",
            close_invalid,
            volume_invalid,
        )

        # ------------------------------------------------------------
        # Hard invariants
        # ------------------------------------------------------------
        if not (df["close"] > 0).all():
            report["is_valid"] = False
            raise ValueError("Invalid close values detected (<= 0)")

        if not (df["volume"] >= 0).all():
            report["is_valid"] = False
            raise ValueError("Invalid volume values detected (< 0)")

        report["is_valid"] = True

        logger.info("STEP 7 complete | dataset is TRD-compliant")

        return df, report

    # =========================================================
    # SYNCHRONIZED CLEANING SUPPORT (Issue #2 Fix)
    # =========================================================

    def get_invalid_mask(
        self, df: pd.DataFrame, after_forward_fill: bool = True
    ) -> pd.Series:
        """
        Return boolean mask of invalid rows (without removing them).

        Used for synchronized cleaning across multiple tickers.
        Returns True for rows that should be removed.

        Args:
            df: DataFrame to check
            after_forward_fill: If True, mark ALL remaining NaN as invalid.
                              If False, only mark gaps > max_gap_fill as invalid.

        Returns:
            Boolean Series (True = invalid row, should be dropped)
        """
        # Start with all False
        invalid_mask = pd.Series(False, index=df.index)

        # OHLCV validation failures
        ohlcv_invalid = (
            (df["high"] < df["low"])
            | (df["open"] <= 0)
            | (df["high"] <= 0)
            | (df["low"] <= 0)
            | (df["close"] <= 0)
            | (df["volume"] < 0)
            | (df["close"] < df["low"])
            | (df["close"] > df["high"])
        )

        invalid_mask |= ohlcv_invalid

        # Missing data handling
        is_missing = df["close"].isna()

        if is_missing.any():
            if after_forward_fill:
                # After forward-fill, ANY remaining NaN is invalid
                # (either initial NaN before first valid, or gap > max_gap_fill)
                invalid_mask |= is_missing
                logger.info(
                    "Marked ALL remaining NaN as invalid (post-forward-fill) | count=%d",
                    is_missing.sum(),
                )
            else:
                # Before forward-fill, only mark gaps > max_gap_fill
                run_id = (is_missing != is_missing.shift()).cumsum()
                gap_lengths = is_missing.groupby(run_id).transform("sum")
                long_gap_mask = is_missing & (gap_lengths > self.max_gap_fill)

                invalid_mask |= long_gap_mask
                logger.info(
                    "Marked long gaps (>%d) as invalid | count=%d",
                    self.max_gap_fill,
                    long_gap_mask.sum(),
                )

        logger.info(
            "Invalid mask computed | total_invalid=%d (%.2f%%)",
            invalid_mask.sum(),
            100 * invalid_mask.sum() / len(df) if len(df) > 0 else 0,
        )

        return invalid_mask
