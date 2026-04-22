"""
src/data/cleaner.py

TRD Stage 2: Data Cleaning (STRICT)

Guarantees:
- Causal (no look-ahead bias)
- Observation-based gap classification
- Bounded forward-fill (≤ MAX_GAP)
- Deterministic long-gap removal
- Strict OHLCV validity
- No NaNs in output OHLCV
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

from constants import MAX_GAP_FILL_BARS


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
        max_gap_fill: int = MAX_GAP_FILL_BARS,
    ) -> None:
        self.max_gap_fill = max_gap_fill
        self.validation_errors: List[str] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Handle empty DataFrame early
        if len(df) == 0:
            logger.info("Empty DataFrame provided to clean(); returning as-is")
            return df

        if df.empty:
            return df

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
        df = self._validate_ohlcv(df)

        # -----------------------------
        # 4. Gap classification (OBSERVATION-BASED)
        # -----------------------------
        gap_info = self._compute_observation_gaps(df)

        # -----------------------------
        # 5. Apply bounded causal forward fill
        # -----------------------------
        df = self._bounded_forward_fill(df, gap_info)

        # -----------------------------
        # 6. Remove long gaps (> 5)
        # -----------------------------
        df = self._remove_long_gaps(df, gap_info)

        # -----------------------------
        # 7. Final invariant enforcement
        # -----------------------------
        df = self._final_validation(df)

        logger.info("Cleaned DataFrame: %d rows", len(df))
        return df

    # =========================================================
    # STEP 3: OHLCV VALIDATION
    # =========================================================

    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(
            "STEP 3 | OHLCV validation started | rows=%d | cols=%d",
            len(df),
            len(df.columns),
        )

        required = ["open", "high", "low", "close", "volume"]

        missing = set(required) - set(df.columns)
        if missing:
            logger.error("OHLCV validation failed | missing_columns=%s", missing)
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("OHLCV validation | required columns present=%s", required)

        initial_rows = len(df)

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

        invalid_count = (~mask).sum()

        logger.info(
            "OHLCV validation | invalid_rows_detected=%d | valid_rows=%d",
            invalid_count,
            mask.sum(),
        )

        if invalid_count > 0:
            # Optional breakdown for debugging
            logger.info(
                "Invalid OHLCV breakdown | high<low=%d | open<=0=%d | high<=0=%d | low<=0=%d | close<=0=%d | volume<0=%d | close_out_of_bounds=%d",
                (df["high"] < df["low"]).sum(),
                (df["open"] <= 0).sum(),
                (df["high"] <= 0).sum(),
                (df["low"] <= 0).sum(),
                (df["close"] <= 0).sum(),
                (df["volume"] < 0).sum(),
                ((df["close"] < df["low"]) | (df["close"] > df["high"])).sum(),
            )

            logger.info(
                "Dropping invalid OHLCV rows | dropped=%d | remaining=%d",
                invalid_count,
                mask.sum(),
            )

        df = df[mask]

        logger.info(
            "STEP 3 | OHLCV validation complete | rows_before=%d | rows_after=%d | dropped=%d",
            initial_rows,
            len(df),
            initial_rows - len(df),
        )

        return df

    # =========================================================
    # STEP 4: GAP CLASSIFICATION (OBSERVATION-BASED)
    # =========================================================

    def _compute_observation_gaps(self, df: pd.DataFrame) -> dict:
        logger.info("STEP 4 | Gap classification started | rows=%d", len(df))

        time_deltas = df.index.to_series().diff()

        expected = time_deltas.median()

        logger.info("Gap classification | inferred_expected_interval=%s", expected)

        gap_sizes = (time_deltas / expected).fillna(0).astype(int)

        gap_start = gap_sizes > 1

        gap_lengths = gap_sizes.cumsum()

        total_gaps = gap_start.sum()
        max_gap = gap_sizes.max()

        logger.info(
            "Gap classification summary | total_gap_starts=%d | max_gap_size=%d",
            total_gaps,
            max_gap,
        )

        # Optional deeper diagnostics
        if total_gaps > 0:
            logger.info(
                "Gap sizes distribution | min=%d | median=%.2f | max=%d",
                gap_sizes.min(),
                gap_sizes.median(),
                gap_sizes.max(),
            )

            logger.info("Gap start indices sample=%s", list(df.index[gap_start][:10]))

        return {
            "gap_start": gap_start,
            "gap_sizes": gap_sizes,
            "gap_lengths": gap_lengths,
            "expected_interval": expected,
        }

    # =========================================================
    # STEP 5: BOUNDED CAUSAL FORWARD FILL
    # =========================================================

    def _bounded_forward_fill(
        self,
        df: pd.DataFrame,
        gap_info: dict,
    ) -> pd.DataFrame:

        logger.info(
            "STEP 5 | Bounded forward fill started | max_gap_fill=%d | rows=%d",
            self.max_gap_fill,
            len(df),
        )

        cols = ["open", "high", "low", "close", "volume"]

        result = df.copy()

        for col in cols:
            logger.info("Forward-fill processing column=%s", col)

            values = result[col].values

            last_valid = None
            gap_count = 0

            filled = 0
            skipped = 0

            for i in range(len(values)):
                if not np.isnan(values[i]):
                    last_valid = values[i]
                    gap_count = 0
                else:
                    if last_valid is not None and gap_count < self.max_gap_fill:
                        values[i] = last_valid
                        gap_count += 1
                        filled += 1
                    else:
                        values[i] = np.nan
                        skipped += 1

            result[col] = values

            logger.info(
                "Forward-fill column complete | column=%s | filled=%d | left_as_nan=%d",
                col,
                filled,
                skipped,
            )

        logger.info("STEP 5 complete | bounded forward fill finished")

        return result

    # =========================================================
    # STEP 6: LONG GAP REMOVAL (> 5 OBSERVATIONS)
    # =========================================================

    def _remove_long_gaps(
        self,
        df: pd.DataFrame,
        gap_info: dict,
    ) -> pd.DataFrame:

        logger.info("STEP 6 | Long gap removal started | rows=%d", len(df))

        is_missing = df["close"].isna()

        run_id = (is_missing != is_missing.shift()).cumsum()
        gap_lengths = is_missing.groupby(run_id).transform("sum")

        long_gap_mask = is_missing & (gap_lengths > self.max_gap_fill)

        long_gap_count = long_gap_mask.sum()

        logger.info(
            "Long gap detection | long_gap_rows=%d | threshold=%d",
            long_gap_count,
            self.max_gap_fill,
        )

        if long_gap_count > 0:
            logger.info(
                "Long gap indices sample=%s", list(df.index[long_gap_mask][:10])
            )

        df = df[~long_gap_mask]

        logger.info(
            "STEP 6 complete | rows_after_removal=%d | removed=%d",
            len(df),
            long_gap_count,
        )

        return df

    # =========================================================
    # STEP 7: FINAL INVARIANT ENFORCEMENT
    # =========================================================

    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("STEP 7 | Final invariant validation started | rows=%d", len(df))

        cols = ["open", "high", "low", "close", "volume"]

        null_count = df[cols].isna().sum().sum()

        if null_count > 0:
            logger.error("Final validation failed | NaNs_remaining=%d", null_count)
            raise ValueError(f"NaNs present in OHLCV after cleaning: {null_count}")

        logger.info("No NaNs in OHLCV confirmed")

        # value sanity checks
        close_invalid = (df["close"] <= 0).sum()
        volume_invalid = (df["volume"] < 0).sum()

        logger.info(
            "Value validation | close_invalid=%d | volume_invalid=%d",
            close_invalid,
            volume_invalid,
        )

        assert (df["close"] > 0).all()
        assert (df["volume"] >= 0).all()

        logger.info("STEP 7 complete | dataset is TRD-compliant")

        return df
