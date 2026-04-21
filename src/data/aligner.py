"""
src/data/aligner.py

TRD Stage 1.5: Temporal Alignment (STRICT)

Responsibility:
- Define canonical SPY-based time index
- Align all tickers to the same timestamp grid
- Preserve missingness (NO imputation, NO filtering)
- Ensure deterministic cross-asset synchronization

STRICT CONSTRAINTS:
- No forward-fill
- No dropna
- No interpolation
- No feature engineering
"""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TickerAligner:
    """SPY-canonical alignment layer (TRD-compliant)."""

    def __init__(
        self,
        benchmark_ticker: str = "SPY",
    ) -> None:
        self.benchmark_ticker = benchmark_ticker

    # =========================================================
    # MAIN ALIGNMENT FUNCTION (PURE)
    # =========================================================

    def align(
        self,
        dfs: Dict[str, pd.DataFrame],
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:

        logger.info(
            "ALIGNMENT START | tickers=%d | benchmark=%s",
            len(dfs),
            self.benchmark_ticker,
        )

        if fields is None:
            fields = ["open", "high", "low", "close", "volume"]

        # =====================================================
        # 1. VALIDATE SPY AS CANONICAL SOURCE
        # =====================================================
        if self.benchmark_ticker not in dfs:
            logger.error("SPY benchmark missing from input dictionary")
            raise ValueError("SPY benchmark ticker is required")

        spy = dfs[self.benchmark_ticker].copy()

        if not isinstance(spy.index, pd.DatetimeIndex):
            logger.info("SPY index not DatetimeIndex → converting")
            spy.index = pd.to_datetime(spy.index, utc=True)

        spy = spy.sort_index()

        if not spy.index.is_monotonic_increasing:
            logger.error("SPY index is not monotonic increasing")
            raise ValueError("SPY index must be sorted and monotonic")

        if spy.index.has_duplicates:
            logger.error("SPY index contains duplicates")
            raise ValueError("SPY index contains duplicate timestamps")

        master_index = spy.index

        logger.info("SPY canonical index established | rows=%d", len(master_index))

        # =====================================================
        # 2. ALIGN ALL TICKERS (PURE REINDEX ONLY)
        # =====================================================
        aligned = {}

        for ticker, df in dfs.items():

            logger.info("Aligning ticker=%s | rows_before=%d", ticker, len(df))

            df = df.copy()

            # ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            else:
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")

            df = df.sort_index()

            # enforce schema BEFORE alignment (no modification after)
            missing_cols = set(fields) - set(df.columns)
            if missing_cols:
                logger.warning("Ticker=%s missing columns=%s", ticker, missing_cols)

            df = df.reindex(columns=fields)

            before_rows = len(df)

            # =================================================
            # CORE ALIGNMENT OPERATION (NO SIDE EFFECTS)
            # =================================================
            df = df.reindex(master_index)

            after_rows = len(df)

            logger.info(
                "Ticker aligned | ticker=%s | rows_before=%d | rows_after=%d | missing_after_alignment=%d",
                ticker,
                before_rows,
                after_rows,
                df.isna().any(axis=1).sum(),
            )

            # IMPORTANT: DO NOT MODIFY VALUES HERE
            # (No FF, no dropna, no interpolation)

            df.columns = pd.MultiIndex.from_product(
                [[ticker], df.columns],
                names=["ticker", "field"],
            )

            aligned[ticker] = df

        # =====================================================
        # 3. CONCATENATION (STRUCTURAL ONLY)
        # =====================================================
        logger.info("Concatenating %d tickers", len(aligned))

        result = pd.concat(aligned.values(), axis=1).sort_index()

        logger.info(
            "ALIGNMENT COMPLETE | shape=(%d, %d)",
            result.shape[0],
            result.shape[1],
        )

        # =====================================================
        # 4. GLOBAL INTEGRITY CHECK (NO CLEANING)
        # =====================================================
        nan_rate = result.isna().mean().mean()

        logger.info(
            "Post-alignment diagnostics | global_nan_rate=%.4f",
            nan_rate,
        )

        if nan_rate > 0.95:
            logger.warning(
                "High missingness detected after alignment (%.2f%%)",
                nan_rate * 100,
            )

        return result

    # =========================================================
    # OPTIONAL SAVE UTILITIES (PURE IO ONLY)
    # =========================================================

    def save_aligned(
        self,
        aligned_df: pd.DataFrame,
        output_dir: str | Path,
    ) -> None:

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tickers = aligned_df.columns.get_level_values("ticker").unique()

        logger.info("Saving aligned dataset | tickers=%d", len(tickers))

        for ticker in tickers:
            df = aligned_df[ticker].copy()

            file_path = output_dir / f"{ticker}.parquet"

            df.to_parquet(file_path)

            logger.info(
                "Saved ticker=%s | shape=(%d,%d) | path=%s",
                ticker,
                df.shape[0],
                df.shape[1],
                file_path,
            )
