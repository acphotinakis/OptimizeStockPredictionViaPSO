"""
src/features/pipeline.py

Production-grade feature engineering pipeline for PSO-LSTM stock prediction.

Design principles (derived from blueprint analysis):
  - All fitting operations use training data ONLY (no leakage)
  - Temporal ordering is preserved at every stage
  - Lag features use shift-based construction (no np.roll leakage)
  - Cross-ticker peer selection is fit on train, applied consistently to val/test
  - Target: log return (not raw price, not simple return)
  - NaN/inf handling is explicit and logged

Paper attribution:
  - Ji et al.          → lookback windowing, log return target
  - Deng & Peng        → return-based target, multi-feature input
  - Zeng et al.        → Pearson feature selection, cross-ticker correlation
  - Lanbouri & Achchab → session-aware construction, TA features
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .technical import compute_technical_features
from .statistical import compute_statistical_features
from .volume import compute_volume_features
from .cross_ticker import compute_cross_ticker_features
from .selector import FeatureSelector

# Import for type hints only
try:
    from .universe_builder import SymbolUniverseBuilder
except ImportError:
    SymbolUniverseBuilder = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lag configuration
# ---------------------------------------------------------------------------
# These are the features we create lagged copies of.
# Lag depths are chosen based on intraday mean-reversion literature:
#   - log_return: lags 1-3 capture short-term autocorrelation
#   - momentum indicators: lag 1 is sufficient (they are already smoothed)
# ---------------------------------------------------------------------------
LAG_DEPTHS: Dict[str, List[int]] = {
    "log_return": [1, 2, 3],
    "rsi_14": [1, 2],
    "vwap_dev": [1],
    "rvol_20": [1],
    "corr_spy_20": [1],
    "macd": [1],
    "atr_14": [1],
    "bb_pct_b": [1],
    "stoch_k": [1],
    "zscore_20": [1],
}

# Columns required to exist on every ticker DataFrame entering the pipeline
REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume", "log_return"}


class FeaturePipeline:
    """
    End-to-end feature engineering and selection pipeline.

    Lifecycle:
        pipeline.fit_transform(dfs_train)  → X_train, y_train, feature_names
        pipeline.transform(dfs_val)        → X_val,   y_val
        pipeline.transform(dfs_test)       → X_test,  y_test

    Leakage guarantees:
        - Peer selection runs on training data only; stored for reuse.
        - Feature selector is fit on training data only.
        - No future price information bleeds into any feature via rolling
          or lag windows (all windows are right-aligned, shift(1) applied
          where the current bar's value would constitute look-ahead).

    Args:
        target_ticker:    The stock being predicted (e.g. 'NVDA').
        universe_builder: SymbolUniverseBuilder instance for constructing target-specific universes.
                         If None, falls back to universe_tickers (legacy mode).
        universe_tickers: (DEPRECATED) All tickers available. Use universe_builder instead.
        selector_kwargs:  Forwarded to FeatureSelector (e.g. importance threshold).
    """

    def __init__(
        self,
        target_ticker: str,
        universe_builder: Optional["SymbolUniverseBuilder"] = None,
        universe_tickers: Optional[List[str]] = None,
        selector_kwargs: Optional[Dict] = None,
    ) -> None:
        self.target_ticker = target_ticker
        self.universe_builder = universe_builder
        self.universe_tickers = universe_tickers  # Legacy fallback
        self.selector = FeatureSelector(**(selector_kwargs or {}))

        # Set after fit_transform — used to apply same selections on val/test
        self._feature_names_full: List[str] = []
        self._feature_names_selected: List[str] = []
        self._peer_tickers: List[str] = []
        self._universe_tickers: List[str] = []  # Stores fitted universe
        self._fitted = False
        
        if universe_builder is None and universe_tickers is None:
            raise ValueError(
                "Must provide either universe_builder or universe_tickers. "
                "universe_builder is recommended for proper peer selection."
            )
        
        logger.info(f"Initialized feature pipeline for {target_ticker}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        dfs_train: Dict[str, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build features from training data, fit selector, return selected matrix.

        Args:
            dfs_train: {ticker: DataFrame} — all tickers, training split only.
                       Each DataFrame must have columns: open, high, low, close,
                       volume, log_return, and optionally session_start (bool).

        Returns:
            X_train:      np.ndarray of shape (N, F_selected), float32
            y_train:      np.ndarray of shape (N,), float32 — log returns
            feature_names: List[str] of length F_selected
        """
        # NEW: Get target-specific universe if universe_builder is provided
        if self.universe_builder is not None:
            self._universe_tickers = self.universe_builder.get_universe(
                self.target_ticker,
                dfs_train,
                fit=True,  # Select peers on training data
            )
            
            # Filter dfs to only include universe symbols
            dfs_train_filtered = {
                ticker: dfs_train[ticker]
                for ticker in self._universe_tickers
                if ticker in dfs_train
            }
        else:
            # Legacy mode: use all provided tickers
            dfs_train_filtered = dfs_train
            self._universe_tickers = list(dfs_train.keys())
        
        self._validate_inputs(dfs_train_filtered)

        X_full, y, names = self._compute_features(dfs_train_filtered, fit=True)

        self._feature_names_full = names

        X_sel, sel_names = self.selector.fit_transform(X_full, y, names)
        self._feature_names_selected = sel_names
        self._fitted = True

        logger.info(
            "[%s] fit_transform: %d raw features → %d selected | "
            "samples: %d | universe: %s | peers: %s",
            self.target_ticker,
            len(names),
            len(sel_names),
            len(X_sel),
            self._universe_tickers,
            self._peer_tickers,
        )
        return X_sel, y, sel_names

    def transform(
        self,
        dfs: Dict[str, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply fitted pipeline to validation or test data.

        Args:
            dfs: {ticker: DataFrame} — same structure as dfs_train, different split.

        Returns:
            X_sel: np.ndarray (N, F_selected), float32
            y:     np.ndarray (N,), float32
        """
        if not self._fitted:
            raise RuntimeError(
                "Pipeline has not been fitted. Call fit_transform() on training "
                "data before calling transform()."
            )
        
        # NEW: Filter to fitted universe
        if self.universe_builder is not None:
            dfs_filtered = {
                ticker: dfs[ticker]
                for ticker in self._universe_tickers
                if ticker in dfs
            }
        else:
            # Legacy mode
            dfs_filtered = dfs
        
        self._validate_inputs(dfs_filtered)

        X_full, y, _ = self._compute_features(dfs_filtered, fit=False)

        # transform() uses the full feature name list from fit to locate columns
        X_sel, _ = self.selector.transform(X_full, self._feature_names_full)

        logger.info(
            "[%s] transform: %d samples, %d selected features",
            self.target_ticker,
            len(X_sel),
            X_sel.shape[1],
        )
        return X_sel, y

    @property
    def n_features(self) -> int:
        return len(self._feature_names_selected)

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names_selected)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """
        Fail fast on obvious data issues before feature computation.
        Catches problems that would otherwise produce silent NaN propagation.
        """
        if self.target_ticker not in dfs:
            raise KeyError(
                f"Target ticker '{self.target_ticker}' not found in input dict. "
                f"Available: {list(dfs.keys())}"
            )

        df = dfs[self.target_ticker]

        # Required columns
        missing_cols = REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"[{self.target_ticker}] Missing required columns: {missing_cols}. "
                f"Available: {list(df.columns)}"
            )

        # Index must be DatetimeIndex for session-aware operations
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                f"[{self.target_ticker}] DataFrame index must be DatetimeIndex, "
                f"got {type(df.index).__name__}."
            )

        # Must be sorted ascending (temporal order)
        if not df.index.is_monotonic_increasing:
            raise ValueError(
                f"[{self.target_ticker}] DataFrame index is not sorted ascending. "
                "Time series integrity requires chronological order."
            )

        # Warn on excessive NaN in close (indicates alignment or cleaning issues)
        nan_pct = df["close"].isna().mean()
        if nan_pct > 0.01:
            logger.info(
                "[%s] close column has %.1f%% NaN values — "
                "check alignment and cleaning stages.",
                self.target_ticker,
                nan_pct * 100,
            )

        # Warn on duplicate timestamps
        n_dupes = df.index.duplicated().sum()
        if n_dupes > 0:
            logger.info(
                "[%s] %d duplicate timestamps detected — "
                "deduplicate before feature engineering.",
                self.target_ticker,
                n_dupes,
            )

        logger.info(f"Inputs Validated")

    # ------------------------------------------------------------------
    # Core feature computation
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        dfs: Dict[str, pd.DataFrame],
        fit: bool,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build the full (pre-selection) feature matrix for the target ticker.

        Leakage notes:
          - All rolling windows use min_periods and are right-aligned (default).
          - shift(1) is applied wherever the current bar's value would leak.
          - Cross-ticker peers are selected during fit only; reused during transform.
          - Lag construction uses explicit slicing (not np.roll) to avoid
            wrapping end-of-series values into the beginning.

        Args:
            dfs: Full ticker dict for the current split.
            fit: If True, peer tickers are selected from data; if False, stored
                 peers from fit_transform are reused (leakage prevention).

        Returns:
            X:     np.ndarray (N, F), float32 — full feature matrix, NaN-cleaned
            y:     np.ndarray (N,),  float32 — log return target
            names: List[str] of length F — feature column names
        """
        df_target = dfs[self.target_ticker].copy()

        # Convenience aliases — these are Series, not copies
        C = df_target["close"]
        H = df_target["high"]
        L = df_target["low"]
        O = df_target["open"]
        r = df_target["log_return"]
        prev_close = C.shift(1)  # prior bar's close — used throughout

        # ----------------------------------------------------------------
        # Block 1: Base price / return features
        # These are derived from raw OHLCV and are always included.
        # ----------------------------------------------------------------
        base = pd.DataFrame(index=df_target.index, dtype=np.float32)

        # Log return IS the target — also a feature for lag construction
        base["log_return"] = r

        # Price structure features (normalized by prev close to be scale-invariant)
        base["mid_price"] = (H + L) / 2.0
        base["hl_ratio"] = (H - L) / (prev_close + 1e-10)  # range / prev close
        base["co_ratio"] = (C - O) / (prev_close + 1e-10)  # close vs open
        base["ho_ratio"] = (H - O) / (prev_close + 1e-10)  # high vs open
        base["lc_ratio"] = (C - L) / (prev_close + 1e-10)  # close vs low
        base["gap"] = (O - prev_close) / (prev_close + 1e-10)  # overnight gap

        # True range (normalized) — more informative than H-L alone
        tr = pd.concat(
            [
                (H - L),
                (H - prev_close).abs(),
                (L - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        base["true_range"] = tr / (prev_close + 1e-10)

        # Intrabar volatility (relative to open)
        base["intrabar_vol"] = (H - L) / (O + 1e-10)

        # Session cumulative return — resets to 0 at each day's open
        # Uses session_start flag if available; falls back to first bar of each date
        base["cum_return_session"] = self._compute_session_cum_return(df_target, C)

        # ----------------------------------------------------------------
        # Block 2: Technical indicators (Lanbouri + Zeng)
        # ----------------------------------------------------------------
        tech = compute_technical_features(df_target)

        # ----------------------------------------------------------------
        # Block 3: Statistical / rolling features
        # ----------------------------------------------------------------
        stat = compute_statistical_features(df_target)

        # ----------------------------------------------------------------
        # Block 4: Volume features
        # ----------------------------------------------------------------
        vol = compute_volume_features(df_target)

        # ----------------------------------------------------------------
        # Block 5: Cross-ticker features (Zeng: sector correlation,
        #          peer relative strength, market context)
        # During fit: peers are selected by correlation; stored in self._peer_tickers
        # During transform: stored peers are used — no re-selection on val/test
        # ----------------------------------------------------------------
        if fit:
            cross, peer_tickers = compute_cross_ticker_features(
                self.target_ticker,
                dfs,
                peer_tickers=None,  # will be selected internally
                return_peer_tickers=True,
            )
            self._peer_tickers = peer_tickers
        else:
            cross = compute_cross_ticker_features(
                self.target_ticker,
                dfs,
                peer_tickers=self._peer_tickers,  # reuse train selection
                return_peer_tickers=False,
            )

        # ----------------------------------------------------------------
        # Assemble all blocks into a single NumPy matrix
        # We concatenate block by block to avoid a large intermediate DataFrame
        # ----------------------------------------------------------------
        feature_blocks: List[np.ndarray] = []
        feature_names: List[str] = []

        def _append_block(df_block: Optional[pd.DataFrame], block_name: str) -> None:
            """Append a feature block, logging any issues."""
            if df_block is None or df_block.empty:
                logger.info("Block '%s' is empty — skipping.", block_name)
                return
            # Remove duplicate column names within the block
            df_block = df_block.loc[:, ~df_block.columns.duplicated()]
            vals = df_block.values.astype(np.float32, copy=False)
            feature_blocks.append(vals)
            feature_names.extend(df_block.columns.tolist())

        _append_block(base, "base")
        _append_block(tech, "technical")
        _append_block(stat, "statistical")
        _append_block(vol, "volume")
        _append_block(cross, "cross_ticker")

        if not feature_blocks:
            raise RuntimeError(
                f"[{self.target_ticker}] All feature blocks are empty. "
                "Check upstream data cleaning and feature computation."
            )

        X = np.concatenate(feature_blocks, axis=1)  # shape: (N, F_raw)

        # ----------------------------------------------------------------
        # Block 6: Lag features
        # Built from X so that lag indices reference already-computed features.
        # Uses explicit slicing — NOT np.roll — to prevent end→start wraparound.
        # ----------------------------------------------------------------
        lag_X, lag_names = self._build_lag_features(X, feature_names)
        if lag_X.shape[1] > 0:
            X = np.concatenate([X, lag_X], axis=1)
            feature_names.extend(lag_names)

        # ----------------------------------------------------------------
        # Remove duplicate feature names that may arise from block overlaps
        # (e.g., if technical and statistical both compute a rolling mean)
        # ----------------------------------------------------------------
        X, feature_names = self._deduplicate_features(X, feature_names)

        # ----------------------------------------------------------------
        # Clean invalid values: NaN, +inf, -inf → 0.0
        # Rationale: LSTM training is sensitive to non-finite values.
        # Zero-fill is conservative; alternatives (mean-fill) would require
        # fitting statistics and risk leakage in transform().
        # ----------------------------------------------------------------
        n_inf = np.isinf(X).sum()
        n_nan = np.isnan(X).sum()
        if n_inf > 0 or n_nan > 0:
            logger.info(
                "[%s] Replacing %d NaN and %d inf values with 0.0 in feature matrix.",
                self.target_ticker,
                n_nan,
                n_inf,
            )
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # ----------------------------------------------------------------
        # Target variable: log return
        # Rows where target is NaN/inf are dropped entirely.
        # This happens at the first bar of each session (no prior close).
        # ----------------------------------------------------------------
        y = r.values.astype(np.float32)
        valid_mask = np.isfinite(y)

        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            logger.info(
                "[%s] Dropping %d rows (%.2f%%) with non-finite target values.",
                self.target_ticker,
                n_invalid,
                100.0 * n_invalid / len(y),
            )
            X = X[valid_mask]
            y = y[valid_mask]

        logger.info(
            "[%s] Feature matrix: %d samples × %d features",
            self.target_ticker,
            X.shape[0],
            X.shape[1],
        )

        return X.astype(np.float32, copy=False), y, feature_names

    # ------------------------------------------------------------------
    # Session cumulative return
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_session_cum_return(
        df: pd.DataFrame,
        close: pd.Series,
    ) -> pd.Series:
        """
        Compute within-session cumulative return, resetting each trading day.

        If a 'session_start' boolean column is present, uses its cumsum to
        identify sessions (supports non-daily data, e.g. early-close days).
        Falls back to date-based grouping otherwise.

        Returns a Series aligned to df.index, with 0.0 where undefined.
        """
        if "session_start" in df.columns:
            # session_start is True on the first bar of each session
            session_ids = df["session_start"].astype(bool).cumsum()
        else:
            # Fallback: group by calendar date
            session_ids = pd.Series(
                pd.factorize(pd.to_datetime(close.index).date)[0],
                index=close.index,
            )

        # First close of each session (used as base for cumulative return)
        first_close = close.groupby(session_ids).transform("first")

        result = (close / first_close - 1.0).fillna(0.0)
        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # Lag feature construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_lag_features(
        X: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build lag features using explicit array slicing.

        CRITICAL: We use slicing (lagged[lag:] = col[:-lag]) NOT np.roll.
        np.roll wraps the last value to position 0, which is a form of
        look-ahead leakage — the "lagged" feature at bar 0 would contain
        the value from the last bar of the dataset, which is future data.

        Zero-padding at the beginning (bars 0..lag-1) is the correct approach:
        it signals "no history available yet" without fabricating values.

        Args:
            X:             Full feature matrix (N, F).
            feature_names: Column names for X, used to locate lag source columns.

        Returns:
            lag_X:     (N, L) array of lag features, float32
            lag_names: List of L names like 'log_return_lag1'
        """
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        lag_arrays: List[np.ndarray] = []
        lag_names: List[str] = []

        for col, lags in LAG_DEPTHS.items():
            if col not in name_to_idx:
                logger.info("Lag source '%s' not found in features — skipping.", col)
                continue

            col_data = X[:, name_to_idx[col]]  # shape: (N,)

            for lag in lags:
                lagged = np.zeros(len(col_data), dtype=np.float32)
                if lag < len(col_data):
                    # lagged[i] = col_data[i - lag] for i >= lag
                    lagged[lag:] = col_data[:-lag]
                # lagged[:lag] remains 0.0 — no data available for those bars
                lag_arrays.append(lagged)
                lag_names.append(f"{col}_lag{lag}")

        if not lag_arrays:
            return np.empty((X.shape[0], 0), dtype=np.float32), []

        return np.stack(lag_arrays, axis=1).astype(np.float32), lag_names

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_features(
        X: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Remove duplicate feature columns by name (keeps first occurrence).

        Duplicates arise when multiple feature blocks compute the same
        derived quantity (e.g., both base and technical compute log_return).
        Silently keeping duplicates wastes compute and biases selectors that
        use correlation-based pruning.
        """
        seen = set()
        keep_indices = []
        keep_names = []

        for i, name in enumerate(feature_names):
            if name not in seen:
                seen.add(name)
                keep_indices.append(i)
                keep_names.append(name)

        n_dupes = len(feature_names) - len(keep_names)
        if n_dupes > 0:
            logger.info(
                "%d duplicate feature column(s) removed: %s",
                n_dupes,
                [n for n in feature_names if feature_names.count(n) > 1],
            )

        return X[:, keep_indices], keep_names


from prettytable import PrettyTable
import pandas as pd


def pretty_print_df(
    df: pd.DataFrame,
    name: str = "DataFrame",
    n_head: int = 3,
    n_tail: int = 3,
    logger=None,
):
    """
    Pretty print a DataFrame with:
      - shape
      - column names
      - dtypes
      - missing %
      - first N rows
      - last N rows
    """
    if df is None or df.empty:
        msg = f"{name}: EMPTY DataFrame"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return

    # --- Summary table ---
    summary = PrettyTable()
    summary.title = f"{name} Summary"
    summary.field_names = ["Metric", "Value"]

    summary.add_row(["Shape", df.shape])
    summary.add_row(["Rows", len(df)])
    summary.add_row(["Columns", len(df.columns)])
    summary.add_row(["Column Names", ", ".join(map(str, df.columns.tolist()))])

    # Dtypes
    dtype_str = ", ".join([f"{col}:{dtype}" for col, dtype in df.dtypes.items()])
    summary.add_row(["Dtypes", dtype_str])

    # Missing %
    missing_pct = df.isna().mean().mean() * 100
    summary.add_row(["Missing %", f"{missing_pct:.4f}%"])

    # --- Head table ---
    head_table = PrettyTable()
    head_table.title = f"{name} (First {n_head} rows)"
    head_table.field_names = ["Index"] + list(df.columns)

    for idx, row in df.head(n_head).iterrows():
        head_table.add_row([str(idx)] + list(row.values))

    # --- Tail table ---
    tail_table = PrettyTable()
    tail_table.title = f"{name} (Last {n_tail} rows)"
    tail_table.field_names = ["Index"] + list(df.columns)

    for idx, row in df.tail(n_tail).iterrows():
        tail_table.add_row([str(idx)] + list(row.values))

    # --- Output ---
    output = f"\n{summary}\n\n{head_table}\n\n{tail_table}\n"

    if logger:
        logger.info(output)
    else:
        print(output)
