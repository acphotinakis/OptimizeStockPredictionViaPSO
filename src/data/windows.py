"""
src/data/windows.py

Sliding-window construction for both LSTM and XGBoost models,
plus walk-forward validation fold generation.

TWO PUBLIC FUNCTIONS
--------------------
build_windows()
    Builds windows from a single contiguous array.
    Used for: normal train / val / test splits.
    Output shapes:
        LSTM    → (M, lookback, F)
        XGBoost → (M, lookback * F)

build_walk_forward_windows()
    Partitions the data into sequential (train, test) folds.
    Each fold calls build_windows() internally.
    Used for: walk-forward validation / backtesting.
    Returns a list of WalkForwardFold objects.

STEP SIZE RULES (from TRD specification)
-----------------------------------------
- step_size is NOT a model parameter. It is a fixed system parameter.
- step_size >= forecast_horizon  ← hard constraint enforced here.
  Violating this causes target leakage between consecutive test windows.
- step_size expressed in raw bars matching the dataset frequency.
- step_size = forecast_horizon is the standard financial setting:
  non-overlapping test targets, cleanest evaluation.

WINDOW TYPES
------------
expanding   Train set grows each fold (train_start fixed at 0).
            Preferred: model always sees all historical data.
rolling     Train set slides forward at fixed size.
            Used when distribution shift makes old data harmful.

SESSION BOUNDARY RULES (carried over from build_windows)
---------------------------------------------------------
- No window crosses a session boundary mid-window.
- No window predicts a target bar that opens a new session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type for walk-forward folds
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardFold:
    """
    One fold produced by build_walk_forward_windows().

    Attributes:
        fold:             Zero-based fold index.
        train_X:          Training feature windows.
        train_y:          Training targets.
        test_X:           Test feature windows.
        test_y:           Test targets.
        train_bar_range:  (first_bar, last_bar) of the raw training slice.
        test_bar_range:   (first_bar, last_bar) of the raw test slice.
        n_train_windows:  Number of valid training windows (set post-init).
        n_test_windows:   Number of valid test windows (set post-init).
    """

    fold: int
    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    train_bar_range: Tuple[int, int]
    test_bar_range: Tuple[int, int]
    n_train_windows: int = field(init=False)
    n_test_windows: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_train_windows = len(self.train_y)
        self.n_test_windows = len(self.test_y)

    def __repr__(self) -> str:
        return (
            f"WalkForwardFold(fold={self.fold}, "
            f"train_bars={self.train_bar_range}, "
            f"test_bars={self.test_bar_range}, "
            f"train_windows={self.n_train_windows}, "
            f"test_windows={self.n_test_windows})"
        )


# ---------------------------------------------------------------------------
# build_windows
# ---------------------------------------------------------------------------


def build_windows(
    X: np.ndarray,
    returns: np.ndarray,
    session_starts: np.ndarray,
    lookback: int,
    model_type: str = "lstm",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows from a contiguous feature matrix.

    A window ending at bar i is valid only if:
      1. No session boundary falls WITHIN bars [window_start+1 ... i].
         The anchor bar (window_start) may itself be a session start.
      2. The target bar (i+1) is NOT the start of a new session.

    Args:
        X:              Feature matrix [N, F], already scaled.
        returns:        Log return targets [N], already scaled.
        session_starts: Boolean array [N]. True on first bar of each session.
        lookback:       Number of bars per window.
        model_type:     "lstm"    → output shape (M, lookback, F)
                        "xgboost" → output shape (M, lookback * F)

    Returns:
        X_out: Windowed feature array.
        y:     Target returns, shape (M,).

    Raises:
        ValueError: On invalid inputs or if no valid windows can be built.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    model_type = model_type.lower().strip()
    if model_type not in {"lstm", "xgboost"}:
        raise ValueError(f"model_type must be 'lstm' or 'xgboost', got '{model_type}'.")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D [N, F], got shape {X.shape}.")
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}.")

    N, F = X.shape

    if len(returns) != N:
        raise ValueError(f"returns length ({len(returns)}) must match X rows ({N}).")
    if len(session_starts) != N:
        raise ValueError(
            f"session_starts length ({len(session_starts)}) must match X rows ({N})."
        )
    if N < lookback + 1:
        raise ValueError(f"Not enough bars: N={N} must be > lookback+1={lookback + 1}.")

    session_starts = np.asarray(session_starts, dtype=bool)

    logger.info(
        "build_windows | model=%s | lookback=%d | N=%d bars | F=%d features",
        model_type,
        lookback,
        N,
        F,
    )

    # ------------------------------------------------------------------
    # Find valid window end indices
    # ------------------------------------------------------------------
    n_sessions = int(session_starts.sum())
    n_rejected_mid = 0
    n_rejected_tgt = 0
    valid_indices: List[int] = []

    for i in range(lookback - 1, N - 1):
        window_start = i - lookback + 1

        # Rule 1: reject if a new session starts WITHIN the window
        # (window_start itself may be a session start — that is fine)
        if np.any(session_starts[window_start + 1 : i + 1]):
            n_rejected_mid += 1
            continue

        # Rule 2: reject if the target bar opens a new session
        if session_starts[i + 1]:
            n_rejected_tgt += 1
            continue

        valid_indices.append(i)

    M = len(valid_indices)

    logger.info(
        "Window filtering | sessions=%d | rejected_mid_boundary=%d | "
        "rejected_target_boundary=%d | valid_windows=%d",
        n_sessions,
        n_rejected_mid,
        n_rejected_tgt,
        M,
    )

    if M == 0:
        raise ValueError(
            f"No valid windows found. lookback={lookback}, N={N}, "
            f"sessions={n_sessions}. "
            f"Check that each session has more bars than the lookback period."
        )

    valid_indices_arr = np.array(valid_indices, dtype=np.int64)

    # ------------------------------------------------------------------
    # Vectorized window extraction via stride tricks
    # sliding_window_view shape: (N - lookback + 1, lookback, F)
    # Row k corresponds to a window ending at bar k + lookback - 1.
    # For a window ending at bar i, the row index is i - (lookback - 1).
    # ------------------------------------------------------------------
    logger.info("Building windowed array via sliding_window_view ...")

    all_windows = np.lib.stride_tricks.sliding_window_view(
        X, window_shape=(lookback, F)
    ).squeeze(
        axis=1
    )  # (N - lookback + 1, lookback, F)

    window_rows = valid_indices_arr - (lookback - 1)
    X_3d = all_windows[window_rows].copy().astype(np.float32)
    y = returns[valid_indices_arr + 1].astype(np.float32)

    # ------------------------------------------------------------------
    # Format for model type
    # ------------------------------------------------------------------
    if model_type == "lstm":
        X_out = X_3d
        logger.info(
            "LSTM output | X_windowed: %s | y: %s | dtype: %s",
            X_out.shape,
            y.shape,
            X_out.dtype,
        )
    else:
        # Flatten (M, lookback, F) → (M, lookback * F)
        # Row-major (C order): oldest bar first, newest bar last.
        X_out = X_3d.reshape(M, lookback * F)
        logger.info(
            "XGBoost output | X_flat: %s | y: %s | dtype: %s",
            X_out.shape,
            y.shape,
            X_out.dtype,
        )

    # ------------------------------------------------------------------
    # Output sanity checks
    # ------------------------------------------------------------------
    assert X_out.shape[0] == M, f"Row count mismatch: {X_out.shape[0]} != {M}"
    assert y.shape == (M,), f"Target shape mismatch: {y.shape} != ({M},)"
    assert np.all(np.isfinite(X_out)), "Non-finite values in X_out after windowing"
    assert np.all(np.isfinite(y)), "Non-finite values in y after windowing"

    retention_pct = 100.0 * M / max(N - lookback, 1)
    logger.info(
        "build_windows complete | M=%d valid windows | "
        "%.1f%% of possible windows retained",
        M,
        retention_pct,
    )

    return X_out, y


# ---------------------------------------------------------------------------
# build_walk_forward_windows
# ---------------------------------------------------------------------------


def build_walk_forward_windows(
    X: np.ndarray,
    returns: np.ndarray,
    session_starts: np.ndarray,
    lookback: int,
    step_size: int,
    forecast_horizon: int,
    min_train_bars: int,
    model_type: str = "lstm",
    window_type: Literal["expanding", "rolling"] = "expanding",
    rolling_train_bars: int | None = None,
) -> List[WalkForwardFold]:
    """
    Generate walk-forward validation folds.

    Each fold produces a (train, test) pair by:
      1. Taking a training slice up to the current fold boundary.
      2. Taking a test slice of exactly step_size bars after that boundary.
      3. Calling build_windows() on each slice independently.
      4. Advancing the boundary by step_size bars for the next fold.

    Session boundary filtering is enforced separately inside each slice
    via build_windows() — windows are never allowed to cross sessions.

    Walk-forward layout (expanding, step_size=H, min_train_bars=T)
    ---------------------------------------------------------------
    Fold 0: train=[0, T)      test=[T,   T+H)
    Fold 1: train=[0, T+H)    test=[T+H, T+2H)
    Fold 2: train=[0, T+2H)   test=[T+2H, T+3H)
    ...

    Walk-forward layout (rolling, rolling_train_bars=W, step_size=H)
    ----------------------------------------------------------------
    Fold 0: train=[0,     W)    test=[W,     W+H)
    Fold 1: train=[H,     W+H)  test=[W+H,   W+2H)
    Fold 2: train=[2H,    W+2H) test=[W+2H,  W+3H)
    ...

    Step size constraint (TRD-level, enforced hard)
    -----------------------------------------------
    step_size >= forecast_horizon

    If violated, consecutive test windows share target bars → leakage.
    Example with step_size=1, forecast_horizon=5:
        Fold 0 test targets: bars [T+1, T+2, T+3, T+4, T+5]
        Fold 1 test targets: bars [T+2, T+3, T+4, T+5, T+6]  ← overlap!

    Args:
        X:                  Feature matrix [N, F], scaled.
        returns:            Log return targets [N], scaled.
        session_starts:     Boolean array [N]. True on first bar of each session.
        lookback:           Bars per window (PSO search parameter).
        step_size:          Bars the test window advances per fold.
                            Must satisfy: step_size >= forecast_horizon.
        forecast_horizon:   How many bars ahead the model forecasts.
                            Used only for the leakage guard — the actual
                            prediction is always next-bar inside build_windows.
        min_train_bars:     Minimum raw bars the training slice must contain.
                            Must be > lookback + 1.
                            Recommended: at least several sessions of data.
        model_type:         "lstm" or "xgboost" — forwarded to build_windows().
        window_type:        "expanding" → training set grows each fold.
                            "rolling"   → training set slides at fixed width.
        rolling_train_bars: Required when window_type="rolling".
                            Number of bars per rolling training slice.

    Returns:
        List[WalkForwardFold]: One fold per valid step. Folds where
        build_windows() fails (e.g. too few bars for even one window) are
        skipped with a WARNING log — they do not raise an exception.

    Raises:
        ValueError: On hard constraint violations or if zero folds are produced.
    """
    # ------------------------------------------------------------------
    # Validate parameters
    # ------------------------------------------------------------------
    model_type = model_type.lower().strip()
    window_type = window_type.lower().strip()

    if model_type not in {"lstm", "xgboost"}:
        raise ValueError(f"model_type must be 'lstm' or 'xgboost', got '{model_type}'.")
    if window_type not in {"expanding", "rolling"}:
        raise ValueError(
            f"window_type must be 'expanding' or 'rolling', got '{window_type}'."
        )
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D [N, F], got shape {X.shape}.")

    N, F = X.shape

    if len(returns) != N or len(session_starts) != N:
        raise ValueError(
            f"X ({N}), returns ({len(returns)}), session_starts "
            f"({len(session_starts)}) must all have the same length."
        )
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}.")
    if forecast_horizon < 1:
        raise ValueError(f"forecast_horizon must be >= 1, got {forecast_horizon}.")
    if step_size < 1:
        raise ValueError(f"step_size must be >= 1, got {step_size}.")

    # Core leakage constraint from TRD spec
    if step_size < forecast_horizon:
        raise ValueError(
            f"step_size ({step_size}) < forecast_horizon ({forecast_horizon}). "
            f"This causes target leakage between consecutive test windows. "
            f"Enforce: step_size >= forecast_horizon."
        )

    if min_train_bars <= lookback + 1:
        raise ValueError(
            f"min_train_bars ({min_train_bars}) must be > lookback+1 "
            f"({lookback + 1}). A training slice this short produces no windows."
        )
    if min_train_bars >= N:
        raise ValueError(
            f"min_train_bars ({min_train_bars}) >= N ({N}). No room for test folds."
        )

    if window_type == "rolling":
        if rolling_train_bars is None:
            raise ValueError(
                "rolling_train_bars must be set when window_type='rolling'."
            )
        if rolling_train_bars <= lookback + 1:
            raise ValueError(
                f"rolling_train_bars ({rolling_train_bars}) must be > "
                f"lookback+1 ({lookback + 1})."
            )
        if rolling_train_bars > N:
            raise ValueError(
                f"rolling_train_bars ({rolling_train_bars}) exceeds N ({N})."
            )

    session_starts = np.asarray(session_starts, dtype=bool)

    # ------------------------------------------------------------------
    # Pre-flight summary log
    # ------------------------------------------------------------------
    logger.info(
        "build_walk_forward_windows | "
        "model=%s | window=%s | lookback=%d | step_size=%d | "
        "forecast_horizon=%d | min_train_bars=%d | N=%d bars | F=%d features",
        model_type,
        window_type,
        lookback,
        step_size,
        forecast_horizon,
        min_train_bars,
        N,
        F,
    )

    if step_size == forecast_horizon:
        logger.info(
            "step_size == forecast_horizon (%d): non-overlapping test targets "
            "(standard financial setting, no target leakage between folds).",
            step_size,
        )
    elif step_size > forecast_horizon:
        logger.info(
            "step_size (%d) > forecast_horizon (%d): gaps between test windows "
            "(coarse backtest mode).",
            step_size,
            forecast_horizon,
        )

    n_expected_folds = max(0, (N - min_train_bars + step_size - 1) // step_size)
    logger.info("Expected folds (approximate): %d", n_expected_folds)

    # ------------------------------------------------------------------
    # Fold generation loop
    # ------------------------------------------------------------------
    folds: List[WalkForwardFold] = []
    fold_idx: int = 0
    skipped: int = 0

    # test_start: first bar of the test slice for this fold
    # Advances by step_size each iteration
    test_start: int = min_train_bars

    while test_start < N:
        test_end = min(test_start + step_size, N)

        # ── Training slice bounds ─────────────────────────────────────
        if window_type == "expanding":
            train_start = 0
            train_end = test_start  # all bars before the test window
        else:  # rolling
            train_end = test_start
            train_start = max(0, train_end - rolling_train_bars)

        actual_train_bars = train_end - train_start
        actual_test_bars = test_end - test_start

        logger.info(
            "Fold %d | train=[%d, %d) %d bars | test=[%d, %d) %d bars",
            fold_idx,
            train_start,
            train_end,
            actual_train_bars,
            test_start,
            test_end,
            actual_test_bars,
        )

        # ── Skip if train slice too small ─────────────────────────────
        if actual_train_bars < min_train_bars:
            logger.warning(
                "Fold %d skipped: train slice has %d bars < min_train_bars=%d.",
                fold_idx,
                actual_train_bars,
                min_train_bars,
            )
            skipped += 1
            test_start += step_size
            fold_idx += 1
            continue

        # ── Skip if test slice too small for even one window ──────────
        # build_windows() needs at least lookback+1 bars to make one window
        if actual_test_bars < lookback + 1:
            logger.warning(
                "Fold %d skipped: test slice has %d bars < lookback+1=%d "
                "(partial final fold — stopping).",
                fold_idx,
                actual_test_bars,
                lookback + 1,
            )
            break

        # ── Slice raw arrays (zero-copy views) ────────────────────────
        X_train_raw = X[train_start:train_end]
        y_train_raw = returns[train_start:train_end]
        ss_train_raw = session_starts[train_start:train_end]

        X_test_raw = X[test_start:test_end]
        y_test_raw = returns[test_start:test_end]
        ss_test_raw = session_starts[test_start:test_end]

        # ── Build train windows ───────────────────────────────────────
        try:
            train_X, train_y = build_windows(
                X_train_raw,
                y_train_raw,
                ss_train_raw,
                lookback=lookback,
                model_type=model_type,
            )
        except ValueError as exc:
            logger.warning(
                "Fold %d skipped: build_windows failed on TRAIN slice. "
                "train=[%d, %d). Error: %s",
                fold_idx,
                train_start,
                train_end,
                exc,
            )
            skipped += 1
            test_start += step_size
            fold_idx += 1
            continue

        # ── Build test windows ────────────────────────────────────────
        try:
            test_X, test_y = build_windows(
                X_test_raw,
                y_test_raw,
                ss_test_raw,
                lookback=lookback,
                model_type=model_type,
            )
        except ValueError as exc:
            logger.warning(
                "Fold %d skipped: build_windows failed on TEST slice. "
                "test=[%d, %d). Error: %s",
                fold_idx,
                test_start,
                test_end,
                exc,
            )
            skipped += 1
            test_start += step_size
            fold_idx += 1
            continue

        # ── Assemble and store fold ───────────────────────────────────
        fold = WalkForwardFold(
            fold=fold_idx,
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            train_bar_range=(train_start, train_end - 1),
            test_bar_range=(test_start, test_end - 1),
        )
        folds.append(fold)

        logger.info(
            "Fold %d | train_windows=%d | test_windows=%d | "
            "train_bars=[%d, %d) | test_bars=[%d, %d)",
            fold_idx,
            fold.n_train_windows,
            fold.n_test_windows,
            train_start,
            train_end,
            test_start,
            test_end,
        )

        # ── Advance to next fold ──────────────────────────────────────
        test_start += step_size
        fold_idx += 1

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    if not folds:
        raise ValueError(
            f"No valid walk-forward folds produced. "
            f"N={N}, lookback={lookback}, step_size={step_size}, "
            f"min_train_bars={min_train_bars}. "
            f"Reduce min_train_bars or step_size, or provide more data."
        )

    total_train_windows = sum(f.n_train_windows for f in folds)
    total_test_windows = sum(f.n_test_windows for f in folds)

    logger.info(
        "build_walk_forward_windows complete | "
        "folds_produced=%d | folds_skipped=%d | "
        "total_train_windows=%d | total_test_windows=%d",
        len(folds),
        skipped,
        total_train_windows,
        total_test_windows,
    )

    return folds


# ---------------------------------------------------------------------------
# Helper: XGBoost feature names for flattened windows
# ---------------------------------------------------------------------------


def get_xgboost_feature_names(
    feature_names: List[str],
    lookback: int,
) -> List[str]:
    """
    Generate column names for the XGBoost flattened feature matrix.

    Column order matches row-major flattening in build_windows:
        [t-(L-1)__feat_0, ..., t-(L-1)__feat_{F-1},
         ...
         t-0__feat_0,     ..., t-0__feat_{F-1}]

    t-0 = most recent bar. t-(L-1) = oldest bar in window.

    Args:
        feature_names: Original feature names, length F.
        lookback:      Window size L.

    Returns:
        List of L*F column name strings.
    """
    names = [
        f"t-{lookback - 1 - bar}__{name}"
        for bar in range(lookback)
        for name in feature_names
    ]
    logger.info(
        "XGBoost feature names generated: %d names (lookback=%d, F=%d)",
        len(names),
        lookback,
        len(feature_names),
    )
    return names
