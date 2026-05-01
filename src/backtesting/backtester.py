from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import logging


# Add project root to path
# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluation.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    cagr,
    calmar_ratio,
    profit_factor,
    win_rate,
)

logger = logging.getLogger(__name__)

SIGNAL_THRESHOLD = 1e-4


class SessionEvent(Enum):
    """Enum for session boundary events."""

    NONE = 0
    OPEN = 1
    CLOSE = 2


@dataclass
class BacktestResult:
    """Container for backtest engine outputs.

    Attributes:
        equity_curve: Per-bar portfolio value, prepended with the initial
            capital seed so ``equity_curve[0] == V0``.
        bar_returns: Per-bar strategy returns (after costs).
        trade_log: One row per closed trade.
        signals: Per-bar ternary signals fed to the engine, length N.
        bar_costs: Per-bar transaction costs incurred (one entry per bar,
            zero on bars without trade events), length N.
        sharpe: Annualised Sharpe ratio.
        sortino: Annualised Sortino ratio.
        mdd: Maximum drawdown.
        cagr_: Compound annual growth rate.
        calmar: Calmar ratio.
        profit_factor_: Gross profit / gross loss.
        win_rate_: Fraction of winning bars.
        n_trades: Number of closed trades.
        turnover: Mean absolute change in signal per bar.
    """

    equity_curve: np.ndarray
    bar_returns: np.ndarray
    trade_log: pd.DataFrame
    signals: np.ndarray
    bar_costs: np.ndarray

    # metrics computed by engine ONLY
    sharpe: float
    sortino: float
    mdd: float
    cagr_: float
    calmar: float
    profit_factor_: float
    win_rate_: float
    n_trades: int
    turnover: float


# ======================================================================
# Signal generation
# ======================================================================


def generate_signals(y_pred: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Convert predicted log returns to ternary trade signals {-1, 0, +1}.

    Args:
        y_pred: Predicted log returns.
        threshold: Minimum absolute value to generate a signal (default: 1bp).

    Returns:
        Array of signals: +1 (long), 0 (flat), -1 (short).
    """
    sig = np.zeros(len(y_pred), dtype=np.float32)
    sig[y_pred > threshold] = 1.0
    sig[y_pred < -threshold] = -1.0
    return sig


def alt_generate_signals(
    predictions: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Convert return predictions to trading signals.

    Rule: signal = sign(predicted_return)
        +1: Long (if prediction > threshold)
        -1: Short (if prediction < -threshold)
         0: Neutral (if |prediction| <= threshold)

    Args:
        predictions: Predicted returns
        threshold: Minimum return to trigger signal (default: 0.0)

    Returns:
        Signals array (+1, 0, -1)
    """
    signals = np.zeros_like(predictions, dtype=np.int8)
    signals[predictions > threshold] = 1  # LONG
    signals[predictions < -threshold] = -1  # SHORT
    # |prediction| <= threshold --> 0 (NEUTRAL)

    n_long = np.sum(signals == 1)
    n_short = np.sum(signals == -1)
    n_neutral = np.sum(signals == 0)

    logger.info(f"Signals generated: {len(signals)} total")
    logger.info(f"  Long: {n_long} ({100*n_long/len(signals):.1f}%)")
    logger.info(f"  Short: {n_short} ({100*n_short/len(signals):.1f}%)")
    logger.info(f"  Neutral: {n_neutral} ({100*n_neutral/len(signals):.1f}%)")

    return signals


class Backtester:
    """Event-driven 1-minute bar backtester with decomposed logic."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        position_fraction: float = 0.02,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        stop_loss: float = 0.02,
        daily_loss_limit: float = 0.05,
        signal_threshold: float = SIGNAL_THRESHOLD,
    ) -> None:
        self.V0 = initial_capital
        self.f = position_fraction
        self.tc = transaction_cost
        self.slip = slippage
        self.stop_loss = stop_loss
        self.daily_limit = daily_loss_limit
        self.threshold = signal_threshold
        # CONFIG LOG
        config = {
            "initial_capital": self.V0,
            "position_fraction": self.f,
            "transaction_cost": self.tc,
            "slippage": self.slip,
            "stop_loss": self.stop_loss,
            "daily_loss_limit": self.daily_limit,
            "signal_threshold": self.threshold,
        }

        logger.info("=" * 80)
        logger.info("BACKTESTER CONFIG")
        logger.info("=" * 80)
        logger.info(config)
        logger.info("=" * 80)

    def run(
        self,
        y_pred: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
        timestamps: pd.DatetimeIndex,
        session_starts: np.ndarray | None = None,
    ) -> BacktestResult:
        """
        Main backtest loop - now a high-level orchestrator.
        """
        N = len(y_pred)
        signals = self._make_signals(y_pred).copy()
        state = self._init_state()

        # Pre-compute timezone index for session detection
        et_index = timestamps.tz_convert("America/New_York")

        for t in range(N):
            # Reset per-bar cost accumulator before any trade events fire.
            state["bar_cost_accumulator"] = 0.0

            # 1. Check session boundaries
            session_event = self._check_session_boundary(
                t, timestamps, session_starts, et_index
            )

            # 2. Update session state if opening
            if session_event == SessionEvent.OPEN:
                state["session_open_equity"] = state["equity"][-1]
                state["daily_halt"] = False

            # 3. Determine target position (respecting daily halt and session close)
            force_flat = (session_event == SessionEvent.CLOSE) or state["daily_halt"]
            target = 0 if force_flat else int(signals[t])

            # 4. Execute position changes
            fill_price = opens[t]
            state = self._update_position(t, target, fill_price, timestamps[t], state)

            # 5. Mark to market and apply risk controls
            state = self._mark_to_market(t, closes[t], state)
            state = self._apply_risk_controls(t, state, signals)

            # 6. Record bar return and per-bar cost
            bar_ret = (state["equity"][-1] - state["equity"][-2]) / (
                state["equity"][-2] + 1e-10
            )
            state["bar_returns"].append(bar_ret)
            state["bar_costs"].append(float(state["bar_cost_accumulator"]))
            state["current_step"] = t

        return self._build_result(state, signals)

    def _init_state(self) -> Dict[str, Any]:
        """Initialize backtest state."""
        return {
            "equity": [self.V0],
            "position": 0,
            "entry_price": 0.0,
            "entry_equity": 0.0,
            "entry_time": None,
            "n_shares_held": 0.0,
            "session_open_equity": self.V0,
            "daily_halt": False,
            "trades": [],
            "bar_returns": [],
            "bar_costs": [],
            "bar_cost_accumulator": 0.0,
            "current_step": 0,
        }

    def _check_session_boundary(
        self,
        t: int,
        timestamps: pd.DatetimeIndex,
        session_starts: np.ndarray | None,
        et_index: pd.DatetimeIndex,
    ) -> SessionEvent:
        """Detect if current bar is session open, close, or mid-session.

        Close takes priority over open at the final bar so end-of-data forces a
        flat position even when the same bar would otherwise be flagged as a
        new session open.
        """
        if session_starts is not None:
            is_open = bool(session_starts[t])
            is_close = (t == len(timestamps) - 1) or bool(session_starts[t + 1])
        else:
            # Time-based detection (9:30 open, 16:00 close ET)
            is_open = et_index[t].hour == 9 and et_index[t].minute == 30
            is_close = (et_index[t].hour == 16 and et_index[t].minute == 0) or (
                t == len(timestamps) - 1
            )

        if is_close:
            return SessionEvent.CLOSE
        elif is_open:
            return SessionEvent.OPEN
        return SessionEvent.NONE

    def _update_position(
        self,
        t: int,
        target: int,
        fill_price: float,
        timestamp: pd.Timestamp,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle position entry and exit logic.
        """
        current_pos = state["position"]

        if target == current_pos:
            return state  # No change

        # Close existing position if different from target
        if current_pos != 0:
            state = self._close_position(t, fill_price, timestamp, state)

        # Open new position if target is non-zero
        if target != 0:
            state = self._open_position(t, target, fill_price, timestamp, state)

        return state

    def _close_position(
        self,
        t: int,
        fill_price: float,
        timestamp: pd.Timestamp,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Close current position and record trade."""
        position = state["position"]
        entry_price = state["entry_price"]
        entry_equity = state["entry_equity"]
        entry_time = state["entry_time"]
        n_shares = state["n_shares_held"]

        # Apply slippage against position direction
        close_slip = fill_price * (1.0 - self.slip * np.sign(position))

        # Calculate P&L
        pnl = position * n_shares * (close_slip - entry_price)
        cost = self.tc * self.f * entry_equity

        # Update equity
        current_equity = state["equity"][-1]
        new_equity = current_equity + pnl - cost
        state["equity"][-1] = new_equity
        state["bar_cost_accumulator"] += cost

        # Record trade
        state["trades"].append(
            {
                "entry_time": entry_time,
                "exit_time": timestamp,
                "direction": position,
                "entry_price": entry_price,
                "exit_price": close_slip,
                "n_shares": n_shares,
                "pnl_gross": pnl,
                "cost": cost,
                "pnl_net": pnl - cost,
            }
        )

        # Reset position state
        state["position"] = 0
        state["entry_price"] = 0.0
        state["entry_equity"] = 0.0
        state["entry_time"] = None
        state["n_shares_held"] = 0.0

        return state

    def _open_position(
        self,
        t: int,
        target: int,
        fill_price: float,
        timestamp: pd.Timestamp,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Open new position."""
        # Apply slippage in direction of trade
        open_slip = fill_price * (1.0 + self.slip * target)

        # Calculate costs
        current_equity = state["equity"][-1]
        cost = self.tc * self.f * current_equity
        state["equity"][-1] = current_equity - cost
        state["bar_cost_accumulator"] += cost

        # Set entry state
        state["position"] = target
        state["entry_price"] = open_slip
        state["entry_equity"] = state["equity"][-1]
        state["entry_time"] = timestamp

        # Lock in share count (constant until exit)
        state["n_shares_held"] = (self.f * state["entry_equity"]) / (open_slip + 1e-10)

        return state

    def _mark_to_market(
        self,
        t: int,
        close_price: float,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate unrealized P&L and update equity."""
        if state["position"] == 0:
            # No position - equity stays flat
            state["equity"].append(state["equity"][-1])
            return state

        # Calculate MTM P&L
        position = state["position"]
        n_shares = state["n_shares_held"]
        entry_price = state["entry_price"]

        mtm_pnl = position * n_shares * (close_price - entry_price)
        new_equity = state["equity"][-1] + mtm_pnl
        state["equity"].append(new_equity)

        return state

    def _apply_risk_controls(
        self,
        t: int,
        state: Dict[str, Any],
        signals: np.ndarray,
    ) -> Dict[str, Any]:
        """Apply stop-loss and daily loss limits.

        Note: stop-losses are simulated as next-bar open exits (the next bar's
        signal is forced to zero). They are not modeled as intra-bar fills.
        """
        if state["position"] == 0:
            return state

        entry_equity = state["entry_equity"]
        current_equity = state["equity"][-1]

        # Check stop-loss
        trade_dd = (entry_equity - current_equity) / (entry_equity + 1e-10)
        if trade_dd > self.stop_loss and t + 1 < len(signals):
            signals[t + 1] = 0  # Force flat next bar

        # Check daily loss limit
        session_open_equity = state["session_open_equity"]
        session_dd = (session_open_equity - current_equity) / (
            session_open_equity + 1e-10
        )
        if session_dd > self.daily_limit:
            state["daily_halt"] = True

        return state

    def _build_result(
        self,
        state: Dict[str, Any],
        signals: np.ndarray,
    ) -> BacktestResult:
        """Compute final metrics and return result object.

        The equity curve is prepended with the initial capital seed ``V0`` so
        downstream metrics like CAGR use the true starting capital as the
        denominator instead of the post-cost mark-to-market of bar 0.
        """
        equity_curve = np.concatenate(
            [np.array([self.V0], dtype=float), np.array(state["equity"][1:])]
        )
        bar_returns = np.array(state["bar_returns"])
        bar_costs = np.array(state["bar_costs"])
        trade_df = pd.DataFrame(state["trades"])

        # Calculate metrics
        sr = sharpe_ratio(bar_returns)
        sor = sortino_ratio(bar_returns)
        mdd = max_drawdown(equity_curve)
        c = cagr(equity_curve)
        cal = calmar_ratio(equity_curve)
        pf = profit_factor(bar_returns)
        wr = win_rate(bar_returns)
        to = float(np.mean(np.abs(np.diff(signals, prepend=0))))

        return BacktestResult(
            equity_curve=equity_curve,
            bar_returns=bar_returns,
            trade_log=trade_df,
            signals=np.asarray(signals),
            bar_costs=bar_costs,
            sharpe=sr,
            sortino=sor,
            mdd=mdd,
            cagr_=c,
            calmar=cal,
            profit_factor_=pf,
            win_rate_=wr,
            n_trades=len(trade_df),
            turnover=to,
        )

    def _make_signals(self, y_pred: np.ndarray) -> np.ndarray:
        """Convert predictions to ternary signals."""
        return generate_signals(y_pred, self.threshold)

    def optimize_threshold(
        self,
        y_pred_val: np.ndarray,
        y_true_val: np.ndarray,
        opens_val: np.ndarray,
        closes_val: np.ndarray,
        timestamps_val: pd.DatetimeIndex,
        session_starts_val: np.ndarray | None = None,
        candidates: Optional[List[float]] = None,
    ) -> float:
        """Grid-search the signal threshold maximising Sharpe."""
        if candidates is None:
            candidates = [0.0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]

        best_theta, best_sr = 0.0, float("-inf")

        for theta in candidates:
            bt = Backtester(
                initial_capital=self.V0,
                position_fraction=self.f,
                transaction_cost=self.tc,
                slippage=self.slip,
                stop_loss=self.stop_loss,
                daily_loss_limit=self.daily_limit,
                signal_threshold=theta,
            )
            result = bt.run(
                y_pred_val.copy(),
                opens_val,
                closes_val,
                timestamps_val,
                session_starts_val,
            )
            if result.sharpe > best_sr:
                best_sr = result.sharpe
                best_theta = theta

        return best_theta
