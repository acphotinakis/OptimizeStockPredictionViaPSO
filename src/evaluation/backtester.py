"""
src/evaluation/backtester.py

Refactored backtesting engine with separated concerns for testability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import pandas as pd

from .metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    cagr,
    calmar_ratio,
    profit_factor,
    win_rate,
    generate_signals,
)

SIGNAL_THRESHOLD = 1e-4


class SessionEvent(Enum):
    """Enum for session boundary events."""

    NONE = 0
    OPEN = 1
    CLOSE = 2


@dataclass
class BacktestState:
    """Immutable state container for backtest iteration."""

    equity: List[float]
    position: int  # -1, 0, +1
    entry_price: float
    entry_equity: float
    entry_time: Optional[pd.Timestamp]
    n_shares_held: float
    session_open_equity: float
    daily_halt: bool
    trades: List[Dict]
    bar_returns: List[float]
    current_step: int

    def __init__(self, initial_capital: float):
        self.equity = [initial_capital]
        self.position = 0
        self.entry_price = 0.0
        self.entry_equity = 0.0
        self.entry_time = None
        self.n_shares_held = 0.0
        self.session_open_equity = initial_capital
        self.daily_halt = False
        self.trades = []
        self.bar_returns = []
        self.current_step = 0

    def copy(self) -> Dict[str, Any]:
        """Return mutable copy for iteration."""
        return {
            "equity": self.equity,
            "position": self.position,
            "entry_price": self.entry_price,
            "entry_equity": self.entry_equity,
            "entry_time": self.entry_time,
            "n_shares_held": self.n_shares_held,
            "session_open_equity": self.session_open_equity,
            "daily_halt": self.daily_halt,
            "trades": self.trades,
            "bar_returns": self.bar_returns,
            "current_step": self.current_step,
        }


@dataclass
class BacktestResult:
    equity_curve: np.ndarray
    bar_returns: np.ndarray
    trade_log: pd.DataFrame
    sharpe: float
    sortino: float
    mdd: float
    cagr_: float
    calmar: float
    profit_factor_: float
    win_rate_: float
    n_trades: int
    turnover: float


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

            # 6. Record bar return
            bar_ret = (state["equity"][-1] - state["equity"][-2]) / (
                state["equity"][-2] + 1e-10
            )
            state["bar_returns"].append(bar_ret)
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
            "current_step": 0,
        }

    def _check_session_boundary(
        self,
        t: int,
        timestamps: pd.DatetimeIndex,
        session_starts: np.ndarray | None,
        et_index: pd.DatetimeIndex,
    ) -> SessionEvent:
        """
        Detect if current bar is session open, close, or mid-session.
        """
        if session_starts is not None:
            is_open = bool(session_starts[t])
            is_close = (t == len(timestamps) - 1) or bool(session_starts[t + 1])
        else:
            # Time-based detection (9:30 open, 16:00 close ET)
            is_open = et_index[t].hour == 9 and et_index[t].minute == 30
            is_close = et_index[t].hour == 16 and et_index[t].minute == 0

        if is_open:
            return SessionEvent.OPEN
        elif is_close:
            return SessionEvent.CLOSE
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
        """Apply stop-loss and daily loss limits."""
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
        """Compute final metrics and return result object."""
        equity_curve = np.array(state["equity"][1:])  # Remove initial seed
        bar_returns = np.array(state["bar_returns"])
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


# """
# src/evaluation/backtester.py

# Realistic backtesting engine implementing the framework defined in
# backtesting_framework.md:
#   - Signals from model predictions (ternary: +1, 0, -1)
#   - Fill at next-bar open + slippage
#   - Transaction costs per trade
#   - Per-trade stop loss
#   - Daily loss limit
#   - Session-end forced flat
# """

# from __future__ import annotations

# from dataclasses import dataclass, field
# from typing import Dict, List, Optional

# import numpy as np
# import pandas as pd

# from .metrics import (
#     sharpe_ratio,
#     sortino_ratio,
#     max_drawdown,
#     cagr,
#     calmar_ratio,
#     profit_factor,
#     win_rate,
#     generate_signals,
# )

# SIGNAL_THRESHOLD = 1e-4  # 1 bp


# @dataclass
# class BacktestResult:
#     equity_curve: np.ndarray
#     bar_returns: np.ndarray
#     trade_log: pd.DataFrame
#     sharpe: float
#     sortino: float
#     mdd: float
#     cagr_: float
#     calmar: float
#     profit_factor_: float
#     win_rate_: float
#     n_trades: int
#     turnover: float


# class Backtester:
#     """Event-driven 1-minute bar backtester.

#     Args:
#         initial_capital: Starting portfolio value in USD.
#         position_fraction: Fraction of capital risked per trade.
#         transaction_cost: One-way commission as a fraction of trade value.
#         slippage: Half-spread slippage as a fraction of fill price.
#         stop_loss: Maximum per-trade drawdown before forced exit.
#         daily_loss_limit: Maximum intraday drawdown before halting new entries.
#         signal_threshold: |y_pred| must exceed this to generate a signal.
#     """

#     def __init__(
#         self,
#         initial_capital: float = 100_000.0,
#         position_fraction: float = 0.02,
#         transaction_cost: float = 0.001,
#         slippage: float = 0.0005,
#         stop_loss: float = 0.02,
#         daily_loss_limit: float = 0.05,
#         signal_threshold: float = SIGNAL_THRESHOLD,
#     ) -> None:
#         self.V0 = initial_capital
#         self.f = position_fraction
#         self.tc = transaction_cost
#         self.slip = slippage
#         self.stop_loss = stop_loss
#         self.daily_limit = daily_loss_limit
#         self.threshold = signal_threshold

#         self.n_shares = 0.0

#     # ------------------------------------------------------------------

#     def run(
#         self,
#         y_pred: np.ndarray,
#         opens: np.ndarray,
#         closes: np.ndarray,
#         timestamps: pd.DatetimeIndex,
#         session_starts: np.ndarray | None = None,
#     ) -> BacktestResult:
#         """Execute the backtest.

#         Args:
#             y_pred:     [N] predicted log returns from the model.
#             opens:      [N] bar open prices.
#             closes:     [N] bar close prices.
#             timestamps: [N] UTC DatetimeIndex.
#             session_starts: [N] bool array, True at first bar of each session.
#                            If None, falls back to time-based detection (9:30 open, 16:00 close).

#         Returns:
#             BacktestResult dataclass.
#         """
#         N = len(y_pred)
#         signals = self._make_signals(y_pred)

#         equity = np.empty(N + 1)
#         equity[0] = self.V0
#         bar_returns = np.zeros(N)

#         position = 0  # -1 / 0 / +1
#         entry_price = 0.0
#         entry_equity = 0.0
#         entry_time = None
#         n_shares_held = 0.0  # shares held — computed at entry, constant until exit
#         session_open_equity = self.V0
#         daily_halt = False

#         trades: List[Dict] = []

#         et_index = timestamps.tz_convert("America/New_York")

#         for t in range(N):
#             # --- Session boundary bookkeeping ---
#             if session_starts is not None:
#                 is_session_open = bool(session_starts[t])
#                 is_session_close = (t == N - 1) or bool(session_starts[t + 1])
#             else:
#                 is_session_open = et_index[t].hour == 9 and et_index[t].minute == 30
#                 is_session_close = et_index[t].hour == 16 and et_index[t].minute == 0

#             if is_session_open:
#                 session_open_equity = equity[t]
#                 daily_halt = False

#             # --- Determine target position ---
#             force_flat = is_session_close or daily_halt
#             target = 0 if force_flat else int(signals[t])

#             # --- Execute trade if signal changes ---
#             fill = opens[t]

#             if target != position:
#                 # ---- Close existing position ----
#                 if position != 0:
#                     close_slip = fill * (1.0 - self.slip * np.sign(position))
#                     # Use share count locked in at entry — not recomputed at exit
#                     pnl = position * n_shares_held * (close_slip - entry_price)
#                     cost = self.tc * self.f * entry_equity
#                     equity[t] = equity[t] + pnl - cost
#                     trades.append(
#                         {
#                             "entry_time": entry_time,
#                             "exit_time": timestamps[t],
#                             "direction": position,
#                             "entry_price": entry_price,
#                             "exit_price": close_slip,
#                             "pnl_gross": pnl,
#                             "cost": cost,
#                             "pnl_net": pnl - cost,
#                         }
#                     )
#                     position = 0
#                     n_shares_held = 0.0

#                 # ---- Open new position ----
#                 if target != 0:
#                     open_slip = fill * (1.0 + self.slip * target)
#                     cost = self.tc * self.f * equity[t]
#                     equity[t] = equity[t] - cost
#                     entry_price = open_slip
#                     entry_equity = equity[t]
#                     entry_time = timestamps[t]
#                     position = target
#                     # Lock in share count at entry based on entry price
#                     n_shares_held = (self.f * entry_equity) / (entry_price + 1e-10)

#             # --- Mark-to-market unrealised P&L using close price ---
#             if position != 0:
#                 mtm_pnl = position * n_shares_held * (closes[t] - entry_price)
#                 equity[t + 1] = equity[t] + mtm_pnl
#                 # Stop-loss: if unrealised drawdown exceeds limit, force flat next bar
#                 trade_dd = (entry_equity - equity[t + 1]) / (entry_equity + 1e-10)
#                 if trade_dd > self.stop_loss and t + 1 < N:
#                     signals[t + 1] = 0  # Override next bar's signal
#             else:
#                 equity[t + 1] = equity[t]

#             # Daily loss-limit check
#             session_dd = (session_open_equity - equity[t + 1]) / (
#                 session_open_equity + 1e-10
#             )
#             if session_dd > self.daily_limit:
#                 daily_halt = True

#             bar_returns[t] = (equity[t + 1] - equity[t]) / (equity[t] + 1e-10)

#         equity_curve = equity[1:]
#         trade_df = pd.DataFrame(trades)

#         # Metrics
#         sr = sharpe_ratio(bar_returns)
#         sor = sortino_ratio(bar_returns)
#         mdd = max_drawdown(equity_curve)
#         c = cagr(equity_curve)
#         cal = calmar_ratio(equity_curve)
#         pf = profit_factor(bar_returns)
#         wr = win_rate(bar_returns)
#         to = float(np.mean(np.abs(np.diff(signals, prepend=0))))

#         return BacktestResult(
#             equity_curve=equity_curve,
#             bar_returns=bar_returns,
#             trade_log=trade_df,
#             sharpe=sr,
#             sortino=sor,
#             mdd=mdd,
#             cagr_=c,
#             calmar=cal,
#             profit_factor_=pf,
#             win_rate_=wr,
#             n_trades=len(trade_df),
#             turnover=to,
#         )

#     # ------------------------------------------------------------------

#     def _make_signals(self, y_pred: np.ndarray) -> np.ndarray:
#         return generate_signals(y_pred, self.threshold)

#     # ------------------------------------------------------------------
#     # Threshold optimisation on validation set
#     # ------------------------------------------------------------------

#     def optimize_threshold(
#         self,
#         y_pred_val: np.ndarray,
#         y_true_val: np.ndarray,
#         opens_val: np.ndarray,
#         closes_val: np.ndarray,
#         timestamps_val: pd.DatetimeIndex,
#         session_starts_val: np.ndarray | None = None,
#         candidates: Optional[List[float]] = None,
#     ) -> float:
#         """Grid-search the signal threshold maximising Sharpe on the val set.

#         Args:
#             y_pred_val: [N] val set predictions.
#             y_true_val: [N] val actual returns (unused here; prices used).
#             opens_val, closes_val: [N] price arrays.
#             timestamps_val: [N] DatetimeIndex.
#             session_starts_val: [N] bool array for session boundaries.
#             candidates: Threshold values to search.

#         Returns:
#             Optimal threshold float.
#         """
#         if candidates is None:
#             candidates = [0.0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]

#         best_theta, best_sr = 0.0, float("-inf")
#         for theta in candidates:
#             bt = Backtester(
#                 initial_capital=self.V0,
#                 position_fraction=self.f,
#                 transaction_cost=self.tc,
#                 slippage=self.slip,
#                 stop_loss=self.stop_loss,
#                 daily_loss_limit=self.daily_limit,
#                 signal_threshold=theta,
#             )
#             result = bt.run(
#                 y_pred_val.copy(),
#                 opens_val,
#                 closes_val,
#                 timestamps_val,
#                 session_starts_val,
#             )
#             if result.sharpe > best_sr:
#                 best_sr = result.sharpe
#                 best_theta = theta

#         return best_theta
