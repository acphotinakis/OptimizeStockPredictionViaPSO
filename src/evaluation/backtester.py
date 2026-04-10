"""
src/evaluation/backtester.py

Realistic backtesting engine implementing the framework defined in
backtesting_framework.md:
  - Signals from model predictions (ternary: +1, 0, -1)
  - Fill at next-bar open + slippage
  - Transaction costs per trade
  - Per-trade stop loss
  - Daily loss limit
  - Session-end forced flat
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

SIGNAL_THRESHOLD = 1e-4  # 1 bp


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
    """Event-driven 1-minute bar backtester.

    Args:
        initial_capital: Starting portfolio value in USD.
        position_fraction: Fraction of capital risked per trade.
        transaction_cost: One-way commission as a fraction of trade value.
        slippage: Half-spread slippage as a fraction of fill price.
        stop_loss: Maximum per-trade drawdown before forced exit.
        daily_loss_limit: Maximum intraday drawdown before halting new entries.
        signal_threshold: |y_pred| must exceed this to generate a signal.
    """

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

        self.n_shares = 0.0

    # ------------------------------------------------------------------

    def run(
        self,
        y_pred: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
        timestamps: pd.DatetimeIndex,
        session_starts: np.ndarray | None = None,
    ) -> BacktestResult:
        """Execute the backtest.

        Args:
            y_pred:     [N] predicted log returns from the model.
            opens:      [N] bar open prices.
            closes:     [N] bar close prices.
            timestamps: [N] UTC DatetimeIndex.
            session_starts: [N] bool array, True at first bar of each session.
                           If None, falls back to time-based detection (9:30 open, 16:00 close).

        Returns:
            BacktestResult dataclass.
        """
        N = len(y_pred)
        signals = self._make_signals(y_pred)

        equity = np.empty(N + 1)
        equity[0] = self.V0
        bar_returns = np.zeros(N)

        position = 0  # -1 / 0 / +1
        entry_price = 0.0
        entry_equity = 0.0
        entry_time = None
        session_open_equity = self.V0
        daily_halt = False

        trades: List[Dict] = []

        et_index = timestamps.tz_convert("America/New_York")

        for t in range(N):
            # --- Session boundary bookkeeping ---
            if session_starts is not None:
                is_session_open = bool(session_starts[t])
                is_session_close = (t == N - 1) or bool(session_starts[t + 1])
            else:
                is_session_open = et_index[t].hour == 9 and et_index[t].minute == 30
                is_session_close = et_index[t].hour == 16 and et_index[t].minute == 0

            if is_session_open:
                session_open_equity = equity[t]
                daily_halt = False

            # --- Determine target position ---
            force_flat = is_session_close or daily_halt
            target = 0 if force_flat else int(signals[t])

            # --- Execute trade if signal changes ---
            fill = opens[t]

            if target != position:
                # ---- Close existing position ----
                if position != 0:
                    close_slip = fill * (1.0 - self.slip * np.sign(position))
                    trade_value = self.f * equity[t]
                    n_shares = trade_value / (entry_price + 1e-10)
                    pnl = position * n_shares * (close_slip - entry_price)
                    cost = self.tc * trade_value
                    equity[t] = equity[t] + pnl - cost
                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": timestamps[t],
                            "direction": position,
                            "entry_price": entry_price,
                            "exit_price": close_slip,
                            "pnl_gross": pnl,
                            "cost": cost,
                            "pnl_net": pnl - cost,
                        }
                    )
                    position = 0

                # ---- Open new position ----
                if target != 0:
                    open_slip = fill * (1.0 + self.slip * target)
                    cost = self.tc * self.f * equity[t]
                    equity[t] = equity[t] - cost
                    entry_price = open_slip
                    entry_equity = equity[t]
                    entry_time = timestamps[t]
                    position = target

            # --- Mark-to-market unrealised P&L using close price ---
            if position != 0:
                position_value = self.f * entry_equity
                self.n_shares = position_value / (entry_price + 1e-10)
                mtm_pnl = position * self.n_shares * (closes[t] - entry_price)
                equity[t + 1] = equity[t] + mtm_pnl
                # Stop-loss: if unrealised drawdown exceeds limit, force flat next bar
                trade_dd = (entry_equity - equity[t + 1]) / (entry_equity + 1e-10)
                if trade_dd > self.stop_loss and t + 1 < N:
                    signals[t + 1] = 0  # Override next bar's signal
            else:
                equity[t + 1] = equity[t]

            # Daily loss-limit check
            session_dd = (session_open_equity - equity[t + 1]) / (
                session_open_equity + 1e-10
            )
            if session_dd > self.daily_limit:
                daily_halt = True

            bar_returns[t] = (equity[t + 1] - equity[t]) / (equity[t] + 1e-10)

        equity_curve = equity[1:]
        trade_df = pd.DataFrame(trades)

        # Metrics
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

    # ------------------------------------------------------------------

    def _make_signals(self, y_pred: np.ndarray) -> np.ndarray:
        return generate_signals(y_pred, self.threshold)

    # ------------------------------------------------------------------
    # Threshold optimisation on validation set
    # ------------------------------------------------------------------

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
        """Grid-search the signal threshold maximising Sharpe on the val set.

        Args:
            y_pred_val: [N] val set predictions.
            y_true_val: [N] val actual returns (unused here; prices used).
            opens_val, closes_val: [N] price arrays.
            timestamps_val: [N] DatetimeIndex.
            session_starts_val: [N] bool array for session boundaries.
            candidates: Threshold values to search.

        Returns:
            Optimal threshold float.
        """
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
            result = bt.run(y_pred_val.copy(), opens_val, closes_val, timestamps_val, session_starts_val)
            if result.sharpe > best_sr:
                best_sr = result.sharpe
                best_theta = theta

        return best_theta
