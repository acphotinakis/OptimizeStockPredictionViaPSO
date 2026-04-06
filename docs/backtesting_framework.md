# Backtesting Framework
## Signal Generation, Risk Management, and Performance Metrics

**Document Version:** 1.0 | March 2026

---

## Table of Contents

1. [Framework Overview](#1-framework-overview)
2. [Signal Generation Logic](#2-signal-generation-logic)
3. [Position Sizing](#3-position-sizing)
4. [Transaction Costs and Slippage](#4-transaction-costs-and-slippage)
5. [Risk Management](#5-risk-management)
6. [Performance Metrics](#6-performance-metrics)
7. [Walk-Forward Backtesting](#7-walk-forward-backtesting)
8. [Backtester Implementation](#8-backtester-implementation)
9. [Limitations and Realistic Assumptions](#9-limitations-and-realistic-assumptions)

---

## 1. Framework Overview

The backtesting framework simulates realistic trading performance based on model predictions. It converts predicted log returns to actionable signals, applies position sizing and risk management rules, subtracts transaction costs and slippage, and computes a full suite of financial performance metrics.

### 1.1 Design Principles

1. **No lookahead bias**: Signals are generated only from predictions available at bar $t$ and executed at the **next bar's open** ($t+1$)
2. **Realistic costs**: Transaction costs and slippage are applied on every fill
3. **Walk-forward**: The backtester is run in rolling windows to prevent overfitting to the test period
4. **Capital allocation**: Fixed fractional position sizing (not all-in)

### 1.2 Backtesting Pipeline

```
Model Predictions ŷ_{t+1}
    │
    ▼
Signal Generation
  [+1 long / -1 short / 0 flat]
    │
    ▼
Position Sizing
  [fraction of capital per bar]
    │
    ▼
Order Execution (next bar open + slippage)
    │
    ▼
P&L Calculation (after transaction costs)
    │
    ▼
Equity Curve Construction
    │
    ▼
Risk-Adjusted Performance Metrics
```

---

## 2. Signal Generation Logic

### 2.1 Threshold-Based Signal

Raw model output $\hat{y}_{t+1}$ (predicted log return) is converted to a ternary signal:

$$\text{signal}_{t+1} = \begin{cases}
+1 & \text{if } \hat{y}_{t+1} > +\theta \\
-1 & \text{if } \hat{y}_{t+1} < -\theta \\
0 & \text{if } |\hat{y}_{t+1}| \leq \theta
\end{cases} \tag{1}$$

**Default threshold:** $\theta = 10^{-4}$ (approximately 1 basis point, or $0.01\%$)

**Rationale:** The threshold filters out predictions near zero that would generate frequent trades with minimal edge, which is especially important given transaction costs. The optimal threshold is determined empirically on the validation set.

### 2.2 Threshold Optimization

The optimal threshold is calibrated on the validation set by grid search over $\theta \in \{0, 5 \times 10^{-5}, 10^{-4}, 2 \times 10^{-4}, 5 \times 10^{-4}, 10^{-3}\}$, maximizing validation-set Sharpe ratio.

```python
def optimize_threshold(y_pred_val, y_true_val, prices_val,
                       thresholds=[0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]):
    best_theta, best_sharpe = 0, -np.inf
    for theta in thresholds:
        signals = generate_signals(y_pred_val, theta)
        sr = compute_sharpe(signals, prices_val)
        if sr > best_sharpe:
            best_theta, best_sharpe = theta, sr
    return best_theta
```

### 2.3 Execution Timing

| Event | Bar |
|---|---|
| Prediction $\hat{y}_{t+1}$ generated | At bar $t$ close |
| Signal $\text{signal}_{t+1}$ computed | At bar $t$ close |
| Order placed | At bar $t$ close |
| Order filled | At bar $t+1$ **open** + slippage |
| P&L realized | At bar $t+1$ close |

This avoids filling on the same bar that generated the signal (which would require the impossible: trading on the bar-close price that the model hasn't yet seen).

---

## 3. Position Sizing

### 3.1 Fixed Fractional (Default)

Each trade uses a fixed fraction $f$ of current portfolio equity:

$$\text{trade\_value}_t = f \times V_t$$

**Default:** $f = 0.02$ (2% of portfolio per position, i.e., 50:1 risk dilution).

### 3.2 Unit Size in Shares

$$\text{shares}_t = \left\lfloor \frac{f \times V_t}{\text{fill\_price}_t} \right\rfloor$$

Fractional shares are not modeled (whole shares only).

### 3.3 Kelly Criterion (Optional Variant)

The theoretical optimal fraction using the Kelly Criterion:

$$f^* = \frac{p \cdot b - q}{b}$$

where $p$ = win probability, $q = 1-p$, and $b$ = average win-to-loss ratio.

Estimated from rolling validation performance: not used in primary experiments (too sensitive to estimation noise), but reported for reference.

---

## 4. Transaction Costs and Slippage

### 4.1 Commissions

**Per-share commission model** (broker-like structure):

$$\text{commission}_t = \max(0.001 \times \text{trade\_value}_t,\ 1.00)$$

This models approximately 10 basis points ($0.10\%$) per transaction.

**Alternative: Fixed cost:**

$$\text{commission}_t = c \times |\text{signal}_{t+1} - \text{signal}_t| \times \text{fill\_price}_t$$

where $c = 0.001$ (10 bps) is applied only when the signal changes (i.e., a trade occurs). Feature: `signal_change_flag`.

### 4.2 Bid-Ask Spread

Half-spread slippage is modeled as:

$$\text{slippage}_t = s \times \text{fill\_price}_t$$

**Default:** $s = 0.0005$ (5 bps half-spread, consistent with high-frequency equities at 1-min resolution).

**Direction:** Longs fill at $O_{t+1} + s \times O_{t+1}$, shorts fill at $O_{t+1} - s \times O_{t+1}$.

### 4.3 Market Impact (Approximation)

For the small trade sizes modeled ($2\%$ of portfolio), market impact is negligible for liquid mid-cap equities. Not modeled explicitly.

### 4.4 Combined Transaction Cost

Per trade (when signal changes):

$$\text{TC}_t = (c + s) \times \text{trade\_value}_t = 0.0015 \times \text{trade\_value}_t \tag{2}$$

This represents 15 bps roundtrip, which is conservative but realistic for a 1-minute HFT strategy.

---

## 5. Risk Management

### 5.1 Stop Loss

A per-trade stop loss exits a position if the mark-to-market loss exceeds a threshold:

$$\text{exit if}: \frac{V_{\text{entry}} - V_t}{V_{\text{entry}}} > \delta_{\text{stop}}$$

**Default:** $\delta_{\text{stop}} = 0.02$ (2% portfolio drawdown per trade triggers exit)

Implementation:
```python
for bar in range(entry_bar + 1, n_bars):
    pnl_pct = (price[bar] - entry_price) / entry_price * direction
    if pnl_pct < -stop_loss:
        exit_bar = bar
        break
```

### 5.2 Daily Loss Limit

If intraday portfolio drawdown exceeds 5% of opening equity, all positions are closed and no new positions are taken for the remainder of the session:

$$\text{halt if}: \frac{V_{\text{session\_open}} - V_t}{V_{\text{session\_open}}} > \delta_{\text{daily}}$$

**Default:** $\delta_{\text{daily}} = 0.05$

### 5.3 Maximum Leverage

Positions are never leveraged (maximum 1:1):

$$\sum_t |\text{position value}_t| \leq V_t$$

For the single-ticker backtest, this means at most 100% exposure in one direction.

### 5.4 Session Close Flat

All open positions are closed at the last bar of each trading session (15:59 ET) to avoid overnight exposure. Overnight holding is out-of-scope for this 1-minute intraday system.

---

## 6. Performance Metrics

### 6.1 Sharpe Ratio (Primary)

$$\text{SR} = \frac{E[R_p - R_f]}{\sigma(R_p - R_f)} \cdot \sqrt{252 \times 390} \tag{3}$$

- $R_p$ = per-bar portfolio return (after costs)
- $R_f = 0$ (zero risk-free rate at bar-level frequency)
- $\sqrt{252 \times 390}$ annualizes from 1-minute bars

**Interpretation guide:**
| Sharpe | Quality |
|---|---|
| < 0 | Loss-making |
| 0 – 0.5 | Poor |
| 0.5 – 1.0 | Acceptable |
| 1.0 – 2.0 | Good |
| > 2.0 | Excellent (institutional grade) |

### 6.2 Maximum Drawdown

$$\text{MDD} = \max_{s \leq t \leq T}\left(\frac{V_s - V_t}{V_s}\right) \tag{4}$$

```python
def max_drawdown(equity_curve: np.ndarray) -> float:
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdown = (rolling_max - equity_curve) / rolling_max
    return drawdown.max()
```

### 6.3 Calmar Ratio

$$\text{Calmar} = \frac{\text{CAGR}}{\text{MDD}} \tag{5}$$

Measures risk-adjusted return with drawdown as the risk denominator.

### 6.4 Compound Annual Growth Rate (CAGR)

$$\text{CAGR} = \left(\frac{V_T}{V_0}\right)^{\frac{252}{N_{\text{trading\_days}}}} - 1 \tag{6}$$

### 6.5 Profit Factor

$$\text{PF} = \frac{\sum_{i: r_i > 0} r_i}{\sum_{i: r_i < 0} |r_i|} \tag{7}$$

PF > 1 indicates profitable; PF > 1.5 is generally considered strong.

### 6.6 Win Rate and Average Win/Loss

$$\text{WR} = \frac{N_{\text{winning trades}}}{N_{\text{total trades}}}, \quad \bar{W} = \frac{\sum_{i: r_i > 0} r_i}{N_{\text{winning trades}}}, \quad \bar{L} = \frac{\sum_{i: r_i < 0} |r_i|}{N_{\text{losing trades}}}$$

**Reward-to-risk ratio:** $\text{RR} = \bar{W} / \bar{L}$

### 6.7 Sortino Ratio

Uses only downside deviation (penalizes harmful volatility only):

$$\text{Sortino} = \frac{E[R_p - R_f]}{\sigma_d} \cdot \sqrt{252 \times 390}$$

where $\sigma_d = \sqrt{E[\min(R_p - R_f, 0)^2]}$.

### 6.8 Information Ratio

Against SPY benchmark:

$$\text{IR} = \frac{E[R_p - R_{\text{SPY}}]}{\sigma(R_p - R_{\text{SPY}})} \cdot \sqrt{252 \times 390}$$

### 6.9 Turnover

$$\text{Turnover} = \frac{\sum_t |\text{signal}_t - \text{signal}_{t-1}|}{N_{\text{bars}}} \tag{8}$$

High turnover × transaction costs = drag on performance.

---

## 7. Walk-Forward Backtesting

### 7.1 Purpose

Walk-forward validation prevents overfitting by simulating the real-world scenario of periodically retraining the model as new data arrives.

### 7.2 Protocol

```
Full timeline: 2019-01-02 to 2024-01-01

Training period: 3 years (growing)
Validation period: 3 months (for threshold optimization)
Test period: 1 month (out-of-sample, never used in training)

Fold structure:
  Fold 1: Train [2019-01 to 2022-01], Val [2022-01 to 2022-04], Test [2022-04 to 2022-05]
  Fold 2: Train [2019-01 to 2022-04], Val [2022-04 to 2022-07], Test [2022-07 to 2022-08]
  ...
  Fold N: Train [2019-01 to 2023-10], Val [2023-10 to 2024-01], Test [2024-01 to 2024-01+1mo]
```

**Step size:** 1 month (expand train, advance val/test windows)

### 7.3 Per-Fold Procedure

```python
for fold in walk_forward_folds:
    # 1. Fit scaler on current training set
    scaler.fit(X_fold_train)
    X_train_scaled = scaler.transform(X_fold_train)
    
    # 2. Run IPSO to find best hyperparameters (on validation set)
    #    (Optional: reuse gbest from full PSO run for efficiency)
    
    # 3. Retrain IPSO-LSTM with gbest params on current training set
    model.fit(X_train_scaled, y_fold_train)
    
    # 4. Optimize threshold on validation set
    theta = optimize_threshold(model.predict(X_fold_val), y_fold_val, prices_fold_val)
    
    # 5. Generate signals on test period
    signals = generate_signals(model.predict(X_fold_test), theta)
    
    # 6. Backtest signals on test period
    fold_results = backtester.run(signals, prices_fold_test)
    results.append(fold_results)

# Aggregate across folds
mean_sharpe = np.mean([r['sharpe'] for r in results])
std_sharpe  = np.std([r['sharpe'] for r in results])
```

---

## 8. Backtester Implementation

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BacktestResult:
    equity_curve: np.ndarray
    trade_log: pd.DataFrame
    sharpe: float
    mdd: float
    cagr: float
    profit_factor: float
    win_rate: float
    calmar: float
    sortino: float
    n_trades: int
    turnover: float

class Backtester:
    def __init__(self,
                 initial_capital: float = 100_000,
                 position_fraction: float = 0.02,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 stop_loss: float = 0.02,
                 daily_loss_limit: float = 0.05,
                 freq_per_year: int = 252 * 390):
        self.V0 = initial_capital
        self.f = position_fraction
        self.tc = transaction_cost
        self.slip = slippage
        self.stop_loss = stop_loss
        self.daily_limit = daily_loss_limit
        self.freq = freq_per_year
    
    def run(self,
            signals: np.ndarray,
            opens: np.ndarray,
            closes: np.ndarray,
            timestamps: pd.DatetimeIndex) -> BacktestResult:
        """
        Args:
            signals:    [N] int array, values in {-1, 0, +1}
            opens:      [N] float array, bar open prices
            closes:     [N] float array, bar close prices
            timestamps: [N] DatetimeIndex (UTC)
        """
        N = len(signals)
        V = np.zeros(N + 1)
        V[0] = self.V0
        
        bar_returns = np.zeros(N)
        position = 0       # Current position: -1, 0, +1
        entry_price = 0.0
        entry_equity = 0.0
        session_open_equity = self.V0
        trades = []
        daily_halt = False
        
        for t in range(N):
            # Check daily loss limit at session start
            if t > 0 and timestamps[t].time() == pd.Timestamp('09:30').time():
                session_open_equity = V[t]
                daily_halt = False
            
            # Session end: force close
            force_close = (timestamps[t].time() >= pd.Timestamp('15:59').time())
            
            # Determine target position
            if daily_halt or force_close:
                target = 0
            else:
                target = int(signals[t])
            
            fill_price = opens[t] * (1 + self.slip * np.sign(target - position))
            
            # Execute trade if position changes
            if target != position:
                # Close existing position
                if position != 0:
                    close_fill = opens[t] * (1 - self.slip * np.sign(position))
                    pnl = position * (close_fill - entry_price) * (V[t] * self.f / entry_price)
                    tc = self.tc * V[t] * self.f
                    V[t] = V[t] + pnl - tc
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamps[t],
                        'direction': position,
                        'entry_price': entry_price,
                        'exit_price': close_fill,
                        'pnl': pnl - tc
                    })
                
                # Open new position (if non-zero)
                if target != 0:
                    entry_price = fill_price
                    entry_time = timestamps[t]
                    entry_equity = V[t]
                    tc = self.tc * V[t] * self.f
                    V[t] = V[t] - tc
                
                position = target
            
            # Mark-to-market unrealized P&L
            if position != 0:
                mtm_pnl = position * (closes[t] - fill_price) * (V[t] * self.f / fill_price)
                V[t + 1] = V[t] + mtm_pnl
                
                # Stop loss check
                drawdown = (entry_equity - V[t + 1]) / entry_equity
                if drawdown > self.stop_loss:
                    # Will force close at next bar open
                    signals[t + 1] = 0  # Override next signal
            else:
                V[t + 1] = V[t]
            
            # Daily halt check
            session_drawdown = (session_open_equity - V[t + 1]) / session_open_equity
            if session_drawdown > self.daily_limit:
                daily_halt = True
            
            bar_returns[t] = (V[t + 1] - V[t]) / V[t] if V[t] > 0 else 0
        
        equity_curve = V[1:]
        trade_df = pd.DataFrame(trades)
        
        # Compute metrics
        sharpe = self._sharpe(bar_returns)
        mdd = self._max_drawdown(equity_curve)
        cagr = self._cagr(equity_curve)
        pf = self._profit_factor(trade_df)
        wr = (trade_df['pnl'] > 0).mean() if len(trade_df) > 0 else 0
        
        return BacktestResult(
            equity_curve=equity_curve,
            trade_log=trade_df,
            sharpe=sharpe,
            mdd=mdd,
            cagr=cagr,
            profit_factor=pf,
            win_rate=wr,
            calmar=cagr / (mdd + 1e-8),
            sortino=self._sortino(bar_returns),
            n_trades=len(trade_df),
            turnover=np.mean(np.abs(np.diff(signals)))
        )
    
    def _sharpe(self, r: np.ndarray) -> float:
        if r.std() < 1e-10: return 0.0
        return (r.mean() / r.std()) * np.sqrt(self.freq)
    
    def _max_drawdown(self, equity: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / (peak + 1e-8)
        return float(dd.max())
    
    def _cagr(self, equity: np.ndarray) -> float:
        n_bars = len(equity)
        n_years = n_bars / self.freq
        return float((equity[-1] / self.V0) ** (1 / n_years) - 1)
    
    def _profit_factor(self, trades: pd.DataFrame) -> float:
        if trades.empty: return 0.0
        gross_profit = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
        gross_loss = trades.loc[trades['pnl'] < 0, 'pnl'].abs().sum()
        return float(gross_profit / (gross_loss + 1e-8))
    
    def _sortino(self, r: np.ndarray) -> float:
        downside = r[r < 0]
        if len(downside) == 0: return np.inf
        downside_std = np.sqrt(np.mean(downside**2))
        return float((r.mean() / downside_std) * np.sqrt(self.freq))
```

---

## 9. Limitations and Realistic Assumptions

### 9.1 What This Backtester Does NOT Model

| Missing Factor | Potential Impact | Mitigation |
|---|---|---|
| Market impact of large orders | Overstates returns for large sizes | Fixed 2% position size keeps orders small |
| Short selling constraints | Shorts may not always be available | Realistic for SPY and large caps |
| Borrowing costs for shorts | ~0.5–2% annualized | Modest for large cap equities; ignored for simplicity |
| Pattern Day Trader rules | <$25K account restricted to 3 day trades/5 days | Not modeled; assume institutional account |
| Execution latency | ~1–10ms for algorithmic orders | Implicit in "fill at open" assumption |
| Order book depth | Assumes all orders fill immediately | Realistic for $<$100K positions in SPY/AAPL |

### 9.2 Known Biases

**Survivorship bias:** All 50 equities in the universe are currently trading (selected in 2026). Some may have been added/removed from indices during the 2019–2024 period. This introduces a mild upward bias in historical returns.

**Mitigation:** Use a survivorship-bias-free index membership list (S&P 500 historical constituents) to construct the universe. Implementation complexity deferred to future work.

**Look-ahead in normalization:** Handled by fitting scalers on training set only. Verified via the data quality checklist.