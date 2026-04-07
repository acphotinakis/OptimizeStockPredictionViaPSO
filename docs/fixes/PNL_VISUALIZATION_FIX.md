# PnL Visualization Fix
## Added Profit & Loss Tracking to LSTM Pipeline

**Date**: April 7, 2026  
**Status**: ✅ Complete  
**Related**: RL Trading Agent Plan (`docs/plans/RL_TRADING_AGENT_PLAN.md`)

---

## Changes Made

### 1. Added PnL Calculation (`scripts/run_lstm_baseline.py`)

**Location**: Lines 560-620

**Features**:
- ✅ Calculates strategy returns: `position[t] × actual_return[t]`
- ✅ Applies transaction costs when position changes
- ✅ Computes cumulative PnL and equity curve
- ✅ Calculates Sharpe ratio (annualized)
- ✅ Computes max drawdown
- ✅ Counts number of trades

**New Columns in CSV**:
- `strategy_return`: Net return after transaction costs
- `pnl`: Cumulative profit/loss ($)
- `equity`: Current portfolio value ($)

### 2. Enhanced Test Plots (`scripts/run_lstm_baseline.py`)

**Added 3rd Panel**: Equity Curve

- Blue line: Portfolio equity over time
- Green/red shading: Profit/loss regions
- Annotation: Final PnL and return percentage
- Title shows: Sharpe ratio and max drawdown

### 3. Updated OHLCV Plots (`plots/plot_lstm.py`)

**Added 4th Panel**: PnL visualization (when OHLCV available)

- Equity curve with initial capital reference line
- Profit/loss shading
- Final PnL annotation with arrow
- Sharpe and max drawdown in title

### 4. Enhanced Signal Summary (`plots/plot_lstm.py`)

**New Section**: PnL Statistics

```
PnL Statistics:
  Initial Capital:  $  100,000.00
  Final Equity:     $  105,234.56
  Total PnL:        $    5,234.56
  Total Return:           5.23%
  Sharpe Ratio:           1.234
  Max Drawdown:          -8.45%
  Number of Trades:         234
```

---

## Usage

### Run Test with PnL

```bash
python scripts/run_lstm_baseline.py \
    --ticker SPY \
    --mode test \
    --seed 42 \
    --initial-capital 100000 \
    --transaction-cost 0.001
```

### View PnL Plots

```bash
python plots/plot_lstm.py --ticker SPY --seed 42
```

**Outputs**:
- `results/plots/lstm_baseline_test_SPY_test_seed42.png` - 3-panel plot with equity curve
- `results/plots/lstm_ohlcv_SPY_seed42.png` - 4-panel OHLCV + PnL chart
- `results/plots/lstm_ohlcv_SPY_seed42.txt` - Text summary with PnL stats

---

## Example Output

### JSON Metrics

```json
{
  "ticker": "SPY",
  "statistical_metrics": {
    "rmse": 0.012345,
    "directional_accuracy": 0.567890
  },
  "signal_statistics": {
    "buy_signals": 8234,
    "sell_signals": 7891,
    "hold_signals": 21531
  },
  "pnl_statistics": {
    "initial_capital": 100000.0,
    "final_equity": 105234.56,
    "total_return": 0.0523,
    "total_pnl": 5234.56,
    "sharpe_ratio": 1.234,
    "max_drawdown": -0.0845,
    "n_trades": 234
  }
}
```

### Console Output

```
PnL Statistics:
  Initial Capital: $100,000.00
  Final Equity: $105,234.56
  Total Return: 5.23%
  Total PnL: $5,234.56
  Sharpe Ratio: 1.234
  Max Drawdown: -8.45%
  Number of Trades: 234
```

---

## Why This Matters

### Before (Prediction-Only)

```
LSTM → Predictions → Metrics (RMSE, DA)
```

**Problem**: Good predictions don't guarantee good trading performance.

### After (Prediction + PnL)

```
LSTM → Predictions → Signals → PnL → Trading Metrics (Sharpe, Drawdown)
```

**Benefit**: Can now evaluate if the model is actually profitable.

---

## Next Step: RL Agent

The PnL calculation is a foundation for the **RL trading agent** (see `docs/plans/RL_TRADING_AGENT_PLAN.md`).

**RL Agent will**:
- Use PnL as the reward signal
- Learn optimal position sizing
- Adapt to market regimes
- Directly optimize trading performance (not prediction accuracy)

**Expected improvement**: Sharpe ratio from 1.2 → 2.0 (+67%)

---

## Files Modified

1. `scripts/run_lstm_baseline.py` - Added PnL calculation and equity curve plot
2. `plots/plot_lstm.py` - Added 4th panel for PnL visualization
3. `docs/plans/RL_TRADING_AGENT_PLAN.md` - Comprehensive RL implementation plan

---

**Status**: ✅ PnL tracking complete. Ready for RL agent implementation.
