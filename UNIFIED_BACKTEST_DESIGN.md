# Unified Backtesting Engine Design Specification

**Version:** 1.0.0  
**Status:** Draft  
**Author:** System Architect  
**Date:** 2026-04-28  
**Source:** Consolidation of `backtest.py`, `backtester.py`, `backtest_results.py`, `run_backtest.py`  
**TRD Reference:** FINAL_PLAN.md §6 (Backtesting), TRD3 (Evaluation)

---

## 1. Architecture Overview

### 1.1 Philosophy

The unified engine follows the **Strategy Pattern** with a **Facade**. The `BacktestEngine` presents a single, simple API to callers. Internally, it delegates to frequency-specific strategy implementations (`DailyStrategy`, `IntradayStrategy`) that reuse the battle-tested logic from the existing codebase. This preserves the simplicity of `backtest.py` for daily use cases while retaining the full power of `backtester.py` for intraday scenarios.

**Design Principles:**
1. **Explicit over implicit**: Frequency is explicitly configured, with optional heuristic auto-detection.
2. **Zero-cost abstraction**: If you only need daily backtesting, you do not pay the complexity cost of intraday state machines.
3. **Immutable inputs**: The engine never mutates input arrays.
4. **Deterministic**: No randomness; same inputs always produce identical outputs.

### 1.2 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI / Orchestrator                        │
│              (pipelines/run_backtest.py)                     │
│                     [Facade Client]                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  BacktestEngine (Facade)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │   Config    │  │    State     │  │   ResultBuilder     │ │
│  │   (Pydantic │  │   (Context)  │  │   (Aggregator)      │ │
│  │   /Dataclass│  │              │  │                     │ │
│  └─────────────┘  └──────────────┘  └─────────────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐
│   DailyStrategy     │    │  IntradayStrategy   │
│  [from backtest.py] │    │ [from backtester.py]│
│                     │    │                     │
│ • Vectorized        │    │ • Event-driven loop │
│ • Returns-only      │    │ • OHLC required     │
│ • Simple costs      │    │ • Session boundaries│
│ • Benchmark B&H     │    │ • Risk controls     │
└─────────────────────┘    └─────────────────────┘
         │                           │
         └─────────────┬─────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│SignalGenerator│ │ CostModel   │ │RiskManager  │
└─────────────┘ └─────────────┘ └─────────────┘
         │             │             │
         └─────────────┴─────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              MetricsCalculator + TradeLogger                 │
│         (Statistical, Trading, Benchmark)                    │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Data Flow

```
Input: predictions[N], actual_returns[N], dates[N]
       Optional: opens[N], closes[N], session_starts[N]

        │
        ▼
┌───────────────┐
│  Validation   │ ──► Check lengths, NaNs, dtypes, monotonic dates
└───────────────┘
        │
        ▼
┌───────────────┐
│   Frequency   │ ──► Explicit config OR heuristic (see §2.4)
│   Resolution  │
└───────────────┘
        │
        ├──► Daily ─────► DailyStrategy.run_vectorized()
        │
        └──► Intraday ──► IntradayStrategy.run_event_driven()
                              │
                              ▼
                    ┌─────────────────┐
                    │  Signal Gen     │ ──► Ternary (-1, 0, +1)
                    │  (threshold)    │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  CostModel      │ ──► Transaction costs + slippage
                    │  (position chg) │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  RiskManager    │ ──► Stop-loss, daily halt (intraday only)
                    │  (optional)     │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Equity Curve    │ ──► MTM + cumulative returns
                    │ Computation     │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  MetricsCalc    │ ──► Sharpe, MDD, DA, etc.
                    │  + TradeLogger  │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  BacktestResult │ ──► Dataclass with full provenance
                    │  (immutable)    │
                    └─────────────────┘
```

### 1.4 Responsibility Assignment

| Component | Responsibility | Source File |
|-----------|---------------|-------------|
| `BacktestEngine` | Public API, input validation, frequency routing, result aggregation | **New** |
| `BacktestConfig` | Immutable configuration dataclass | **New** |
| `DailyStrategy` | Vectorized daily backtest logic | `backtest.py` (refactored) |
| `IntradayStrategy` | Event-driven intraday backtest logic | `backtester.py` (refactored) |
| `SignalGenerator` | Convert predictions to ternary signals | `evaluation/metrics.py` (reused) |
| `CostModel` | Transaction costs + slippage | Extracted from both |
| `RiskManager` | Stop-loss, daily loss limits | `backtester.py` (extracted) |
| `MetricsCalculator` | Statistical, trading, benchmark metrics | `evaluation/metrics.py` + `backtest.py` |
| `TradeLogger` | Detailed trade record keeping | `backtester.py` (extracted) |
| `BacktestResult` | Structured output container | `backtest_results.py` (refactored) |
| `ResultPersistence` | Save/load JSON/CSV/YAML/Markdown | `backtest_results.py` (refactored) |

---

## 2. Core Engine (`BacktestEngine`)

### 2.1 API Design

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
from enum import Enum

import numpy as np
import pandas as pd


class FrequencyMode(Enum):
    AUTO = "auto"
    DAILY = "daily"
    INTRADAY = "intraday"


@dataclass(frozen=True)
class BacktestConfig:
    """
    Immutable configuration for the backtest engine.
    
    All parameters have sensible defaults aligned with FINAL_PLAN.md.
    """
    # Core parameters
    transaction_cost: float = 0.0015      # One-way (0.15%)
    initial_capital: float = 100_000.0
    
    # Optional intraday parameters
    slippage: Optional[float] = None      # Default: 0.0 for daily, 0.0005 for intraday
    stop_loss: Optional[float] = None     # e.g., 0.02 = 2%
    daily_loss_limit: Optional[float] = None  # e.g., 0.05 = 5%
    position_fraction: float = 1.0        # 1.0 = fully invested (daily), 0.02 = 2% (intraday)
    
    # Signal generation
    signal_threshold: float = 0.0         # Minimum |prediction| to trigger signal
    signal_mode: Literal["sign", "threshold"] = "sign"
    
    # Frequency detection
    frequency_mode: FrequencyMode = FrequencyMode.AUTO
    
    # Detailed logging
    detailed_trade_log: bool = False      # True = emit per-trade DataFrame
    
    # Benchmark
    compute_benchmark: bool = True        # Buy-and-hold comparison
    
    def __post_init__(self):
        if self.transaction_cost < 0:
            raise ValueError("transaction_cost must be non-negative")
        if self.stop_loss is not None and not (0 < self.stop_loss < 1):
            raise ValueError("stop_loss must be in (0, 1)")
        if self.daily_loss_limit is not None and not (0 < self.daily_loss_limit < 1):
            raise ValueError("daily_loss_limit must be in (0, 1)")


class BacktestEngine:
    """
    Unified backtesting engine supporting both daily and intraday frequencies.
    
    Usage:
        engine = BacktestEngine(config=BacktestConfig(transaction_cost=0.001))
        result = engine.run(
            predictions=y_pred,
            actual_returns=y_test,
            dates=date_index,
            opens=opens,      # Required only for intraday
            closes=closes,    # Required only for intraday
        )
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._strategy: Optional[Any] = None
    
    def run(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        dates: pd.DatetimeIndex,
        opens: Optional[np.ndarray] = None,
        closes: Optional[np.ndarray] = None,
        session_starts: Optional[np.ndarray] = None,
    ) -> "BacktestResult":
        """
        Execute backtest.
        
        Args:
            predictions: Model predicted returns, shape (N,)
            actual_returns: Actual market returns, shape (N,)
            dates: DatetimeIndex aligned with predictions
            opens: Opening prices, shape (N,). Required for intraday.
            closes: Closing prices, shape (N,). Required for intraday.
            session_starts: Boolean array marking session open bars.
            
        Returns:
            BacktestResult containing equity curve, metrics, and optional trade log.
        """
        ...
    
    def detect_frequency(
        self,
        dates: pd.DatetimeIndex,
        has_ohlc: bool,
    ) -> Literal["daily", "intraday"]:
        """
        Heuristic frequency detection.
        
        Rules:
        1. If opens/closes provided → intraday
        2. If median time delta < 1 hour → intraday
        3. If explicit config set → use config
        4. Default → daily
        """
        ...
```

### 2.2 Internal Architecture

The engine uses **composition over inheritance**. The strategy is selected at runtime based on frequency.

```python
class _BacktestContext:
    """
    Mutable internal state passed through the backtest pipeline.
    Not exposed to callers.
    """
    predictions: np.ndarray
    actual_returns: np.ndarray
    dates: pd.DatetimeIndex
    signals: np.ndarray
    strategy_returns: np.ndarray
    equity_curve: np.ndarray
    trade_costs: np.ndarray
    trade_log: List[Dict[str, Any]]
    benchmark_equity: np.ndarray
```

#### Strategy Interface

```python
from typing import Protocol

class BacktestStrategy(Protocol):
    def run(self, ctx: _BacktestContext, config: BacktestConfig) -> None:
        """Mutates ctx in-place with backtest results."""
        ...
```

### 2.3 Design Decisions & Justifications

#### Decision 1: Strategy Pattern vs. Monolithic Class

**Options:**
- **A. Single class with if-statements**: Simple but violates Single Responsibility Principle. Daily logic is vectorized; intraday is event-driven. Mixing them creates a God class.
- **B. Inheritance (BaseBacktest → Daily/Intraday)**: Tempting, but the two strategies share almost no internal state. Intraday needs entry prices, session tracking, daily halts; daily needs none of this.
- **C. Strategy Pattern (chosen)**: Clean separation. Daily strategy is a pure function over arrays. Intraday strategy is a state machine. Both conform to the same protocol.

**Justification:** The existing `backtest.py` and `backtester.py` are fundamentally different algorithms. Forcing them into an inheritance hierarchy would create an awkward base class with many `Optional` attributes. The Strategy pattern lets each algorithm exist independently while exposing a unified facade.

#### Decision 2: Frequency Detection

**Question:** How to reliably detect daily vs intraday?

**Answer:** Three-tier approach:
1. **Explicit config (`frequency_mode`)**: User knows their data. This is the primary path.
2. **OHLC availability**: If `opens` and `closes` are provided, assume intraday. Daily backtests only need returns.
3. **Heuristic fallback**: Compute median time delta. If median delta < 1 hour → intraday. If >= 1 day → daily. If mixed/irregular → raise error requiring explicit config.

```python
def detect_frequency(self, dates, has_ohlc):
    if self.config.frequency_mode != FrequencyMode.AUTO:
        return self.config.frequency_mode.value
    
    if has_ohlc:
        return "intraday"
    
    deltas = pd.Series(dates).diff().dropna()
    median_delta = deltas.median()
    
    if median_delta < pd.Timedelta(hours=1):
        return "intraday"
    elif median_delta >= pd.Timedelta(hours=20):
        return "daily"
    else:
        raise ValueError(
            f"Ambiguous frequency (median delta: {median_delta}). "
            f"Please set frequency_mode explicitly."
        )
```

**Edge case handling:**
- **Mixed frequency**: Raises explicit error. We do not support mixed-frequency data in a single backtest.
- **Irregular gaps**: Heuristic uses median, not mean, to resist outliers (weekends, holidays).
- **Daily data with OHLC**: If user provides OHLC but wants daily, they must set `frequency_mode=FrequencyMode.DAILY`. The engine will use close-to-close returns from the closes array.

#### Decision 3: Risk Controls as Optional Parameters

**Question:** Should stop-loss/daily limits be optional or always applied?

**Answer:** Optional via `Optional[float]` config fields. When `None`, the risk control is disabled.

**Rationale:**
- Daily strategies rarely need intraday stop-losses (no intra-bar data).
- Backtests for research often want to isolate signal quality from risk management.
- However, production intraday backtests require them. Making them optional satisfies both use cases.

**Edge case:** Stop-loss at last bar. If a stop-loss triggers at the final bar, we cannot exit (no next bar). The engine logs this as a "forced exit at close" and applies costs.

#### Decision 4: Session Boundaries

**Question:** How to handle half-days, holidays, extended hours?

**Answer:** Session boundaries are detected via a `session_starts` boolean array provided by the upstream data pipeline. If not provided, the engine falls back to time-based detection (9:30 and 16:00 ET).

**Rationale:**
- The upstream data pipeline already knows about market calendars (holidays, half-days).
- Pushing calendar logic into the backtester creates duplication and coupling.
- The `session_starts` array is the cleanest contract: one boolean per bar.

```python
# Preferred: explicit session markers
if session_starts is not None:
    is_open = session_starts[t]
    is_close = session_starts[t+1] if t+1 < N else True
else:
    # Fallback: time-based (9:30 ET, 16:00 ET)
    et_time = dates.tz_convert("America/New_York")[t]
    is_open = et_time.hour == 9 and et_time.minute == 30
    is_close = et_time.hour == 16 and et_time.minute == 0
```

#### Decision 5: Stateless vs. Stateful Engine

**Question:** Should the engine be stateless or stateful?

**Answer:** The public API (`BacktestEngine.run()`) is **stateless** (functional). The internal strategies use **mutable context** (`_BacktestContext`) for performance.

**Rationale:**
- Stateless public API is thread-safe and easy to test.
- Internal mutation is acceptable because `_BacktestContext` is private and discarded after `run()` returns.
- For walk-forward validation, the caller simply instantiates a new `BacktestEngine` per fold.

#### Decision 6: Position Sizing

**Question:** How to unify daily (binary long/short) and intraday (fractional capital) position sizing?

**Answer:** Configurable `position_fraction`.

- **Daily mode**: `position_fraction=1.0` means full notional long/short. This preserves the exact behavior of `backtest.py`.
- **Intraday mode**: `position_fraction=0.02` means 2% of capital per trade. This preserves `backtester.py` behavior.

The strategy implementations interpret this parameter appropriately:
- `DailyStrategy`: `position = signal * 1.0` (full notional). `position_fraction` is ignored unless we add leverage later.
- `IntradayStrategy`: `n_shares = (position_fraction * equity) / price`.

#### Decision 7: Signal Generation

**Question:** How to handle different signal generation modes?

**Answer:** Delegated to `SignalGenerator` utility (already exists in `evaluation/metrics.py`).

```python
def generate_signals(predictions: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    signals = np.zeros_like(predictions, dtype=np.int8)
    signals[predictions > threshold] = 1
    signals[predictions < -threshold] = -1
    return signals
```

The engine calls this once before strategy execution. Both daily and intraday strategies receive the same `signals` array.

#### Decision 8: Cost Model

**Question:** How to structure transaction costs and slippage?

**Answer:** Extracted into a `CostModel` class used by both strategies.

```python
@dataclass(frozen=True)
class CostModel:
    transaction_cost: float
    slippage: Optional[float]
    
    def apply(
        self,
        position_changes: np.ndarray,
        entry_prices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (trade_costs, slippage_costs).
        """
        ...
```

For daily mode, slippage is ignored (no price data). For intraday, slippage is applied asymmetrically against the trade direction, matching `backtester.py`.

### 2.4 Daily Strategy Specification

```python
class DailyStrategy:
    """
    Vectorized daily backtest.
    
    Reproduces the exact behavior of CanonicalBacktest from backtest.py.
    """
    
    def run(self, ctx: _BacktestContext, config: BacktestConfig) -> None:
        # 1. Signals already generated by engine
        signals = ctx.signals
        
        # 2. Detect position changes
        position = np.concatenate([[0], signals])  # Start neutral
        position_changes = np.abs(np.diff(position))
        
        # 3. Apply costs
        trade_costs = position_changes * config.transaction_cost
        # Daily: no slippage (no price data)
        
        # 4. Strategy returns
        ctx.strategy_returns = signals * ctx.actual_returns - trade_costs[1:]
        ctx.trade_costs = trade_costs[1:]
        
        # 5. Equity curve
        cumulative = (1 + ctx.strategy_returns).cumprod()
        ctx.equity_curve = config.initial_capital * cumulative
        
        # 6. Benchmark (buy and hold)
        if config.compute_benchmark:
            bench_cumulative = (1 + ctx.actual_returns).cumprod()
            ctx.benchmark_equity = config.initial_capital * bench_cumulative
        
        # 7. Trade log (simplified for daily)
        if config.detailed_trade_log:
            ctx.trade_log = self._build_daily_trade_log(ctx, config)
```

### 2.5 Intraday Strategy Specification

```python
class IntradayStrategy:
    """
    Event-driven intraday backtest.
    
    Reproduces the exact behavior of Backtester from backtester.py.
    """
    
    def run(self, ctx: _BacktestContext, config: BacktestConfig) -> None:
        # Requires opens/closes to be present in context
        # Iterates bar-by-bar with full state machine
        # Applies session boundaries, stop-loss, daily limits, slippage
        ...
```

The intraday strategy is a direct refactoring of `backtester.py` into the strategy interface. The loop logic, session detection, position management, and risk controls are preserved verbatim but operate on `_BacktestContext` instead of returning a separate `BacktestResult`.

### 2.6 Metrics Calculation

Metrics are computed **after** the strategy runs, by a shared `MetricsCalculator`. This avoids duplicating metric logic across strategies.

```python
class MetricsCalculator:
    def compute_all(
        self,
        ctx: _BacktestContext,
        config: BacktestConfig,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Returns (statistical_metrics, trading_metrics, benchmark_metrics).
        """
        stat = self._statistical(ctx.predictions, ctx.actual_returns)
        trade = self._trading(ctx.equity_curve, ctx.strategy_returns)
        bench = self._benchmark(ctx.benchmark_equity) if config.compute_benchmark else {}
        return stat, trade, bench
```

**Statistical Metrics** (from `evaluation/metrics.py`):
- `mse`, `mae`, `rmse`, `r2`, `directional_accuracy`, `mape`

**Trading Metrics** (from `backtest.py` + `backtester.py`):
- `total_return`, `cagr`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`
- `max_drawdown`, `calmar_ratio`, `win_rate`, `profit_factor`, `total_trades`

**Benchmark Metrics**:
- `benchmark_total_return`, `benchmark_sharpe`, `benchmark_max_drawdown`
- `alpha`, `beta`, `information_ratio` (optional, if benchmark returns available)

---

## 3. Results Container (`BacktestResult`)

### 3.1 Dataclass Specification

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """
    Immutable container for complete backtest results.
    
    All array fields are guaranteed to have the same length (aligned).
    """
    
    # ── Identification ─────────────────────────────────────────────
    model_type: str = "unknown"
    ticker: str = "unknown"
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    
    # ── Time Series (all length N, aligned) ────────────────────────
    dates: pd.DatetimeIndex = field(repr=False)
    predictions: np.ndarray = field(repr=False)
    actual_returns: np.ndarray = field(repr=False)
    signals: np.ndarray = field(repr=False)
    strategy_returns: np.ndarray = field(repr=False)
    equity_curve: np.ndarray = field(repr=False)
    trade_costs: np.ndarray = field(repr=False)
    
    # ── Metrics ────────────────────────────────────────────────────
    statistical_metrics: Dict[str, float] = field(default_factory=dict)
    trading_metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_metrics: Dict[str, float] = field(default_factory=dict)
    
    # ── Detailed Trade Log (optional) ──────────────────────────────
    trade_log: Optional[pd.DataFrame] = None
    
    # ── Provenance ─────────────────────────────────────────────────
    config: Dict[str, Any] = field(default_factory=dict, repr=False)
    engine_version: str = "1.0.0"
    
    def __post_init__(self):
        # Alignment invariant
        n = len(self.predictions)
        for field_name in ["actual_returns", "signals", "strategy_returns", 
                          "equity_curve", "trade_costs"]:
            arr = getattr(self, field_name)
            if len(arr) != n:
                raise ValueError(
                    f"Length mismatch: predictions ({n}) vs {field_name} ({len(arr)})"
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (excludes large arrays)."""
        ...
    
    def summary(self) -> str:
        """Human-readable summary string."""
        ...
```

### 3.2 Versioning

The result container includes `engine_version` for backward compatibility. When loading old results, a migration layer checks the version and transforms fields if necessary.

```python
# In result persistence layer
def load_backtest_results(path: Path) -> BacktestResult:
    data = json.loads(path.read_text())
    version = data.get("engine_version", "0.0.0")
    
    if version == "0.0.0":
        # Migrate from old format
        data = _migrate_v0_to_v1(data)
    
    return BacktestResult(**data)
```

---

## 4. CLI Orchestrator (`pipelines/run_backtest.py`)

### 4.1 Refactored Architecture

The CLI becomes a **thin orchestration layer** (~150 lines). All backtesting logic is delegated to `BacktestEngine`.

```python
#!/usr/bin/env python3
"""
Unified Backtesting Pipeline - Production Entry Point

Thin orchestration layer. All backtesting logic lives in BacktestEngine.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.engine import BacktestEngine, BacktestConfig, FrequencyMode
from src.backtesting.results import BacktestResult, save_backtest_results
from src.evaluation.model_loader import load_model
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Backtesting Pipeline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-type", required=True, 
                       choices=["pso_lstm", "lstm_baseline", "xgboost"])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frequency", choices=["auto", "daily", "intraday"], 
                       default="auto")
    return parser.parse_args()


def load_test_data(data_path: Path) -> dict:
    """Load preprocessed test data."""
    X = np.load(data_path / "X_test.npy")
    y = np.load(data_path / "y_test.npy")
    dates = np.load(data_path / "test_index.npy", allow_pickle=True)
    
    data = {
        "X_test": X,
        "y_test": y,
        "dates": pd.to_datetime(dates),
    }
    
    # Optional OHLC for intraday
    for key in ["opens", "closes", "session_starts"]:
        path = data_path / f"{key}.npy"
        if path.exists():
            data[key] = np.load(path)
    
    return data


def main():
    args = parse_args()
    setup_logger(log_file=args.output_dir / "backtest.log")
    
    # Load config and model
    config = load_config(args.config)
    model = load_model(args.model_type, args.model_path, config)
    
    # Load data
    data = load_test_data(args.data_path)
    predictions = model.predict(data["X_test"])
    
    # Determine frequency mode
    freq_map = {
        "auto": FrequencyMode.AUTO,
        "daily": FrequencyMode.DAILY,
        "intraday": FrequencyMode.INTRADAY,
    }
    
    # Build engine config from unified config
    bt_cfg = config.backtesting
    engine_config = BacktestConfig(
        transaction_cost=bt_cfg.transaction_cost,
        initial_capital=bt_cfg.initial_capital,
        slippage=getattr(bt_cfg, "slippage", None),
        stop_loss=getattr(bt_cfg, "stop_loss", None),
        daily_loss_limit=getattr(bt_cfg, "daily_loss_limit", None),
        position_fraction=getattr(bt_cfg, "position_fraction", 1.0),
        frequency_mode=freq_map[args.frequency],
        detailed_trade_log=True,
        compute_benchmark=True,
    )
    
    # Run unified backtest
    engine = BacktestEngine(config=engine_config)
    result = engine.run(
        predictions=predictions,
        actual_returns=data["y_test"],
        dates=data["dates"],
        opens=data.get("opens"),
        closes=data.get("closes"),
        session_starts=data.get("session_starts"),
    )
    
    # Save results
    save_backtest_results(result, args.output_dir)
    
    # Log summary
    logger.info("=" * 80)
    logger.info("BACKTEST COMPLETE")
    logger.info(f"Sharpe: {result.trading_metrics['sharpe']:.2f}")
    logger.info(f"Max DD: {result.trading_metrics['max_drawdown']:.2%}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
```

### 4.2 Key Refactoring Changes

| Issue in Current Code | Resolution |
|----------------------|------------|
| `backtest_baseline_lstm` duplicates `run_unified_backtest` | Deleted. Single `engine.run()` path for all models. |
| Commented-out code (lines 248-254) | Removed. Engine handles frequency internally. |
| Hardcoded model-type branching | Eliminated. Model loader adapter pattern handles all types. |
| Inline metric computation | Delegated to `MetricsCalculator`. |
| Inline plotting | Delegated to `evaluation.plotting` (unchanged). |

---

## 5. Migration Strategy

### 5.1 Deprecation Plan

**Phase 1 (Weeks 1-2):** Introduce new engine alongside old code. Old APIs remain functional.

**Phase 2 (Weeks 3-4):** Add deprecation warnings to old classes:

```python
# In backtest.py
class CanonicalBacktest:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "CanonicalBacktest is deprecated. Use BacktestEngine instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
```

**Phase 3 (Weeks 5-6):** Update all internal pipelines to use `BacktestEngine`.

**Phase 4 (Week 7):** Remove old implementations (or move to `src/backtesting/_legacy/`).

### 5.2 Equivalence Testing

Before deprecating, we run **deterministic equivalence tests**:

```python
def test_daily_equivalence():
    """New engine must produce identical results to CanonicalBacktest."""
    old = CanonicalBacktest(transaction_cost=0.0015)
    old_result = old.run_backtest(pred, ret, dates)
    
    new = BacktestEngine(BacktestConfig(transaction_cost=0.0015, frequency_mode="daily"))
    new_result = new.run(pred, ret, dates)
    
    np.testing.assert_allclose(
        old_result["capital"].values,
        new_result.equity_curve,
        rtol=1e-10,
    )

def test_intraday_equivalence():
    """New engine must produce identical results to Backtester."""
    old = Backtester(...)
    old_result = old.run(pred, opens, closes, dates)
    
    new = BacktestEngine(BacktestConfig(..., frequency_mode="intraday"))
    new_result = new.run(pred, ret, dates, opens, closes)
    
    np.testing.assert_allclose(old_result.equity_curve, new_result.equity_curve)
    pd.testing.assert_frame_equal(old_result.trade_log, new_result.trade_log)
```

### 5.3 Saved Results Migration

Old results use a flat structure. The new `ResultPersistence` layer includes a migration function:

```python
def _migrate_v0_to_v1(data: dict) -> dict:
    """Migrate BacktestResults from old format to v1.0.0."""
    data["engine_version"] = "1.0.0"
    data["benchmark_metrics"] = {}
    data["config"] = data.pop("backtest_config", {})
    return data
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

| Test | Description |
|------|-------------|
| `test_config_validation` | Invalid configs raise ValueError |
| `test_frequency_detection` | Heuristics correctly classify daily/intraday/mixed |
| `test_signal_generation` | Thresholds produce correct ternary signals |
| `test_daily_cost_model` | Costs applied only on position changes |
| `test_intraday_slippage` | Slippage applied asymmetrically |
| `test_stop_loss_trigger` | Stop-loss forces flat at next bar |
| `test_daily_halt` | Daily limit halts trading until next session |
| `test_session_boundary` | Force flat at market close |
| `test_alignment_invariant` | Result arrays always same length |
| `test_benchmark_computation` | B&H equity curve computed correctly |

### 6.2 Integration Tests

| Test | Description |
|------|-------------|
| `test_end_to_end_daily` | Full pipeline on synthetic daily data |
| `test_end_to_end_intraday` | Full pipeline on synthetic 1-min data |
| `test_model_agnostic` | Engine works with dummy model, LSTM adapter, XGB adapter |
| `test_result_persistence` | Save/load roundtrip preserves all data |

### 6.3 Equivalence Tests

| Test | Description |
|------|-------------|
| `test_equivalence_canonical_backtest` | New daily = old `CanonicalBacktest` |
| `test_equivalence_backtester` | New intraday = old `Backtester` |
| `test_equivalence_edge_cases` | Empty data, single sample, all-neutral signals |

### 6.4 Edge Cases

```python
# All-neutral signals (no trades)
predictions = np.zeros(100)
result = engine.run(predictions, returns, dates)
assert result.trading_metrics["total_trades"] == 0
assert np.allclose(result.equity_curve, config.initial_capital)

# Single sample
predictions = np.array([0.01])
result = engine.run(predictions, returns, dates)
assert len(result.equity_curve) == 1

# NaN inputs
with pytest.raises(ValueError):
    engine.run(predictions_with_nan, returns, dates)
```

---

## 7. Specific Questions Answered

### Q1: Frequency Detection
**A:** Explicit config primary, OHLC availability secondary, median-delta heuristic tertiary. Mixed/irregular raises explicit error.

### Q2: Risk Controls
**A:** Optional via `Optional[float]`. When `None`, disabled. Stop-loss at last bar triggers "exit at close" with costs.

### Q3: Session Boundaries
**A:** Preferred: explicit `session_starts` boolean array from upstream data pipeline. Fallback: timezone-aware time detection (9:30/16:00 ET). Half-days handled by upstream marker.

### Q4: State Management
**A:** Public API is stateless. Internal strategies use mutable `_BacktestContext` for performance. No multi-asset support yet (future extension via `PortfolioContext`).

### Q5: Backward Compatibility
**A:** Old APIs preserved with `DeprecationWarning` for one release cycle. Equivalence tests guarantee identical outputs. Saved results auto-migrated on load.

### Q6: Performance
**A:** Daily strategy is fully vectorized (NumPy). Intraday strategy is loop-based (necessary for state machine). No unnecessary copies. Acceptable trade-off: intraday correctness requires sequential execution.

---

## Appendix A: File Layout (Target)

```
src/
└── backtesting/
    ├── __init__.py
    ├── engine.py              # BacktestEngine + BacktestConfig
    ├── strategies/
    │   ├── __init__.py
    │   ├── base.py            # BacktestStrategy protocol
    │   ├── daily.py           # DailyStrategy (vectorized)
    │   └── intraday.py        # IntradayStrategy (event-driven)
    ├── components/
    │   ├── __init__.py
    │   ├── signal_generator.py
    │   ├── cost_model.py
    │   ├── risk_manager.py
    │   ├── metrics.py         # MetricsCalculator
    │   └── trade_logger.py
    ├── results.py             # BacktestResult + persistence
    └── _legacy/               # Old implementations (temporary)
        ├── backtest.py
        ├── backtester.py
        └── backtest_results.py
```

---

## Appendix B: TRD Compliance Matrix

| TRD Requirement | Implementation |
|-----------------|----------------|
| §6.1: Signal Generation | `SignalGenerator.generate_signals()` with threshold |
| §6.2: Transaction Costs | `CostModel` applied on position changes only |
| §6.3: No Lookahead | Engine never uses future data |
| §6.4: Performance Metrics | `MetricsCalculator` computes all required metrics |
| §7.1: Buy-and-Hold Benchmark | Computed in daily mode, stored in `benchmark_metrics` |
| Temporal Integrity | Strict chronological ordering enforced |