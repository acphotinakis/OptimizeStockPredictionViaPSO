# Prompt: Design a Unified Backtesting Engine

## Context

You are a senior quantitative systems engineer tasked with analyzing and consolidating **three different backtesting implementations** into a single, production-grade, unified backtesting engine for a financial ML prediction system.

The system currently has:
1. **`pipelines/run_backtest.py`** - CLI entry point (orchestration layer)
2. **`src/backtesting/backtest.py`** - Simple canonical backtest with transaction costs
3. **`src/backtesting/backtester.py`** - Event-driven 1-minute bar backtester with advanced risk controls
4. **`src/backtesting/backtest_results.py`** - Results storage and persistence

---

## Current Architecture Analysis

### Component 1: `pipelines/run_backtest.py` (412 lines)
**Purpose:** CLI orchestration layer

**Responsibilities:**
- Parse command-line arguments
- Load models (PSO-LSTM, Baseline LSTM, XGBoost)
- Load test data
- Coordinate backtesting execution
- Compute metrics (statistical + trading)
- Save results and generate visualizations
- Logging and error handling

**Key Features:**
- Multi-model support (3 model types)
- Model loading via adapter pattern (`load_model()`)
- Unified metrics computation
- Plot generation
- Structured output directory management

**Issues:**
- Contains incomplete/commented code (e.g., lines 248-254, 269)
- Mixes high-level orchestration with implementation details
- Redundant functions (`backtest_baseline_lstm` vs `run_unified_backtest`)
- Hardcoded logic for different model types

---

### Component 2: `src/backtesting/backtest.py` (317 lines)
**Purpose:** Simple canonical backtesting engine

**Responsibilities:**
- Signal generation (directional: +1, 0, -1)
- Transaction cost application (one-way cost on position changes)
- Strategy return computation
- Performance metrics (Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Profit Factor)
- Buy-and-hold benchmark comparison

**Key Features:**
- Clean, functional API: `run_backtest(predictions, actual_returns, dates)`
- Transaction costs applied correctly (only on position changes)
- Cumulative returns calculation
- Comprehensive performance metrics
- Benchmark comparison (buy-and-hold)
- TRD-compliant (references `FINAL_PLAN.md`)

**Strengths:**
- Well-documented with TRD references
- Simple, stateless design
- Clear separation of concerns (signal generation → cost application → metrics)
- Easy to test and reason about

**Limitations:**
- **No intraday risk controls** (stop-loss, daily limits)
- **No slippage modeling**
- **Assumes daily frequency** (not suitable for 1-minute bars)
- **No session boundary handling** (open/close logic)
- **Simple position sizing** (binary long/short/neutral)

---

### Component 3: `src/backtesting/backtester.py` (437 lines)
**Purpose:** Event-driven high-frequency backtester

**Responsibilities:**
- Intraday position management (open/close at specific bars)
- Session boundary detection (market open at 9:30 ET, close at 16:00 ET)
- Risk controls:
  - **Stop-loss**: Force flat next bar if trade drawdown > 2%
  - **Daily loss limit**: Halt trading if session drawdown > 5%
- Slippage modeling (0.05% default)
- Position sizing (2% of capital per trade)
- Trade logging (entry/exit times, P&L, costs)
- Mark-to-market equity calculation

**Key Features:**
- Event-driven architecture (step-by-step simulation)
- Realistic intraday trading mechanics:
  - Entry at open price, MTM at close price
  - Position sizing via fractional capital allocation
  - Slippage applied asymmetrically (against trade direction)
- Risk management:
  - Trade-level stop-loss
  - Session-level daily halt
- Session awareness:
  - Force flat at market close (16:00 ET)
  - Reset daily halt at market open (9:30 ET)
- Threshold optimization via grid search

**Strengths:**
- Production-grade for high-frequency strategies
- Comprehensive risk controls
- Detailed trade logging
- Handles timezone-aware timestamps
- Realistic slippage and cost modeling

**Limitations:**
- **More complex** (harder to understand and maintain)
- **Requires OHLC data** (not just returns)
- **Requires timezone-aware timestamps**
- **Not compatible with daily-frequency data** (session logic assumes intraday)
- **No benchmark comparison**

---

### Component 4: `src/backtesting/backtest_results.py` (320 lines)
**Purpose:** Results storage and persistence

**Responsibilities:**
- Define `BacktestResults` dataclass (predictions, signals, equity curve, metrics)
- Save results to disk (JSON, CSV, YAML, Markdown)
- Load results from disk
- Generate human-readable Markdown report

**Key Features:**
- Structured output format:
  - `backtest_results.json` - Metrics and summary
  - `equity_curve.csv` - Time series data
  - `predictions.csv` - Predictions vs actuals
  - `metadata.yaml` - Model and config metadata
  - `backtest_report.md` - Human-readable summary
- Array length alignment (handles LSTM windowing edge cases)
- Comprehensive metadata tracking

**Strengths:**
- Clean dataclass design
- Well-structured output format
- Good documentation
- Handles edge cases (length mismatches)

**Issues:**
- Tightly coupled to specific data structures
- No versioning for backward compatibility
- Limited extensibility for custom metrics

---

## Problem Statement

The current codebase has **three separate backtesting implementations** with different capabilities:

| Feature | `backtest.py` | `backtester.py` | Desired Unified Engine |
|---------|---------------|-----------------|------------------------|
| **Frequency Support** | Daily | Intraday (1-min) | **Both** |
| **Transaction Costs** | ✅ Yes | ✅ Yes | ✅ Required |
| **Slippage** | ❌ No | ✅ Yes | ✅ Optional |
| **Stop-Loss** | ❌ No | ✅ Yes | ✅ Optional |
| **Daily Limits** | ❌ No | ✅ Yes | ✅ Optional |
| **Session Boundaries** | ❌ No | ✅ Yes | ✅ For intraday |
| **Benchmark Comparison** | ✅ Yes | ❌ No | ✅ Required |
| **Trade Logging** | ❌ No | ✅ Yes | ✅ Optional |
| **Complexity** | Low | High | **Moderate** |
| **Testability** | High | Medium | **High** |

**Key Issues:**
1. **Duplication**: Two separate backtesting engines with overlapping logic
2. **Inconsistency**: Different APIs, different output formats
3. **Complexity**: Hard to maintain and extend
4. **Incomplete**: CLI orchestrator has commented/unused code
5. **Coupling**: Components not cleanly separated

---

## Design Requirements

### Functional Requirements

1. **Multi-Frequency Support**
   - Must handle both daily and intraday data seamlessly
   - Automatically detect frequency and apply appropriate logic
   - Session boundaries only for intraday (ignore for daily)

2. **Flexible Risk Controls**
   - Transaction costs (mandatory)
   - Slippage (optional, configurable)
   - Stop-loss (optional, intraday only)
   - Daily loss limits (optional, intraday only)

3. **Comprehensive Metrics**
   - Statistical: RMSE, MAE, R², Directional Accuracy, MAPE
   - Trading: Sharpe, Sortino, CAGR, Max Drawdown, Calmar, Profit Factor, Win Rate
   - Benchmark: Buy-and-hold comparison

4. **Trade Logging**
   - Optional detailed trade log (entry/exit, P&L, costs)
   - Required for analysis and debugging

5. **Results Persistence**
   - Structured output format (JSON, CSV, YAML, Markdown)
   - Backward-compatible versioning
   - Metadata tracking (model, config, timestamp)

6. **Model Agnostic**
   - Support any model type (LSTM, PSO-LSTM, XGBoost, etc.)
   - Use adapter pattern for model loading

### Non-Functional Requirements

1. **Simplicity**
   - Clear, readable code
   - Well-documented with docstrings
   - Easy to test and reason about

2. **Modularity**
   - Clean separation of concerns
   - Reusable components
   - Minimal coupling

3. **Extensibility**
   - Easy to add new metrics
   - Easy to add new risk controls
   - Easy to add new output formats

4. **Performance**
   - Fast execution (vectorized operations where possible)
   - Memory-efficient (avoid unnecessary copies)

5. **Testability**
   - Unit testable components
   - Deterministic results (no randomness)
   - Clear error messages

---

## Deliverables

### 1. Unified Backtesting Engine Design Document

Create a comprehensive design document (`UNIFIED_BACKTEST_DESIGN.md`) that specifies:

#### Section 1: Architecture Overview
- High-level component diagram
- Data flow diagram
- Responsibility assignment
- Interface contracts

#### Section 2: Core Engine (`BacktestEngine`)
- **API Design:**
  ```python
  class BacktestEngine:
      def __init__(
          self,
          transaction_cost: float = 0.0015,
          slippage: float = 0.0,
          stop_loss: Optional[float] = None,
          daily_loss_limit: Optional[float] = None,
          position_fraction: float = 0.02,
          initial_capital: float = 1.0,
          signal_threshold: float = 0.0,
      ): ...
      
      def run(
          self,
          predictions: np.ndarray,
          actual_returns: np.ndarray,
          dates: pd.DatetimeIndex,
          opens: Optional[np.ndarray] = None,
          closes: Optional[np.ndarray] = None,
      ) -> BacktestResult: ...
  ```

- **Responsibilities:**
  - Detect frequency (daily vs intraday)
  - Generate signals
  - Apply transaction costs
  - Apply slippage (if provided)
  - Apply risk controls (if enabled and intraday)
  - Compute equity curve
  - Compute all metrics
  - Generate trade log (if detailed mode)

- **Design Decisions:**
  - Which components to reuse from existing implementations?
  - How to handle daily vs intraday logic branching?
  - How to structure internal state management?
  - How to ensure backward compatibility?

#### Section 3: Results Container (`BacktestResult`)
- **Data Structure:**
  ```python
  @dataclass
  class BacktestResult:
      # Time series
      dates: pd.DatetimeIndex
      predictions: np.ndarray
      actual_returns: np.ndarray
      signals: np.ndarray
      strategy_returns: np.ndarray
      equity_curve: np.ndarray
      trade_costs: np.ndarray
      
      # Metrics
      statistical_metrics: Dict[str, float]
      trading_metrics: Dict[str, float]
      benchmark_metrics: Dict[str, float]
      
      # Optional detailed trade log
      trade_log: Optional[pd.DataFrame]
      
      # Metadata
      model_type: str
      ticker: str
      timestamp: str
      config: Dict[str, Any]
  ```

#### Section 4: CLI Orchestrator (`pipelines/run_backtest.py`)
- Refactored to be clean orchestration layer only
- Remove implementation details
- Use unified engine for all model types
- Standard error handling and logging

#### Section 5: Migration Strategy
- How to preserve existing functionality?
- How to test equivalence with old implementations?
- How to deprecate old components?
- What to do with existing saved results?

#### Section 6: Testing Strategy
- Unit tests for core engine
- Integration tests for CLI
- Equivalence tests (new vs old on same data)
- Edge case tests (empty data, single sample, NaN handling)

---

### 2. Implementation Plan

Create a step-by-step implementation plan:

1. **Phase 1: Core Engine**
   - Implement unified `BacktestEngine` class
   - Reuse logic from `backtest.py` for daily frequency
   - Reuse logic from `backtester.py` for intraday frequency
   - Add frequency auto-detection
   - Write unit tests

2. **Phase 2: Results Container**
   - Refactor `BacktestResults` dataclass
   - Add versioning support
   - Update save/load functions
   - Add backward compatibility layer

3. **Phase 3: CLI Refactor**
   - Simplify `run_backtest.py`
   - Remove duplication
   - Use unified engine
   - Update documentation

4. **Phase 4: Testing**
   - Write comprehensive tests
   - Validate against old implementations
   - Performance benchmarking

5. **Phase 5: Documentation**
   - Update README
   - Update TRD3 (evaluation spec)
   - Create usage examples
   - Migration guide for existing code

6. **Phase 6: Cleanup**
   - Deprecate old components
   - Remove commented code
   - Delete obsolete files

---

## Specific Questions to Answer

1. **Frequency Detection:**
   - How to reliably detect daily vs intraday frequency?
   - Use heuristics (time deltas) or explicit config parameter?
   - What if frequency is mixed or irregular?

2. **Risk Controls:**
   - Should stop-loss/daily limits be optional or always applied?
   - How to handle edge case where stop-loss triggers at last bar?
   - Should risk controls be pluggable (strategy pattern)?

3. **Session Boundaries:**
   - How to handle half-days, holidays, extended hours?
   - Should session detection be timezone-aware or config-driven?
   - What if OHLC data not available for daily frequency?

4. **State Management:**
   - Should engine be stateless (functional) or stateful (object)?
   - How to handle multi-asset portfolios (future extension)?
   - How to support walk-forward validation integration?

5. **Backward Compatibility:**
   - Should old APIs be preserved with deprecation warnings?
   - How to migrate existing pipeline scripts?
   - How to handle saved results from old format?

6. **Performance:**
   - Should we optimize for speed or readability?
   - Are there opportunities for vectorization?
   - What are acceptable trade-offs?

---

## Output Format

Produce two documents:

### 1. `UNIFIED_BACKTEST_DESIGN.md`
- Complete design specification (as outlined above)
- Include code examples for all interfaces
- Include diagrams (ASCII art or Mermaid)
- Include decision justifications with trade-off analysis

### 2. `UNIFIED_BACKTEST_IMPLEMENTATION_PLAN.md`
- Step-by-step implementation plan
- File-by-file changes required
- Testing strategy for each phase
- Timeline estimates (in story points or hours)
- Risk assessment and mitigation strategies

---

## Constraints

1. **No Breaking Changes:** Existing pipelines must continue to work (with deprecation warnings if needed)
2. **TRD Compliance:** Must align with `plans/design_specs/FINAL_PLAN.md` and TRD3
3. **Code Quality:** Must maintain high standards (type hints, docstrings, tests)
4. **Simplicity:** Prefer simple over clever (optimize for readability)
5. **No External Dependencies:** Use only existing dependencies (no new libraries)

---

## Success Criteria

A successful design will:

1. ✅ Unify all three backtesting implementations into one engine
2. ✅ Support both daily and intraday frequencies seamlessly
3. ✅ Provide optional risk controls (stop-loss, daily limits, slippage)
4. ✅ Compute comprehensive metrics (statistical, trading, benchmark)
5. ✅ Be simpler than the sum of current implementations
6. ✅ Be fully tested with >90% coverage
7. ✅ Be backward compatible with existing pipelines
8. ✅ Be well-documented with usage examples
9. ✅ Pass equivalence tests against old implementations
10. ✅ Enable easy extension for future features

---

## Example Usage (Target API)

```python
# Daily backtesting (simple)
from src.backtesting.engine import BacktestEngine

engine = BacktestEngine(
    transaction_cost=0.0015,  # 0.15% one-way
    initial_capital=100_000.0,
)

result = engine.run(
    predictions=y_pred,
    actual_returns=y_test,
    dates=dates,
)

print(f"Sharpe Ratio: {result.trading_metrics['sharpe']:.2f}")
print(f"Max Drawdown: {result.trading_metrics['max_drawdown']:.2%}")

# Intraday backtesting (advanced)
engine = BacktestEngine(
    transaction_cost=0.001,
    slippage=0.0005,
    stop_loss=0.02,
    daily_loss_limit=0.05,
    position_fraction=0.02,
    initial_capital=100_000.0,
)

result = engine.run(
    predictions=y_pred,
    actual_returns=y_test,
    dates=dates,
    opens=opens,  # Required for intraday
    closes=closes,  # Required for intraday
)

# Access detailed trade log
print(result.trade_log.head())

# Save results
from src.backtesting.results import save_backtest_results

save_backtest_results(result, output_dir="results/backtest/AAPL")
```

---

## Notes

- Prioritize **correctness** over performance
- Prioritize **simplicity** over extensibility (YAGNI principle)
- **Document all design decisions** with rationale
- **Include error handling** for edge cases
- **Add logging** at key decision points
- **Reference TRD sections** where applicable

---

**Your task:** Produce the two deliverable documents based on deep analysis of the existing code and this specification. Think carefully about design trade-offs and justify all decisions.
