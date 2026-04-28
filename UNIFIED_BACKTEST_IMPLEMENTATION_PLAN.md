# Unified Backtesting Engine Implementation Plan

**Version:** 1.0.0  
**Status:** Draft  
**Author:** System Architect  
**Date:** 2026-04-28  
**Estimated Duration:** 4-5 weeks  
**Team Size:** 1-2 engineers  

---

## Overview

This plan consolidates `backtest.py`, `backtester.py`, `backtest_results.py`, and `run_backtest.py` into a single unified backtesting engine. The implementation proceeds in six phases, each with defined deliverables, tests, and success criteria.

---

## Phase 1: Core Engine & Strategies (Week 1)

### Goal
Implement `BacktestEngine`, `BacktestConfig`, `DailyStrategy`, and `IntradayStrategy` with 100% equivalence to existing implementations.

### File Changes

| File | Action | Lines | Owner |
|------|--------|-------|-------|
| `src/backtesting/engine.py` | **Create** | ~200 | Backend |
| `src/backtesting/strategies/base.py` | **Create** | ~20 | Backend |
| `src/backtesting/strategies/daily.py` | **Create** | ~80 | Backend |
| `src/backtesting/strategies/intraday.py` | **Create** | ~250 | Backend |
| `src/backtesting/components/signal_generator.py` | **Create** | ~30 | Backend |
| `src/backtesting/components/cost_model.py` | **Create** | ~50 | Backend |
| `src/backtesting/components/risk_manager.py` | **Create** | ~60 | Backend |
| `tests/backtesting/test_equivalence.py` | **Create** | ~150 | QA |

### Detailed Tasks

#### 1.1 BacktestConfig & Validation
```python
# src/backtesting/engine.py
@dataclass(frozen=True)
class BacktestConfig:
    ...
```
- Implement frozen dataclass with `__post_init__` validation.
- Support all parameters from `config/default_config.yaml` §backtesting.

#### 1.2 DailyStrategy (Vectorized)
- Port logic from `CanonicalBacktest.run_backtest()` and `apply_transaction_costs()`.
- Must produce **bit-identical** equity curves and costs.
- Use NumPy vectorized operations only.

#### 1.3 IntradayStrategy (Event-Driven)
- Port logic from `Backtester.run()` and helper methods (`_update_position`, `_close_position`, `_open_position`, `_mark_to_market`, `_apply_risk_controls`).
- Preserve session boundary detection (explicit `session_starts` + time fallback).
- Preserve slippage asymmetry: `close_slip = fill_price * (1.0 - self.slip * np.sign(position))`.
- Preserve trade log format.

#### 1.4 SignalGenerator
- Extract `generate_signals()` from `evaluation/metrics.py`.
- Ensure threshold logic matches existing implementations.

#### 1.5 CostModel
- Unify transaction cost logic from both old implementations.
- Daily: costs on position changes only.
- Intraday: costs + slippage on entry/exit.

#### 1.6 RiskManager
- Extract stop-loss and daily loss limit from `backtester.py`.
- Make optional (disabled when config fields are `None`).

### Testing (Phase 1)

| Test | Type | Criteria |
|------|------|----------|
| `test_daily_equivalence` | Equivalence | New daily equity curve == old `CanonicalBacktest` within `rtol=1e-10` |
| `test_intraday_equivalence` | Equivalence | New intraday equity curve == old `Backtester` within `rtol=1e-10` |
| `test_trade_log_equivalence` | Equivalence | New trade log DataFrame equals old trade log |
| `test_config_validation` | Unit | Invalid configs raise `ValueError` |
| `test_frequency_heuristic` | Unit | Auto-detection works for 1-min, daily, and raises on mixed |

### Success Criteria
- [ ] All equivalence tests pass.
- [ ] Code coverage > 90% for new engine files.
- [ ] No regressions in existing test suite.

### Risk: Intraday Logic Drift
**Mitigation:** Copy `backtester.py` methods verbatim into `IntradayStrategy`, then refactor incrementally. Run equivalence test after every method extraction.

---

## Phase 2: Results Container & Persistence (Week 1-2)

### Goal
Refactor `BacktestResult` into an immutable, versioned dataclass with backward-compatible persistence.

### File Changes

| File | Action | Lines | Owner |
|------|--------|-------|-------|
| `src/backtesting/results.py` | **Rewrite** | ~300 | Backend |
| `src/backtesting/components/metrics.py` | **Create** | ~200 | Backend |
| `tests/backtesting/test_results.py` | **Create** | ~100 | QA |

### Detailed Tasks

#### 2.1 BacktestResult Dataclass
- Immutable (`frozen=True`).
- Alignment invariant in `__post_init__`.
- `engine_version` field for future migrations.

#### 2.2 MetricsCalculator
- Extract metric functions from `evaluation/metrics.py` and `backtest.py`.
- Single source of truth for:
  - Statistical: MSE, MAE, RMSE, R², DA, MAPE
  - Trading: Sharpe, Sortino, CAGR, MDD, Calmar, Profit Factor, Win Rate
  - Benchmark: Total return, Sharpe, MDD

#### 2.3 ResultPersistence
- `save_backtest_results()`: JSON (metrics), CSV (equity curve), YAML (metadata), Markdown (report).
- `load_backtest_results()`: Auto-detect version, migrate if needed.
- Handle array length alignment (existing logic from `backtest_results.py`).

#### 2.4 Migration Layer
```python
def _migrate_v0_to_v1(data: dict) -> dict: ...
```
- Map old flat structure to new nested structure.

### Testing (Phase 2)

| Test | Type | Criteria |
|------|------|----------|
| `test_result_alignment` | Unit | Mismatched array lengths raise `ValueError` |
| `test_save_load_roundtrip` | Integration | Save + load produces identical `BacktestResult` |
| `test_migration_v0_to_v1` | Unit | Old JSON loads correctly into new dataclass |
| `test_metrics_calculator` | Unit | All metrics computed correctly on synthetic data |

### Success Criteria
- [ ] Old saved results load without errors.
- [ ] New results save in all four formats.
- [ ] Markdown report generates correctly.

### Risk: Breaking Old Results
**Mitigation:** Keep old `BacktestResults` dataclass in `_legacy/` until Phase 6. Run migration tests on sample data from production.

---

## Phase 3: CLI Refactor (Week 2)

### Goal
Simplify `run_backtest.py` into a thin orchestration layer (~150 lines).

### File Changes

| File | Action | Lines | Owner |
|------|--------|-------|-------|
| `pipelines/run_backtest.py` | **Rewrite** | ~150 | Backend |
| `src/evaluation/model_loader.py` | **Modify** | ~+20 | Backend |

### Detailed Tasks

#### 3.1 Remove Duplication
- Delete `backtest_baseline_lstm()` function.
- Delete commented-out code (lines 248-254, 269).
- Single `engine.run()` call for all model types.

#### 3.2 Unified Data Loading
```python
def load_test_data(data_path: Path) -> dict:
    # Load X_test, y_test, dates
    # Optionally load opens, closes, session_starts
    ...
```

#### 3.3 Model Adapter Integration
- Ensure `load_model()` returns an object with `.predict()` method.
- No model-type branching in CLI.

#### 3.4 Output Directory Structure
```
results/experiments/
└── {ticker}_{timeframe}_{model_type}_{run_id}/
    ├── backtest/
    │   ├── backtest_results.json
    │   ├── equity_curve.csv
    │   ├── predictions.csv
    │   ├── metadata.yaml
    │   └── backtest_report.md
    └── plots/
        └── (generated by evaluation.plotting)
```

### Testing (Phase 3)

| Test | Type | Criteria |
|------|------|----------|
| `test_cli_daily` | Integration | CLI runs successfully on daily data |
| `test_cli_intraday` | Integration | CLI runs successfully on intraday data |
| `test_cli_all_model_types` | Integration | PSO-LSTM, Baseline LSTM, XGBoost all produce results |

### Success Criteria
- [ ] CLI executes without errors for all three model types.
- [ ] Output files match expected structure.
- [ ] Log files contain expected summary metrics.

### Risk: CLI Argument Changes
**Mitigation:** Preserve all existing CLI arguments. Add `--frequency` as optional. Update README with new usage examples.

---

## Phase 4: Comprehensive Testing (Week 3)

### Goal
Achieve >90% test coverage and validate equivalence against old implementations on real data.

### File Changes

| File | Action | Lines | Owner |
|------|--------|-------|-------|
| `tests/backtesting/test_engine.py` | **Create** | ~300 | QA |
| `tests/backtesting/test_strategies.py` | **Create** | ~200 | QA |
| `tests/backtesting/test_edge_cases.py` | **Create** | ~150 | QA |
| `tests/backtesting/test_integration.py` | **Create** | ~100 | QA |

### Test Categories

#### 4.1 Unit Tests (Engine)
- Config validation (negative costs, invalid stop-loss)
- Frequency detection (daily, intraday, mixed, irregular)
- Signal generation (threshold variations)
- Empty inputs, single sample, all-neutral signals

#### 4.2 Unit Tests (Strategies)
- Daily: cost application, cumulative returns, benchmark
- Intraday: session open/close, position sizing, slippage direction
- Intraday: stop-loss trigger, daily halt reset

#### 4.3 Edge Cases
```python
# All same predictions
predictions = np.full(100, 0.5)
# All zero predictions
predictions = np.zeros(100)
# Single bar
predictions = np.array([0.01])
# NaN in inputs (should raise)
# Inf in inputs (should raise)
# Length mismatch (should raise)
```

#### 4.4 Integration Tests
- End-to-end with synthetic data (deterministic).
- End-to-end with real AAPL 1-min data.
- End-to-end with real AAPL daily data.

#### 4.5 Equivalence Tests
- Run old and new engines on identical inputs.
- Compare every field in `BacktestResult` to old output.
- Tolerance: `rtol=1e-10` for floats, exact match for DataFrames.

### Success Criteria
- [ ] Test coverage > 90% for `src/backtesting/`.
- [ ] All equivalence tests pass on real data.
- [ ] No flaky tests (run 100x, 100% pass rate).

### Risk: Equivalence Test Failures
**Mitigation:** If differences found, determine if they are:
1. **Bugs in new code**: Fix immediately.
2. **Bugs in old code**: Document and accept new behavior as correct.
3. **Floating point drift**: Increase tolerance if justified.

---

## Phase 5: Documentation (Week 3-4)

### Goal
Update all documentation to reflect the unified engine.

### Deliverables

| Document | Action | Owner |
|----------|--------|-------|
| `README.md` | Update backtesting section | Tech Writer |
| `docs/BACKTESTING.md` | **Create** usage guide | Tech Writer |
| `docs/TRD3.md` | Update evaluation spec | Architect |
| `UNIFIED_BACKTEST_DESIGN.md` | Finalize (this doc) | Architect |
| `MIGRATION_GUIDE.md` | **Create** for existing users | Tech Writer |

### Documentation Contents

#### 5.1 Usage Guide
```python
# Daily backtesting
from src.backtesting.engine import BacktestEngine, BacktestConfig

engine = BacktestEngine(BacktestConfig(
    transaction_cost=0.0015,
    initial_capital=100_000.0,
))
result = engine.run(predictions, actual_returns, dates)

# Intraday backtesting
engine = BacktestEngine(BacktestConfig(
    transaction_cost=0.001,
    slippage=0.0005,
    stop_loss=0.02,
    daily_loss_limit=0.05,
    position_fraction=0.02,
))
result = engine.run(predictions, actual_returns, dates, opens, closes)
```

#### 5.2 Migration Guide
- How to replace `CanonicalBacktest` with `BacktestEngine`.
- How to replace `Backtester` with `BacktestEngine`.
- CLI argument changes (minimal).
- Result format changes (versioned, auto-migrated).

#### 5.3 TRD3 Updates
- Reference new `BacktestEngine` API.
- Update metric definitions to use `MetricsCalculator`.
- Add frequency detection specification.

### Success Criteria
- [ ] All docs reviewed by at least one other engineer.
- [ ] Usage examples are copy-paste runnable.
- [ ] Migration guide covers all existing use cases.

---

## Phase 6: Cleanup & Deprecation (Week 4-5)

### Goal
Remove old implementations after a grace period.

### File Changes

| File | Action | Owner |
|------|--------|-------|
| `src/backtesting/backtest.py` | **Move** to `_legacy/` + deprecation warning | Backend |
| `src/backtesting/backtester.py` | **Move** to `_legacy/` + deprecation warning | Backend |
| `src/backtesting/backtest_results.py` | **Move** to `_legacy/` + deprecation warning | Backend |
| `src/backtesting/__init__.py` | Update exports | Backend |

### Deprecation Warnings

```python
# src/backtesting/_legacy/backtest.py
import warnings

class CanonicalBacktest:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "CanonicalBacktest is deprecated and will be removed in v2.0. "
            "Use BacktestEngine from src.backtesting.engine instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # ... existing init
```

### Cleanup Checklist
- [ ] Old files moved to `_legacy/`.
- [ ] Deprecation warnings added to all public classes/functions.
- [ ] Internal imports updated to use new engine.
- [ ] No references to old classes in `pipelines/` or `src/` (except `_legacy/`).
- [ ] CI passes with deprecation warnings treated as non-fatal.

### Final Verification
- [ ] Run full pipeline on AAPL daily data.
- [ ] Run full pipeline on AAPL 1-min data.
- [ ] Compare outputs to pre-cleanup baseline.
- [ ] Tag release `v1.5.0` (unified engine) / `v2.0.0` (old code removed).

---

## Timeline Summary

| Phase | Duration | Start | End | Key Deliverable |
|-------|----------|-------|-----|-----------------|
| 1: Core Engine | 1 week | Day 1 | Day 7 | `BacktestEngine` + equivalence tests |
| 2: Results | 3-4 days | Day 6 | Day 10 | `BacktestResult` + persistence |
| 3: CLI Refactor | 2-3 days | Day 10 | Day 13 | Clean `run_backtest.py` |
| 4: Testing | 1 week | Day 13 | Day 20 | >90% coverage, all equivalence pass |
| 5: Documentation | 3-4 days | Day 18 | Day 23 | Updated docs + migration guide |
| 6: Cleanup | 3-4 days | Day 23 | Day 27 | Old code deprecated/removed |

**Total: 4-5 weeks** (with 1 engineer)  
**Total: 3 weeks** (with 2 engineers, phases 1-3 parallelized)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Intraday equivalence drift | Medium | High | Port `backtester.py` verbatim first, refactor later. Pin behavior with tests. |
| Breaking existing pipelines | Low | High | Preserve old APIs with deprecation warnings. Run full integration tests before merge. |
| Performance regression (intraday) | Low | Medium | Event-driven loop is inherently sequential. Accept for correctness. Optimize hot paths later if needed. |
| Config schema mismatch | Medium | Medium | Derive `BacktestConfig` directly from `config/default_config.yaml` schema. Validate at startup. |
| Team bandwidth | High | Medium | Phase 1 is highest priority. If time-constrained, stop after Phase 3 and defer cleanup. |

---

## Story Points Estimate (Agile)

| Phase | Stories | Points |
|-------|---------|--------|
| 1.1 Config & Engine scaffold | 1 | 3 |
| 1.2 DailyStrategy | 1 | 5 |
| 1.3 IntradayStrategy | 1 | 8 |
| 1.4 Components (signal, cost, risk) | 1 | 5 |
| 1.5 Equivalence tests | 1 | 5 |
| 2.1 BacktestResult refactor | 1 | 5 |
| 2.2 MetricsCalculator | 1 | 5 |
| 2.3 Persistence + migration | 1 | 3 |
| 3.1 CLI refactor | 1 | 5 |
| 3.2 Integration tests | 1 | 3 |
| 4.1 Edge case tests | 1 | 3 |
| 4.2 Real data validation | 1 | 5 |
| 5.1 Documentation | 1 | 3 |
| 6.1 Deprecation + cleanup | 1 | 3 |
| **Total** | **14** | **61 points** |

At 15 points/week per engineer = **4 weeks** for 1 engineer, **2 weeks** for 2 engineers.

---

## Success Criteria Checklist

- [ ] **Functional:** Unified engine supports daily and intraday seamlessly.
- [ ] **Metrics:** Computes all statistical, trading, and benchmark metrics.
- [ ] **Equivalence:** Produces identical results to old implementations.
- [ ] **Quality:** >90% test coverage, type hints, docstrings.
- [ ] **Compatibility:** Old pipelines work with deprecation warnings.
- [ ] **Documentation:** Usage guide and migration guide complete.
- [ ] **Performance:** No significant regression (<5% slower).
- [ ] **Clean:** Old code deprecated or removed.
