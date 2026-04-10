# `REVIEW.md`

## 1. Executive Summary

The pipeline architecture for ClaudePaper is functional but suffers from significant redundancies, inconsistent abstraction layers, and critical "dead ends" in the training scripts. While the core feature engineering and model implementations (XGBoost, LSTM) are robust, the orchestration layer (the `pipelines/` directory) shows signs of "script sprawl"—where logic was copied and modified across multiple files instead of being consolidated into reusable components.

**Key Systemic Risks:**
*   **Redundant Training:** Both `evaluate.py` and `backtest.py` re-train models from scratch instead of loading persisted artifacts. This leads to massive compute waste and risks evaluating a different model than the one being backtested.
*   **Hard-coded Pipeline Break:** `run_lstm_baseline.py` is currently non-functional as a training script due to a hard-coded `sys.exit(0)` before the training loop.
*   **Data Leakage Risk:** Peer selection in `compute_cross_ticker_features` defaults to using the full dataset (including val/test) if peer tickers are not explicitly provided, creating look-ahead bias.
*   **Inefficient Feature Engineering:** Cross-ticker features like market breadth are re-calculated for every single ticker in the universe, despite being identical for all stocks.

**Overall Maintainability Score:** 6/10 (Requires consolidation of orchestration logic).

---

## 2. Critical Issues (Must Fix)

### Issue 1: Hard-coded Exit in Training Loop
*   **File Path:** `pipelines/run_lstm_baseline.py`
*   **Function / Class Name:** `run_train`
*   **Issue Type:** Broken Assumption / Dead Code
*   **Problem Description:** The training function prints debug info about the scaler and then calls `sys.exit(0)`, preventing the model from ever being trained.
*   **Why this is an issue:** It renders the baseline training pipeline completely unusable for its primary purpose.
*   **Evidence:** Lines 233 in `pipelines/run_lstm_baseline.py`: `sys.exit(0)` is called immediately after logging target metrics.
*   **Proposed Fix:** Remove the debug print section and the `sys.exit(0)` call to allow execution to proceed to the `build_windows` and `model.fit` calls.

### Issue 2: Data Leakage in Peer Selection
*   **File Path:** `src/features/cross_ticker.py`
*   **Function / Class Name:** `compute_cross_ticker_features`
*   **Issue Type:** Feature Leakage
*   **Problem Description:** If `peer_tickers` is None, the function calculates correlations on the `dfs` provided. In some contexts (like `FeaturePipeline` fit/transform), this might be safe, but the function itself warns that it uses the "FULL series which may include validation/test data."
*   **Why this is an issue:** Selecting peers based on their correlation over the entire dataset allows information from the future (test set) to influence which features are used during training.
*   **Evidence:** `src/features/cross_ticker.py`, Line 116: `logger.info(f"peer_tickers is None... Computing correlations on FULL series which may include validation/test data.")`
*   **Proposed Fix:** Ensure `FeaturePipeline` always passes only the training split to the correlation calculation or pre-select peers in a dedicated, split-aware stage.

### Issue 3: Massive Redundant Re-training
*   **File Path:** `pipelines/evaluate.py`, `pipelines/backtest.py`
*   **Function / Class Name:** `main`
*   **Issue Type:** Inefficiency / Inconsistency
*   **Problem Description:** Both scripts re-train the models they are supposed to evaluate/backtest.
*   **Why this is an issue:** 1) Extreme compute waste (training a model for every evaluation run). 2) Inconsistency: the model evaluated in `evaluate.py` might differ from the one used in `backtest.py` due to stochastic training processes (even with seeds, hardware/state differences can occur).
*   **Evidence:** `pipelines/evaluate.py`, Line 158: `y_pred_ipso, metrics_ipso = train_and_evaluate_model(...)` which calls `ipso_trainer.fit(...)`.
*   **Proposed Fix:** Update evaluation and backtesting scripts to load trained model weights (e.g., `.pth` or `.ubj` files) saved by the training scripts (`run_lstm_baseline.py`, `run_xgboost.py`).

---

## 3. Major Inefficiencies

### Issue 1: Redundant Cross-Ticker Computations
*   **File Path:** `src/features/cross_ticker.py`
*   **Function / Class Name:** `compute_cross_ticker_features`
*   **Issue Type:** Performance Inefficiency
*   **Problem Description:** `mkt_breadth` and `universe_mean_ret` are calculated by aggregating all tickers. This is done 50+ times (once for each target ticker) during feature building.
*   **Why this is an issue:** It increases feature engineering time linearly with the number of tickers, even though the result is the same for every stock in the aligned universe.
*   **Evidence:** `src/features/cross_ticker.py`, Lines 143-145: `all_returns = pd.DataFrame({...})` and `out["mkt_breadth"] = (all_returns > 0).mean(axis=1)`.
*   **Proposed Fix:** Compute market-level features once in `run_build_features.py` and pass them as a pre-computed block to the pipeline.

### Issue 2: Pseudo-Parallelism
*   **File Path:** `pipelines/run_build_features.py`
*   **Function / Class Name:** `main`
*   **Issue Type:** Performance Inefficiency
*   **Problem Description:** The script accepts an `--n-jobs` argument but processes tickers in a standard for-loop.
*   **Why this is an issue:** Feature engineering is a CPU-bound task that could be significantly accelerated using `joblib` or `multiprocessing`.
*   **Evidence:** `pipelines/run_build_features.py`, Lines 285-294: `for ticker in tickers_to_process: process_ticker(...)`. The `n_jobs` variable is logged but never utilized.
*   **Proposed Fix:** Implement `joblib.Parallel` or a similar pool to process tickers in parallel.

---

## 4. Minor Cleanups

### Issue 1: Unused Arguments
*   **File Path:** `pipelines/ingest_data.py`
*   **Issue Type:** Dead Code
*   **Problem Description:** The `--save-combined` argument is defined but has no implementation in the script.
*   **Proposed Fix:** Remove the argument or implement the logic to save the combined universe Parquet (which is currently redundantly handled in `run_build_features.py`).

### Issue 2: Redundant Helper Functions
*   **File Path:** `pipelines/utils_pipelines.py`
*   **Function Name:** `load_artefacts` vs `load_model`
*   **Issue Type:** Duplication
*   **Problem Description:** Both functions perform almost identical logic to load an XGBoost model and its metadata.
*   **Proposed Fix:** Consolidate into a single `load_xgb_model` function.

---

## 5. Data Flow & Integrity Risks

*   **Temporal Ordering:** While `FeaturePipeline` uses `shift(1)` correctly for lags, the merging logic in `run_build_features.py` (Lines 185-212) performs a `pd.concat` on individual ticker files. If any ticker has a misaligned index or missing timestamps not handled by `ingest_data.py`, this could introduce "holes" in the MultiIndex DataFrame.
*   **Train/Val Contamination:** In `pipelines/run_xgboost.py`, the `train_tune` function optionally retrains on `Train+Val`. If this model is then evaluated on the `Val` set in `run_val`, the metrics will be over-optimistic (leakage).
*   **Schema Mismatch:** `utils_pipelines.py` contains a `load_prices` function that generates "synthetic prices" if raw data is missing. This could lead to a pipeline silently running on fake data instead of failing fast, which is dangerous for financial applications.

---

## 6. Dead Code & Redundancy Map

| Component | Redundancy / Status |
| :--- | :--- |
| `ingest_data.py` | Redundant cleaning of SPY in every mode; `save-combined` is dead code. |
| `run_build_features.py` | Contains merging logic that overlaps with `ingest_data.py`. Parallelism is dead code. |
| `backtest.py` | Redundant training logic; duplicates `run_lstm_baseline.py` model creation. |
| `evaluate.py` | Redundant training logic; duplicates `run_lstm_baseline.py` and `run_xgboost.py`. |
| `utils_pipelines.py` | `load_artefacts` and `load_model` are duplicates. |
| `alpaca_ingestor.py` | `_load_bars` (static) and `load_bars` are duplicates. |

---

## 7. Recommended Architectural Improvements

1.  **Stateless Execution:** Move all training logic out of `evaluate.py` and `backtest.py`. These scripts should be purely for inference and reporting on existing artifacts.
2.  **Centralized Data Manager:** Consolidate the "Merge individual tickers into universe" logic into a single class (possibly `TickerAligner` or a new `UniverseManager`) to avoid the fragmentation between `ingest_data.py` and `run_build_features.py`.
3.  **Shared Pre-computations:** Create a `MarketFeatures` class to compute breadth and mean returns once per universe, rather than once per ticker.
4.  **Unified Backtesting:** Consolidate the manual PnL logic in `run_lstm_baseline.py` into the `src.evaluation.Backtester` class used by XGBoost to ensure metrics like Sharpe and Drawdown are calculated identically across all models.
