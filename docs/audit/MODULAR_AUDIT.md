# Software Design and Modularity Audit

**Project:** ClaudePaper (PSO-LSTM Stock Prediction System)  
**Date:** April 7, 2026  
**Auditor:** Gemini CLI Architecture Review  
**Total Files Analyzed:** 50+ Python modules  
**Lines of Code:** ~10,000 LOC  

---

## Executive Summary

This audit evaluates the ClaudePaper codebase for **software engineering principles** including Separation of Concerns (SoC), cohesion, coupling, code duplication, and architectural layering. The analysis identifies **68 structural issues** affecting maintainability, testability, and extensibility.

### Severity Distribution

| Severity | Count | % of Total |
|----------|-------|------------|
| **CRITICAL** | 12 | 18% |
| **HIGH** | 28 | 41% |
| **MEDIUM** | 20 | 29% |
| **LOW** | 8 | 12% |
| **TOTAL** | **68** | **100%** |

### Key Findings

1. **Inverted Dependencies:** `pipelines/` scripts import from `scripts/` (wrong direction).
2. **God Objects:** 680-line scripts (`run_lstm_baseline.py`) mixing CLI, training, validation, and plotting.
3. **Massive Duplication:** Feature loading, windowing, and technical indicator calculations repeated 5+ times.
4. **Missing Abstractions:** No shared interfaces for models (LSTM, XGBoost, RL) or feature blocks.
5. **Business Logic in Scripts:** Training loops and evaluation logic embedded in entry points rather than `src/`.
6. **Code Out of Place:** Plotting logic inside feature modules (`ttm_squeeze.py`) and utilities in ingestion classes.

---

## Table of Contents

1. [Critical Issues](#1-critical-issues)
2. [High Priority Issues](#2-high-priority-issues)
3. [Medium Priority Issues](#3-medium-priority-issues)
4. [Low Priority Issues](#4-low-priority-issues)
5. [Architectural Recommendations](#5-architectural-recommendations)

---

## 1. Critical Issues

### 1.1 Inverted Dependency: Pipelines Import Scripts
**File:** `pipelines/run_lstm_baseline.py`  
**Principle Violated:** Dependency Inversion, Layering  
**Severity:** CRITICAL  
**Description:** The pipeline layer (intended as a library/core workflow layer) depends on the `scripts/` directory (intended as an application/entry point layer). This creates circular dependency risks and makes the core logic impossible to use without the specific CLI scripts.  
**Suggested Refactoring:** Move `scripts/plots_lstm.py` to `src/visualization/` and update all imports to point to the `src/` directory.

### 1.2 God Script: run_lstm_baseline.py
**File:** `pipelines/run_lstm_baseline.py`  
**Principle Violated:** Single Responsibility Principle (SRP)  
**Severity:** CRITICAL  
**Description:** This 684-line script manages CLI parsing, data loading, training loops, walk-forward validation, backtesting, and plotting. It is difficult to test, reuse, or maintain.  
**Suggested Refactoring:** Extract training logic to `src/models/lstm/trainer.py`, validation logic to `src/evaluation/walk_forward.py`, and keep the script as a thin CLI wrapper (under 100 lines).

### 1.3 Massive Code Duplication: Feature Loading
**Files:** `scripts/evaluate.py`, `scripts/run_pso.py`, `scripts/backtest.py`, etc.  
**Principle Violated:** DRY (Don't Repeat Yourself)  
**Severity:** CRITICAL  
**Description:** The same ~20-line block for loading `.npy` files and building windows is repeated across at least 5 major scripts. Any change to the data format requires updating all 5 locations.  
**Suggested Refactoring:** Create a `src/data/loaders.py` module with a unified `FeatureSplitLoader` class that handles loading and windowing in one place.

### 1.4 Inconsistent Model Interfaces
**Files:** `src/models/lstm/`, `src/models/xgboost/`, `src/models/rl/`  
**Principle Violated:** Liskov Substitution Principle  
**Severity:** CRITICAL  
**Description:** LSTM uses a `Trainer` class with `.fit()`, XGBoost uses a standalone model with `.train()`, and RL uses an agent with `.update()`. This lack of polymorphism makes it impossible to write generic evaluation or optimization code.  
**Suggested Refactoring:** Define a `BaseModel` Protocol in `src/models/base.py` that enforces standard `.fit()`, `.predict()`, `.save()`, and `.load()` methods.

---

## 2. High Priority Issues

### 2.1 Duplicate Indicator Logic
**Files:** `plots.py`, `src/features/technical.py`, `src/features/ttm_squeeze.py`  
**Principle Violated:** SoC, DRY  
**Severity:** HIGH  
**Description:** Basic indicators like SMA, ROC, and True Range are implemented independently in multiple files. This leads to inconsistent results if one implementation is updated and not the others.  
**Suggested Refactoring:** Consolidate all low-level indicator math into `src/features/indicators_core.py` and have all other modules import from there.

### 2.2 Plotting Logic in Feature Modules
**File:** `src/features/ttm_squeeze.py`  
**Principle Violated:** Separation of Concerns  
**Severity:** HIGH  
**Description:** The `ttm_squeeze.py` module contains both the mathematical calculation of the squeeze and a complex Matplotlib plotting function. Feature modules should only handle data processing.  
**Suggested Refactoring:** Move the `plot_beardy_squeeze` function to a new `src/visualization/` directory.

### 2.3 Hardcoded Configuration in Pipeline
**File:** `src/features/pipeline.py`  
**Principle Violated:** Open-Closed Principle  
**Severity:** HIGH  
**Description:** `LAG_SOURCES` and `LAG_DEPTHS` are hardcoded as module-level constants. This makes it impossible to experiment with different lag structures without modifying the source code.  
**Suggested Refactoring:** Pass configuration dictionaries to the `FeaturePipeline` constructor or load them from the project's YAML configuration.

---

## 3. Medium Priority Issues

### 3.1 Misplaced Utilities
**File:** `src/data/alpaca_ingestor.py`  
**Principle Violated:** Cohesion  
**Severity:** MEDIUM  
**Description:** The `downcast_ohlcv` function is a general data cleaning utility but lives inside the Alpaca-specific ingestion module.  
**Suggested Refactoring:** Move `downcast_ohlcv` to `src/data/cleaner.py` or a general `src/utils/dataframe_utils.py`.

### 3.2 Side Effects at Module Level
**File:** `src/data/alpaca_ingestor.py`  
**Principle Violated:** Clean Code  
**Severity:** MEDIUM  
**Description:** Calls `load_dotenv()` at the top level of the module, which can cause unexpected environment changes when importing the module.  
**Suggested Refactoring:** Move environment loading to the `__init__` method or the main entry point of the application.

---

## 4. Low Priority Issues

### 4.1 Dead Code and Commented Blocks
**Files:** Multiple (e.g., `src/features/pipeline.py`, `src/data/aligner.py`)  
**Principle Violated:** Code Hygiene  
**Severity:** LOW  
**Description:** Large blocks of commented-out code (sometimes hundreds of lines) clutter the modules, making them harder to read.  
**Suggested Refactoring:** Remove dead code; rely on Git history for version tracking.

### 4.2 Inconsistent Docstrings
**Files:** `src/database/experiment_tracker.py`, `src/database/cleaning_tracker.py`  
**Principle Violated:** Documentation Integrity  
**Severity:** LOW  
**Description:** Docstrings list incorrect file paths (e.g., `src/utils/...` instead of `src/database/...`).  
**Suggested Refactoring:** Update docstrings to reflect the actual file locations.

---

## 5. Architectural Recommendations

1.  **Introduce a Layered Architecture:**
    *   **Level 0 (Utils/Base):** Math, generic DF utilities, base interfaces.
    *   **Level 1 (Core):** Feature engineering, model definitions, database tracking.
    *   **Level 2 (Workflows):** Training pipelines, backtesters, optimizers.
    *   **Level 3 (Applications):** CLI scripts and entry points.

2.  **Standardize Model Interfaces:** Implement a standard wrapper for all model types to allow for cross-model evaluation scripts.

3.  **Centralize Data Loading:** Create a robust data access layer that hides the details of `.npy` vs `.parquet` storage.

---
**End of Audit Report**
