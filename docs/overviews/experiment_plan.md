# Experiment Plan
## PSO-LSTM Stock Price Prediction Study

**Document Version:** 1.0 | March 2026

---

## Table of Contents

1. [Research Questions](#1-research-questions)
2. [PSO Configuration](#2-pso-configuration)
3. [Hyperparameter Search Ranges](#3-hyperparameter-search-ranges)
4. [Evaluation Budget](#4-evaluation-budget)
5. [Baseline Definitions](#5-baseline-definitions)
6. [Metrics](#6-metrics)
7. [Experiment Variants](#7-experiment-variants)
8. [Reporting Plan](#8-reporting-plan)

---

## 1. Research Questions

This study is designed to answer three primary research questions:

| ID | Research Question |
|---|---|
| RQ1 | Does IPSO-tuned LSTM produce lower forecast error than baseline models (persistence, vanilla LSTM, XGBoost) under a fixed evaluation budget? |
| RQ2 | Which LSTM hyperparameters have the greatest impact on prediction accuracy and trading performance? |
| RQ3 | Does IPSO (tanh inertia + adaptive mutation) outperform standard PSO on this problem in terms of convergence speed and final gbest quality? |

---

## 2. PSO Configuration

### 2.1 IPSO Settings

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Swarm size | $M$ | 30 | Ji et al. (2021); Deng & Peng (2025) |
| Max iterations | $T$ | 50 | Budget constraint |
| Min inertia weight | $\omega_{\min}$ | 0.4 | Ji et al. (2021) |
| Max inertia weight | $\omega_{\max}$ | 0.9 | Ji et al. (2021) |
| Cognitive coefficient | $c_1$ | 1.5 | Ji et al. (2021); Deng & Peng (2025) |
| Social coefficient | $c_2$ | 1.5 | Ji et al. (2021); Deng & Peng (2025) |
| Velocity clamp fraction | — | 0.2 | Standard |
| Random seed | — | 42 | Reproducibility |

### 2.2 Standard PSO Settings (Baseline Optimizer)

Identical to IPSO except:
- **Linear** inertia weight decay (instead of tanh)
- **No adaptive mutation**

$$\omega_{\text{linear}}^t = 0.9 - \frac{(0.9 - 0.4) \cdot t}{50} = 0.9 - 0.01t$$

This allows a direct apples-to-apples comparison (RQ3).

---

## 3. Hyperparameter Search Ranges

### 3.1 LSTM Architecture Search Space

| Hyperparameter | Type | Range / Choices | Notes |
|---|---|---|---|
| `num_layers` | Integer | {1, 2, 3, 4} | Stacked LSTM depth |
| `hidden_units` | Integer | [32, 512] (steps of 32) | Units per layer |
| `dropout` | Float | [0.0, 0.5] | Applied inter-layer and pre-output |
| `learning_rate` | Float (log scale) | [$10^{-5}$, $10^{-1}$] | Adam learning rate |
| `lookback` | Categorical | {10, 30, 60, 120} | Minutes of history |

### 3.2 Encoding in Continuous Particle Space

| Dim | Continuous Range | Decoded Value |
|---|---|---|
| $x_1 \in [1.0, 4.99]$ | → | $\lfloor x_1 \rfloor \in \{1,2,3,4\}$ |
| $x_2 \in [32, 512]$ | → | Round to nearest 32 |
| $x_3 \in [0.0, 0.5]$ | → | Direct |
| $x_4 \in [\ln 10^{-5}, \ln 10^{-1}]$ | → | $\exp(x_4)$ |
| $x_5 \in [0.0, 3.99]$ | → | `[10,30,60,120][int(x_5)]` |

---

## 4. Evaluation Budget

### 4.1 Total Fitness Evaluations

$$\text{Budget} = M \times T = 30 \times 50 = 1{,}500 \text{ evaluations per ticker}$$

### 4.2 Estimated Runtime

| Component | Time per eval | Total (no parallel) | Total (8-parallel) |
|---|---|---|---|
| LSTM training (early stop ~30 epochs) | ~60s (GPU) | 25 hours | ~3 hours |
| Val evaluation + fitness | ~5s | 2 hours | ~15 min |
| **Per-ticker total** | — | **~27 hours** | **~3.5 hours** |
| **Full universe (51 tickers)** | — | — | **~180 hours** |

**Practical plan:** Focus primary experiments on 5 representative tickers (1 per sector) for the course project submission:
- `AAPL` (Technology)
- `JPM` (Financials)
- `JNJ` (Healthcare)
- `AMZN` (Consumer)
- `BA` (Industrials)

Full universe analysis is the "stretch goal" if compute allows.

### 4.3 Compute Resources

| Resource | Specification |
|---|---|
| CPU | Intel i7-12th gen (8 cores, 16 threads) |
| GPU | NVIDIA RTX 4060 (8GB VRAM) |
| RAM | 32 GB |
| Storage | 1 TB SSD |

---

## 5. Baseline Definitions

### 5.1 Persistence Model (Naive Baseline)

$$\hat{y}_{t+1} = y_t = r_t$$

- No training required
- Represents the "no skill" floor
- RMSE of persistence = realized return volatility

### 5.2 Vanilla LSTM (Manual Tuning Baseline)

Fixed hyperparameters based on prior literature defaults:

| Hyperparameter | Value | Reference |
|---|---|---|
| `num_layers` | 2 | Zeng et al. (2025); Ji et al. (2021) |
| `hidden_units` | 128 | Moderate capacity |
| `dropout` | 0.2 | Standard regularization |
| `learning_rate` | 0.001 | Adam default |
| `lookback` | 30 | Lanbouri & Achchab (2020) |

Training procedure identical to IPSO-LSTM (same epochs, batch size, early stopping).

### 5.3 XGBoost Regressor (Non-Deep-Learning Baseline)

```python
xgb.XGBRegressor(
    n_estimators    = 500,
    max_depth       = 6,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    min_child_weight= 1,
    reg_alpha       = 0.1,   # L1
    reg_lambda      = 1.0,   # L2
    tree_method     = 'hist',
    random_state    = 42
)
```

Input: **Flattened** feature vector (no temporal structure; all $T \times F$ features concatenated into a single vector of length $T_{\max} \times F$). $T_{\max} = 30$ (fixed lookback for XGBoost).

### 5.4 Standard PSO-LSTM (Ablation Baseline)

Same as IPSO-LSTM but with:
- Linear inertia weight decay
- No adaptive mutation

This directly quantifies the contribution of the IPSO improvements to RQ3.

---

## 6. Metrics

### 6.1 Statistical / Prediction Metrics

| Metric | Symbol | Formula | Objective |
|---|---|---|---|
| Root Mean Squared Error | RMSE | $\sqrt{\frac{1}{N}\sum(\hat{y}-y)^2}$ | Minimize |
| Mean Absolute Error | MAE | $\frac{1}{N}\sum|\hat{y}-y|$ | Minimize |
| Directional Accuracy | DA | $\frac{|\{i: \text{sign}(\hat{y}_i) = \text{sign}(y_i)\}|}{N}$ | Maximize |
| F1 Score (Ternary) | F1 | $2 \cdot \frac{P \cdot R}{P + R}$ | Maximize |
| R-Squared | $R^2$ | $1 - \frac{\sum(\hat{y}-y)^2}{\sum(y-\bar{y})^2}$ | Maximize |

**Ternary classification threshold:** Returns with $|r| < 10^{-4}$ are classified as "flat" (class 0), $r > 10^{-4}$ as "up" (class +1), and $r < -10^{-4}$ as "down" (class -1).

### 6.2 Trading / Financial Metrics

| Metric | Symbol | Formula | Objective |
|---|---|---|---|
| Annualized Sharpe Ratio | SR | $\frac{E[R_p - R_f]}{\sigma(R_p - R_f)} \cdot \sqrt{252 \times 390}$ | Maximize |
| Maximum Drawdown | MDD | $\max_{s \leq t}\frac{V_s - V_t}{V_s}$ | Minimize |
| CAGR | — | $\left(\frac{V_T}{V_0}\right)^{1/n_{\text{years}}} - 1$ | Maximize |
| Profit Factor | PF | $\frac{\sum \text{gross profits}}{\sum \text{gross losses}}$ | Maximize |
| Win Rate | WR | $\frac{\text{winning trades}}{\text{total trades}}$ | Maximize |
| Number of Trades | — | Count of signal changes | — |

### 6.3 PSO Convergence Metrics

| Metric | Description |
|---|---|
| Fitness curve | gbest fitness vs. iteration number |
| Swarm diversity | $D_t = \frac{1}{M}\sum_i \|\mathbf{x}_i^t - \bar{\mathbf{x}}^t\|_2$ |
| Convergence iteration | First iteration where gbest fitness improves < 0.001 for 5 consecutive steps |
| Hyperparameter distribution | Scatter matrix of gbest hyperparameters across 5 ticker runs |

---

## 7. Experiment Variants

### Experiment 1 — Main Comparison (RQ1)

**Goal:** Compare IPSO-LSTM vs. all baselines on 5 tickers.

**Protocol:**
1. Run IPSO (30 particles, 50 iterations) on validation set
2. Retrain IPSO-LSTM with gbest params on Train+Val
3. Evaluate all models on held-out Test set
4. Report all statistical and trading metrics

**Expected output:** Table 1 (statistical metrics) and Table 2 (trading metrics) for all 5 tickers × 4 models.

### Experiment 2 — IPSO vs. Standard PSO (RQ3)

**Goal:** Isolate the contribution of tanh inertia + mutation.

**Protocol:**
1. Run Standard PSO (linear inertia, no mutation) with same budget
2. Run IPSO with same budget
3. Compare: fitness curves, final gbest, and test-set performance

**Expected output:** Figure 1 (convergence curves) and Table 3 (final metrics).

### Experiment 3 — Lookback Window Sensitivity (RQ2)

**Goal:** Understand how lookback window size affects performance.

**Protocol:**
1. Fix all other hyperparameters to gbest values from Experiment 1
2. Sweep `lookback` ∈ {10, 30, 60, 120} and retrain
3. Compare test-set RMSE and Sharpe across windows

**Expected output:** Figure 2 (RMSE/Sharpe vs. lookback).

### Experiment 4 — Feature Ablation (RQ2)

**Goal:** Understand feature category contributions.

**Protocol:**
1. Full feature set (baseline)
2. Remove cross-ticker features
3. Remove volume features
4. Remove statistical moment features
5. Use only OHLCV (no engineered features)

For each variant, run IPSO and evaluate on test set.

**Expected output:** Table 4 (ablation results by feature category).

### Experiment 5 — Walk-Forward Validation

**Goal:** Ensure performance is stable across different market regimes.

**Protocol:**
1. Use expanding window (3-month fold, 1-month step)
2. Retrain IPSO-LSTM at each fold boundary
3. Aggregate metrics across all folds (mean ± std)

**Expected output:** Figure 3 (rolling Sharpe over time) and Table 5 (fold statistics).

---

## 8. Reporting Plan

### 8.1 Figures

| Figure | Description |
|---|---|
| Fig 1 | IPSO vs. PSO convergence curves (gbest fitness vs. iteration) |
| Fig 2 | RMSE and Sharpe vs. lookback window (Experiment 3) |
| Fig 3 | Equity curves for all models on AAPL test set |
| Fig 4 | Swarm diversity over iterations (IPSO vs. PSO) |
| Fig 5 | Predicted vs. actual log returns scatter plot |
| Fig 6 | Walk-forward rolling Sharpe ratio |
| Fig 7 | Feature importance bar chart (XGBoost) |
| Fig 8 | Hyperparameter distribution across tickers (gbest values) |

### 8.2 Tables

| Table | Description |
|---|---|
| Table 1 | Statistical metrics: RMSE, DA, F1, R² — all models × 5 tickers |
| Table 2 | Trading metrics: Sharpe, MDD, CAGR — all models × 5 tickers |
| Table 3 | IPSO vs. PSO convergence comparison |
| Table 4 | Feature ablation results |
| Table 5 | Walk-forward validation statistics |
| Table 6 | IPSO gbest hyperparameters by ticker |

### 8.3 Statistical Significance

To account for the stochastic nature of PSO, run IPSO 5 times with different seeds (42, 123, 456, 789, 1024) per ticker. Report:
- Mean and standard deviation of test-set RMSE and Sharpe across 5 runs
- Best-run results (to compare with literature)

### 8.4 Expected Results (Informed Hypotheses)

Based on prior literature (Ji et al., 2021; Zeng et al., 2025; Deng & Peng, 2025):

| Hypothesis | Based on |
|---|---|
| IPSO-LSTM will achieve 10–30% lower RMSE than vanilla LSTM | Ji et al. (2021): 20.54% improvement |
| IPSO-LSTM will achieve higher Sharpe than persistence | Lanbouri & Achchab (2020): LSTM profitable on HFT |
| XGBoost will match or beat vanilla LSTM but lag IPSO-LSTM | Zeng et al. (2025): PSO-LSTM > XGB |
| Lookback = 30–60 minutes will be optimal | Lanbouri (2020) suggests short windows better for HFT |
| tanh inertia provides 5–15% fitness improvement over linear | Ji et al. (2021) ablation |