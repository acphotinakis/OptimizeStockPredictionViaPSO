# Literature Review
## PSO-LSTM Hybrid Models for Stock Price Prediction

**Document Version:** 1.0 | March 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Paper Summaries](#2-paper-summaries)
3. [Comparison of Methods](#3-comparison-of-methods)
4. [Identified Research Gaps](#4-identified-research-gaps)
5. [How This Work Improves on Prior Art](#5-how-this-work-improves-on-prior-art)
6. [References](#6-references)

---

## 1. Overview

This review synthesizes four key papers provided for this project, spanning the themes of LSTM for high-frequency stock prediction, PSO-LSTM hybrid optimization, improved PSO algorithms, and hybrid LSTM-PSO models for stock index forecasting. Together, these works establish the theoretical and empirical foundation for our IPSO-LSTM system.

---

## 2. Paper Summaries

### Paper 1: Ji, Liew & Yang (2021) — *A Novel Improved Particle Swarm Optimization with Long-Short Term Memory Hybrid Model for Stock Indices Forecast*

**Venue:** IEEE Access, DOI: 10.1109/ACCESS.2021.3056713

**Problem:** Standard PSO for LSTM hyperparameter optimization suffers from premature convergence to local optima, limiting the quality of the final model.

**Approach:**

The authors propose IPSO-LSTM, which improves the standard PSO algorithm in two complementary ways:

1. **Non-linear inertia weight (tanh decay):**

$$\omega_i^t = \omega_{\max} - (\omega_{\max} - \omega_{\min}) \cdot \tanh\!\left(\frac{4t}{T_{\max}}\right)$$

Each particle maintains its own inertia weight, creating diversity in the swarm's search behavior. The tanh function maintains large inertia (global exploration) early and transitions rapidly to small inertia (local exploitation) in later iterations — superior to linear decay.

2. **Adaptive mutation factor:**

$$\mu_{\text{mf}} = 0.3 \cdot \frac{t}{T_{\max}} + 0.7$$

When a uniform random draw $\xi > \mu_{\text{mf}}$, the particle's position is randomly reinitialized. This escape mechanism decreases in probability as training progresses, balancing exploration and exploitation dynamically.

**Particles** encode: number of LSTM training iterations, number of nodes in two hidden layers, and learning rate (4-dimensional).

**Data:** S&P/ASX200 daily closing prices (2010–2020); also validated on DJI, IXIC, HSI, and N225.

**Key Results:**
- IPSO-LSTM outperforms SVR, LSTM, and PSO-LSTM on all four metrics (RMSE, MAE, MAPE, R²)
- Optimal lookback period: 20 trading days
- R² values for IPSO-LSTM: [0.9596, 0.9760]
- Improvement over PSO-LSTM: MAPE -29.90%, RMSE -20.54%, MAE -30.71%

**Strengths:** Rigorous ablation comparing linear vs. tanh inertia; multi-market validation demonstrates generalizability.

**Limitations:** Uses only daily closing prices (no intraday features); does not incorporate a trading/backtest evaluation; particle encoding limited to 4 dimensions.

---

### Paper 2: Deng & Peng (2025) — *Application of LSTM Model based on Particle Swarm Optimization Algorithm in Stock Market Trend Prediction*

**Venue:** DEAI 2025 (ACM), DOI: 10.1145/3745238.3745261

**Problem:** LSTM hyperparameters (hidden neurons, dropout, batch size) are typically set by human intuition, introducing subjectivity and suboptimality.

**Approach:**

Standard PSO is applied to optimize three LSTM hyperparameters: number of hidden layer neurons, dropout rate, and batch size. The fitness function combines MSE and the sum of squared weights (MSW):

$$f(x) = \gamma \cdot \text{MSE} + (1-\gamma) \cdot \text{MSW}, \quad \gamma = 0.9$$

The MSW term penalizes large weights, acting as an L2 regularizer within the PSO objective. This enhances generalization beyond raw prediction error minimization.

**PSO settings:** $N=2$ particles, $t=20$ iterations, $D=4$ dimensions, $c_1=c_2=1.5$, $w=0.5$.

**Data:** Shanghai Pudong Development Bank stock (600000.SH), 1,481 daily records including PE ratio, PB ratio, volume, turnover.

**Key Results:**
- PSO-LSTM achieves MAE = 0.046, MSE = 0.090, RMSE = 0.095 on return prediction
- Predicted trend aligns with true trend visually
- Model demonstrates improved accuracy over simple LSTM

**Strengths:** Innovative MSW regularization in the PSO fitness function; practical Chinese stock market application.

**Limitations:** Very small swarm ($N=2$) and few iterations (20); only daily data; no comparison against non-neural baselines; no trading/backtest evaluation.

---

### Paper 3: Zeng, Liang, Yang, Wang & Cai (2025) — *Enhancing Stock Index Prediction: A Hybrid LSTM-PSO Model for Improved Forecasting Accuracy*

**Venue:** PLoS ONE, DOI: 10.1371/journal.pone.0310296

**Problem:** Manual LSTM hyperparameter tuning is labor-intensive and yields suboptimal configurations; stock data is noisy and requires preprocessing.

**Approach:**

A three-stage pipeline:

1. **Data preprocessing:** Wavelet transform (DWT with Haar function) for denoising, followed by Pearson correlation analysis to remove highly correlated features ($|\rho| > 0.95$)
2. **PSO optimization:** Optimizes number of hidden neurons (range [50,300]) and training epochs (range [50,300]) for LSTM models with 1, 2, and 3 hidden layers
3. **LSTM prediction:** Uses best PSO-identified parameters

PSO settings: $N=20$ particles, $K=50$ iterations, $D=6$, $w=0.8$, $c_1=c_2=1.5$.

**Data:** Six global stock indices — DJIA, S&P 500, Hang Seng, Nikkei 225, CSI300, Nifty50 (2008/07 to 2016/09). 80/20 train/test split.

**Feature set (18 features):** OHLCV, MACD, CCI, ATR, Bollinger, EMA20, MA5/10, MTM6/12, ROC, SMI, WVAD, exchange rate, interest rate.

**Key Results:**
- PSO-LSTM achieves best performance on Hang Seng, Nikkei 225, and Nifty50 (lowest RMSE, MAE, MAPE)
- Statistical significance confirmed: p-value (sign test) = 0.0078; p-value (Wilcoxon) = 0.005
- Lookback sensitivity: 7-day, 20-day, and 50-day lookbacks show similar accuracy (robust to lookback choice)
- PSO-LSTM consistently in top 3 vs. 9 baseline methods (XGB, RF, KNN, SVM, MLP, LSTM, BiLSTM, RNN, GRU)

**Strengths:** Multi-market validation with statistical significance testing; comprehensive baseline comparison (9 methods); lookback sensitivity analysis.

**Limitations:** Optimizes only 2 LSTM hyperparameters (neurons and epochs); does not optimize dropout or learning rate; no trading performance evaluation; uses daily data.

---

### Paper 4: Lanbouri & Achchab (2020) — *Stock Market Prediction on High Frequency Data using Long-Short Term Memory*

**Venue:** Procedia Computer Science (MobiSPC 2020), DOI: 10.1016/j.procs.2020.07.087

**Problem:** High-frequency (1-minute) intraday stock prediction is more challenging than daily prediction due to noise, latency constraints, and the importance of technical indicators.

**Approach:**

An LSTM regression model is applied to 1-minute bar S&P500 data from 11/09/2017 to 16/02/2018 (43,148 bars for 484 stocks). The model predicts closing price 1, 5, and 10 minutes ahead.

Two model variants:
1. **Without technical indicators:** 5 input features (OHLCV). Input layer: 5 → 10 hidden nodes.
2. **With technical indicators:** 10 input features (OHLCV + EMA12, EMA25, MACD, Bollinger Upper/Lower)

**Normalization:** Min-Max scaling $\tilde{x} = (x - \text{Min}) / (\text{Max} - \text{Min})$

**Data split:** Train (11/09/2017–17/01/2018), Validation (17/01/2018–16/02/2018)

**Key Results:**

| Model | 1-min RMSE | 5-min RMSE | 10-min RMSE |
|---|---|---|---|
| Without TI | **0.0018** | **0.0046** | **0.0046** |
| With TI | 0.0721 | 0.0201 | 0.0108 |

**Counterintuitive finding:** The model WITHOUT technical indicators outperforms the one WITH them, especially at short horizons (1 minute). The authors suggest this may be because adding technical indicators introduces noise for very short-term prediction, and because the raw OHLCV already captures the relevant signal.

**Strengths:** Only paper to directly address 1-minute HFT prediction; establishes that LSTM is viable for intraday prediction; practical RMSE baseline values for comparison.

**Limitations:** No PSO or hyperparameter optimization; no trading performance or cost analysis; very short dataset (5 months); no cross-ticker features; single LSTM configuration tested.

---

### Additional Context: CSCI 633 Project Proposal (Group Project, 2026)

The group project proposal directly defines the specific system to be built:

- **5-year 1-minute OHLCV data** for 51 tickers (50 equities + SPY)
- **PSO search space:** 5 dimensions (layers, units, dropout, lr, lookback)
- **Fitness function:** Sharpe Ratio, MDD, RMSE
- **Baselines:** Persistence, vanilla LSTM, XGBoost
- **Research questions:** Does IPSO outperform baselines? Which hyperparameters matter most?

This proposal directly motivates our hybrid system design.

---

## 3. Comparison of Methods

### 3.1 Architecture Comparison

| Study | Model | Data Freq | Tickers | Features | PSO Dims | Baselines |
|---|---|---|---|---|---|---|
| Ji et al. (2021) | IPSO-LSTM | Daily | 5 indices | Closing price only | 4 | SVR, LSTM, PSO-LSTM |
| Deng & Peng (2025) | PSO-LSTM | Daily | 1 stock | OHLCV + 5 indicators | 4 | LSTM only |
| Zeng et al. (2025) | PSO-LSTM | Daily | 6 indices | OHLCV + 12 TI + 2 macro | 2 | 9 methods |
| Lanbouri & Achchab (2020) | LSTM | 1-min | SPY (500 stocks) | OHLCV + 5 TI | N/A | Baseline LSTM |
| **This work** | **IPSO-LSTM** | **1-min** | **51 equities** | **117 features** | **5** | **4 models** |

### 3.2 PSO Configuration Comparison

| Study | Particles | Iterations | Dimensions | Inertia | Mutation |
|---|---|---|---|---|---|
| Ji et al. (2021) | Not specified | Not specified | 4 | Nonlinear (tanh) | Adaptive |
| Deng & Peng (2025) | 2 | 20 | 4 | Fixed (0.5) | None |
| Zeng et al. (2025) | 20 | 50 | 2 | Fixed (0.8) | None |
| **This work** | **30** | **50** | **5** | **Nonlinear (tanh)** | **Adaptive** |

### 3.3 Fitness Function Comparison

| Study | Primary Metric | Secondary Metrics | Trading Component |
|---|---|---|---|
| Ji et al. (2021) | RMSE (MSE) | None | No |
| Deng & Peng (2025) | MSE + MSW (combined) | None | No |
| Zeng et al. (2025) | RMSE | None | No |
| **This work** | **Composite (RMSE + Sharpe + MDD)** | **All 6 metrics** | **Yes** |

### 3.4 Performance Summary

| Study | Best Model | RMSE | Sharpe | Notes |
|---|---|---|---|---|
| Ji et al. (2021) | IPSO-LSTM | Lower than PSO-LSTM by 20.54% | Not reported | ASX200, daily |
| Deng & Peng (2025) | PSO-LSTM | 0.095 (return scale) | Not reported | Single stock, daily |
| Zeng et al. (2025) | PSO-LSTM | 83.38 (DJIA price scale) | Not reported | R² = 0.983 |
| Lanbouri (2020) | LSTM (no TI) | 0.0018 (1-min standardized) | Not reported | 1-min HFT |

---

## 4. Identified Research Gaps

The synthesis of the four papers reveals several clear gaps that this project addresses:

### Gap 1: No High-Frequency PSO-LSTM

All four PSO-LSTM hybrid studies use **daily data**. Lanbouri (2020) uses 1-minute data but without PSO. **No study combines PSO-based hyperparameter optimization with intraday (1-minute) LSTM prediction.** This project fills that gap directly.

### Gap 2: No Trading-Aware Fitness Function

All prior studies optimize LSTM solely on statistical metrics (RMSE, MSE, MAE). None incorporate trading performance (Sharpe, drawdown) in the PSO fitness function. This means their optimized hyperparameters may minimize prediction error but not necessarily trading quality. **Our composite fitness function directly optimizes for risk-adjusted trading performance.**

### Gap 3: Limited Feature Engineering

- Ji et al. (2021) use only closing price
- Deng & Peng (2025) use OHLCV + 5 basic indicators
- Zeng et al. (2025) use 18 features including OHLCV + 12 TI + 2 macro

**No study designs a 100+ feature set** incorporating cross-ticker correlations, market breadth, time-of-day effects, and VWAP-based liquidity features.

### Gap 4: No Multi-Ticker Cross-Asset Features

All studies treat each ticker/index in isolation. The inter-market information flow (e.g., SPY → individual equities) is ignored. **Our system explicitly includes beta to SPY, rolling correlations with SPY, peer correlations, and market breadth features.**

### Gap 5: No Walk-Forward Backtesting

All studies report final test-set metrics but none perform walk-forward validation or proper backtesting with transaction costs and slippage. This overstates practical performance. **Our backtesting framework with transaction costs (15 bps roundtrip) and stop losses provides a more honest assessment.**

### Gap 6: Small Swarm Sizes

Deng & Peng (2025) use only 2 particles — far too few for a 4-dimensional search space. **We use 30 particles**, providing sufficient coverage for meaningful global optimization.

### Gap 7: Low-Dimensional Search

All studies optimize 2–4 hyperparameters. **Our 5-dimensional space** additionally includes lookback window as an optimized parameter, which Lanbouri (2020) identifies as critical for HFT performance.

---

## 5. How This Work Improves on Prior Art

| Improvement | Specific Enhancement |
|---|---|
| **Data frequency** | 1-minute bars vs. daily; captures intraday momentum and microstructure |
| **Feature richness** | 117 raw features (vs. ≤18 in prior work); includes cross-ticker, time-of-day, VWAP, peer correlations |
| **PSO improvements** | Adopts Ji et al.'s IPSO (tanh + mutation) but extends to 5D with lookback as a PSO variable |
| **Fitness function** | Composite of RMSE + Sharpe + MDD; directly optimizes for trading quality |
| **Universe breadth** | 51 tickers across 5 sectors (vs. 1–6 in prior work); tests generalizability |
| **Walk-forward validation** | 1-month out-of-sample rolling backtests (vs. single train/test split) |
| **Transaction costs** | 15 bps roundtrip + stop losses (vs. no transaction costs in prior work) |
| **Feature selection** | Two-stage: XGBoost importance + optional PSO mask (novel combination) |
| **Statistical rigor** | 5 random seeds; mean ± std reported (vs. single run in most prior work) |

---

## 6. References

1. Ji, Y., Liew, A. W.-C., & Yang, L. (2021). A Novel Improved Particle Swarm Optimization With Long-Short Term Memory Hybrid Model for Stock Indices Forecast. *IEEE Access*, 9, 23660–23671. https://doi.org/10.1109/ACCESS.2021.3056713

2. Deng, C., & Peng, J. (2025). Application of LSTM Model based on Particle Swarm Optimization Algorithm in Stock Market Trend Prediction. In *Proceedings of DEAI 2025*. ACM. https://doi.org/10.1145/3745238.3745261

3. Zeng, X., Liang, C., Yang, Q., Wang, F., & Cai, J. (2025). Enhancing stock index prediction: A hybrid LSTM-PSO model for improved forecasting accuracy. *PLoS ONE*, 20(1), e0310296. https://doi.org/10.1371/journal.pone.0310296

4. Lanbouri, Z., & Achchab, S. (2020). Stock Market prediction on High frequency data using Long-Short Term Memory. *Procedia Computer Science*, 175, 603–608. https://doi.org/10.1016/j.procs.2020.07.087

5. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. In *Proceedings of ICNN'95*, 4, 1942–1948.

6. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

7. Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654–669.

8. Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. In *Proceedings of IEEE ICEC*, 69–73.