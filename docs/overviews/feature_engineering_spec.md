# Feature Engineering Specification
## PSO-LSTM Stock Price Prediction System

**Document Version:** 1.0 | March 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Price & Return Features](#2-price--return-features)
3. [Technical Indicator Features](#3-technical-indicator-features)
4. [Statistical Moment Features](#4-statistical-moment-features)
5. [Volume & Liquidity Features](#5-volume--liquidity-features)
6. [Cross-Ticker & Market Features](#6-cross-ticker--market-features)
7. [Lag Features](#7-lag-features)
8. [Optional Dimensionality Reduction](#8-optional-dimensionality-reduction)
9. [Normalization Strategy](#9-normalization-strategy)
10. [Feature Selection Strategy](#10-feature-selection-strategy)
11. [Feature Dictionary Summary Table](#11-feature-dictionary-summary-table)

---

## 1. Overview

Feature engineering transforms raw 1-minute OHLCV bars into a rich representation capturing price dynamics, momentum, volatility, liquidity, and cross-asset relationships. All features are computed **on the training set only**; scaling parameters are fitted on train and applied to validation and test sets to prevent lookahead bias.

**Total features: 117** (before selection)

| Category | Count |
|---|---|
| Price & Return | 10 |
| Technical Indicators | 42 |
| Statistical Moments | 20 |
| Volume & Liquidity | 20 |
| Cross-Ticker & Market | 15 |
| Lag Features | 10 |
| **Total** | **117** |

After XGBoost importance filtering (top 70% cumulative importance): approximately **70–85 features** are retained per ticker.

---

## 2. Price & Return Features

These are the fundamental price-based inputs, computed from raw OHLCV bars.

### 2.1 Mid-Price

$$\text{mid}_t = \frac{\text{high}_t + \text{low}_t}{2}$$

### 2.2 Log Return (Prediction Target)

$$r_t = \ln\left(\frac{\text{close}_t}{\text{close}_{t-1}}\right)$$

The **target variable** is $r_{t+1}$ (next-period log return).

### 2.3 OHLCV Ratios

These encode the intrabar price structure without requiring cross-bar normalization:

| Feature | Formula |
|---|---|
| `hl_ratio` | $(H_t - L_t) / C_{t-1}$ |
| `co_ratio` | $(C_t - O_t) / C_{t-1}$ |
| `ho_ratio` | $(H_t - O_t) / C_{t-1}$ |
| `lc_ratio` | $(C_t - L_t) / C_{t-1}$ |

### 2.4 Gap

$$\text{gap}_t = \frac{O_t - C_{t-1}}{C_{t-1}}$$

### 2.5 True Range

$$\text{TR}_t = \max\left(H_t - L_t,\ |H_t - C_{t-1}|,\ |L_t - C_{t-1}|\right)$$

---

## 3. Technical Indicator Features

All windows computed in minutes to suit 1-minute bar data.

### 3.1 Moving Averages (6 features)

| Feature | Formula | Window |
|---|---|---|
| `sma_5` | $\frac{1}{5}\sum_{i=0}^{4} C_{t-i}$ | 5 |
| `sma_20` | Simple moving average | 20 |
| `sma_60` | Simple moving average | 60 |
| `ema_12` | $\text{EMA}_{12,t} = \alpha C_t + (1-\alpha)\text{EMA}_{12,t-1}$, $\alpha = 2/13$ | 12 |
| `ema_26` | Exponential moving average | 26 |
| `ema_60` | Exponential moving average | 60 |

### 3.2 MACD (3 features)

$$\text{MACD}_t = \text{EMA}_{12,t} - \text{EMA}_{26,t}$$
$$\text{Signal}_t = \text{EMA}_{9}(\text{MACD}_t)$$
$$\text{Histogram}_t = \text{MACD}_t - \text{Signal}_t$$

Features: `macd`, `macd_signal`, `macd_hist`

### 3.3 RSI — Relative Strength Index (2 features)

$$\text{RS}_{k,t} = \frac{\overline{U}_{k,t}}{\overline{D}_{k,t}}, \quad \text{RSI}_{k,t} = 100 - \frac{100}{1 + \text{RS}_{k,t}}$$

where $\overline{U}_{k,t}$ = average gain and $\overline{D}_{k,t}$ = average loss over the last $k$ periods.

Features: `rsi_14`, `rsi_30`

### 3.4 Bollinger Bands (4 features)

$$\text{BB\_mid}_{k,t} = \text{SMA}_{k,t}, \quad \text{BB\_upper}_{k,t} = \text{SMA}_{k,t} + 2\sigma_{k,t}$$
$$\text{BB\_lower}_{k,t} = \text{SMA}_{k,t} - 2\sigma_{k,t}, \quad \text{BB\_width}_{k,t} = \frac{\text{BB\_upper} - \text{BB\_lower}}{\text{BB\_mid}}$$
$$\text{BB\%B}_{k,t} = \frac{C_t - \text{BB\_lower}}{\text{BB\_upper} - \text{BB\_lower}}$$

$k = 20$ minutes. Features: `bb_upper`, `bb_lower`, `bb_width`, `bb_pct_b`

### 3.5 Stochastic Oscillator (2 features)

$$\%K_t = 100 \cdot \frac{C_t - \min_{14}(L)}{\max_{14}(H) - \min_{14}(L)}, \quad \%D_t = \text{SMA}_3(\%K)$$

Features: `stoch_k`, `stoch_d`

### 3.6 Average True Range (2 features)

$$\text{ATR}_{k,t} = \frac{1}{k}\sum_{i=0}^{k-1}\text{TR}_{t-i}$$

Features: `atr_14`, `atr_30`

### 3.7 Commodity Channel Index (1 feature)

$$\text{CCI}_t = \frac{\text{TP}_t - \text{SMA}_{20}(\text{TP})}{ 0.015 \cdot \text{MAD}_{20}(\text{TP})}$$

where $\text{TP}_t = (H_t + L_t + C_t)/3$ and MAD is mean absolute deviation. Feature: `cci_20`

### 3.8 Williams %R (1 feature)

$$\%R_t = -100 \cdot \frac{\max_{14}(H) - C_t}{\max_{14}(H) - \min_{14}(L)}$$

Feature: `williams_r`

### 3.9 Rate of Change (2 features)

$$\text{ROC}_{k,t} = \frac{C_t - C_{t-k}}{C_{t-k}} \times 100$$

Features: `roc_5`, `roc_10`

### 3.10 On-Balance Volume Momentum (1 feature)

$$\Delta\text{OBV}_t = \frac{\text{OBV}_t - \text{OBV}_{t-20}}{\text{OBV}_{t-20}}$$

Feature: `obv_momentum_20`

### 3.11 Money Flow Index (2 features)

$$\text{MFI}_{k,t} = 100 - \frac{100}{1 + \frac{\text{Positive Money Flow}_{k,t}}{\text{Negative Money Flow}_{k,t}}}$$

where $\text{Money Flow}_t = \text{TP}_t \times V_t$.

Features: `mfi_14`, `mfi_30`

### 3.12 Ichimoku Components (4 features)

$$\text{Tenkan} = \frac{\max_9(H) + \min_9(L)}{2}, \quad \text{Kijun} = \frac{\max_{26}(H) + \min_{26}(L)}{2}$$
$$\text{Senkou A} = \frac{\text{Tenkan} + \text{Kijun}}{2}, \quad \text{Chikou} = C_{t-26}$$

Features: `ichi_tenkan`, `ichi_kijun`, `ichi_senkou_a`, `ichi_chikou`

### 3.13 Parabolic SAR (2 features)

Standard Parabolic SAR with acceleration factor $\alpha_0 = 0.02$, $\alpha_{\max} = 0.2$.

Features: `psar_value`, `psar_signal` (1=long, -1=short)

### 3.14 Donchian Channel (2 features)

$$\text{DC\_upper}_t = \max_{20}(H_t), \quad \text{DC\_lower}_t = \min_{20}(L_t)$$

Feature: `dc_upper`, `dc_lower`

### 3.15 Linear Regression Slope (2 features)

$$\text{slope}_{k,t} = \frac{\sum_{i=0}^{k-1}(i - \bar{i})(C_{t-i} - \bar{C})}{\sum_{i=0}^{k-1}(i - \bar{i})^2}$$

Normalized by $C_t$. Features: `lr_slope_10`, `lr_slope_30`

### 3.16 Z-Score of Close (2 features)

$$z_{k,t} = \frac{C_t - \text{SMA}_{k,t}}{\sigma_{k,t}}$$

Features: `zscore_20`, `zscore_60`

---

## 4. Statistical Moment Features

Rolling windows over $k \in \{10, 20, 60, 120\}$ minutes of log returns $r_t$.

### 4.1 Rolling Mean (Return)

$$\mu_{k,t} = \frac{1}{k}\sum_{i=0}^{k-1} r_{t-i}$$

Features: `ret_mean_10`, `ret_mean_20`, `ret_mean_60`, `ret_mean_120`

### 4.2 Rolling Variance (Realized Variance)

$$\sigma^2_{k,t} = \frac{1}{k-1}\sum_{i=0}^{k-1}(r_{t-i} - \mu_{k,t})^2$$

Features: `ret_var_10`, `ret_var_20`, `ret_var_60`

### 4.3 Rolling Skewness

$$\text{skew}_{k,t} = \frac{\frac{1}{k}\sum_{i=0}^{k-1}(r_{t-i} - \mu)^3}{\sigma^3}$$

Features: `ret_skew_20`, `ret_skew_60`

### 4.4 Rolling Kurtosis (Excess)

$$\text{kurt}_{k,t} = \frac{\frac{1}{k}\sum_{i=0}^{k-1}(r_{t-i} - \mu)^4}{\sigma^4} - 3$$

Features: `ret_kurt_20`, `ret_kurt_60`

### 4.5 Autocorrelation Lag-1

$$\rho_{1,k,t} = \text{corr}(r_{t-k:t-1},\ r_{t-k+1:t})$$

Features: `ret_autocorr_1_20`, `ret_autocorr_1_60`

### 4.6 Rolling Min/Max Ratio (Price Range Compression)

$$\text{range\_ratio}_{k,t} = \frac{\max_k(C) - \min_k(C)}{\text{SMA}_k(C)}$$

Features: `range_ratio_20`, `range_ratio_60`

### 4.7 Realized Volatility

$$\text{RV}_{k,t} = \sqrt{\sum_{i=0}^{k-1} r^2_{t-i}}$$

Features: `rv_10`, `rv_30`

---

## 5. Volume & Liquidity Features

### 5.1 VWAP (Volume-Weighted Average Price)

$$\text{VWAP}_t = \frac{\sum_{i=t_{\text{open}}}^{t} \text{TP}_i \cdot V_i}{\sum_{i=t_{\text{open}}}^{t} V_i}$$

Reset at session open (09:30 ET). Feature: `vwap`, `price_to_vwap`

### 5.2 VWAP Deviation

$$\text{VWAP\_dev}_t = \frac{C_t - \text{VWAP}_t}{\text{VWAP}_t}$$

Feature: `vwap_dev`

### 5.3 Volume Ratio (Relative Volume)

$$\text{RVOL}_{k,t} = \frac{V_t}{\text{SMA}_k(V)}$$

Features: `rvol_20`, `rvol_60`

### 5.4 On-Balance Volume

$$\text{OBV}_t = \text{OBV}_{t-1} + \begin{cases} V_t & \text{if } C_t > C_{t-1} \\ -V_t & \text{if } C_t < C_{t-1} \\ 0 & \text{otherwise} \end{cases}$$

Feature: `obv`, `obv_ema_20` (EMA of OBV)

### 5.5 Accumulation/Distribution Line

$$\text{CLV}_t = \frac{(C_t - L_t) - (H_t - C_t)}{H_t - L_t}$$
$$\text{ADL}_t = \text{ADL}_{t-1} + \text{CLV}_t \cdot V_t$$

Feature: `adl`, `adl_slope_10`

### 5.6 Chaikin Money Flow

$$\text{CMF}_{k,t} = \frac{\sum_{i=0}^{k-1}\text{CLV}_{t-i} \cdot V_{t-i}}{\sum_{i=0}^{k-1} V_{t-i}}$$

Feature: `cmf_20`

### 5.7 Force Index

$$\text{FI}_t = r_t \cdot V_t$$

Features: `force_1`, `force_ema_13` (EMA-13 of Force Index)

### 5.8 Ease of Movement

$$\text{EOM}_t = \frac{(H_t + L_t)/2 - (H_{t-1} + L_{t-1})/2}{V_t / (H_t - L_t)}$$

Feature: `eom_14`

### 5.9 Volume Price Trend

$$\text{VPT}_t = \text{VPT}_{t-1} + V_t \cdot r_t$$

Feature: `vpt`

### 5.10 Intraday Turnover Velocity

$$\text{ITV}_t = \frac{V_t}{\sum_{i=0}^{k-1} V_{t-i}} \quad k=20$$

Feature: `itv_20`

### 5.11 Bar Activity Score

$$\text{BAS}_t = \log(1 + V_t) \cdot \frac{H_t - L_t}{C_{t-1}}$$

Feature: `bas`

### 5.12 Time-of-Day Encoding (Cyclical)

$$\text{tod\_sin}_t = \sin\!\left(\frac{2\pi \cdot \text{minutes\_since\_open}_t}{390}\right)$$
$$\text{tod\_cos}_t = \cos\!\left(\frac{2\pi \cdot \text{minutes\_since\_open}_t}{390}\right)$$

Features: `tod_sin`, `tod_cos` (390 minutes per full trading session)

### 5.13 Day-of-Week Encoding (Cyclical)

$$\text{dow\_sin}_t = \sin\!\left(\frac{2\pi \cdot \text{day\_of\_week}_t}{5}\right), \quad \text{dow\_cos}_t = \cos\!\left(\frac{2\pi \cdot \text{day\_of\_week}_t}{5}\right)$$

Features: `dow_sin`, `dow_cos`

---

## 6. Cross-Ticker & Market Features

These features capture the relationship between the target ticker and the SPY ETF (market benchmark) as well as the broader universe.

### 6.1 Beta to SPY

$$\hat{\beta}_t = \frac{\text{Cov}_k(r_{\text{ticker}},\ r_{\text{SPY}})}{\text{Var}_k(r_{\text{SPY}})} \quad k=60$$

Feature: `beta_spy_60`

### 6.2 Rolling Correlation with SPY

$$\rho_{k,t}^{\text{SPY}} = \text{corr}_k(r_{\text{ticker}},\ r_{\text{SPY}})$$

Features: `corr_spy_20`, `corr_spy_60`

### 6.3 Residual Return (Alpha)

$$\alpha_t = r_{\text{ticker},t} - \hat{\beta}_t \cdot r_{\text{SPY},t}$$

Feature: `alpha_spy`

### 6.4 Relative Strength vs. SPY

$$\text{RS}_t = \frac{\text{close}_{\text{ticker},t} / \text{close}_{\text{ticker},t-k}}{\text{close}_{\text{SPY},t} / \text{close}_{\text{SPY},t-k}} \quad k=20$$

Feature: `rel_strength_spy_20`

### 6.5 SPY Momentum

$$\text{SPY\_mom}_t = r_{\text{SPY}, t-1}$$

Feature: `spy_lag_return_1`

### 6.6 Sector Rotation Score

$$\text{SRS}_t = \frac{r_{\text{ticker},t-k:t}}{\sigma_k(r_{\text{ticker}})} - \frac{r_{\text{SPY},t-k:t}}{\sigma_k(r_{\text{SPY}})} \quad k=20$$

Feature: `sector_rotation_20`

### 6.7 Cross-Ticker Realized Correlation (Top-3 Correlated Tickers)

For each target ticker, compute the rolling 60-minute correlation with its 3 most correlated peer tickers (computed on the training set). This provides sector-cohesion signals.

Features: `peer_corr_1`, `peer_corr_2`, `peer_corr_3`

### 6.8 Market Breadth (Universe-Level)

$$\text{breadth}_t = \frac{|\{i : r_{i,t} > 0\}|}{N}$$

where $N = 51$ (all tickers). Feature: `mkt_breadth`

### 6.9 VIX Proxy (Realized)

Proxy VIX from SPY's 30-minute rolling realized volatility:

$$\text{VIX\_proxy}_t = \sqrt{252 \times 390} \cdot \text{RV}_{30}(\text{SPY})$$

Feature: `vix_proxy`

---

## 7. Lag Features

Lagged values of the top-5 most predictive features (from XGBoost pre-selection) at lags 1, 2, and 3 minutes.

| Feature | Lags |
|---|---|
| `log_return` | t-1, t-2, t-3 |
| `rsi_14` | t-1, t-2 |
| `vwap_dev` | t-1 |
| `rvol_20` | t-1 |
| `corr_spy_20` | t-1 |

Features: 10 lag features total

---

## 8. Optional Dimensionality Reduction

### 8.1 PCA

Applied to reduce correlated technical indicator features to orthogonal principal components explaining 95% of variance.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_train_technical)
# Typically reduces 42 technical features → ~20 PCA components
```

### 8.2 Autoencoder Compression

A shallow autoencoder (encoder: 117→64→32, decoder: 32→64→117) trained on the training set as a nonlinear alternative to PCA.

```python
# Encoder architecture
nn.Linear(117, 64), nn.ReLU(),
nn.Linear(64, 32)  # Bottleneck (latent representation)
```

### 8.3 PSO Feature Mask (Optional Extension)

A binary feature mask can be appended to the particle encoding as an additional 117-bit binary vector. This extends the search space to $5 + 117 = 122$ dimensions. Due to computational cost, this is treated as an optional experiment rather than the default configuration.

---

## 9. Normalization Strategy

**Critical rule:** All normalization parameters (mean, standard deviation, min, max) are **fitted exclusively on the training set** and **applied without re-fitting** to the validation and test sets.

### 9.1 RobustScaler (Default)

Uses median and interquartile range (IQR) instead of mean and standard deviation, making it robust to the fat-tailed distributions common in financial returns.

$$\tilde{x} = \frac{x - \text{median}(x_{\text{train}})}{\text{IQR}(x_{\text{train}})}$$

Applied to: all continuous features.

### 9.2 MinMaxScaler (Price Ratios)

For features bounded by construction (e.g., RSI ∈ [0,100], %K ∈ [0,100]):

$$\tilde{x} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Applied to: RSI, Stochastic, Williams %R, %B.

### 9.3 No Normalization

Applied to: time-of-day and day-of-week cyclical encodings (already bounded in [-1, 1]).

### 9.4 Target Variable

Log returns are **not** normalized. The LSTM output layer predicts raw log returns. RMSE is computed on de-normalized returns for interpretability.

---

## 10. Feature Selection Strategy

### 10.1 Stage 1 — XGBoost Importance Filtering

Train an XGBoost regressor on the training set (flattened: each sample is a single timestep, not a sequence) to predict next-period log returns. Rank features by their gain-based importance score. Retain features with cumulative importance ≥ 70%.

```python
import xgboost as xgb
model = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05)
model.fit(X_train_flat, y_train)
importance = model.feature_importances_
cumulative = np.cumsum(np.sort(importance)[::-1]) / importance.sum()
threshold_idx = np.argmax(cumulative >= 0.70)
selected_features = np.argsort(importance)[::-1][:threshold_idx+1]
```

**Expected output:** ~70–85 features retained (from 117).

### 10.2 Stage 2 — Variance Threshold

Remove features with near-zero variance on the training set (threshold: $\sigma^2 < 10^{-6}$). This catches degenerate features that may arise during low-volatility periods.

### 10.3 Stage 3 — Pearson Correlation Deduplication

Remove one of any pair of features with $|\rho| > 0.98$ on the training set, retaining the one with higher XGBoost importance.

### 10.4 Stage 4 (Optional) — PSO Binary Feature Mask

In the extended experiment, append a binary feature mask to the PSO particle encoding. Each bit $m_j \in \{0, 1\}$ indicates whether feature $j$ is included. The mutation operator applies bit-flip mutations on the feature mask separately from the continuous hyperparameter dimensions.

---

## 11. Feature Dictionary Summary Table

| ID | Feature Name | Category | Formula / Source | Window |
|---|---|---|---|---|
| F01 | `log_return` | Price | $\ln(C_t/C_{t-1})$ | — |
| F02 | `mid_price` | Price | $(H+L)/2$ | — |
| F03 | `hl_ratio` | Price | $(H-L)/C_{t-1}$ | — |
| F04 | `co_ratio` | Price | $(C-O)/C_{t-1}$ | — |
| F05 | `ho_ratio` | Price | $(H-O)/C_{t-1}$ | — |
| F06 | `lc_ratio` | Price | $(C-L)/C_{t-1}$ | — |
| F07 | `gap` | Price | $(O-C_{t-1})/C_{t-1}$ | — |
| F08 | `true_range` | Price | $\max(H-L, |H-C_{t-1}|, |L-C_{t-1}|)$ | — |
| F09 | `cum_return_session` | Price | $C_t/O_{\text{open}} - 1$ | session |
| F10 | `intrabar_vol` | Price | $(H-L)/O$ | — |
| F11 | `sma_5` | Technical | SMA of close | 5 |
| F12 | `sma_20` | Technical | SMA of close | 20 |
| F13 | `sma_60` | Technical | SMA of close | 60 |
| F14 | `ema_12` | Technical | EMA of close | 12 |
| F15 | `ema_26` | Technical | EMA of close | 26 |
| F16 | `ema_60` | Technical | EMA of close | 60 |
| F17 | `macd` | Technical | EMA12 – EMA26 | 26 |
| F18 | `macd_signal` | Technical | EMA9(MACD) | 9 |
| F19 | `macd_hist` | Technical | MACD – Signal | — |
| F20 | `rsi_14` | Technical | RSI formula | 14 |
| F21 | `rsi_30` | Technical | RSI formula | 30 |
| F22 | `bb_upper` | Technical | SMA20 + 2σ | 20 |
| F23 | `bb_lower` | Technical | SMA20 – 2σ | 20 |
| F24 | `bb_width` | Technical | (Upper-Lower)/Mid | 20 |
| F25 | `bb_pct_b` | Technical | %B formula | 20 |
| F26 | `stoch_k` | Technical | %K formula | 14 |
| F27 | `stoch_d` | Technical | SMA3(%K) | 3 |
| F28 | `atr_14` | Technical | ATR formula | 14 |
| F29 | `atr_30` | Technical | ATR formula | 30 |
| F30 | `cci_20` | Technical | CCI formula | 20 |
| F31 | `williams_r` | Technical | %R formula | 14 |
| F32 | `roc_5` | Technical | ROC formula | 5 |
| F33 | `roc_10` | Technical | ROC formula | 10 |
| F34 | `obv_momentum_20` | Technical | ΔOBV/OBV | 20 |
| F35 | `mfi_14` | Technical | MFI formula | 14 |
| F36 | `mfi_30` | Technical | MFI formula | 30 |
| F37 | `ichi_tenkan` | Technical | (max9H+min9L)/2 | 9 |
| F38 | `ichi_kijun` | Technical | (max26H+min26L)/2 | 26 |
| F39 | `ichi_senkou_a` | Technical | (Tenkan+Kijun)/2 | — |
| F40 | `ichi_chikou` | Technical | C_{t-26} | 26 |
| F41 | `psar_value` | Technical | Parabolic SAR | — |
| F42 | `psar_signal` | Technical | ±1 | — |
| F43 | `dc_upper` | Technical | max20(H) | 20 |
| F44 | `dc_lower` | Technical | min20(L) | 20 |
| F45 | `lr_slope_10` | Technical | OLS slope | 10 |
| F46 | `lr_slope_30` | Technical | OLS slope | 30 |
| F47 | `zscore_20` | Technical | (C–SMA)/σ | 20 |
| F48 | `zscore_60` | Technical | (C–SMA)/σ | 60 |
| F49 | `mfi_14` | Technical | Money Flow Index | 14 |
| F50 | `adx_14` | Technical | Average Directional Index | 14 |
| F51 | `dmi_plus` | Technical | +DI | 14 |
| F52 | `dmi_minus` | Technical | -DI | 14 |
| F53 | `ret_mean_10` | Statistical | Rolling mean of $r$ | 10 |
| F54 | `ret_mean_20` | Statistical | Rolling mean of $r$ | 20 |
| F55 | `ret_mean_60` | Statistical | Rolling mean of $r$ | 60 |
| F56 | `ret_mean_120` | Statistical | Rolling mean of $r$ | 120 |
| F57 | `ret_var_10` | Statistical | Rolling variance | 10 |
| F58 | `ret_var_20` | Statistical | Rolling variance | 20 |
| F59 | `ret_var_60` | Statistical | Rolling variance | 60 |
| F60 | `ret_skew_20` | Statistical | Rolling skewness | 20 |
| F61 | `ret_skew_60` | Statistical | Rolling skewness | 60 |
| F62 | `ret_kurt_20` | Statistical | Rolling kurtosis | 20 |
| F63 | `ret_kurt_60` | Statistical | Rolling kurtosis | 60 |
| F64 | `ret_autocorr_1_20` | Statistical | Lag-1 autocorr | 20 |
| F65 | `ret_autocorr_1_60` | Statistical | Lag-1 autocorr | 60 |
| F66 | `range_ratio_20` | Statistical | (maxC–minC)/SMA | 20 |
| F67 | `range_ratio_60` | Statistical | (maxC–minC)/SMA | 60 |
| F68 | `rv_10` | Statistical | Realized vol | 10 |
| F69 | `rv_30` | Statistical | Realized vol | 30 |
| F70 | `hurst_exp_60` | Statistical | Hurst exponent | 60 |
| F71 | `vwap` | Volume | VWAP (session) | session |
| F72 | `price_to_vwap` | Volume | C/VWAP | session |
| F73 | `vwap_dev` | Volume | (C–VWAP)/VWAP | session |
| F74 | `rvol_20` | Volume | V/SMA20(V) | 20 |
| F75 | `rvol_60` | Volume | V/SMA60(V) | 60 |
| F76 | `obv` | Volume | OBV formula | — |
| F77 | `obv_ema_20` | Volume | EMA20(OBV) | 20 |
| F78 | `adl` | Volume | Accum/Distrib | — |
| F79 | `adl_slope_10` | Volume | OLS slope of ADL | 10 |
| F80 | `cmf_20` | Volume | Chaikin MF | 20 |
| F81 | `force_1` | Volume | r×V | — |
| F82 | `force_ema_13` | Volume | EMA13(r×V) | 13 |
| F83 | `eom_14` | Volume | Ease of Movement | 14 |
| F84 | `vpt` | Volume | Vol Price Trend | — |
| F85 | `itv_20` | Volume | V / sum20(V) | 20 |
| F86 | `bas` | Volume | log(1+V)×(H-L)/C | — |
| F87 | `tod_sin` | Volume | sin(2π×tod/390) | — |
| F88 | `tod_cos` | Volume | cos(2π×tod/390) | — |
| F89 | `dow_sin` | Volume | sin(2π×dow/5) | — |
| F90 | `dow_cos` | Volume | cos(2π×dow/5) | — |
| F91 | `beta_spy_60` | Cross | Cov/Var (SPY) | 60 |
| F92 | `corr_spy_20` | Cross | Pearson corr | 20 |
| F93 | `corr_spy_60` | Cross | Pearson corr | 60 |
| F94 | `alpha_spy` | Cross | r – β×r_SPY | — |
| F95 | `rel_strength_spy_20` | Cross | Relative strength | 20 |
| F96 | `spy_lag_return_1` | Cross | r_SPY(t-1) | — |
| F97 | `sector_rotation_20` | Cross | Normalized momentum diff | 20 |
| F98 | `peer_corr_1` | Cross | Corr w/ peer 1 | 60 |
| F99 | `peer_corr_2` | Cross | Corr w/ peer 2 | 60 |
| F100 | `peer_corr_3` | Cross | Corr w/ peer 3 | 60 |
| F101 | `mkt_breadth` | Cross | Fraction up moves | — |
| F102 | `vix_proxy` | Cross | RV30(SPY)×annualized | 30 |
| F103 | `spy_atr_14` | Cross | ATR14 of SPY | 14 |
| F104 | `spy_volume_ratio` | Cross | RVOL20 of SPY | 20 |
| F105 | `universe_mean_ret` | Cross | Mean r across 51 tickers | — |
| F106 | `ret_lag1` | Lag | $r_{t-1}$ | — |
| F107 | `ret_lag2` | Lag | $r_{t-2}$ | — |
| F108 | `ret_lag3` | Lag | $r_{t-3}$ | — |
| F109 | `rsi_lag1` | Lag | $\text{RSI}_{t-1}$ | — |
| F110 | `rsi_lag2` | Lag | $\text{RSI}_{t-2}$ | — |
| F111 | `vwap_dev_lag1` | Lag | $\text{vwap\_dev}_{t-1}$ | — |
| F112 | `rvol_lag1` | Lag | $\text{rvol}_{t-1}$ | — |
| F113 | `corr_spy_lag1` | Lag | $\rho^{\text{SPY}}_{t-1}$ | — |
| F114 | `macd_lag1` | Lag | $\text{MACD}_{t-1}$ | — |
| F115 | `atr_lag1` | Lag | $\text{ATR}_{t-1}$ | — |
| F116 | `bb_pct_b_lag1` | Lag | $\text{\%B}_{t-1}$ | — |
| F117 | `stoch_k_lag1` | Lag | $\%K_{t-1}$ | — |

**Total: 117 features before selection; ~75 after XGBoost importance filtering.**