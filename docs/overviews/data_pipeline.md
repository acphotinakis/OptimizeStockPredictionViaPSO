# Data Pipeline
## Alpaca API Integration, Cleaning, Alignment, and Splitting

**Document Version:** 1.0 | March 2026

---

## Table of Contents

1. [Alpaca API Integration](#1-alpaca-api-integration)
2. [Data Schema](#2-data-schema)
3. [Cleaning Steps](#3-cleaning-steps)
4. [Multi-Ticker Alignment](#4-multi-ticker-alignment)
5. [Train/Val/Test Split Logic](#5-trainvaltest-split-logic)
6. [Sliding Window Construction](#6-sliding-window-construction)
7. [Session Filtering](#7-session-filtering)
8. [Data Quality Checklist](#8-data-quality-checklist)

---

## 1. Alpaca API Integration

### 1.1 API Overview

The [Alpaca Markets](https://alpaca.markets) REST API provides historical bar data at 1-minute resolution for US equities. A free paper-trading account is sufficient for data access.

**Endpoint:**
```
GET https://data.alpaca.markets/v2/stocks/{symbol}/bars
```

**Authentication:**
```http
APCA-API-KEY-ID: <your_key>
APCA-API-SECRET-KEY: <your_secret>
```

### 1.2 Request Parameters

| Parameter | Value | Notes |
|---|---|---|
| `timeframe` | `1Min` | 1-minute bars |
| `start` | `2019-01-02` | Inclusive |
| `end` | `2024-01-01` | Exclusive |
| `adjustment` | `split` | Adjust for stock splits only (not dividends, to preserve return signal) |
| `feed` | `sip` | Consolidated tape (most complete) |
| `limit` | `10000` | Max bars per page |

### 1.3 Python Implementation

```python
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
from pathlib import Path
import time
import logging

class AlpacaIngestor:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.logger = logging.getLogger(__name__)
    
    def download_bars(self,
                      ticker: str,
                      start: str = '2019-01-02',
                      end: str = '2024-01-01') -> pd.DataFrame:
        """
        Download 1-minute bars for a single ticker.
        Handles pagination automatically.
        """
        try:
            bars = self.api.get_bars(
                ticker,
                TimeFrame.Minute,
                start=start,
                end=end,
                adjustment='split',
                feed='sip'
            ).df
        except Exception as e:
            self.logger.error(f"Failed to download {ticker}: {e}")
            return pd.DataFrame()
        
        if bars.empty:
            self.logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        
        # Standardize columns
        bars = bars[['open', 'high', 'low', 'close', 'volume']].copy()
        bars.index = pd.to_datetime(bars.index, utc=True)
        bars.index.name = 'timestamp'
        bars['ticker'] = ticker
        return bars
    
    def download_universe(self,
                          tickers: list,
                          output_dir: Path,
                          start: str = '2019-01-02',
                          end: str = '2024-01-01') -> None:
        """
        Download all tickers and save as parquet files.
        Respects Alpaca's 200 req/min rate limit.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, ticker in enumerate(tickers):
            output_path = output_dir / f"{ticker}.parquet"
            if output_path.exists():
                self.logger.info(f"[{i+1}/{len(tickers)}] {ticker} already cached, skipping")
                continue
            
            self.logger.info(f"[{i+1}/{len(tickers)}] Downloading {ticker}...")
            df = self.download_bars(ticker, start, end)
            if not df.empty:
                df.to_parquet(output_path)
            
            # Rate limiting: 200 req/min → sleep 0.35s between calls
            time.sleep(0.35)
```

### 1.4 Ticker Universe

The 51-ticker universe consists of:

**SPY (benchmark):**
- SPY (S&P 500 ETF)

**50 Equities (diversified by sector):**

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, NVDA, GOOGL, META, AMD, INTC, CRM, ADBE, QCOM |
| Financials | JPM, BAC, GS, MS, WFC, C, BLK, AXP, COF, USB |
| Healthcare | JNJ, UNH, PFE, ABBV, MRK, LLY, TMO, DHR, MDT, AMGN |
| Consumer | AMZN, TSLA, HD, MCD, NKE, SBUX, TGT, COST, WMT, LOW |
| Industrials | BA, CAT, GE, UPS, HON, MMM, RTX, LMT, DE, FDX |

This cross-sector diversity ensures the feature selection and model generalize beyond a single industry's patterns.

---

## 2. Data Schema

### 2.1 Raw Bar Schema (Per Ticker, After Download)

| Column | Type | Description | Example |
|---|---|---|---|
| `timestamp` | `DatetimeTZDtype[ns, UTC]` | Bar close time (UTC) | 2019-01-02 14:30:00+00:00 |
| `open` | `float64` | Opening price | 249.54 |
| `high` | `float64` | High price | 249.80 |
| `low` | `float64` | Low price | 249.40 |
| `close` | `float64` | Closing price | 249.75 |
| `volume` | `int64` | Number of shares traded | 1,452,300 |
| `ticker` | `str` | Ticker symbol | "AAPL" |

### 2.2 Processed Schema (After Cleaning)

Same columns plus:

| Column | Type | Description |
|---|---|---|
| `log_return` | `float64` | $\ln(C_t / C_{t-1})$ |
| `session_minute` | `int16` | Minutes since 09:30 (0–389) |
| `is_session_open` | `bool` | True if within trading hours |
| `gap_flag` | `bool` | True if bar was forward-filled |
| `outlier_flag` | `bool` | True if return > 5σ (clipped) |

### 2.3 Aligned Multi-Ticker Schema (Master DataFrame)

After alignment, a single MultiIndex DataFrame:

```
Index: DatetimeIndex (UTC, 1-min frequency, NYSE session hours only)
Columns: MultiIndex with levels ['ticker', 'field']
         50 tickers × 6 fields = 306 columns

Example:
         AAPL                              MSFT              ...
         open   high   low    close  vol   open   high  ...
2019-01-02 14:30  249.54 249.80 249.40 249.75 1452300 99.82 99.90
```

---

## 3. Cleaning Steps

### 3.1 Pipeline Overview

```
Raw OHLCV
    │
    ├─ Step 1: Timestamp Standardization (UTC → ET, then re-UTC)
    ├─ Step 2: Session Filtering (09:30–16:00 ET only)
    ├─ Step 3: Zero-Volume Bar Removal
    ├─ Step 4: Price Sanity Check (no zero or negative prices)
    ├─ Step 5: Missing Value Handling
    ├─ Step 6: Return Outlier Detection and Clipping
    └─ Step 7: OHLC Consistency Check
```

### 3.2 Step 1: Timestamp Standardization

All timestamps are stored as UTC. For session filtering, they are converted to Eastern Time (ET) on-the-fly.

```python
import pytz
eastern = pytz.timezone('America/New_York')
df_et = df.copy()
df_et.index = df.index.tz_convert(eastern)
```

### 3.3 Step 2: Session Filtering

Retain only bars during regular NYSE trading hours:
- Start: 09:30:00 ET
- End: 15:59:00 ET (the 16:00 bar is excluded as it captures close auction noise)

```python
df_session = df_et[
    (df_et.index.time >= pd.Timestamp('09:30').time()) &
    (df_et.index.time <= pd.Timestamp('15:59').time())
]
```

**Pre-market and after-hours data:** Excluded. These sessions have significantly lower liquidity, wider spreads, and different price dynamics that would confuse the model.

### 3.4 Step 3: Zero-Volume Bar Removal

Bars with $V_t = 0$ indicate no trading activity and typically occur at exchange-level data gaps:

```python
df_clean = df_session[df_session['volume'] > 0]
```

### 3.5 Step 4: Price Sanity Check

Remove bars with non-positive prices (corrupted data):

```python
price_cols = ['open', 'high', 'low', 'close']
df_clean = df_clean[(df_clean[price_cols] > 0).all(axis=1)]
```

Also verify OHLC consistency: $L \leq O \leq H$, $L \leq C \leq H$:
```python
ohlc_ok = (df_clean['low'] <= df_clean['open']) & \
           (df_clean['open'] <= df_clean['high']) & \
           (df_clean['low'] <= df_clean['close']) & \
           (df_clean['close'] <= df_clean['high'])
df_clean = df_clean[ohlc_ok]
```

### 3.6 Step 5: Missing Value Handling

After session filtering, some 1-minute bars may be missing (no trades). Two strategies:

**Strategy A — Forward Fill (≤ 5 consecutive bars):**

```python
# Create a complete 1-min index for all trading sessions
full_index = pd.date_range(...)  # All 1-min bars in the date range (session only)
df_reindexed = df_clean.reindex(full_index)

# Forward fill gaps of 5 bars or fewer
gap_size = df_reindexed['close'].isna().astype(int)
gap_groups = (gap_size != gap_size.shift()).cumsum()
gap_lengths = gap_size.groupby(gap_groups).transform('sum')

df_reindexed.loc[gap_lengths <= 5] = df_reindexed.loc[gap_lengths <= 5].ffill()
df_reindexed['gap_flag'] = gap_lengths > 0
```

**Strategy B — Drop (> 5 consecutive bars):**

Gaps exceeding 5 minutes are genuine data outages. The surrounding bars are marked as non-contiguous; sliding windows that span a gap are excluded from the dataset.

```python
# Mark bars that follow a gap > 5 bars
df_reindexed['post_gap'] = gap_lengths > 5
# Exclude any window that contains a True value in post_gap
```

### 3.7 Step 6: Return Outlier Detection and Clipping

Extreme returns (> 5σ over a 60-minute rolling window) indicate possible data errors or circuit breaker events. These are **clipped** (not removed) to preserve the timestamp index:

```python
r = np.log(df['close'] / df['close'].shift(1))
rolling_mean = r.rolling(60).mean()
rolling_std = r.rolling(60).std()
z_score = (r - rolling_mean) / rolling_std

# Clip returns at ±5σ
clip_mask = z_score.abs() > 5
df.loc[clip_mask, 'outlier_flag'] = True
# Recompute close prices after clipping returns
for idx in df[clip_mask].index:
    sign = np.sign(r[idx])
    clipped_r = sign * 5 * rolling_std[idx]
    df.loc[idx, 'close'] = df['close'].shift(1)[idx] * np.exp(clipped_r)
```

---

## 4. Multi-Ticker Alignment

### 4.1 Problem

Each ticker has a slightly different trading schedule (circuit breakers, early closes, halts). A naive outer join would introduce massive missing values; an inner join would discard too much data.

### 4.2 Alignment Strategy

**Step 1:** Create a master timestamp index from SPY (the most liquid and complete ticker).
**Step 2:** Forward-fill each ticker to the master index (up to 5 bars).
**Step 3:** Drop timestamps where more than 5% of tickers (>3) have missing data.

```python
def align_universe(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Align 51 ticker DataFrames to a common master timestamp index.
    """
    # Use SPY as master index (most complete)
    master_index = dfs['SPY'].index
    
    aligned = {}
    for ticker, df in dfs.items():
        # Reindex to master and forward fill (max 5 bars)
        df_aligned = df.reindex(master_index).ffill(limit=5)
        aligned[ticker] = df_aligned
    
    # Stack into MultiIndex DataFrame
    result = pd.concat(aligned, axis=1)
    result.columns = pd.MultiIndex.from_tuples(
        [(ticker, field) for ticker, df in aligned.items()
         for field in df.columns],
        names=['ticker', 'field']
    )
    
    # Drop rows where >5% of tickers have NaN close prices
    close_cols = result.xs('close', axis=1, level='field')
    missing_fraction = close_cols.isna().mean(axis=1)
    result = result[missing_fraction <= 0.05]
    
    return result
```

### 4.3 Expected Data Volume

| Period | Trading days | 1-min bars/day | Total bars per ticker |
|---|---|---|---|
| Training (3yr) | ~756 | 390 | ~294,840 |
| Validation (1yr) | ~252 | 390 | ~98,280 |
| Test (1yr) | ~252 | 390 | ~98,280 |
| **Total** | **~1,260** | **390** | **~491,400** |

With 51 tickers: $\approx 25$ million bar records total (before deduplication and alignment). Stored as Parquet, this is approximately 800 MB.

---

## 5. Train/Val/Test Split Logic

### 5.1 Chronological Split (No Shuffling)

To prevent lookahead bias, the split is strictly chronological:

```
|──────── Train (3 years) ──────────|──── Val (1 year) ────|─── Test (1 year) ───|
2019-01-02                        2022-01-03             2023-01-03           2024-01-01
```

**Critical rules:**
1. **No shuffling** across the split boundaries
2. Normalization scalers fitted **only** on the training set
3. PSO fitness evaluated **only** on the validation set
4. Test set touched **only once** for final reporting

### 5.2 Split Implementation

```python
def temporal_split(df: pd.DataFrame,
                   train_end: str = '2022-01-03',
                   val_end: str = '2023-01-03') -> tuple:
    """
    Chronological train/val/test split.
    Returns (df_train, df_val, df_test).
    """
    df_train = df.loc[:train_end]
    df_val   = df.loc[train_end:val_end].iloc[1:]  # Exclude last training bar
    df_test  = df.loc[val_end:].iloc[1:]            # Exclude last val bar
    return df_train, df_val, df_test
```

### 5.3 Session Boundary Handling

The sliding window must not cross session (overnight) boundaries. Overnight gaps represent structural breaks in the 1-minute time series.

**Implementation:** Mark the first bar of each trading session as a "session start". Any sliding window that includes a session start as a non-first element is excluded from the dataset.

```python
df['session_start'] = (df.index.time == pd.Timestamp('09:30').time())

def build_windows(df, lookback, target_col='log_return'):
    X, y = [], []
    session_starts = df['session_start'].values
    close = df['close'].values
    features = df[feature_cols].values
    
    for i in range(lookback, len(df)):
        window = features[i-lookback:i]
        
        # Skip windows that cross a session boundary
        if session_starts[i-lookback+1:i].any():
            continue
        
        X.append(window)
        y.append(np.log(close[i] / close[i-1]))  # Next-bar log return
    
    return np.array(X), np.array(y)
```

---

## 6. Sliding Window Construction

### 6.1 Window Parameters

| Parameter | Value | Notes |
|---|---|---|
| Lookback $T$ | {10, 30, 60, 120} | PSO-tuned |
| Step size | 1 | Predict every bar |
| Target | $r_{t+1}$ | 1-step ahead log return |
| Session crossing | Excluded | No overnight windows |

### 6.2 Memory-Efficient Construction

For large datasets, pre-allocating the full tensor is memory-efficient:

```python
def build_windows_efficient(features: np.ndarray,
                            returns: np.ndarray,
                            session_starts: np.ndarray,
                            lookback: int) -> tuple:
    N = len(features)
    F = features.shape[1]
    
    # Pre-count valid samples
    valid_mask = np.ones(N, dtype=bool)
    valid_mask[:lookback] = False
    for i in range(lookback, N):
        if session_starts[i-lookback+1:i].any():
            valid_mask[i] = False
    
    n_valid = valid_mask.sum()
    X = np.empty((n_valid, lookback, F), dtype=np.float32)
    y = np.empty(n_valid, dtype=np.float32)
    
    idx = 0
    for i in range(lookback, N):
        if valid_mask[i]:
            X[idx] = features[i-lookback:i]
            y[idx] = returns[i]
            idx += 1
    
    return X, y
```

---

## 7. Session Filtering

### 7.1 Trading Calendar

US NYSE regular trading hours: **09:30:00 to 15:59:00 ET**, Monday through Friday, excluding federal holidays and early-close days.

Early-close days (e.g., day before Thanksgiving, day before Christmas) end at 13:00 ET. These are handled by the Alpaca API which returns no data after the early close.

### 7.2 Holiday Filtering

```python
import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date='2019-01-01', end_date='2024-01-01')
valid_days = schedule.index.date
df_filtered = df[df.index.date.isin(valid_days)]
```

### 7.3 First and Last Bar of Session

The first bar of the session (09:30 ET) captures overnight gap dynamics. The last bar (15:59 ET) captures closing auction dynamics. Both are **retained** as they contain informative features (e.g., gap feature, session-end volume surge).

---

## 8. Data Quality Checklist

Run before proceeding to feature engineering:

| Check | Method | Pass Criterion |
|---|---|---|
| Completeness | % non-null close prices per ticker | ≥ 95% |
| OHLC consistency | max(L > O, L > C, H < O, H < C) | 0% violations |
| Zero volume bars | count(V = 0) after cleaning | 0% |
| Session coverage | bars per day per ticker | 385–390 (allow early closes) |
| Return distribution | mean, std, skew per ticker | mean ≈ 0, std < 0.005/bar |
| Outlier flag rate | count(outlier_flag) / N | < 0.1% |
| Gap flag rate | count(gap_flag) / N | < 5% |
| Alignment | % tickers with NaN at any timestamp | < 5% |
| Split leak | any val/test feature stats computed from train? | Must be False |

```python
def run_quality_checks(df_aligned: pd.DataFrame,
                       df_train: pd.DataFrame) -> dict:
    results = {}
    close = df_aligned.xs('close', axis=1, level='field')
    
    results['completeness'] = close.notna().mean().min()
    results['bars_per_day'] = len(df_aligned) / (df_aligned.index.normalize().nunique())
    results['gap_rate'] = df_aligned.xs('gap_flag', axis=1, level='field').mean().max()
    results['outlier_rate'] = df_aligned.xs('outlier_flag', axis=1, level='field').mean().max()
    results['alignment_rate'] = close.isna().mean(axis=1).max()
    
    all_pass = all([
        results['completeness'] >= 0.95,
        results['gap_rate'] <= 0.05,
        results['outlier_rate'] <= 0.001,
        results['alignment_rate'] <= 0.05
    ])
    results['all_pass'] = all_pass
    return results
```