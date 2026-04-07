from __future__ import annotations
from matplotlib.ticker import FuncFormatter, MaxNLocator

"""
ttm_squeeze.py — Beardy Squeeze Pro–style TTM Squeeze (Python)

What you get:
- Beardy Squeeze Pro–matched calculations:
  * BB basis = SMA(close, length)
  * BB dev   = BB_mult * stdev(close, length)  [population std]
  * KC basis = SMA(close, length)
  * KC width = SMA(TrueRange, length)
  * KC bands at mults: 1.0, 1.5, 2.0
  * Multi-level squeeze dot colors: orange/red/black/green (High/Mid/Low/No)
  * Momentum = ta.linreg(close - avg(avg(highest,lowest), sma(close)), length, 0)
  * Momentum histogram colors: aqua/blue/red/yellow (same rules as Pine)

- Plot:
  * Top panel: candlestick chart
  * Bottom panel: colorized momentum histogram + colored squeeze dots at zero line

Dependencies:
  pip install pandas numpy matplotlib yfinance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


# =========================================================
# 1) Utilities (TradingView-like)
# =========================================================
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def stdev_pop(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).std(ddof=0)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def linreg_ta(y: pd.Series, n: int) -> pd.Series:
    """
    TradingView ta.linreg(x, length, 0) returns the regression line value at the
    current bar over each rolling window.

    Fit y ~ a + b*x with x=0..n-1 and return y_hat at x=n-1.
    """
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)

    def linreg_last(arr: np.ndarray) -> float:
        yv = arr.astype(float)
        if np.any(~np.isfinite(yv)):
            return np.nan
        y_mean = yv.mean()
        b = np.sum((x - x_mean) * (yv - y_mean)) / denom
        a = y_mean - b * x_mean
        return float(a + b * (n - 1))

    return y.rolling(n, min_periods=n).apply(
        lambda a: linreg_last(a.to_numpy()), raw=False
    )


def _col_as_series(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Robustly extract a column as a Series even if df[name] returns a DataFrame.
    If multiple columns share the same label, take the first.
    """
    x = df[name]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x.astype(float)


# =========================================================
# 2) Beardy Squeeze Pro calculations (matched to Pine)
# =========================================================
def beardy_squeeze_pro(
    df: pd.DataFrame,
    length: int = 20,
    bb_mult: float = 2.0,
    kc_mult_high: float = 1.0,
    kc_mult_mid: float = 1.5,
    kc_mult_low: float = 2.0,
) -> pd.DataFrame:
    out = df.copy()

    for c in ["open", "high", "low", "close"]:
        if c not in out.columns:
            raise ValueError(f"Missing column: {c}. Got columns: {list(out.columns)}")

    high = _col_as_series(out, "high")
    low = _col_as_series(out, "low")
    close = _col_as_series(out, "close")

    # --- Bollinger Bands
    out["BB_basis"] = sma(close, length)
    out["BB_dev"] = bb_mult * stdev_pop(close, length)
    out["BB_upper"] = out["BB_basis"] + out["BB_dev"]
    out["BB_lower"] = out["BB_basis"] - out["BB_dev"]

    # --- Keltner Channels (SMA close mid, SMA(TR) width)
    tr = true_range(high, low, close)
    out["KC_basis"] = sma(close, length)
    out["devKC"] = sma(tr, length)

    out["KC_upper_high"] = out["KC_basis"] + out["devKC"] * kc_mult_high
    out["KC_lower_high"] = out["KC_basis"] - out["devKC"] * kc_mult_high

    out["KC_upper_mid"] = out["KC_basis"] + out["devKC"] * kc_mult_mid
    out["KC_lower_mid"] = out["KC_basis"] - out["devKC"] * kc_mult_mid

    out["KC_upper_low"] = out["KC_basis"] + out["devKC"] * kc_mult_low
    out["KC_lower_low"] = out["KC_basis"] - out["devKC"] * kc_mult_low

    # --- Squeeze states (exact boolean logic in Pine)
    out["NoSqz"] = (out["BB_lower"] < out["KC_lower_low"]) | (
        out["BB_upper"] > out["KC_upper_low"]
    )
    out["LowSqz"] = (out["BB_lower"] >= out["KC_lower_low"]) | (
        out["BB_upper"] <= out["KC_upper_low"]
    )
    out["MidSqz"] = (out["BB_lower"] >= out["KC_lower_mid"]) | (
        out["BB_upper"] <= out["KC_upper_mid"]
    )
    out["HighSqz"] = (out["BB_lower"] >= out["KC_lower_high"]) | (
        out["BB_upper"] <= out["KC_upper_high"]
    )

    # --- Momentum baseline (match Pine)
    highest_h = high.rolling(length, min_periods=length).max()
    lowest_l = low.rolling(length, min_periods=length).min()
    mid_hl = (highest_h + lowest_l) / 2.0
    baseline = (mid_hl + out["BB_basis"]) / 2.0

    deviation = close - baseline
    out["mom"] = linreg_ta(deviation, length)

    no_sqz_shifted = out["NoSqz"].shift(1).fillna(False).astype(bool)
    no_sqz_curr = out["NoSqz"].fillna(False).astype(bool)

    out["sqz_started"] = no_sqz_shifted & (~no_sqz_curr)
    out["sqz_fired"] = no_sqz_curr & (~no_sqz_shifted)

    return out
    # # --- Start/Fired (Pine alert semantics)
    # out["sqz_started"] = out["NoSqz"].shift(1).fillna(False) & (
    #     ~out["NoSqz"].fillna(False)
    # )
    # out["sqz_fired"] = out["NoSqz"].fillna(False) & (
    #     ~out["NoSqz"].shift(1).fillna(False)
    # )

    # return out


# =========================================================
# 3) Coloring rules (match Pine)
# =========================================================
def mom_colors(mom: pd.Series) -> np.ndarray:
    m = mom.to_numpy(dtype=float)
    prev = np.r_[np.nan, m[:-1]]

    colors = np.empty(len(m), dtype=object)

    pos = m > 0
    inc = m > prev
    colors[pos & inc] = "#00BCD4"  # aqua
    colors[pos & ~inc] = "#2962FF"  # blue

    neg = ~pos
    dec = m < prev
    colors[neg & dec] = "#F44336"  # red
    colors[neg & ~dec] = "#FFEB3B"  # yellow

    # FIX: Use string "none" instead of tuple (0,0,0,0)
    # This prevents NumPy from trying to unpack the tuple into the array slice.
    colors[~np.isfinite(m)] = "none"
    return colors


def squeeze_dot_colors(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    c = np.empty(n, dtype=object)

    high = df["HighSqz"].fillna(False).to_numpy()
    mid = df["MidSqz"].fillna(False).to_numpy()
    low = df["LowSqz"].fillna(False).to_numpy()

    c[high] = "#FF9800"  # orange
    c[~high & mid] = "#F44336"  # red
    c[~high & ~mid & low] = "#000000"  # black
    c[~high & ~mid & ~low] = "#4CAF50"  # green

    return c


# =========================================================
# 4) Plotting
# =========================================================
def plot_beardy_squeeze(
    df: pd.DataFrame,
    title: str = "Beardy Squeeze Pro (Python)",
    last_n: int = 600,
    save_path: str | None = None,
):
    d = df.copy()

    if last_n is not None and len(d) > last_n:
        d = d.iloc[-last_n:]

    # --- NEW: Use an integer index (0, 1, 2...) instead of dates for X-axis
    x = np.arange(len(d))

    # We still need the actual dates for the labels
    dates = d.index

    o = _col_as_series(d, "open").to_numpy()
    h = _col_as_series(d, "high").to_numpy()
    l = _col_as_series(d, "low").to_numpy()
    c = _col_as_series(d, "close").to_numpy()

    mom = d["mom"].astype(float)
    mom_c = mom_colors(mom)
    dot_c = squeeze_dot_colors(d)

    # --- NEW: Width is now simple because x is just 0, 1, 2...
    width = 0.8

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.05)
    ax_price = fig.add_subplot(gs[0, 0])
    ax_sqz = fig.add_subplot(gs[1, 0], sharex=ax_price)

    # Candles
    for xi, oi, hi, li, ci in zip(x, o, h, l, c):
        up = ci >= oi
        body_color = "#26A69A" if up else "#EF5350"

        # Plot lines and rectangles using the integer index 'xi'
        ax_price.vlines(
            xi, li, hi, linewidth=1, color="#787b86"
        )  # added grey wick color
        y0 = min(oi, ci)
        height = abs(ci - oi) if abs(ci - oi) > 0 else 1e-9
        ax_price.add_patch(
            Rectangle(
                (xi - width / 2, y0),
                width,
                height,
                facecolor=body_color,
                edgecolor=body_color,
                linewidth=1,
            )
        )

    ax_price.set_title(title)
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.2)

    # Bottom: momentum histogram + squeeze dots
    ax_sqz.axhline(0, linewidth=1, color="gray", alpha=0.5)

    # --- NEW: Plot bars at integer locations
    ax_sqz.bar(x, mom.to_numpy(dtype=float), width=width, color=mom_c, align="center")
    ax_sqz.scatter(x, np.zeros_like(x), s=18, c=dot_c, zorder=3)

    ax_sqz.set_ylabel("Squeeze / Mom")
    ax_sqz.grid(True, alpha=0.2)

    # --- NEW: Custom Formatter to map Index -> Date String
    def format_date(x_val, pos=None):
        # x_val is the float position on the axis
        index = int(np.round(x_val))
        if 0 <= index < len(dates):
            # Format however you like:
            return dates[index].strftime("%Y-%m-%d\n%H:%M")
        return ""

    # Set the formatter
    ax_sqz.xaxis.set_major_formatter(FuncFormatter(format_date))

    # Ensure we don't have too many ticks cluttering the view
    ax_sqz.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

    plt.setp(ax_price.get_xticklabels(), visible=False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.show()


# def plot_beardy_squeeze(
#     df: pd.DataFrame,
#     title: str = "Beardy Squeeze Pro (Python)",
#     last_n: int = 600,
#     save_path: str | None = None,
# ):
#     d = df.copy()

#     if last_n is not None and len(d) > last_n:
#         d = d.iloc[-last_n:]

#     if not isinstance(d.index, pd.DatetimeIndex):
#         raise ValueError("DataFrame index must be a DatetimeIndex for plotting.")

#     x = mdates.date2num(d.index.to_pydatetime())

#     o = _col_as_series(d, "Open").to_numpy()
#     h = _col_as_series(d, "High").to_numpy()
#     l = _col_as_series(d, "Low").to_numpy()
#     c = _col_as_series(d, "Close").to_numpy()

#     mom = d["mom"].astype(float)
#     mom_c = mom_colors(mom)
#     dot_c = squeeze_dot_colors(d)

#     dx = np.median(np.diff(x)) if len(x) >= 2 else 1 / 24
#     width = dx * 0.70

#     fig = plt.figure(figsize=(16, 8))
#     gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.05)
#     ax_price = fig.add_subplot(gs[0, 0])
#     ax_sqz = fig.add_subplot(gs[1, 0], sharex=ax_price)

#     # Candles
#     for xi, oi, hi, li, ci in zip(x, o, h, l, c):
#         up = ci >= oi
#         body_color = "#26A69A" if up else "#EF5350"
#         ax_price.vlines(xi, li, hi, linewidth=1)
#         y0 = min(oi, ci)
#         height = abs(ci - oi) if abs(ci - oi) > 0 else 1e-9
#         ax_price.add_patch(
#             Rectangle(
#                 (xi - width / 2, y0),
#                 width,
#                 height,
#                 facecolor=body_color,
#                 edgecolor=body_color,
#                 linewidth=1,
#             )
#         )

#     ax_price.set_title(title)
#     ax_price.set_ylabel("Price")
#     ax_price.grid(True, alpha=0.2)

#     # Bottom: momentum histogram + squeeze dots
#     ax_sqz.axhline(0, linewidth=1)
#     ax_sqz.bar(x, mom.to_numpy(dtype=float), width=width, color=mom_c, align="center")
#     ax_sqz.scatter(x, np.zeros_like(x), s=18, c=dot_c)

#     ax_sqz.set_ylabel("Squeeze / Mom")
#     ax_sqz.grid(True, alpha=0.2)

#     ax_sqz.xaxis_date()
#     ax_sqz.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
#     plt.setp(ax_price.get_xticklabels(), visible=False)

#     fig.tight_layout()

#     if save_path:
#         fig.savefig(save_path, dpi=160)
#     plt.show()


# =========================================================
# 5) Data normalization (FIXED FOR YOUR ERROR)
# =========================================================
def normalize_yfinance_single_ticker(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize yfinance output into a single-ticker OHLCV DataFrame with columns:
      Open, High, Low, Close, Volume (and maybe Adj Close)

    Handles:
    - MultiIndex columns: (ticker, field)
    - Flattened columns
    - BUG/edge case: columns are repeated ticker strings:
        ['AAPL','AAPL','AAPL','AAPL','AAPL']  -> rename based on column count
    """
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError(f"No data returned for ticker={ticker}")

    # Case 1: MultiIndex (most common for multi-ticker or some yfinance configs)
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            data = data[ticker].copy()
        else:
            # fallback: flatten to last level
            data = data.copy()
            data.columns = [c[-1] for c in data.columns]

    # Case 2: Columns are all the ticker string (your exact failure)
    cols = list(data.columns)
    if len(cols) >= 4 and all(str(c) == ticker for c in cols):
        # Rename by count. With auto_adjust=True, yfinance often returns 5 cols: OHLC + Volume
        if len(cols) == 5:
            data = data.copy()
            data.columns = ["Open", "High", "Low", "Close", "Volume"]
        elif len(cols) == 6:
            data = data.copy()
            data.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        else:
            raise ValueError(
                f"Got repeated ticker columns for {ticker} but unexpected count={len(cols)}. "
                f"Columns={cols}"
            )

    # Case 3: Some weird flattened names like "AAPL_Open" or similar
    # Try to map them back to OHLCV if possible.
    if not {"Open", "High", "Low", "Close"}.issubset(set(data.columns)):
        mapped = {}
        for c in data.columns:
            cs = str(c).lower().replace(" ", "")
            if "open" in cs:
                mapped[c] = "Open"
            elif "high" in cs:
                mapped[c] = "High"
            elif "low" in cs:
                mapped[c] = "Low"
            elif "close" in cs and "adj" not in cs:
                mapped[c] = "Close"
            elif "adj" in cs and "close" in cs:
                mapped[c] = "Adj Close"
            elif "volume" in cs:
                mapped[c] = "Volume"

        if mapped:
            data = data.rename(columns=mapped)

    required = {"Open", "High", "Low", "Close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing columns after normalization for {ticker}: {missing}. "
            f"Got: {list(data.columns)}"
        )

    return data


def load_csv(path: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[timestamp_col]).set_index(timestamp_col)
    return df


# =========================================================
# 6) Main
# =========================================================
if __name__ == "__main__":
    import yfinance as yf

    ticker = "AAPL"

    # If you still see weird columns, you can also try group_by="column" explicitly.
    data = yf.download(
        ticker,
        interval="5m",
        period="30d",
        auto_adjust=True,
        progress=False,
        # group_by="column",  # optional: try uncommenting if your environment behaves oddly
    )

    data = normalize_yfinance_single_ticker(data, ticker)
    data.index.name = "timestamp"

    out = beardy_squeeze_pro(
        data,
        length=20,
        bb_mult=2.0,
        kc_mult_high=1.0,
        kc_mult_mid=1.5,
        kc_mult_low=2.0,
    )

    plot_beardy_squeeze(
        out,
        title=f"{ticker} - Beardy Squeeze Pro (5m)",
        last_n=600,
        save_path="beardy_squeeze.png",  # e.g. "beardy_squeeze.png"
    )
