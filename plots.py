import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pathlib import Path
import pandas as pd
import mplfinance as mpf
from pathlib import Path
from src.features.technical import compute_technical_features
from src.features.ttm_squeeze import beardy_squeeze_pro, plot_beardy_squeeze


# -------------------------------------------------
# Utilities for indicators
# -------------------------------------------------
def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def roc(series: pd.Series, n: int) -> pd.Series:
    return series.pct_change(n) * 100


def mfi(high, low, close, volume, n=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    pos_sum = positive_flow.rolling(n).sum()
    neg_sum = negative_flow.rolling(n).sum()
    mfi_val = 100 * pos_sum / (pos_sum + neg_sum)
    return mfi_val


# -------------------------------------------------
# Candlestick plot with indicators
# -------------------------------------------------
def plot_candlestick_with_indicators(
    df, title="Candlestick", width=0.8, save_path=None
):
    fig, ax = plt.subplots(figsize=(18, 6))
    for idx, row in df.iterrows():
        color = "#26A69A" if row["close"] >= row["open"] else "#EF5350"
        ax.vlines(idx, row["low"], row["high"], color="grey", linewidth=1)
        ax.add_patch(
            Rectangle(
                (
                    idx - pd.Timedelta(minutes=width * 30),
                    min(row["open"], row["close"]),
                ),
                pd.Timedelta(minutes=width * 60),
                abs(row["close"] - row["open"]),
                facecolor=color,
                edgecolor=color,
            )
        )
    # Overlay SMA
    ax.plot(df.index, sma(df["close"], 20), color="blue", label="SMA20")
    ax.plot(df.index, sma(df["close"], 50), color="orange", label="SMA50")
    # Bollinger Bands
    bb20 = sma(df["close"], 20)
    bb_std = df["close"].rolling(20).std()
    ax.fill_between(
        df.index,
        bb20 + 2 * bb_std,
        bb20 - 2 * bb_std,
        color="grey",
        alpha=0.2,
        label="Bollinger Bands",
    )
    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    if save_path:
        fig.savefig(save_path, dpi=160)
    plt.show()


# -------------------------------------------------
# ROC plot
# -------------------------------------------------
def plot_roc(df, n=12, title="Rate of Change (ROC)", save_path=None):
    r = roc(df["close"], n)
    plt.figure(figsize=(18, 4))
    plt.plot(df.index, r, label=f"ROC{n}", color="purple")
    plt.axhline(0, color="grey", lw=1)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylabel("% Change")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


# -------------------------------------------------
# MFI plot
# -------------------------------------------------
def plot_mfi(df, n=14, title="Money Flow Index (MFI)", save_path=None):
    m = mfi(df["high"], df["low"], df["close"], df["volume"], n)
    plt.figure(figsize=(18, 4))
    plt.plot(df.index, m, color="green", label=f"MFI{n}")
    plt.axhline(80, color="red", linestyle="--")
    plt.axhline(20, color="blue", linestyle="--")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylabel("MFI")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


plot_dir = Path("plots")
plot_dir.mkdir(exist_ok=True)
processed_dir = Path("data/processed")
tickers = ["AAPL", "NVDA"]

# -------------------------------------------------
# Example usage with your Beardy Squeeze Pro dataframe
# -------------------------------------------------
if __name__ == "__main__":

    for ticker in tickers:
        parquet_file = processed_dir / f"{ticker}.parquet"
        if not parquet_file.exists():
            print(f"{ticker} processed data not found, skipping...")
            continue

        # Load 1-minute OHLCV data
        df = pd.read_parquet(parquet_file)
        df.index = pd.to_datetime(df.index)

        # --- Resample to daily ---
        daily = (
            df.resample("1D")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        # Compute technical indicators
        tech_df = compute_technical_features(daily)

        # --- Moving Averages overlay ---
        ma_overlays = [
            mpf.make_addplot(tech_df["sma_5"], color="blue", width=1, ylabel="SMA/EMA"),
            mpf.make_addplot(tech_df["sma_20"], color="orange", width=1),
            mpf.make_addplot(tech_df["ema_12"], color="green", width=1),
            mpf.make_addplot(tech_df["ema_26"], color="red", width=1),
        ]

        # --- Bollinger Bands overlay ---
        bb_overlays = [
            mpf.make_addplot(
                tech_df["bb_upper"], color="gray", linestyle="--", width=1
            ),
            mpf.make_addplot(
                tech_df["bb_lower"], color="gray", linestyle="--", width=1
            ),
        ]
        ma_overlays.extend(bb_overlays)

        # --- MACD panel ---
        macd_panel = [
            mpf.make_addplot(
                tech_df["macd"], panel=1, color="blue", width=1, ylabel="MACD"
            ),
            mpf.make_addplot(tech_df["macd_signal"], panel=1, color="red", width=1),
            mpf.make_addplot(
                tech_df["macd_hist"], type="bar", panel=1, color="gray", alpha=0.5
            ),
        ]

        # --- RSI panel ---
        rsi_panel = [
            mpf.make_addplot(
                tech_df["rsi_14"], panel=2, color="purple", width=1, ylabel="RSI"
            )
        ]

        all_addplots = ma_overlays + macd_panel + rsi_panel

        # --- Custom Legend ---
        legend_labels = ["SMA 5", "SMA 20", "EMA 12", "EMA 26", "BB Upper", "BB Lower"]

        # --- Plot candlestick chart ---
        candlestick_file = plot_dir / f"{ticker}_daily_candlestick.png"
        mpf.plot(
            daily[["open", "high", "low", "close", "volume"]],
            type="candle",
            style="yahoo",
            volume=True,
            addplot=all_addplots,
            title=f"{ticker} Daily Candlestick with Technical Indicators",
            ylabel="Price",
            figratio=(20, 12),  # Bigger figure
            figscale=1.5,
            tight_layout=True,
            savefig=dict(fname=candlestick_file, dpi=200),
        )

        plot_candlestick_with_indicators(daily, title=f"{ticker} Daily Candlestick")
        plot_roc(daily, n=12, title=f"{ticker} ROC12")
        plot_mfi(daily, n=14, title=f"{ticker} MFI14")
        squeeze_df = beardy_squeeze_pro(daily)
        plot_beardy_squeeze(
            squeeze_df,
            title=f"{ticker} - Beardy Squeeze Pro (Daily)",
            last_n=180,
            save_path=f"plots/{ticker}_ttm_squeeze.png",
        )

        print(f"{ticker} daily candlestick chart saved to {candlestick_file}")
