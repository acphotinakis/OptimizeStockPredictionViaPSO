#!/usr/bin/env python3
"""
Grouped visualizations for ticker features.
Generates heatmaps, grouped distributions, grouped boxplots,
correlation maps, and rolling stats with subplots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import pandas as pd

# Config
tickers = ["AAPL", "NVDA"]
features_dir = Path("data/features")
plot_dir = Path("plots")
plot_dir.mkdir(exist_ok=True)

# Define feature groups for logical plotting
feature_groups = {
    "price": ["log_return", "true_range", "rv_10", "rv_30", "rvol_20"],
    "momentum": ["cci_20", "force_1", "co_ratio"],
    "stats": ["ret_skew_20", "ret_skew_60", "ret_kurt_60", "hurst_exp_60"],
    "market": ["corr_spy_60", "peer_corr_NVDA", "peer_corr_AMD", "mkt_breadth"],
}


# --- Functions ---
def plot_grouped_distributions(df, ticker, groups):
    for group_name, cols in groups.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        n = len(cols)
        fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            sns.histplot(df[col], kde=True, bins=50, ax=ax)
            ax.set_title(f"{ticker} - {col} Distribution")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{ticker}_dist_{group_name}.png")
        plt.close()


def plot_grouped_boxplots(df, ticker, groups):
    for group_name, cols in groups.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[cols])
        plt.xticks(rotation=45)
        plt.title(f"{ticker} - {group_name.capitalize()} Features Boxplot")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{ticker}_boxplot_{group_name}.png")
        plt.close()


def plot_grouped_rolling_stats(df, ticker, groups, window=20):
    for group_name, cols in groups.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        n = len(cols)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            rolling_mean = df[col].rolling(window).mean()
            rolling_std = df[col].rolling(window).std()
            ax.plot(df[col], label="Original", alpha=0.5)
            ax.plot(rolling_mean, label=f"{window}-step Mean", color="green")
            ax.plot(rolling_std, label=f"{window}-step Std", color="red")
            ax.set_title(f"{ticker} - {col} Rolling Stats")
            ax.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{ticker}_rolling_{group_name}.png")
        plt.close()


def plot_correlation_heatmap(df, ticker):
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(f"{ticker} - Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{ticker}_correlation.png")
    plt.close()


# --- Main Loop ---
for ticker in tickers:
    ticker_dir = features_dir / ticker
    if not ticker_dir.exists():
        print(f"{ticker} data not found, skipping...")
        continue

    # Load data
    X_train = np.load(ticker_dir / "X_train.npy")
    with open(ticker_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    feature_names = metadata["feature_names"]

    print(f"\n=== {ticker} ===")
    print("Metadata:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")
    print("\nTraining data (first 5 samples):")
    print(X_train[:5])

    # Standardize features for heatmap
    X_std = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-10)

    # # --- Feature heatmap ---
    # plt.figure(figsize=(14, 6))
    # im = plt.imshow(X_std.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    # plt.colorbar(im, label="Standardized value (z-score)")
    # plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
    # plt.xlabel("Samples")
    # plt.title(f"{ticker} Feature Matrix (Standardized)")
    # plt.tight_layout()
    # plt.savefig(plot_dir / f"{ticker}_features.png")
    # plt.close()

    # # Convert to DataFrame for grouped seaborn plots
    # df = pd.DataFrame(X_train, columns=feature_names)

    # # --- Grouped distributions ---
    # plot_grouped_distributions(df, ticker, feature_groups)

    # # --- Grouped boxplots ---
    # plot_grouped_boxplots(df, ticker, feature_groups)

    # # --- Correlation heatmap ---
    # plot_correlation_heatmap(df, ticker)

    # # --- Grouped rolling stats ---
    # plot_grouped_rolling_stats(df, ticker, feature_groups)

print("All grouped plots saved to 'plots/' directory.")


import pandas as pd
import mplfinance as mpf
from pathlib import Path
from src.features.technical import compute_technical_features

plot_dir = Path("plots")
plot_dir.mkdir(exist_ok=True)
processed_dir = Path("data/processed")
tickers = ["AAPL", "NVDA"]

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
        mpf.make_addplot(tech_df["bb_upper"], color="gray", linestyle="--", width=1),
        mpf.make_addplot(tech_df["bb_lower"], color="gray", linestyle="--", width=1),
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

    print(f"{ticker} daily candlestick chart saved to {candlestick_file}")

# #!/usr/bin/env python3
# """
# Comprehensive visualization for ticker features.
# Generates heatmaps, distributions, boxplots, correlation maps, and rolling stats.
# """

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import pickle
# import pandas as pd

# # Config
# tickers = ["AAPL", "NVDA"]
# features_dir = Path("data/features")
# plot_dir = Path("plots")
# plot_dir.mkdir(exist_ok=True)


# # Function for univariate distribution plots
# def plot_distributions(df, ticker):
#     for col in df.columns:
#         plt.figure(figsize=(6, 4))
#         sns.histplot(df[col], kde=True, bins=50)
#         plt.title(f"{ticker} - {col} Distribution")
#         plt.tight_layout()
#         plt.savefig(plot_dir / f"{ticker}_{col}_dist.png")
#         plt.close()


# # Function for boxplots
# def plot_boxplots(df, ticker):
#     plt.figure(figsize=(12, 6))
#     sns.boxplot(data=df)
#     plt.xticks(rotation=45)
#     plt.title(f"{ticker} - Feature Boxplots")
#     plt.tight_layout()
#     plt.savefig(plot_dir / f"{ticker}_boxplot.png")
#     plt.close()


# # Function for correlation heatmaps
# def plot_correlation_heatmap(df, ticker):
#     corr = df.corr()
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
#     plt.title(f"{ticker} - Feature Correlation Heatmap")
#     plt.tight_layout()
#     plt.savefig(plot_dir / f"{ticker}_correlation.png")
#     plt.close()


# # Function for rolling statistics
# def plot_rolling_stats(df, ticker, window=20):
#     for col in df.columns:
#         rolling_mean = df[col].rolling(window).mean()
#         rolling_std = df[col].rolling(window).std()
#         plt.figure(figsize=(10, 4))
#         plt.plot(df[col], label="Original", alpha=0.5)
#         plt.plot(rolling_mean, label=f"{window}-step Mean", color="green")
#         plt.plot(rolling_std, label=f"{window}-step Std", color="red")
#         plt.title(f"{ticker} - {col} Rolling Stats")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(plot_dir / f"{ticker}_{col}_rolling.png")
#         plt.close()


# # Main loop over tickers
# for ticker in tickers:
#     ticker_dir = features_dir / ticker
#     if not ticker_dir.exists():
#         print(f"{ticker} data not found, skipping...")
#         continue

#     # Load data
#     X_train = np.load(ticker_dir / "X_train.npy")
#     with open(ticker_dir / "metadata.pkl", "rb") as f:
#         metadata = pickle.load(f)
#     feature_names = metadata["feature_names"]

#     print(f"\n=== {ticker} ===")
#     print("Metadata:")
#     for k, v in metadata.items():
#         print(f"  {k}: {v}")

#     print("\nTraining data (first 5 samples):")
#     print(X_train[:5])

#     # Standardize features for heatmap
#     X_std = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-10)

#     # --- Feature heatmap ---
#     plt.figure(figsize=(14, 6))
#     im = plt.imshow(X_std.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
#     plt.colorbar(im, label="Standardized value (z-score)")
#     plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
#     plt.xlabel("Samples")
#     plt.title(f"{ticker} Feature Matrix (Standardized)")
#     plt.tight_layout()
#     plt.savefig(plot_dir / f"{ticker}_features.png")
#     plt.close()

#     # Convert to DataFrame for seaborn plots
#     df = pd.DataFrame(X_train, columns=feature_names)

#     # --- Univariate distributions ---
#     plot_distributions(df, ticker)

#     # --- Boxplots ---
#     plot_boxplots(df, ticker)

#     # --- Correlation heatmap ---
#     plot_correlation_heatmap(df, ticker)

#     # --- Rolling stats ---
#     plot_rolling_stats(df, ticker)

# print("All plots saved to 'plots/' directory.")
# # #!/usr/bin/env python3
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from pathlib import Path
# # import pickle

# # # List of tickers to plot
# # tickers = [
# #     "AAPL",
# #     "NVDA",
# # ]

# # features_dir = Path("data/features")
# # PLOT_DIR = Path("plots")
# # PLOT_DIR.mkdir(exist_ok=True)

# # for ticker in tickers:
# #     ticker_dir = features_dir / ticker
# #     if not ticker_dir.exists():
# #         print(f"{ticker} data not found, skipping...")
# #         continue

# #     # Load data
# #     X_train = np.load(ticker_dir / "X_train.npy")
# #     with open(ticker_dir / "metadata.pkl", "rb") as f:
# #         metadata = pickle.load(f)
# #     feature_names = metadata["feature_names"]

# #     print(f"\n=== {ticker} ===")
# #     print("Metadata:")
# #     for k, v in metadata.items():
# #         print(f"  {k}: {v}")

# #     print("\nTraining data (first 5 samples):")
# #     print(X_train[:5])

# #     # Standardize features for better visualization
# #     X_std = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-10)

# #     # Plot
# #     plt.figure(figsize=(14, 6))
# #     im = plt.imshow(X_std.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
# #     plt.colorbar(im, label="Standardized value (z-score)")
# #     plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
# #     plt.xlabel("Samples")
# #     plt.title(f"{ticker} Feature Matrix (Standardized)")
# #     plt.tight_layout()
# #     plot_file = PLOT_DIR / f"{ticker}_features.png"
# #     plt.savefig(plot_file, dpi=150)
# #     plt.show()
# # # #!/usr/bin/env python3
# # # """
# # # Plot feature matrices for each stock ticker.
# # # Loads X_train and metadata, then generates a heatmap for visual inspection.
# # # """

# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from pathlib import Path
# # # import pickle

# # # # Paths
# # # FEATURE_DIR = Path("data/features")
# # # PLOT_DIR = Path("plots")
# # # PLOT_DIR.mkdir(exist_ok=True)

# # # # Only plot these tickers
# # # TICKERS = ["AAPL", "NVDA"]


# # # # Iterate over all tickers in features folder
# # # for ticker_dir in FEATURE_DIR.iterdir():
# # #     if not ticker_dir.is_dir():
# # #         continue

# # #     ticker = ticker_dir.name
# # #     if ticker not in TICKERS:
# # #         continue  # Skip tickers not in the list

# # #     x_train_path = ticker_dir / "X_train.npy"
# # #     metadata_path = ticker_dir / "metadata.pkl"

# # #     if not x_train_path.exists() or not metadata_path.exists():
# # #         print(f"Skipping {ticker} - missing data")
# # #         continue

# # #     # Load training data and metadata
# # #     X_train = np.load(x_train_path)
# # #     with open(metadata_path, "rb") as f:
# # #         metadata = pickle.load(f)

# # #     # --- Print values ---
# # #     print(f"\n=== {ticker} ===")
# # #     print("Metadata:")
# # #     for key, value in metadata.items():
# # #         print(f"  {key}: {value}")

# # #     print("\nTraining data (first 5 samples):")
# # #     print(X_train[:5, :])  # print first 5 rows for readability

# # #     # --- Plotting ---
# # #     feature_names = metadata["feature_names"]
# # #     plt.figure(figsize=(12, 6))
# # #     plt.imshow(X_train.T, aspect="auto", cmap="viridis")
# # #     plt.colorbar(label="Feature Value")
# # #     plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
# # #     plt.xlabel("Samples")
# # #     plt.title(f"{ticker} Feature Matrix")
# # #     plt.tight_layout()

# # #     # Save plot
# # #     plot_file = PLOT_DIR / f"{ticker}_features.png"
# # #     plt.savefig(plot_file, dpi=150)
# # #     plt.close()
# # #     print(f"Saved plot for {ticker} -> {plot_file}")
