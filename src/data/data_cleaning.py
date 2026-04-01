import logging
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

logger = logging.getLogger(__name__)


def handle_missing_data(df, short_gap=5, long_gap=15):
    """
    Cleans missing data in a time-series using a two-tiered strategy:
    1. Short Gaps: Forward-fill values to maintain temporal continuity.
    2. Long Gaps: Drop sequences to avoid training on artificial data.

    This approach is inspired by the handling of missing values in high-frequency
    trading data.

    Args:
        df (pd.DataFrame): The input financial data with a DatetimeIndex.
        short_gap (int): Maximum consecutive missing minutes to forward-fill.
        long_gap (int): Consecutive missing minutes beyond which data is dropped.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    if df.empty:
        return df

    logger.info(
        f"Handling missing data (Short gap: {short_gap}m, Long gap: {long_gap}m)"
    )

    # Identify gaps in the index
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1min")
    df_reindexed = df.reindex(full_index)

    # Identify blocks of missing data
    is_null = df_reindexed["close"].isnull()
    null_groups = (is_null != is_null.shift()).cumsum()
    gap_lengths = is_null.groupby(null_groups).transform("sum")

    # Apply tiered strategy
    # 1. Forward-fill short gaps
    df_cleaned = df_reindexed.ffill(limit=short_gap)

    # 2. Re-identify remaining nulls that exceeded short_gap but are below long_gap
    # or just drop everything that remains null after ffill and exceeds long_gap logic
    # In a production pipeline, we often drop rows that still have NaNs after ffill
    final_nulls = df_cleaned["close"].isnull()
    final_gap_lengths = final_nulls.groupby(null_groups).transform("sum")

    # Drop rows belonging to gaps that are longer than the threshold
    df_cleaned = df_cleaned[~(final_nulls & (final_gap_lengths >= long_gap))]

    # Drop any remaining NaNs to ensure model stability
    df_cleaned = df_cleaned.dropna()

    logger.info(f"Cleaned data shape: {df_cleaned.shape}")
    return df_cleaned


def remove_duplicates(df):
    """
    Removes duplicate timestamps from the index to ensure a unique time-series.
    """
    original_len = len(df)
    df_unique = df[~df.index.duplicated(keep="first")]

    removed = original_len - len(df_unique)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate timestamps from the dataset.")

    return df_unique


def handle_outliers(df, method="winsorize"):
    """
    Manages extreme return spikes or anomalies.
    Winsorization is preferred over deletion to preserve the sample size
    and capture high-volatility events without destabilizing LSTM gradients.

    Args:
        df (pd.DataFrame): The input data.
        method (str): "winsorize" to clip values at percentiles.

    Returns:
        pd.DataFrame: Data with adjusted outlier values.
    """
    df_out = df.copy()
    cols_to_clean = ["open", "high", "low", "close", "volume"]

    if method == "winsorize":
        for col in cols_to_clean:
            if col in df_out.columns:
                # Limits [0.01, 0.01] clips the bottom and top 1% of data
                df_out[col] = winsorize(df_out[col], limits=[0.01, 0.01])
        logger.info("Outliers handled using Winsorization (1st and 99th percentiles).")
    else:
        logger.warning(
            f"Outlier method '{method}' not implemented. Returning original data."
        )

    return df_out


def run_cleaning_pipeline(df, cfg):
    """
    Orchestrates the cleaning process based on Hydra configuration settings.

    Args:
        df (pd.DataFrame): Raw ingested data.
        cfg: Hydra configuration (cfg.data.cleaning).
    """
    logger.info("Starting data cleaning pipeline...")

    # 1. Remove duplicates first to ensure index integrity
    df = remove_duplicates(df)

    # 2. Handle missing data using project thresholds
    df = handle_missing_data(
        df,
        short_gap=cfg.data.cleaning.short_gap_threshold,
        long_gap=cfg.data.cleaning.long_gap_threshold,
    )

    # 3. Address anomalies
    df = handle_outliers(df, method=cfg.data.validation.handle_outliers)

    logger.info("Data cleaning complete.")
    return df
