import logging
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def check_missing_timestamps(df, freq="1min"):
    """
    Checks for gaps in the time-series index based on the expected frequency.

    Args:
        df (pd.DataFrame): The input financial dataframe with a DatetimeIndex.
        freq (str): The expected frequency (e.g., '1min' for intraday data ).

    Returns:
        dict: Summary of missing timestamps and the calculated missing percentage.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("Dataframe must have a DatetimeIndex for timestamp validation.")
        return {"error": "Invalid index type"}

    # Generate the perfect range based on start/end of the current dataframe
    expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    missing_ts = expected_range.difference(df.index)

    missing_pct = (
        len(missing_ts) / len(expected_range) if len(expected_range) > 0 else 0
    )

    return {
        "missing_count": len(missing_ts),
        "missing_pct": missing_pct,
        "missing_indices": missing_ts.tolist()[:10],  # Return first 10 for debugging
    }


def check_duplicates(df):
    """
    Identifies duplicate timestamps in the index.
    """
    duplicate_mask = df.index.duplicated()
    duplicate_count = duplicate_mask.sum()

    return {"duplicate_count": int(duplicate_count), "is_valid": duplicate_count == 0}


def check_missing_values(df):
    """
    Checks for NaN or null values across all OHLCV columns.
    """
    null_counts = df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())

    return {
        "null_counts": null_counts,
        "total_nulls": total_nulls,
        "is_valid": total_nulls == 0,
    }


def detect_outliers(df, method="iqr"):
    """
    Detects anomalies in price returns to flag extreme market shocks or bad data.

    Args:
        df (pd.DataFrame): The input data.
        method (str): "iqr" (Interquartile Range) or "zscore".

    Returns:
        dict: Count of outliers found per column.
    """
    outlier_report = {}

    # We focus on price columns for outlier detection
    cols_to_check = ["open", "high", "low", "close"]

    for col in cols_to_check:
        if col not in df.columns:
            continue

        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        else:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df.iloc[np.where(z_scores > 3)]

        outlier_report[col] = len(outliers)

    return outlier_report


def run_validation_suite(df, cfg):
    """
    Orchestrates all checks and compares against project configuration thresholds.

    Returns:
        dict: A structured report indicating if the data is fit for training.
    """
    # Extract frequency from config (default to '1min' per proposal )
    freq = getattr(cfg.data, "timeframe", "1min")
    max_allowed_missing = cfg.data.validation.max_missing_pct

    logger.info("Running financial data validation suite...")

    ts_report = check_missing_timestamps(df, freq)
    dup_report = check_duplicates(df)
    null_report = check_missing_values(df)
    outlier_report = detect_outliers(df)

    # Final pass/fail logic based on configuration
    failed_checks = []
    if ts_report["missing_pct"] > max_allowed_missing:
        failed_checks.append(
            f"Missing percentage ({ts_report['missing_pct']:.2%}) exceeds limit ({max_allowed_missing:.2%})"
        )

    if not dup_report["is_valid"]:
        failed_checks.append("Duplicate timestamps detected in index")

    is_fit_for_training = len(failed_checks) == 0

    report = {
        "is_fit_for_training": is_fit_for_training,
        "failed_reasons": failed_checks,
        "timestamp_integrity": ts_report,
        "duplicates": dup_report,
        "null_values": null_report,
        "outliers": outlier_report,
    }

    if is_fit_for_training:
        logger.info("Data validation passed. Dataset is ready for feature engineering.")
    else:
        logger.warning(f"Data validation failed: {failed_checks}")

    return report
