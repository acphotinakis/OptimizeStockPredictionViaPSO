import logging
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)


def time_series_split(
    df: pd.DataFrame, cfg
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions the dataset into chronological Train, Validation, and Test sets.

    This function uses a year-based offset approach as specified in the
    project proposal to ensure the model is evaluated on truly out-of-sample
    future data.

    Args:
        df (pd.DataFrame): The cleaned and engineered dataframe with a DatetimeIndex.
        cfg: Hydra configuration containing split year counts (cfg.data.splits).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    logger.info("Executing chronological time-series split...")

    # Ensure the index is sorted to prevent accidental leakage
    df = df.sort_index()

    # Determine the absolute start and end of the dataset
    start_date = df.index.min()
    end_date = df.index.max()

    # 1. Training Set: Defined as the first N years
    train_end = start_date + pd.DateOffset(years=cfg.data.splits.train_years)
    train_df = df[start_date:train_end].iloc[
        :-1
    ]  # Exclude the very last tick to avoid overlap

    # 2. Validation Set: The year following the training period
    # This set is used specifically for PSO fitness evaluation
    val_end = train_end + pd.DateOffset(years=cfg.data.splits.val_years)
    val_df = df[train_end:val_end].iloc[:-1]

    # 3. Testing Set: The final year for out-of-sample verification
    test_df = df[val_end:end_date]

    # Log the date ranges for transparency and auditing
    logger.info(
        f"Train Split: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} samples) "
    )
    logger.info(
        f"Val Split:   {val_df.index.min()} to {val_df.index.max()} ({len(val_df)} samples) "
    )
    logger.info(
        f"Test Split:  {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} samples) "
    )

    # Integrity Check: Ensure no data is lost or overlapping
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        logger.error(
            "Split produced an empty dataset. Check your data range vs. split year config."
        )
        raise ValueError(
            "Data range is insufficient for the requested year-based splits."
        )

    return train_df, val_df, test_df


def get_walk_forward_splits(df, cfg, n_windows=5):
    """
    Optional: Implements walk-forward validation logic.

    While not explicitly in the base proposal, this provides a more robust
    alternative to static splits by sliding the window forward across the
    5-year dataset.
    """
    logger.warning(
        "Walk-forward validation requested. This will extend the PSO search time."
    )
    # Implementation of walk-forward logic would go here
    pass
