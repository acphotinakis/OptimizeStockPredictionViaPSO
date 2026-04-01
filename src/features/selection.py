import logging
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def correlation_filter(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Removes highly correlated features using Pearson correlation filtering.

    This technique identifies pairs of features that provide redundant
    information to the LSTM. Reducing redundancy helps prevent overfitting
    and speeds up the PSO optimization process.

    Args:
        df (pd.DataFrame): Dataframe containing engineered features.
        threshold (float): Pearson correlation coefficient threshold (e.g., 0.95).

    Returns:
        pd.DataFrame: Dataframe with redundant features removed.
    """
    logger.info(f"Applying Pearson correlation filter (Threshold: {threshold})...")

    # Calculate absolute correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if to_drop:
        logger.info(f"Dropped {len(to_drop)} redundant features: {to_drop}")
        df_filtered = df.drop(columns=to_drop)
    else:
        logger.info("No redundant features found above threshold.")
        df_filtered = df.copy()

    return df_filtered


def xgboost_importance_filter(
    df: pd.DataFrame, target: pd.Series, top_n: int = 20
) -> pd.DataFrame:
    """
    Optional: Uses an XGBoost regressor to rank features by gain/importance.

    As proposed in the project design, XGBoost serves as a robust non-deep
    learning baseline. Using it here allows us to 'pre-filter'
    the 100+ generated features down to the most impactful candidates.

    Args:
        df (pd.DataFrame): Feature set.
        target (pd.Series): The target mid-price movement.
        top_n (int): Number of top features to retain.

    Returns:
        pd.DataFrame: Dataframe containing only the top N features.
    """
    logger.info(f"Ranking features via XGBoost (Selecting top {top_n})...")

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(df, target)

    # Get feature importance and sort
    importances = pd.Series(model.feature_importances_, index=df.columns)
    top_features = importances.sort_values(ascending=False).head(top_n).index.tolist()

    logger.info(f"Selected top features: {top_features}")
    return df[top_features]


def run_selection_pipeline(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Orchestrates feature selection based on Hydra configuration.

    Args:
        df (pd.DataFrame): Dataframe with all engineered features.
        cfg: Hydra configuration (cfg.features.selection).
    """
    # Pearson filter is the primary requirement
    df = correlation_filter(df, cfg.pearson_threshold)

    # The target variable is 'target_return' or similar, ensure it's handled
    # if using model-based importance
    if cfg.use_xgboost_importance and "target" in df.columns:
        target = df["target"]
        features = df.drop(columns=["target"])
        df_selected = xgboost_importance_filter(features, target)
        # Re-attach target for the next pipeline stage
        df = pd.concat([df_selected, target], axis=1)

    return df
