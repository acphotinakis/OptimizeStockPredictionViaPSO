import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def create_target_variable(df: pd.DataFrame, method: str = "mid_price_return", horizon: int = 1) -> pd.DataFrame:
    """
    Creates the target variable for LSTM prediction.
    
    As specified in the project proposal, the target is the mid-price movement
    at the next time step.
    
    Args:
        df (pd.DataFrame): Dataframe with OHLCV data.
        method (str): Method for target creation. Options:
            - "mid_price_return": (high + low) / 2 return
            - "close_return": Close price return
            - "close_price": Raw close price (for regression)
        horizon (int): Prediction horizon in time steps (default: 1 for next-step prediction).
    
    Returns:
        pd.DataFrame: Original dataframe with 'target' column added.
    """
    logger.info(f"Creating target variable using method: {method}, horizon: {horizon}")
    
    df_with_target = df.copy()
    
    if method == "mid_price_return":
        # Calculate mid-price
        mid_price = (df["high"] + df["low"]) / 2
        
        # Calculate returns (percentage change)
        # target[t] = (mid_price[t+horizon] - mid_price[t]) / mid_price[t]
        df_with_target["target"] = mid_price.pct_change(periods=horizon).shift(-horizon)
        
        logger.info("Target: Mid-price percentage return")
        
    elif method == "close_return":
        # Simple close-to-close return
        df_with_target["target"] = df["close"].pct_change(periods=horizon).shift(-horizon)
        
        logger.info("Target: Close price percentage return")
        
    elif method == "close_price":
        # Raw close price (for direct price prediction)
        df_with_target["target"] = df["close"].shift(-horizon)
        
        logger.info("Target: Raw close price")
        
    else:
        raise ValueError(f"Unknown target method: {method}")
    
    # Drop rows with NaN targets (last 'horizon' rows)
    original_len = len(df_with_target)
    df_with_target = df_with_target.dropna(subset=["target"])
    dropped = original_len - len(df_with_target)
    
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with NaN targets (last {horizon} time steps)")
    
    logger.info(f"Target variable created. Shape: {df_with_target.shape}")
    logger.info(f"Target statistics: mean={df_with_target['target'].mean():.6f}, "
                f"std={df_with_target['target'].std():.6f}, "
                f"min={df_with_target['target'].min():.6f}, "
                f"max={df_with_target['target'].max():.6f}")
    
    return df_with_target


def validate_target(df: pd.DataFrame) -> bool:
    """
    Validates that the target variable is properly formed.
    
    Args:
        df (pd.DataFrame): Dataframe with 'target' column.
    
    Returns:
        bool: True if target is valid, False otherwise.
    """
    if "target" not in df.columns:
        logger.error("Target column not found in dataframe")
        return False
    
    # Check for NaN values
    if df["target"].isna().any():
        logger.error(f"Target contains {df['target'].isna().sum()} NaN values")
        return False
    
    # Check for infinite values
    if np.isinf(df["target"]).any():
        logger.error("Target contains infinite values")
        return False
    
    # Check for reasonable range (returns should typically be < 100%)
    if (df["target"].abs() > 1.0).any():
        logger.warning(f"Target contains extreme values (>{100}% return). Max: {df['target'].max():.2%}")
    
    logger.info("Target validation passed")
    return True
