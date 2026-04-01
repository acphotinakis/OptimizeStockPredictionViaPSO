import logging
import joblib
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Configure module-level logger
logger = logging.getLogger(__name__)


def fit_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
    """
    Fits a MinMaxScaler on the training data.

    To prevent data leakage, this function must ONLY be called on the
    training split (typically the first 3 years of data).
    The range is set to [-1, 1] to optimize for LSTM 'tanh' and 'sigmoid'
    activation functions.

    Args:
        train_df (pd.DataFrame): The training dataset containing engineered features.

    Returns:
        MinMaxScaler: The fitted scaler object.
    """
    logger.info("Fitting MinMaxScaler on training data (Range: [-1, 1]).")

    # Initialize scaler with the project-standard range [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Fit only on the training data
    scaler.fit(train_df)

    logger.info("Scaler fitting complete.")
    return scaler


def transform_data(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """
    Transforms a dataset using a pre-fitted scaler.

    This function is used for the training, validation, and testing sets
    using the parameters derived from the training fit.

    Args:
        df (pd.DataFrame): The data to scale.
        scaler (MinMaxScaler): A fitted scaler object.

    Returns:
        pd.DataFrame: A new DataFrame with scaled values, preserving index and columns.
    """
    logger.info(f"Transforming dataset of shape {df.shape}.")

    # Transform the data
    scaled_values = scaler.transform(df)

    # Reconstruct DataFrame to maintain column names and DatetimeIndex for downstream modules
    scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)

    return scaled_df


def save_scaler(scaler: MinMaxScaler, path: str):
    """
    Persists the fitted scaler to disk for future inference or test-set reproduction.

    Args:
        scaler (MinMaxScaler): The fitted scaler to save.
        path (str): The destination file path (e.g., 'models/scalers/scaler.joblib').
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(scaler, save_path)
        logger.info(f"Scaler successfully saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save scaler: {str(e)}")
        raise


def load_scaler(path: str) -> MinMaxScaler:
    """
    Loads a saved scaler object from disk.

    Args:
        path (str): Path to the saved .joblib file.

    Returns:
        MinMaxScaler: The loaded scaler object.
    """
    load_path = Path(path)
    if not load_path.exists():
        logger.error(f"Scaler file not found at: {load_path}")
        raise FileNotFoundError(f"No scaler found at {load_path}")

    try:
        scaler = joblib.load(load_path)
        logger.info(f"Scaler loaded from: {load_path}")
        return scaler
    except Exception as e:
        logger.error(f"Failed to load scaler: {str(e)}")
        raise
