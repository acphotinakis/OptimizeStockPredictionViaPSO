import torch
from torch.utils.data import Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
    """
    A custom PyTorch Dataset for financial time-series forecasting.

    This class implements a sliding window approach to convert tabular data
    into temporal sequences, allowing the LSTM to learn from a specified
    historical context (lookback window) to predict future movements .
    """

    def __init__(self, data, target, lookback):
        """
        Initializes the dataset with input features and prediction targets.

        Args:
            data (np.ndarray or pd.DataFrame): Feature matrix of shape (N, features).
            target (np.ndarray or pd.Series): Target variable vector (e.g., mid-price movement).
            lookback (int): The number of prior 1-minute intervals to include in each sequence.
        """
        # Convert inputs to torch tensors for GPU-compatibility and gradient tracking
        self.X = torch.tensor(
            data.values if hasattr(data, "values") else data, dtype=torch.float32
        )
        self.y = torch.tensor(
            target.values if hasattr(target, "values") else target, dtype=torch.float32
        )
        self.lookback = lookback

    def __len__(self):
        """
        Calculates the total number of valid sequences.

        The total count is N - lookback, ensuring every window has a
        corresponding next-step target value.
        """
        return len(self.X) - self.lookback

    def __getitem__(self, idx):
        """
        Generates a single sliding window sequence and its target.

        For a given index 'i', the sequence is defined as the slice:
        $$X_{window} = [x_{i}, x_{i+1}, ..., x_{i+lookback-1}]$$

        The corresponding target is the value at the very next interval:
        $$y_{target} = y_{i+lookback}$$

        Returns:
            tuple: (X_sequence, y_target)
        """
        # Extract the historical window
        x_window = self.X[idx : idx + self.lookback]

        # Extract the target for the immediate next time step
        y_label = self.y[idx + self.lookback]

        return x_window, y_label
