import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """
    A configurable LSTM neural network for financial time-series forecasting.

    This model supports varying depths, widths, and regularization settings
    to allow the PSO algorithm to find the optimal architecture for
    high-frequency stock data.
    """

    def __init__(
        self, input_size, hidden_size, num_layers, dropout=0.2, bidirectional=False
    ):
        """
        Initializes the LSTM architecture.

        Args:
            input_size (int): Number of input features per time step .
            hidden_size (int): Number of neurons in each LSTM layer.
            num_layers (int): Number of recurrent layers to stack.
            dropout (float): Dropout probability for regularization.
            bidirectional (bool): If True, becomes a Bidirectional LSTM.
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Define the core LSTM layer
        # batch_first=True ensures compatibility with our (Batch, Seq, Feature) dataset shape
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout if num_layers > 1 else 0
            ),  # Dropout only applies between stacked layers
            bidirectional=bidirectional,
        )

        # The output of a bidirectional LSTM is double the hidden_size
        fc_input_dim = hidden_size * 2 if bidirectional else hidden_size

        # Fully connected layer to map LSTM features to a single mid-price prediction
        self.fc = nn.Linear(fc_input_dim, 1)

        logger.debug(
            f"Initialized LSTM with {num_layers} layers, {hidden_size} hidden units (Bidi: {bidirectional})"
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, features).

        Returns:
            torch.Tensor: Predicted mid-price movement.
        """
        # Initialize hidden and cell states with zeros
        # h0/c0 shape: (num_layers * num_directions, batch, hidden_size)
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.num_layers * num_directions, x.size(0), self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * num_directions, x.size(0), self.hidden_size
        ).to(x.device)

        # Pass input through LSTM layers
        # out: tensor of shape (batch, seq_len, num_directions * hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step only (many-to-one architecture)
        out = self.fc(out[:, -1, :])

        return out
