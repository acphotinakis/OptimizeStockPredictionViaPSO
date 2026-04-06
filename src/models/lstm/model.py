"""
models/lstm/model.py
---------------------
Bidirectional LSTM / GRU model with layer norm, dropout,
and multi-class softmax output.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LSTMClassifier(nn.Module):
    """
    2-layer LSTM (optionally Bidirectional or GRU) for ternary classification.

    Architecture
    ------------
    Input --> LayerNorm --> LSTM/GRU layers --> Dropout --> Dense --> Softmax

    Parameters
    ----------
    input_size : int
        Number of features per timestep.
    hidden_size : int
    num_layers : int
    num_classes : int
        3 for up/neutral/down.
    dropout : float
    bidirectional : bool
    use_gru : bool
        If True, use GRU cells instead of LSTM.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_gru: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.input_norm = nn.LayerNorm(input_size)

        rnn_cls = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        fc_in = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_in, fc_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_in // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_size)

        Returns
        -------
        logits : torch.Tensor, shape (batch, num_classes)
        """
        x = self.input_norm(x)
        out, _ = self.rnn(x)
        # Take last timestep output
        out = out[:, -1, :]
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


class MultiHorizonLSTM(nn.Module):
    """
    Shared LSTM encoder with separate classification heads per horizon.

    Parameters
    ----------
    input_size : int
    hidden_size : int
    num_layers : int
    horizons : list[int]
        e.g. [1, 5, 15] – one output head per horizon.
    num_classes : int
    dropout : float
    bidirectional : bool
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        horizons: Optional[list] = None,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if horizons is None:
            horizons = [1, 5, 15]
        self.horizons = horizons
        num_directions = 2 if bidirectional else 1

        self.input_norm = nn.LayerNorm(input_size)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        fc_in = hidden_size * num_directions

        # One classification head per horizon
        self.heads = nn.ModuleDict(
            {
                f"h{h}": nn.Sequential(
                    nn.Linear(fc_in, fc_in // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fc_in // 2, num_classes),
                )
                for h in horizons
            }
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_size)

        Returns
        -------
        dict mapping horizon name --> logits tensor (batch, num_classes)
        """
        x = self.input_norm(x)
        out, _ = self.rnn(x)
        enc = self.dropout(out[:, -1, :])
        return {
            f"h{h}": head(enc) for h, head in zip(self.horizons, self.heads.values())
        }
