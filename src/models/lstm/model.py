"""
src/models/lstm/model.py

LSTMModel — PyTorch nn.Module only.
No training loop, no data loading.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Stacked LSTM with many-to-one output.

    Args:
        input_size:        Feature dimension F.
        num_layers:        Number of stacked LSTM layers.
        hidden_units:      Hidden state dimension.
        dropout:           Dropout rate (applied between layers if num_layers > 1).
        output_size:       Output dimension (1 for regression).
        use_checkpointing: Enable gradient checkpointing to save GPU memory.
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        hidden_units: int,
        dropout: float,
        output_size: int = 1,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.dropout_rate = dropout
        self.use_checkpointing = use_checkpointing

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_units, output_size)
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier / orthogonal init; forget-gate bias = 1 (Jozefowicz et al.)."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.zero_()
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            lstm_out, _ = checkpoint(self.lstm, x, use_reentrant=False)
        else:
            lstm_out, _ = self.lstm(x)

        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def architecture_summary(self) -> dict:
        """Return architecture metadata for serialisation."""
        return {
            "input_size": self.input_size,
            "num_layers": self.num_layers,
            "hidden_units": self.hidden_units,
            "dropout": self.dropout_rate,
            "use_checkpointing": self.use_checkpointing,
        }
