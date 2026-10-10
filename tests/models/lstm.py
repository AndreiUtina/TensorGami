from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """2-layer LSTM over a sequence; classifies from the last layer's final hidden state.

    ``h[-1]`` is plain integer indexing, which torch 2.2 does not report to
    ``__torch_function__``: it exercises the importer's view recovery.
    """

    def __init__(self, features: int = 8, hidden: int = 32, classes: int = 4) -> None:
        super().__init__()
        self.lstm = nn.LSTM(features, hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])
