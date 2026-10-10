"""ConvLSTM (Shi et al. 2015): a custom cell, stacked, looped over time in Python."""

from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden: int, kernel: int = 3) -> None:
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(in_channels + hidden, 4 * hidden, kernel, padding=kernel // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.split(self.gates(torch.cat([x, h], dim=1)), self.hidden, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """Two stacked ConvLSTM layers over T steps, then a 1×1 convolution on the last state."""

    def __init__(self, channels: int = 1, hidden: int = 8, layers: int = 2) -> None:
        super().__init__()
        self.hidden = hidden
        self.cells = nn.ModuleList(
            [ConvLSTMCell(channels if k == 0 else hidden, hidden) for k in range(layers)]
        )
        self.head = nn.Conv2d(hidden, channels, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:  # B, T, C, H, W
        b, steps, _, height, width = frames.shape
        states = [
            (
                frames.new_zeros(b, self.hidden, height, width),
                frames.new_zeros(b, self.hidden, height, width),
            )
            for _ in self.cells
        ]
        for t in range(steps):
            x = frames[:, t]
            for k, cell in enumerate(self.cells):
                states[k] = cell(x, states[k])
                x = states[k][0]
        return self.head(x)
