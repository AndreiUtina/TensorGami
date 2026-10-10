from __future__ import annotations

import math

import torch
from torch import nn


class TokenTransformer(nn.Module):
    """Token embedding + sinusoidal positions (a buffer) + 2 encoder layers + output head."""

    def __init__(
        self,
        vocab: int = 100,
        d_model: int = 32,
        heads: int = 4,
        layers: int = 2,
        max_len: int = 50,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens) + self.pe[:, : tokens.size(1)]
        return self.head(self.encoder(x))
