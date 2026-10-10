"""Small reference PyTorch models for the six target families (requirements FR9).

``REFERENCE`` maps a name to ``(model factory, example input factory)``. The models are tiny
(CPU, milliseconds) but structurally faithful: residual merges, recurrent loops, custom cells,
conditional inputs, sampling and attention are all present.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from models.cgan import Discriminator, Generator
from models.cnn import SmallCNN
from models.convlstm import ConvLSTMForecaster
from models.lstm import LSTMClassifier
from models.transformer import TokenTransformer
from models.vae import VAE

REFERENCE: dict[str, tuple[Callable[[], torch.nn.Module], Callable[[], Any]]] = {
    "cnn": (SmallCNN, lambda: torch.randn(1, 3, 32, 32)),
    "lstm": (LSTMClassifier, lambda: torch.randn(1, 5, 8)),
    "cgan_generator": (Generator, lambda: (torch.randn(1, 16), torch.tensor([3]))),
    "cgan_discriminator": (
        Discriminator,
        lambda: (torch.randn(1, 1, 16, 16), torch.tensor([3])),
    ),
    "vae": (VAE, lambda: torch.randn(1, 784)),
    "convlstm": (ConvLSTMForecaster, lambda: torch.randn(1, 3, 1, 16, 16)),
    "transformer": (TokenTransformer, lambda: torch.randint(0, 100, (1, 6))),
}
