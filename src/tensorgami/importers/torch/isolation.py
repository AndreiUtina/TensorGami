"""Run a model without side effects on it or on the caller (NFR3)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn


def model_device(model: nn.Module) -> torch.device | None:
    for param in model.parameters():
        return param.device
    for buffer in model.buffers():
        return buffer.device
    return None


@contextmanager
def isolated(model: nn.Module) -> Iterator[None]:
    """No gradients, eval mode (BatchNorm statistics stay put), and a forked RNG (sampling in
    the model does not advance the caller's random state). Every module's training flag is
    restored afterwards, even if the model raises."""
    flags = [(m, m.training) for m in model.modules()]
    cuda = sorted(
        {
            t.device.index or 0
            for t in [*model.parameters(), *model.buffers()]
            if t.device.type == "cuda"
        }
    )
    model.eval()
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=cuda):
            yield
    finally:
        for module, training in flags:
            module.training = training
