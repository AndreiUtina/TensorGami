"""Summaries of tensors as the model runs (roadmap step 14d), made the moment each operation
returns: an in-place operation later in the pass (``ReLU(inplace=True)``, ``out += x``) would
otherwise change what an earlier one seemed to produce."""

from __future__ import annotations

import torch

from tensorgami.activations import Activation, summarize


def activation(tensor: torch.Tensor, *, image: bool = False) -> Activation | None:
    """A tensor's summary (:func:`~tensorgami.activations.summarize`); ``None`` for a tensor
    without values (on the meta device, sparse, empty)."""
    if not isinstance(tensor, torch.Tensor):
        return None
    with torch._C.DisableTorchFunction():  # not part of the model: unseen by the recorder
        if tensor.is_meta or tensor.is_sparse or tensor.numel() == 0:
            return None
        values = tensor.detach()
        if values.is_complex():
            values = values.abs()
        values = values.to("cpu", torch.float32).numpy()
    return summarize(values, image=image)
