"""PyTorch importer (ADR-0003): ``trace(model, example_input)`` → :class:`tensorgami.ir.Graph`."""

from __future__ import annotations

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "tensorgami.importers.torch needs PyTorch. Install it the way your project does "
        "(e.g. `pip install torch`), or use the extra: `pip install tensorgami[torch]`."
    ) from exc

from tensorgami.importers.torch.importer import TorchImporter, trace

__all__ = ["TorchImporter", "trace"]
