"""The PyTorch importer: ``trace(model, example_input) → ir.Graph`` (ADR-0003)."""

from __future__ import annotations

import re
import sys
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tensorgami.importers.base import ImporterError, UnclassifiedOpWarning
from tensorgami.importers.torch.catalog import LEAF_MODULES
from tensorgami.importers.torch.convert import ConversionReport, GraphConverter
from tensorgami.importers.torch.isolation import isolated, model_device
from tensorgami.importers.torch.overrides import Overrides
from tensorgami.importers.torch.recorder import Recorder, mro_names
from tensorgami.ir import Graph


class TorchImporter:
    """Captures a ``torch.nn.Module`` by running it once on an example input.

    ``overrides`` sets category, label or leaf-ness per module path or class name (see
    :mod:`~tensorgami.importers.torch.overrides`); ``leaf_types`` adds module classes that are
    drawn as single operations without tracing their internals.
    """

    def __init__(
        self,
        *,
        overrides: Mapping[str, Mapping[str, Any]] | None = None,
        leaf_types: tuple[type[nn.Module], ...] = (),
    ) -> None:
        self._overrides = Overrides(overrides)
        self._leaf_types = tuple(leaf_types)

    def import_model(self, model: nn.Module, example_input: Any) -> Graph:
        if isinstance(model, type) and issubclass(model, nn.Module):
            raise ImporterError(
                f"got the class {model.__name__}, not a model: create it first, "
                f"e.g. {model.__name__}(), and pass that"
            )
        if not isinstance(model, nn.Module):
            raise ImporterError(f"expected a torch.nn.Module, got {type(model).__name__}")
        device = model_device(model)
        example_input = _to_device(example_input, device)
        recorder = Recorder(model, self._is_leaf)
        try:
            with isolated(model), recorder, recorder.mode():
                output = _call(model, example_input)
        except Exception as exc:  # noqa: BLE001 - re-raised with an explanation
            raise ImporterError(_explain(model, example_input, exc)) from exc

        graph, report = GraphConverter(self._overrides).convert(
            recorder,
            list(_labelled(example_input, "input")),
            list(_labelled(output, "output")),
            name=type(model).__name__,
            root_type=type(model).__name__,
            metadata={"framework": "torch", "torch_version": torch.__version__},
        )
        _warn(report)
        return graph

    def _is_leaf(self, module: nn.Module, path: str) -> bool:
        names = mro_names(module)
        override = self._overrides.lookup(path, names)
        if override.leaf is not None:
            return override.leaf
        if self._leaf_types and isinstance(module, self._leaf_types):
            return True
        return any(name in LEAF_MODULES for name in names)


def trace(
    model: nn.Module,
    example_input: Any,
    *,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    leaf_types: tuple[type[nn.Module], ...] = (),
) -> Graph:
    """Capture ``model`` as an IR graph by running it once on ``example_input``.

    ``example_input`` is a tensor, a tuple of positional arguments, or a dict of keyword
    arguments; tensors are moved to the model's device. The model, its training flags and the
    caller's random state are left untouched.
    """
    return TorchImporter(overrides=overrides, leaf_types=leaf_types).import_model(
        model, example_input
    )


# --------------------------------------------------------------------------- helpers

_SHAPE_ERROR = re.compile(r"\b\d+-?d\b|dimension|shape|size mismatch|mat1 and mat2")
_CHANNELS = re.compile(r"to have (\d+) channels, but got (\d+) channels")


def _call(model: nn.Module, example_input: Any) -> Any:
    if isinstance(example_input, Mapping):
        return model(**example_input)
    if isinstance(example_input, tuple):
        return model(*example_input)
    return model(example_input)


def _to_device(obj: Any, device: torch.device | None) -> Any:
    if device is None or device.type == "meta":
        return obj
    if isinstance(obj, torch.Tensor):
        return obj if obj.device == device else obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_to_device(o, device) for o in obj)
    if isinstance(obj, list):
        return [_to_device(o, device) for o in obj]
    if isinstance(obj, Mapping):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def _labelled(obj: Any, prefix: str) -> Iterator[tuple[str, torch.Tensor]]:
    """(label, tensor) pairs: 'input' for a single tensor, 'input 0', 'input 1' for tuples,
    dict keys for keyword arguments and dict-like outputs."""
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            yield from _labelled(value, str(key))
    elif isinstance(obj, (tuple, list)):
        tensors = [o for o in obj if isinstance(o, (torch.Tensor, tuple, list, Mapping))]
        if len(tensors) == 1 and isinstance(tensors[0], torch.Tensor):
            yield prefix, tensors[0]
            return
        for index, item in enumerate(obj):
            yield from _labelled(item, f"{prefix} {index}")


def _shapes(obj: Any) -> str:
    shapes = [f"{label} {tuple(t.shape)}" for label, t in _labelled(obj, "input")]
    return ", ".join(shapes) or "no tensors"


def _explain(model: nn.Module, example_input: Any, exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    lines = [
        f"running {type(model).__name__} on the example input failed:",
        f"  {text.splitlines()[0] if text else type(exc).__name__}",
        f"  example input: {_shapes(example_input)}",
    ]
    lower = text.lower()
    if channels := _CHANNELS.search(lower):
        expected, got = channels.groups()
        lines.append(
            f"  hint: a layer expects {expected} input channels but gets {got}: check the "
            "channel dimension (dim 1). A model that takes several images stacked along the "
            "channels needs them concatenated, e.g. torch.cat([a, b], 1)"
        )
    elif _SHAPE_ERROR.search(lower):
        lines.append(
            "  hint: check the input shape; a missing batch dimension is the most "
            "common cause, e.g. pass x.unsqueeze(0)"
        )
    if "long" in lower or ("int" in lower and "index" in lower):
        lines.append("  hint: the model may expect integer token ids: use torch.long tensors")
    if "same device" in lower:
        lines.append("  hint: move the example input to the model's device")
    if isinstance(example_input, list):
        lines.append("  note: a list is passed as one argument; use a tuple for several")
    return "\n".join(lines)


_PACKAGE = str(Path(__file__).parents[2])  # src/tensorgami


def _user_stacklevel() -> int:
    """The ``stacklevel`` that points a warning from :func:`_warn` at the first frame outside
    Tensorgami: the user's ``tg.draw(...)`` or ``trace(...)`` line."""
    level, frame = 1, sys._getframe(1)  # stacklevel 1 is the caller of warnings.warn: _warn
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        level, frame = level + 1, frame.f_back
    return level


def _warn(report: ConversionReport) -> None:
    stacklevel = _user_stacklevel()
    if report.unclassified:
        listed = ", ".join(f"{op} ×{n}" for op, n in sorted(report.unclassified.items()))
        warnings.warn(
            UnclassifiedOpWarning(
                f"{len(report.unclassified)} operation type(s) are not in Tensorgami's "
                f"catalogue and are drawn as 'other': {listed}. Set a category with "
                f"overrides={{'<module or class>': {{'category': '...'}}}} if they matter."
            ),
            stacklevel=stacklevel,
        )
    if report.fused:
        warnings.warn(
            UnclassifiedOpWarning(
                f"fused operation(s) {sorted(report.fused)} hide the structure inside them "
                "(e.g. the Transformer fast path); the figure shows them as single boxes."
            ),
            stacklevel=stacklevel,
        )
