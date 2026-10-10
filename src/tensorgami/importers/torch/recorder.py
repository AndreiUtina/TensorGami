"""Records one forward pass: module invocations (hooks) and tensor operations
(``TorchFunctionMode``). Recording only; turning records into a graph is ``convert.py``."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.overrides import TorchFunctionMode

from tensorgami.importers.torch.catalog import normalize_op_name
from tensorgami.ir import ROOT, invocation_id

LeafPolicy = Callable[[nn.Module, str], bool]
_SCALARS = (int, float, bool, str)


def iter_tensors(obj: Any) -> Iterator[torch.Tensor]:
    """Tensors inside nested tuples, lists and dicts, in order."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from iter_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from iter_tensors(item)


def mro_names(obj: object) -> tuple[str, ...]:
    return tuple(cls.__name__ for cls in type(obj).__mro__)


@dataclass
class GroupRecord:
    id: str
    type_name: str
    parent: str
    module: str
    call_index: int
    mro: tuple[str, ...]


@dataclass
class OpRecord:
    op: str
    is_module: bool
    group: str
    module: str  # path of the leaf module ("" for functional ops)
    call_index: int
    inputs: list[torch.Tensor]  # data inputs, parameters and buffers excluded
    outputs: list[torch.Tensor]
    weights: tuple[str, ...] = ()  # names of parameters/buffers read by a functional op
    scalars: dict[str, Any] = field(default_factory=dict)
    mro: tuple[str, ...] = ()
    module_obj: nn.Module | None = None


@dataclass
class _Frame:
    module: nn.Module
    kind: str  # "group" | "leaf" | "inner" (inside a leaf: not recorded)
    group: str = ROOT
    call_index: int = 0


class Recorder:
    """Install with ``with recorder:``; run the model inside ``recorder.mode()``."""

    def __init__(self, model: nn.Module, is_leaf: LeafPolicy) -> None:
        self.model = model
        self._is_leaf = is_leaf
        self._paths: dict[int, str] = {}
        for path, module in model.named_modules():
            self._paths.setdefault(id(module), path)
        self._weights: dict[int, str] = {}
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
            self._weights.setdefault(id(tensor), name)
        self._calls: Counter[int] = Counter()
        self._stack: list[_Frame] = []
        self._handles: list[Any] = []
        self.groups: list[GroupRecord] = []
        self.ops: list[OpRecord] = []

    # ------------------------------------------------------------------ hooks

    def __enter__(self) -> Recorder:
        for module in self.model.modules():
            self._handles.append(module.register_forward_pre_hook(self._pre, with_kwargs=True))
            self._handles.append(
                module.register_forward_hook(self._post, with_kwargs=True, always_call=True)
            )
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def mode(self) -> TorchFunctionMode:
        return _FunctionMode(self)

    @property
    def _inside_leaf(self) -> bool:
        return any(f.kind in ("leaf", "inner") for f in self._stack)

    @property
    def _group(self) -> str:
        for frame in reversed(self._stack):
            if frame.kind == "group":
                return frame.group
        return ROOT

    def _pre(self, module: nn.Module, args: tuple, kwargs: dict) -> None:
        if self._inside_leaf:
            self._stack.append(_Frame(module, "inner"))
            return
        call = self._calls[id(module)]
        self._calls[id(module)] += 1
        path = self._paths.get(id(module), "")
        if self._is_leaf(module, path):
            self._stack.append(_Frame(module, "leaf", self._group, call))
            return
        if module is self.model and call == 0:
            group_id = ROOT
        else:
            group_id = invocation_id(path or type(module).__name__, call)
            self.groups.append(
                GroupRecord(
                    group_id, type(module).__name__, self._group, path, call, mro_names(module)
                )
            )
        self._stack.append(_Frame(module, "group", group_id, call))

    def _post(self, module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
        frame = self._stack.pop()
        if frame.kind != "leaf" or output is None:  # output is None when forward raised
            return
        path = self._paths.get(id(module), "")
        self.ops.append(
            OpRecord(
                op=type(module).__name__,
                is_module=True,
                group=frame.group,
                module=path,
                call_index=frame.call_index,
                inputs=self._data_tensors((args, kwargs)),
                outputs=list(iter_tensors(output)),
                mro=mro_names(module),
                module_obj=module,
            )
        )

    # ------------------------------------------------------------------ functions

    def on_function(self, func: Any, args: tuple, kwargs: dict, output: Any) -> None:
        if self._inside_leaf:
            return
        name = _function_name(func)
        outputs = list(iter_tensors(output))
        if name == "setitem" and args and isinstance(args[0], torch.Tensor):
            outputs = [args[0]]  # x[...] = y mutates x: x now depends on y
        if not outputs:
            return
        everything = list(iter_tensors((args, kwargs)))
        weights = tuple(
            dict.fromkeys(self._weights[id(t)] for t in everything if id(t) in self._weights)
        )
        self.ops.append(
            OpRecord(
                op=name,
                is_module=False,
                group=self._group,
                module="",
                call_index=0,
                inputs=[t for t in everything if id(t) not in self._weights],
                outputs=outputs,
                weights=weights,
                scalars=_scalars(args, kwargs),
            )
        )

    def _data_tensors(self, obj: Any) -> list[torch.Tensor]:
        return [t for t in iter_tensors(obj) if id(t) not in self._weights]


class _FunctionMode(TorchFunctionMode):
    def __init__(self, recorder: Recorder) -> None:
        super().__init__()
        self._recorder = recorder

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        self._recorder.on_function(func, args, kwargs, output)
        return output


def _function_name(func: Any) -> str:
    name = getattr(func, "__name__", "") or type(func).__name__
    if name == "__get__":  # properties such as .T or .mT arrive as descriptor getters
        name = getattr(getattr(func, "__self__", None), "__name__", "attribute")
    return normalize_op_name(name)


def _scalars(args: tuple, kwargs: dict) -> dict[str, Any]:
    """Plain Python settings of a functional call (``dim=1``, ``p=0.1``, split sizes, …)."""
    out: dict[str, Any] = {}
    positional = [a for a in args if isinstance(a, _SCALARS)]
    if positional:
        out["args"] = tuple(positional)
    for key, value in kwargs.items():
        if isinstance(value, _SCALARS):
            out[key] = value
    return out
