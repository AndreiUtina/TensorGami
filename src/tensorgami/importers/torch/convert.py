"""Turns a :class:`Recorder`'s records into an IR :class:`Graph`.

Data flow is recovered from tensor identity. Two PyTorch-specific gaps are closed here:

* **untraced views** — basic indexing such as ``x[0]`` or ``h[-1]`` bypasses
  ``__torch_function__`` in torch 2.2. A tensor with no recorded producer that shares storage
  with a recorded tensor gets a synthetic ``getitem`` node, so the flow stays connected and the
  consumer still sees the right shape;
* **shape-only reads** — ``randn_like(mu)`` or ``zeros_like(h)`` become sources without data
  edges (ADR-0002).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from tensorgami.activations import Activation
from tensorgami.importers.torch.capture import activation
from tensorgami.importers.torch.catalog import (
    FUSED,
    SHAPE_ONLY,
    function_category,
    guess_group_category,
    module_category,
    module_param_names,
)
from tensorgami.importers.torch.overrides import Overrides
from tensorgami.importers.torch.recorder import OpRecord, Recorder
from tensorgami.ir import Graph, GraphBuilder, ParamValue, Shape
from tensorgami.ir.activations import ActivationStore
from tensorgami.vocabulary import EdgeKind, OpCategory

Port = tuple[str, int]  # (node id, output index)


@dataclass
class ConversionReport:
    unclassified: Counter[str] = field(default_factory=Counter)
    fused: set[str] = field(default_factory=set)
    recovered_views: int = 0
    constants: int = 0


def shape_of(t: torch.Tensor) -> Shape:
    return tuple(int(d) for d in t.shape)


def to_param(value: Any) -> ParamValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list, torch.Size)):
        return tuple(to_param(v) for v in value)
    return str(value)


class GraphConverter:
    def __init__(self, overrides: Overrides, *, activations: bool = False) -> None:
        self._overrides = overrides
        self._capture = activations
        self.activations: dict[str, Activation] = {}  # node id → its output's summary

    def convert(
        self,
        recorder: Recorder,
        inputs: Sequence[tuple[str, torch.Tensor]],
        outputs: Sequence[tuple[str, torch.Tensor]],
        *,
        name: str,
        root_type: str,
        metadata: dict[str, str],
    ) -> tuple[Graph, ConversionReport]:
        self._b = GraphBuilder(name, root_type=root_type, metadata=metadata)
        self._report = ConversionReport()
        self._producer: dict[int, Port] = {}
        self._by_storage: dict[int, Port] = {}
        self._keep: list[torch.Tensor] = []  # ids stay unique while we hold the tensors

        for g in recorder.groups:
            override = self._overrides.lookup(g.module, g.mro)
            self._b.group(
                g.id,
                g.type_name,
                g.parent,
                module=g.module,
                call_index=g.call_index,
                label=override.label or "",
                category=override.category or guess_group_category(g.type_name),
            )
        for label, tensor in inputs:
            node = self._b.input(shape_of(tensor), label=label)
            self._summarize(node, tensor, image=True)
            self._produce(tensor, (node, 0))
        for record in recorder.ops:
            self._add_op(record)
        for label, tensor in outputs:
            port = self._resolve(tensor, group="")
            if port is not None:
                node = self._b.output(port[0], port=port[1], label=label)
                self._summarize(node, tensor)
        return self._b.build(), self._report

    # ------------------------------------------------------------------ operations

    def _add_op(self, rec: OpRecord) -> None:
        if rec.op in FUSED:
            self._report.fused.add(rec.op)
        shape_only = not rec.is_module and rec.op in SHAPE_ONLY
        sources: list[Port] = []
        if not shape_only:
            for tensor in rec.inputs:
                port = self._resolve(tensor, group=rec.group)
                if port is not None and port not in sources:  # e.g. attention(x, x, x)
                    sources.append(port)
        category, label, params = self._describe(rec, tensor_inputs=len(sources))
        node = self._b.node(
            rec.op,
            category=category,
            label=label,
            group=rec.group,
            call_index=rec.call_index,
            params=params,
            outputs=[shape_of(t) for t in rec.outputs],
            module=rec.module,
        )
        if self._capture and rec.summary is not None:
            self.activations[node] = rec.summary
        for src, out_port in sources:
            noise = self._b.get(src).category is OpCategory.STOCHASTIC
            kind = EdgeKind.STOCHASTIC if noise else EdgeKind.DATA
            self._b.edge(src, node, kind, source_port=out_port)
        for index, tensor in enumerate(rec.outputs):
            self._produce(tensor, (node, index))

    def _describe(self, rec: OpRecord, tensor_inputs: int) -> tuple[OpCategory, str, dict]:
        if rec.is_module:
            override = self._overrides.lookup(rec.module, rec.mro)
            category = override.category or module_category(rec.mro) or guess_group_category(rec.op)
            if category is None:
                category = OpCategory.OTHER
                self._report.unclassified[rec.op] += 1
            params = {
                k: to_param(getattr(rec.module_obj, k))
                for k in module_param_names(rec.mro)
                if hasattr(rec.module_obj, k)
            }
            return category, override.label or "", params
        category, known = function_category(rec.op, tensor_inputs)
        if not known:
            self._report.unclassified[rec.op] += 1
        params: dict[str, ParamValue] = {k: to_param(v) for k, v in rec.scalars.items()}
        if rec.weights:
            params["weights"] = rec.weights
        return category, "", params

    # ------------------------------------------------------------------ data flow

    def _produce(self, tensor: torch.Tensor, port: Port) -> None:
        self._keep.append(tensor)
        self._producer[id(tensor)] = port
        key = _storage_key(tensor)
        if key is not None:
            self._by_storage[key] = port

    def _resolve(self, tensor: torch.Tensor, group: str) -> Port | None:
        """Where does this tensor come from? Recorded producer, else a recovered view of a
        recorded tensor, else a constant created outside the trace."""
        port = self._producer.get(id(tensor))
        if port is not None:
            return port
        key = _storage_key(tensor)
        base = self._by_storage.get(key) if key is not None else None
        if base is not None:
            view = self._b.node(
                "getitem",
                category=OpCategory.RESHAPE,
                group=group,
                params={"recovered": True},
                outputs=[shape_of(tensor)],
            )
            self._b.edge(base[0], view, source_port=base[1])
            self._report.recovered_views += 1
        else:
            view = self._b.node(
                "constant", category=OpCategory.TENSOR, group=group, outputs=[shape_of(tensor)]
            )
            self._report.constants += 1
        self._summarize(view, tensor)
        self._produce(tensor, (view, 0))
        return view, 0

    def _summarize(self, node: str, tensor: torch.Tensor, *, image: bool = False) -> None:
        if self._capture:
            summary = activation(tensor, image=image)
            if summary is not None:
                self.activations[node] = summary


def _storage_key(tensor: torch.Tensor) -> int | None:
    """Identity of the memory a tensor views; None for meta/sparse tensors."""
    try:
        ptr = tensor.untyped_storage().data_ptr()
    except (RuntimeError, NotImplementedError):
        return None
    return ptr or None


def call_store(graph: Graph, converter: GraphConverter, recorder: Recorder) -> ActivationStore:
    """What the model produced: every operation's output, and every module call's."""
    return ActivationStore(graph, dict(converter.activations), dict(recorder.calls))
