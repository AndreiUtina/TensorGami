"""The intermediate representation: an immutable, framework-neutral computation graph.

See ADR-0002. A :class:`Graph` holds

* :class:`Node` — one operation (or a graph input/output), in execution order;
* :class:`Edge` — a tensor flowing from an output port of one node to an input port of another;
* :class:`Group` — one *invocation* of a module, arranged as a tree (the module hierarchy);
* :class:`Repeat` — structurally identical groups (stacked layers or recurrent time steps).

Nothing here knows about PyTorch or about drawing.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from types import MappingProxyType
from typing import TypeAlias

from tensorgami.vocabulary import EdgeKind, OpCategory

Dim: TypeAlias = int | None  # None: unknown / symbolic size
Shape: TypeAlias = tuple[Dim, ...]
ParamValue: TypeAlias = int | float | str | bool | None | tuple["ParamValue", ...]

ROOT = ""  # id of the root group (the model itself)
INPUT_OP, OUTPUT_OP = "input", "output"


def _frozen(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class Node:
    """One operation.

    ``group`` is the module invocation the operation ran in; ``module`` is the path of the
    leaf module that *is* this operation (``"layers.0.linear1"``), empty for functional ops
    such as ``x + y``. ``params`` holds selected hyper-parameters (kernel size, channels…).
    """

    id: str
    op: str
    category: OpCategory = OpCategory.OTHER
    label: str = ""
    group: str = ROOT
    call_index: int = 0
    params: Mapping[str, ParamValue] = field(default_factory=dict)
    output_shapes: tuple[Shape, ...] = ()
    module: str = ""
    count: int = 1  # > 1: stands for N identical invocations (folded time steps)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", _frozen(self.params))
        object.__setattr__(self, "output_shapes", tuple(tuple(s) for s in self.output_shapes))

    def __hash__(self) -> int:  # params is a read-only mapping, so hash the identity fields
        return hash((self.id, self.op, self.group, self.module))

    @property
    def display_label(self) -> str:
        return self.label or self.op

    @property
    def is_input(self) -> bool:
        return self.op == INPUT_OP

    @property
    def is_output(self) -> bool:
        return self.op == OUTPUT_OP


@dataclass(frozen=True)
class Edge:
    """A tensor from ``source``'s output ``source_port`` to ``target``'s input ``target_port``."""

    source: str
    target: str
    kind: EdgeKind = EdgeKind.DATA
    source_port: int = 0
    target_port: int = 0
    shape: Shape | None = None

    def __post_init__(self) -> None:
        if self.shape is not None:
            object.__setattr__(self, "shape", tuple(self.shape))


@dataclass(frozen=True)
class Group:
    """One invocation of a module. ``id`` is the module path, suffixed ``#k`` for the k-th
    repeated call (``cell``, ``cell#1``, …); the root group has id :data:`ROOT`."""

    id: str
    type_name: str
    parent: str | None = ROOT
    module: str = ""
    call_index: int = 0
    label: str = ""
    category: OpCategory | None = None  # explicit override, e.g. a custom recurrent cell
    count: int = 1  # > 1: stands for N identical invocations (a folded ×N repeat)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def display_label(self) -> str:
        return self.label or self.type_name


# Unrolled recurrence: a group of type UNROLLED is a time × layer grid; its nodes carry their
# place in ``params``: STEP (0-based), LAYER (0 = reads the input) and ROLE ("cell", or "in" /
# "out" for the per-step tensors x_t / h_t). Written by the unrolled view, read by its layout.
UNROLLED = "unrolled"
STEP, LAYER, ROLE = "time_step", "time_layer", "time_role"

# Encoder–decoder with nested skips (U-Net): a group of type USHAPE is laid out as a U; its
# nodes carry U_LEVEL (0 = the outermost skip) and U_SIDE ("enc", "dec" or "bottom").
USHAPE = "u-shape"
U_LEVEL, U_SIDE = "u_level", "u_side"


class RepeatKind(StrEnum):
    LAYERS = "layers"  # different modules with identical structure (Transformer layers)
    TIME = "time"  # the same module called repeatedly with state passed along (RNN cells)


@dataclass(frozen=True)
class Repeat:
    """Groups recognised as repetitions of one structure, in order."""

    id: str
    kind: RepeatKind
    groups: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))

    @property
    def count(self) -> int:
        return len(self.groups)


@dataclass(frozen=True)
class Graph:
    """Immutable graph with lookup helpers. Build one with :class:`GraphBuilder`."""

    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    groups: tuple[Group, ...]
    repeats: tuple[Repeat, ...] = ()
    name: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attr in ("nodes", "edges", "groups", "repeats"):
            object.__setattr__(self, attr, tuple(getattr(self, attr)))
        object.__setattr__(self, "metadata", _frozen(self.metadata))

    def __hash__(self) -> int:
        return hash((self.name, self.nodes, self.edges, self.groups, self.repeats))

    # ------------------------------------------------------------------ indices

    @cached_property
    def _nodes_by_id(self) -> dict[str, Node]:
        return {n.id: n for n in self.nodes}

    @cached_property
    def _groups_by_id(self) -> dict[str, Group]:
        return {g.id: g for g in self.groups}

    @cached_property
    def _in_edges(self) -> dict[str, tuple[Edge, ...]]:
        out: dict[str, list[Edge]] = {}
        for e in self.edges:
            out.setdefault(e.target, []).append(e)
        return {k: tuple(v) for k, v in out.items()}

    @cached_property
    def _out_edges(self) -> dict[str, tuple[Edge, ...]]:
        out: dict[str, list[Edge]] = {}
        for e in self.edges:
            out.setdefault(e.source, []).append(e)
        return {k: tuple(v) for k, v in out.items()}

    @cached_property
    def _children(self) -> dict[str, tuple[Group, ...]]:
        out: dict[str, list[Group]] = {}
        for g in self.groups:
            if g.parent is not None:
                out.setdefault(g.parent, []).append(g)
        return {k: tuple(v) for k, v in out.items()}

    @cached_property
    def _members(self) -> dict[str, tuple[Node, ...]]:
        out: dict[str, list[Node]] = {}
        for n in self.nodes:
            out.setdefault(n.group, []).append(n)
        return {k: tuple(v) for k, v in out.items()}

    # ------------------------------------------------------------------ queries

    def node(self, node_id: str) -> Node:
        try:
            return self._nodes_by_id[node_id]
        except KeyError:
            raise KeyError(f"unknown node '{node_id}'") from None

    def group(self, group_id: str) -> Group:
        try:
            return self._groups_by_id[group_id]
        except KeyError:
            raise KeyError(f"unknown group '{group_id}'") from None

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes_by_id

    def has_group(self, group_id: str) -> bool:
        return group_id in self._groups_by_id

    @property
    def root(self) -> Group:
        return self.group(ROOT)

    @property
    def inputs(self) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes if n.is_input)

    @property
    def outputs(self) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes if n.is_output)

    def in_edges(self, node_id: str) -> tuple[Edge, ...]:
        return self._in_edges.get(node_id, ())

    def out_edges(self, node_id: str) -> tuple[Edge, ...]:
        return self._out_edges.get(node_id, ())

    def predecessors(self, node_id: str) -> tuple[Node, ...]:
        return tuple(self.node(e.source) for e in self.in_edges(node_id))

    def successors(self, node_id: str) -> tuple[Node, ...]:
        return tuple(self.node(e.target) for e in self.out_edges(node_id))

    def children(self, group_id: str) -> tuple[Group, ...]:
        """Direct child groups, in order of first invocation."""
        return self._children.get(group_id, ())

    def members(self, group_id: str) -> tuple[Node, ...]:
        """Nodes directly inside a group (not in its child groups), in execution order."""
        return self._members.get(group_id, ())

    def descendants(self, group_id: str) -> Iterator[Group]:
        for child in self.children(group_id):
            yield child
            yield from self.descendants(child.id)

    def nodes_under(self, group_id: str) -> tuple[Node, ...]:
        """All nodes in a group and its descendants, in execution order."""
        ids = {group_id, *(g.id for g in self.descendants(group_id))}
        return tuple(n for n in self.nodes if n.group in ids)

    def ancestors(self, group_id: str) -> tuple[Group, ...]:
        """From the group's parent up to the root."""
        chain = []
        current = self.group(group_id)
        while current.parent is not None:
            current = self.group(current.parent)
            chain.append(current)
        return tuple(chain)

    def depth(self, group_id: str) -> int:
        """0 for the root, 1 for its children, …"""
        return len(self.ancestors(group_id))

    def summary(self, *, max_rows: int | None = None) -> str:
        """Plain-text table of the nodes (see :func:`tensorgami.ir.summary.summarize`)."""
        from tensorgami.ir.summary import summarize

        return summarize(self, max_rows=max_rows)

    def categories(self) -> list[OpCategory]:
        """Node categories in order of first appearance."""
        return list(dict.fromkeys(n.category for n in self.nodes))
