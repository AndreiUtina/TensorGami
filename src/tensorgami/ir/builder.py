"""Incremental construction of a :class:`Graph` (used by importers and tests)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from tensorgami.ir.model import (
    INPUT_OP,
    OUTPUT_OP,
    ROOT,
    Edge,
    Graph,
    Group,
    Node,
    ParamValue,
    Repeat,
    RepeatKind,
    Shape,
)
from tensorgami.ir.validation import GraphError, GraphValidator
from tensorgami.vocabulary import EdgeKind, OpCategory

_AUTO = object()  # sentinel: derive the edge shape from the source node


def invocation_id(module: str, call_index: int) -> str:
    """Group id of the ``call_index``-th call of ``module``: ``cell``, ``cell#1``, …"""
    return module if call_index == 0 else f"{module}#{call_index}"


class GraphBuilder:
    """Collects nodes, edges, groups and repeats, then freezes them into a validated graph.

    Node ids default to ``n0, n1, …`` in insertion (execution) order; edge target ports
    default to the order in which a node's inputs are connected; edge shapes default to the
    source's output shape on that port.
    """

    def __init__(
        self,
        name: str = "",
        root_type: str = "Model",
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self._name = name
        self._metadata = dict(metadata or {})
        self._groups: dict[str, Group] = {ROOT: Group(ROOT, root_type, parent=None)}
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._repeats: list[Repeat] = []
        self._next_port: dict[str, int] = {}

    # ------------------------------------------------------------------ structure

    def group(
        self,
        group_id: str,
        type_name: str,
        parent: str = ROOT,
        *,
        module: str | None = None,
        call_index: int = 0,
        label: str = "",
        category: OpCategory | None = None,
    ) -> str:
        if group_id in self._groups:
            raise GraphError(f"duplicate group id '{group_id}'")
        if parent not in self._groups:
            raise GraphError(f"unknown parent group '{parent}' for '{group_id}'")
        self._groups[group_id] = Group(
            group_id,
            type_name,
            parent=parent,
            module=group_id.split("#")[0] if module is None else module,
            call_index=call_index,
            label=label,
            category=category,
        )
        return group_id

    def node(
        self,
        op: str,
        *,
        node_id: str | None = None,
        category: OpCategory = OpCategory.OTHER,
        label: str = "",
        group: str = ROOT,
        call_index: int = 0,
        params: Mapping[str, ParamValue] | None = None,
        outputs: Sequence[Shape] = (),
        module: str = "",
    ) -> str:
        node_id = node_id or f"n{len(self._nodes)}"
        if node_id in self._nodes:
            raise GraphError(f"duplicate node id '{node_id}'")
        if group not in self._groups:
            raise GraphError(f"node '{node_id}' placed in unknown group '{group}'")
        self._nodes[node_id] = Node(
            node_id, op, category, label, group, call_index, params or {}, tuple(outputs), module
        )
        return node_id

    def get(self, node_id: str) -> Node:
        """A node added so far (the graph itself is only available after :meth:`build`)."""
        try:
            return self._nodes[node_id]
        except KeyError:
            raise GraphError(f"unknown node '{node_id}'") from None

    def input(self, shape: Shape, *, node_id: str | None = None, label: str = "input") -> str:
        return self.node(
            INPUT_OP, node_id=node_id, category=OpCategory.IO, label=label, outputs=[shape]
        )

    def output(
        self, source: str, *, port: int = 0, node_id: str | None = None, label: str = "output"
    ) -> str:
        """Add a graph output fed by ``source``'s output ``port``."""
        shape = self._source_shape(source, port)
        out = self.node(
            OUTPUT_OP,
            node_id=node_id,
            category=OpCategory.IO,
            label=label,
            outputs=[shape] if shape is not None else [],
        )
        self.edge(source, out, source_port=port)
        return out

    def edge(
        self,
        source: str,
        target: str,
        kind: EdgeKind = EdgeKind.DATA,
        *,
        source_port: int = 0,
        target_port: int | None = None,
        shape: Shape | None | object = _AUTO,
    ) -> Edge:
        for end in (source, target):
            if end not in self._nodes:
                raise GraphError(f"edge {source} → {target}: unknown node '{end}'")
        if target_port is None:
            target_port = self._next_port.get(target, 0)
        self._next_port[target] = max(self._next_port.get(target, 0), target_port + 1)
        resolved = self._source_shape(source, source_port) if shape is _AUTO else shape
        e = Edge(source, target, kind, source_port, target_port, resolved)  # type: ignore[arg-type]
        self._edges.append(e)
        return e

    def chain(self, ids: Iterable[str], kind: EdgeKind = EdgeKind.DATA) -> None:
        """Connect consecutive nodes: a → b → c."""
        ids = list(ids)
        for a, b in zip(ids, ids[1:], strict=False):
            self.edge(a, b, kind)

    def repeat(
        self, groups: Sequence[str], kind: RepeatKind, *, repeat_id: str | None = None
    ) -> str:
        repeat_id = repeat_id or f"r{len(self._repeats)}"
        self._repeats.append(Repeat(repeat_id, kind, tuple(groups)))
        return repeat_id

    # ------------------------------------------------------------------ output

    def build(self, *, validate: bool = True) -> Graph:
        graph = Graph(
            nodes=tuple(self._nodes.values()),
            edges=tuple(self._edges),
            groups=tuple(self._groups.values()),
            repeats=tuple(self._repeats),
            name=self._name,
            metadata=self._metadata,
        )
        return GraphValidator().ensure_valid(graph) if validate else graph

    def _source_shape(self, source: str, port: int) -> Shape | None:
        if source not in self._nodes:
            raise GraphError(f"unknown node '{source}'")
        outputs = self._nodes[source].output_shapes
        return outputs[port] if 0 <= port < len(outputs) else None
