"""Transform protocol, pipelines, and shared graph-rewriting helpers.

Every transform is ``Graph → Graph`` and returns a new, validated graph; the input is never
modified (graphs are immutable).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol

from tensorgami.ir import Edge, Graph, GraphValidator, Group, Node, Repeat
from tensorgami.vocabulary import EdgeKind, OpCategory


class GraphTransform(Protocol):
    name: str

    def apply(self, graph: Graph) -> Graph: ...


class Pipeline:
    """Applies transforms in order; is itself a transform, so pipelines nest."""

    name = "pipeline"

    def __init__(self, transforms: Sequence[GraphTransform]) -> None:
        self.transforms = list(transforms)

    def apply(self, graph: Graph) -> Graph:
        for transform in self.transforms:
            graph = transform.apply(graph)
        return graph

    def __repr__(self) -> str:
        return f"Pipeline([{', '.join(t.name for t in self.transforms)}])"


# --------------------------------------------------------------------------- helpers


def rebuild(
    graph: Graph,
    *,
    nodes: Iterable[Node] | None = None,
    edges: Iterable[Edge] | None = None,
    groups: Iterable[Group] | None = None,
    repeats: Iterable[Repeat] | None = None,
) -> Graph:
    """A new graph with some parts replaced; repeats pointing at removed groups are dropped
    and edges are normalised. Always validated: a transform bug fails loudly here."""
    groups = tuple(graph.groups if groups is None else groups)
    group_ids = {g.id for g in groups}
    repeats = tuple(graph.repeats if repeats is None else repeats)
    kept_repeats = []
    for r in repeats:
        alive = tuple(g for g in r.groups if g in group_ids)
        if len(alive) >= 2:
            kept_repeats.append(dataclasses.replace(r, groups=alive))
    new = dataclasses.replace(
        graph,
        nodes=tuple(graph.nodes if nodes is None else nodes),
        edges=tuple(normalize_edges(graph.edges if edges is None else edges)),
        groups=groups,
        repeats=tuple(kept_repeats),
    )
    return GraphValidator().ensure_valid(new)


def normalize_edges(edges: Iterable[Edge]) -> list[Edge]:
    """Drop duplicates (same source port, target and kind) and number each target's input
    ports 0, 1, … in order."""
    seen: set[tuple] = set()
    unique: list[Edge] = []
    for e in edges:
        key = (e.source, e.source_port, e.target, e.kind)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    next_port: dict[str, int] = {}
    out = []
    for e in unique:
        port = next_port.get(e.target, 0)
        next_port[e.target] = port + 1
        out.append(e if e.target_port == port else dataclasses.replace(e, target_port=port))
    return out


def data_inputs(graph: Graph, node_id: str) -> list[Edge]:
    return [e for e in graph.in_edges(node_id) if e.kind in (EdgeKind.DATA, EdgeKind.STOCHASTIC)]


def without_groups(graph: Graph, removed: set[str]) -> list[Group]:
    """Groups minus ``removed`` and all their descendants."""
    gone = set(removed)
    for gid in removed:
        gone.update(g.id for g in graph.descendants(gid))
    return [g for g in graph.groups if g.id not in gone]


def collapse(
    graph: Graph,
    units: Mapping[str, tuple[set[str], Callable[[list[tuple[str, int]]], Node]]],
    *,
    removed_groups: set[str] = frozenset(),
) -> Graph:
    """Replace each unit (a set of node ids) by one summary node.

    ``units`` maps a key to ``(node ids, make_node)``; ``make_node`` receives the unit's
    distinct outgoing (node, port) pairs in order and returns the summary node, whose output
    port *k* stands for the *k*-th pair. Edges inside a unit disappear; edges crossing its
    border are re-attached to the summary node.
    """
    owner: dict[str, str] = {}
    for key, (members, _) in units.items():
        for node_id in members:
            owner[node_id] = key
    exits: dict[str, list[tuple[str, int]]] = {key: [] for key in units}
    for e in graph.edges:
        key = owner.get(e.source)
        if key is not None and owner.get(e.target) != key:
            pair = (e.source, e.source_port)
            if pair not in exits[key]:
                exits[key].append(pair)
    summaries = {key: make(exits[key]) for key, (_, make) in units.items()}

    nodes: list[Node] = []
    placed: set[str] = set()
    for n in graph.nodes:  # each summary takes the place of its unit's first node
        key = owner.get(n.id)
        if key is None:
            nodes.append(n)
        elif key not in placed:
            placed.add(key)
            nodes.append(summaries[key])
    for key, summary in summaries.items():  # units without nodes still get their summary
        if key not in placed:
            nodes.append(summary)

    edges: list[Edge] = []
    for e in graph.edges:
        src_key, tgt_key = owner.get(e.source), owner.get(e.target)
        if src_key is not None and src_key == tgt_key:
            continue
        source, port = e.source, e.source_port
        if src_key is not None:
            source, port = summaries[src_key].id, exits[src_key].index((e.source, e.source_port))
        target = summaries[tgt_key].id if tgt_key is not None else e.target
        edges.append(dataclasses.replace(e, source=source, source_port=port, target=target))
    return rebuild(graph, nodes=nodes, edges=edges, groups=without_groups(graph, removed_groups))


def exit_shapes(graph: Graph, exits: list[tuple[str, int]]) -> tuple:
    shapes = []
    for node_id, port in exits:
        outputs = graph.node(node_id).output_shapes
        shapes.append(outputs[port] if port < len(outputs) else ())
    return tuple(shapes)


def summary_node(
    graph: Graph,
    node_id: str,
    exits: list[tuple[str, int]],
    *,
    op: str,
    category: OpCategory,
    group: str,
    label: str = "",
    module: str = "",
    call_index: int = 0,
    size: int = 0,
) -> Node:
    return Node(
        node_id,
        op,
        category,
        label,
        group,
        call_index,
        {"collapsed": size} if size else {},
        exit_shapes(graph, exits),
        module,
    )
