"""Collapse module invocations into single summary nodes (levels of detail, survey R1).

A collapsed group keeps its :class:`~tensorgami.ir.Group` record (title, ×N count, place in
the hierarchy) and now contains exactly one node standing for everything it did.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial

from tensorgami.ir import Graph, Group, Node
from tensorgami.transforms.base import collapse, summary_node
from tensorgami.vocabulary import OpCategory

#: Categories that describe *what a block computes* (used to colour a collapsed block).
_COMPUTE = frozenset(
    {
        OpCategory.CONV,
        OpCategory.LINEAR,
        OpCategory.RECURRENT,
        OpCategory.ATTENTION,
        OpCategory.EMBEDDING,
        OpCategory.RESAMPLE,
        OpCategory.NORM,
    }
)


def content_category(graph: Graph, group_id: str) -> OpCategory:
    """The category most of a group's compute operations share (e.g. a ``features`` block
    of convolutions and pooling → convolution); ``module`` when there is no clear majority."""
    counts: dict[OpCategory, int] = {}
    for n in graph.nodes_under(group_id):
        if n.category in _COMPUTE:
            counts[n.category] = counts.get(n.category, 0) + 1
    if not counts:
        return OpCategory.MODULE
    top = max(counts, key=lambda c: (counts[c], -list(OpCategory).index(c)))
    return top if counts[top] * 2 > sum(counts.values()) else OpCategory.MODULE


def _group_summary(graph: Graph, g: Group, size: int, exits: list[tuple[str, int]]) -> Node:
    return summary_node(
        graph,
        f"<{g.id}>",
        exits,
        op=g.type_name,
        category=g.category or content_category(graph, g.id),
        group=g.id,
        label=g.label,
        module=g.module,
        call_index=g.call_index,
        size=size,
    )


def _collapse_groups(graph: Graph, targets: list[Group]) -> Graph:
    units = {}
    removed: set[str] = set()
    for g in targets:
        members = {n.id for n in graph.nodes_under(g.id)}
        removed.update(child.id for child in graph.children(g.id))
        if len(members) <= 1 and not graph.children(g.id):
            continue  # already a single operation
        units[g.id] = (members, partial(_group_summary, graph, g, len(members)))
    if not units and not removed:
        return graph
    return collapse(graph, units, removed_groups=removed)


def _outermost(graph: Graph, selected: Callable[[Group], bool]) -> list[Group]:
    chosen = []
    for g in graph.groups:
        if g.parent is None or not selected(g):
            continue
        if not any(selected(a) for a in graph.ancestors(g.id) if a.parent is not None):
            chosen.append(g)
    return chosen


class CollapseCells:
    """Custom recurrent cells and attention modules become one node per invocation: at the
    layer level a ConvLSTM cell is a layer, not twelve gate operations."""

    name = "collapse-cells"

    def __init__(
        self, categories: Iterable[OpCategory] = (OpCategory.RECURRENT, OpCategory.ATTENTION)
    ) -> None:
        self.categories = frozenset(categories)

    def apply(self, graph: Graph) -> Graph:
        return _collapse_groups(graph, _outermost(graph, lambda g: g.category in self.categories))


class CollapseGroups:
    """Collapse every module invocation at ``depth`` (1 = the model's direct children)."""

    name = "collapse-groups"

    def __init__(self, depth: int = 1) -> None:
        if depth < 1:
            raise ValueError("depth must be ≥ 1 (use CollapseToSystem for the whole model)")
        self.depth = depth

    def apply(self, graph: Graph) -> Graph:
        return _collapse_groups(graph, [g for g in graph.groups if graph.depth(g.id) == self.depth])


class CollapseToSystem:
    """Level 0: inputs → the whole model as one node → outputs."""

    name = "collapse-to-system"

    def apply(self, graph: Graph) -> Graph:
        members = {n.id for n in graph.nodes if not (n.is_input or n.is_output)}
        if not members:
            return graph
        root = graph.root

        def make(exits):
            return summary_node(
                graph,
                "<model>",
                exits,
                op=root.type_name,
                category=OpCategory.MODULE,
                group=root.id,
                label=root.label,
                size=len(members),
            )

        removed = {g.id for g in graph.children(root.id)}
        return collapse(graph, {"model": (members, make)}, removed_groups=removed)
