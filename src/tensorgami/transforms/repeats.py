"""Repeated structure: detect ×N layer stacks, and fold repeats into one counted copy."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Iterable

from tensorgami.ir import Graph, Repeat, RepeatKind
from tensorgami.transforms.base import rebuild, without_groups
from tensorgami.vocabulary import EdgeKind


def signature(graph: Graph, group_id: str) -> tuple:
    """Structure of a group: its type, its operations (op, category, shapes) in order, its
    internal wiring and its sub-groups. Equal signatures mean interchangeable copies."""
    nodes = graph.nodes_under(group_id)
    index = {n.id: i for i, n in enumerate(nodes)}
    ops = tuple((n.op, n.category, n.output_shapes) for n in nodes)
    wiring = tuple(
        sorted(
            (index[e.source], e.source_port, index[e.target], e.kind)
            for n in nodes
            for e in graph.out_edges(n.id)
            if e.target in index
        )
    )
    subgroups = tuple(g.type_name for g in graph.descendants(group_id))
    return graph.group(group_id).type_name, ops, wiring, subgroups


def _connected(graph: Graph, a: str, b: str) -> bool:
    targets = {n.id for n in graph.nodes_under(b)}
    return any(e.target in targets for n in graph.nodes_under(a) for e in graph.out_edges(n.id))


class DetectLayerRepeats:
    """Consecutive sibling groups with identical structure, each feeding the next, form a
    LAYERS repeat (e.g. the 6 layers of a Transformer encoder)."""

    name = "detect-layer-repeats"

    def apply(self, graph: Graph) -> Graph:
        in_repeat = {g for r in graph.repeats for g in r.groups}
        repeats = list(graph.repeats)
        for parent in graph.groups:
            run: list[str] = []
            run_sig = None
            for child in [*graph.children(parent.id), None]:  # None flushes the last run
                sig = signature(graph, child.id) if child and child.id not in in_repeat else None
                extends = (
                    sig is not None and sig == run_sig and _connected(graph, run[-1], child.id)
                )
                if extends:
                    run.append(child.id)
                    continue
                if len(run) >= 2:
                    repeats.append(Repeat(f"layers:{run[0]}", RepeatKind.LAYERS, run))
                run, run_sig = ([child.id], sig) if sig is not None else ([], None)
        return rebuild(graph, repeats=repeats)


class FoldRepeats:
    """Keep the first copy of each repeat of the given kinds and mark it ``count = N``.

    * LAYERS: edges chaining copy *k* to copy *k+1* disappear (they are the stack itself);
      edges from outside into, or out of, any copy are re-attached to the first copy.
    * TIME: edges between steps become ``state`` loops on the first step (folded recurrence),
      and per-step inputs/outputs are re-attached to it. Recurrent *leaf* modules (a cell drawn
      as one node per step) are folded the same way.

    Copies whose structure differs (data-dependent branches) are left alone.
    """

    name = "fold-repeats"

    def __init__(self, kinds: Iterable[RepeatKind] = (RepeatKind.LAYERS, RepeatKind.TIME)) -> None:
        self.kinds = frozenset(kinds)

    def apply(self, graph: Graph) -> Graph:
        # one kind at a time, layers first: a node may sit in a layer copy *and* a time step
        for kind in (RepeatKind.LAYERS, RepeatKind.TIME):
            if kind in self.kinds:
                graph = self._fold(graph, kind)
        return graph

    def _fold(self, graph: Graph, fold_kind: RepeatKind) -> Graph:
        mapping: dict[str, str] = {}  # node of a later copy → equivalent node of the first
        unit_of: dict[str, tuple[str, int, RepeatKind]] = {}  # node → (plan, copy, kind)
        counts: dict[str, int] = {}  # kept group/node id → count
        removed_groups: set[str] = set()
        folded: set[str] = set()

        plans: list[tuple[str, RepeatKind, list[list[str]], str | None, list[str]]] = []
        for r in graph.repeats:
            if r.kind is fold_kind:
                units = [[n.id for n in graph.nodes_under(g)] for g in r.groups]
                plans.append((r.id, r.kind, units, r.groups[0], list(r.groups[1:])))
        if fold_kind is RepeatKind.TIME:
            plans += _leaf_time_plans(graph)

        for plan_id, kind, units, keep_group, drop_groups in plans:
            first = units[0]
            if any(not _same_ops(graph, first, u) for u in units[1:]):
                continue
            folded.add(plan_id)
            for copy, unit in enumerate(units):
                for position, node_id in enumerate(unit):
                    unit_of[node_id] = (plan_id, copy, kind)
                    if copy:
                        mapping[node_id] = first[position]
            if keep_group is not None:
                counts[keep_group] = len(units)
                removed_groups.update(drop_groups)
            else:
                counts[first[0]] = len(units)

        if not folded:
            return graph
        edges = []
        for e in graph.edges:
            a, b = unit_of.get(e.source), unit_of.get(e.target)
            same_plan = a is not None and b is not None and a[0] == b[0] and a[1] != b[1]
            if same_plan and a[2] is RepeatKind.LAYERS:
                continue  # the chain between stacked copies
            kind = EdgeKind.STATE if same_plan else e.kind
            source, target = mapping.get(e.source, e.source), mapping.get(e.target, e.target)
            if source == target and kind is not EdgeKind.STATE:
                continue
            edges.append(dataclasses.replace(e, source=source, target=target, kind=kind))
        nodes = [
            dataclasses.replace(n, count=counts[n.id]) if n.id in counts else n
            for n in graph.nodes
            if n.id not in mapping
        ]
        groups = [
            dataclasses.replace(g, count=counts[g.id]) if g.id in counts else g
            for g in without_groups(graph, removed_groups)
        ]
        repeats = [r for r in graph.repeats if r.id not in folded]
        return rebuild(graph, nodes=nodes, edges=edges, groups=groups, repeats=repeats)


def _same_ops(graph: Graph, a: list[str], b: list[str]) -> bool:
    return len(a) == len(b) and all(
        graph.node(x).op == graph.node(y).op for x, y in zip(a, b, strict=True)
    )


def _leaf_time_plans(graph: Graph) -> list:
    """Leaf modules called once per step (a cell drawn as one node), linked by state edges.
    Nodes inside a TIME-repeated group are folded with their group, not again here."""
    covered = {
        n.id
        for r in graph.repeats
        if r.kind is RepeatKind.TIME
        for g in r.groups
        for n in graph.nodes_under(g)
    }
    calls: dict[str, list[str]] = defaultdict(list)
    for n in graph.nodes:
        if n.module and n.id not in covered:
            calls[n.module].append(n.id)
    plans = []
    for module, ids in calls.items():
        if len(ids) < 2:
            continue
        linked = all(
            any(e.target == b and e.kind is EdgeKind.STATE for e in graph.out_edges(a))
            for a, b in zip(ids, ids[1:], strict=False)
        )
        if linked:
            plans.append((f"time-leaf:{module}", RepeatKind.TIME, [[i] for i in ids], None, []))
    return plans
