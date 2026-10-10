"""Recognise recurrence in an unrolled trace (ADR-0002: loops are captured as they ran)."""

from __future__ import annotations

import dataclasses
from collections import defaultdict

from tensorgami.ir import Graph, Repeat, RepeatKind
from tensorgami.transforms.base import rebuild
from tensorgami.vocabulary import EdgeKind


def invocations(graph: Graph, node_id: str) -> dict[str, int]:
    """Module path → call index for every module invocation a node runs inside, including the
    node's own leaf module (a ConvLSTM gate in step 2 of layer 0: ``{"cells.0": 2, …}``)."""
    node = graph.node(node_id)
    chain: dict[str, int] = {}
    group = graph.group(node.group)
    for g in (group, *graph.ancestors(group.id)):
        if g.parent is not None and g.module:
            chain[g.module] = g.call_index
    if node.module:
        chain[node.module] = node.call_index
    return chain


class DetectRecurrence:
    """Edges from one call of a module to a later call of the *same* module carry state
    between time steps: their kind becomes ``state``. Group invocations linked this way
    step after step become a TIME repeat."""

    name = "detect-recurrence"

    def apply(self, graph: Graph) -> Graph:
        chains = {n.id: invocations(graph, n.id) for n in graph.nodes}
        edges, links = [], defaultdict(set)  # module → {(from call, to call)}
        for e in graph.edges:
            a, b = chains[e.source], chains[e.target]
            crossing = [m for m in a if m in b and a[m] != b[m]]
            if crossing and e.kind is EdgeKind.DATA:
                e = dataclasses.replace(e, kind=EdgeKind.STATE)
            for m in crossing:
                links[m].add((a[m], b[m]))
            edges.append(e)

        in_repeat = {g for r in graph.repeats for g in r.groups}
        by_module: dict[str, list] = defaultdict(list)
        for g in graph.groups:
            if g.parent is not None and g.module and g.id not in in_repeat:
                by_module[g.module].append(g)
        repeats = list(graph.repeats)
        for module, groups in by_module.items():
            groups.sort(key=lambda g: g.call_index)
            consecutive = all((k, k + 1) in links[module] for k in range(len(groups) - 1))
            same_parent = len({g.parent for g in groups}) == 1
            if len(groups) >= 2 and consecutive and same_parent:
                repeats.append(Repeat(f"time:{module}", RepeatKind.TIME, [g.id for g in groups]))
        return rebuild(graph, edges=edges, repeats=repeats)
