"""Encoder–decoder models with nested skips (U-Net, pix2pix) folded into a U (step 11b).

A U-Net's skips nest: the skip of encoder level 1 ends at the last merge, the skip of level 2
at the one before, and so on. Drawn along one line, the model is as long as all its layers;
folded, each level is one row (encoder on the left, decoder on the right, the skip a short
arrow across the middle) and the bottleneck sits in a row below, so the figure is about as
wide as one level.
"""

from __future__ import annotations

import dataclasses

from tensorgami.ir import U_LEVEL, U_SIDE, USHAPE, Graph, Group
from tensorgami.transforms.base import data_inputs, rebuild
from tensorgami.transforms.cleanup import PruneEmptyGroups
from tensorgami.vocabulary import EdgeKind, OpCategory


class FoldUShape:
    """Find at least ``min_levels`` skip connections nested inside each other (an encoder
    tensor reaching the merge of its decoder level) and move the nodes from the outermost
    skip's source to its merge into one group of type ``USHAPE``, each marked with its level
    and side (``U_LEVEL``, ``U_SIDE``). The modules inside lose their frames: the U shows the
    levels instead. Residual networks (skips one after another, not nested) are untouched.
    """

    name = "fold-u-shape"

    def __init__(self, min_levels: int = 3) -> None:
        self.min_levels = min_levels

    def apply(self, graph: Graph) -> Graph:
        index = {n.id: i for i, n in enumerate(graph.nodes)}
        skips = _nested_skips(graph, index)
        if len(skips) < self.min_levels:
            return graph
        sources = [index[a] for a, _ in skips]  # increasing: outermost level first
        merges = [index[b] for _, b in skips]  # decreasing
        home = _common_group(graph, skips[0][0], skips[0][1])
        under = {n.id for n in graph.nodes_under(home)}
        region = [
            n
            for n in graph.nodes
            if n.id in under
            and sources[0] <= index[n.id] <= merges[0]
            and n.category is not OpCategory.IO
        ]
        gid = f"{USHAPE}:{home or 'model'}"
        nodes = []
        inside = {n.id for n in region}
        for n in graph.nodes:
            if n.id in inside:
                level, side = _place(index[n.id], sources, merges)
                params = {**n.params, U_LEVEL: level, U_SIDE: side}
                n = dataclasses.replace(n, group=gid, params=params)
            nodes.append(n)
        groups = [*graph.groups, Group(gid, USHAPE, parent=home)]
        return PruneEmptyGroups().apply(rebuild(graph, nodes=nodes, groups=groups))


def _nested_skips(graph: Graph, index: dict[str, int]) -> list[tuple[str, str]]:
    """The longest run of skips (source → merge) nested inside each other, outermost first."""
    skips = []
    for e in graph.edges:
        merge = graph.node(e.target)
        if e.kind is not EdgeKind.DATA or merge.category is not OpCategory.MERGE:
            continue
        if len(data_inputs(graph, merge.id)) >= 2 and index[merge.id] - index[e.source] >= 3:
            skips.append((index[e.source], index[merge.id], e.source, merge.id))
    skips.sort()
    best = [1] * len(skips)
    previous = [-1] * len(skips)
    for j, (a2, b2, _, _) in enumerate(skips):
        for i, (a1, b1, _, _) in enumerate(skips[:j]):
            if a1 < a2 and b2 < b1 and best[i] + 1 > best[j]:
                best[j], previous[j] = best[i] + 1, i
    if not skips:
        return []
    j = max(range(len(skips)), key=lambda k: best[k])
    chain = []
    while j >= 0:
        chain.append((skips[j][2], skips[j][3]))
        j = previous[j]
    return chain[::-1]


def _common_group(graph: Graph, a: str, b: str) -> str:
    def chain(node_id: str) -> list[str]:
        gid = graph.node(node_id).group
        return [gid, *(g.id for g in graph.ancestors(gid))]

    other = set(chain(b))
    return next(gid for gid in chain(a) if gid in other)


def _place(i: int, sources: list[int], merges: list[int]) -> tuple[int, str]:
    """Level and side of the node at position ``i``: encoder level k runs up to its skip's
    source, decoder level k from its merge; between the innermost pair is the bottleneck."""
    for k, source in enumerate(sources):
        if i <= source:
            return k, "enc"
    if i < merges[-1]:
        return len(sources), "bottom"
    for k in range(len(merges) - 1, -1, -1):
        if k == 0 or i < merges[k - 1]:
            return k, "dec"
    return 0, "dec"
