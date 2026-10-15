"""Encoder–decoder models with nested skips (U-Net, pix2pix) folded into a U (step 11b).

A U-Net's skips nest: the skip of encoder level 1 ends at the last merge, the skip of level 2
at the one before, and so on. Drawn along one line, the model is as long as all its layers;
folded, each level is one row (encoder on the left, decoder on the right, the skip a short
arrow across the middle) and the bottleneck sits in a row below, so the figure is about as
wide as one level.
"""

from __future__ import annotations

import dataclasses

from tensorgami.ir import U_LEVEL, U_SIDE, USHAPE, Edge, Graph, Group
from tensorgami.transforms.base import data_inputs, rebuild
from tensorgami.transforms.cleanup import PruneEmptyGroups
from tensorgami.vocabulary import EdgeKind, OpCategory


class FoldUShape:
    """Find at least ``min_levels`` skip connections nested inside each other (an encoder
    tensor reaching the merge of its decoder level) and move the encoder–decoder — every node
    of the group holding the outermost skip — into one group of type ``USHAPE``, each marked
    with its level and side (``U_LEVEL``, ``U_SIDE``): the layers before the outermost skip
    open the top encoder row, those after its merge close the top decoder row (U-Net's first
    convolutions and its 1 × 1 output layer). The modules inside lose their frames: the U
    shows the levels instead. Residual networks (skips one after another, not nested) are
    untouched.
    """

    name = "fold-u-shape"

    def __init__(self, min_levels: int = 3, max_levels: int | None = None) -> None:
        """``max_levels``: draw at most this many levels; a deeper U keeps its first
        ``max_levels - 1`` levels and the innermost one, bridged across the gap (the layout
        marks it with ⋮)."""
        self.min_levels = min_levels
        self.max_levels = max_levels

    def apply(self, graph: Graph) -> Graph:
        index = {n.id: i for i, n in enumerate(graph.nodes)}
        skips = _nested_skips(graph, index)
        if len(skips) < self.min_levels:
            return graph
        sources = [index[a] for a, _ in skips]  # increasing: outermost level first
        merges = [index[b] for _, b in skips]  # decreasing
        home = _common_group(graph, skips[0][0], skips[0][1])
        under = {n.id for n in graph.nodes_under(home)}
        region = [n for n in graph.nodes if n.id in under and n.category is not OpCategory.IO]
        gid = f"{USHAPE}:{home or 'model'}"
        place = {n.id: _place(index[n.id], sources, merges) for n in region}
        hidden = self._hidden(len(skips))
        nodes = []
        for n in graph.nodes:
            if n.id in place:
                level, side = place[n.id]
                if level in hidden:
                    continue
                params = {**n.params, U_LEVEL: level, U_SIDE: side}
                n = dataclasses.replace(n, group=gid, params=params)
            nodes.append(n)
        gone = {n for n, (level, _) in place.items() if level in hidden}
        edges = [e for e in graph.edges if e.source not in gone and e.target not in gone]
        if hidden:
            edges += _bridges(graph, place, hidden, skips)
        groups = [*graph.groups, Group(gid, USHAPE, parent=home)]
        return PruneEmptyGroups().apply(rebuild(graph, nodes=nodes, edges=edges, groups=groups))

    def _hidden(self, levels: int) -> set[int]:
        """The middle levels left out when the U is deeper than ``max_levels``."""
        if self.max_levels is None or levels <= self.max_levels:
            return set()
        return set(range(self.max_levels - 1, levels - 1))


def _bridges(graph: Graph, place: dict, hidden: set[int], skips) -> list[Edge]:
    """Across the hidden levels: the encoder hands on to the innermost level, which hands
    back up to the merge of the last level shown above the gap."""
    above, below = min(hidden) - 1, max(hidden) + 1
    order = {n.id: i for i, n in enumerate(graph.nodes)}

    def side(level: int, name: str) -> list[str]:
        return sorted((n for n, p in place.items() if p == (level, name)), key=order.get)

    down = Edge(skips[above][0], side(below, "enc")[0])
    up = Edge(side(below, "dec")[-1], skips[above][1])
    return [down, up]


def _nested_skips(graph: Graph, index: dict[str, int]) -> list[tuple[str, str]]:
    """The longest run of skips (source → merge) nested inside each other, outermost first.
    A skip leaves an encoder layer (not a graph input) and stays inside one model."""
    skips = []
    for e in graph.edges:
        merge, source = graph.node(e.target), graph.node(e.source)
        if e.kind is not EdgeKind.DATA or merge.category is not OpCategory.MERGE:
            continue
        if source.is_input or _model(source.id) != _model(merge.id):
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


def _model(node_id: str) -> str:
    """The model a node belongs to in a figure of several models (ids ``model/n3``)."""
    return node_id.partition("/")[0] if "/" in node_id else ""


def _common_group(graph: Graph, a: str, b: str) -> str:
    def chain(node_id: str) -> list[str]:
        gid = graph.node(node_id).group
        return [gid, *(g.id for g in graph.ancestors(gid))]

    other = set(chain(b))
    return next(gid for gid in chain(a) if gid in other)


def _place(i: int, sources: list[int], merges: list[int]) -> tuple[int, str]:
    """Level and side of the node at position ``i``: encoder level k runs up to its skip's
    source (level 0 from the start), decoder level k from its merge (level 0 to the end);
    between the innermost pair is the bottleneck."""
    for k, source in enumerate(sources):
        if i <= source:
            return k, "enc"
    if i < merges[-1]:
        return len(sources), "bottom"
    for k in range(len(merges) - 1, -1, -1):
        if k == 0 or i < merges[k - 1]:
            return k, "dec"
    return 0, "dec"
