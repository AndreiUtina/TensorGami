"""Remove or fold operations that add no information at the layer level (survey R5)."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Iterable

from tensorgami.ir import Graph, Node
from tensorgami.transforms.base import rebuild
from tensorgami.vocabulary import EdgeKind, OpCategory


def _names(text: str) -> frozenset[str]:
    return frozenset(text.split())


#: Pure data-layout operations: the tensor's values are unchanged, only its arrangement.
LAYOUT_OPS = _names("""view reshape contiguous permute transpose t T mT H mH unsqueeze squeeze
                       expand expand_as to float double half bfloat16 type clone detach view_as
                       reshape_as getitem data movedim swapaxes swapdims Identity""")

#: Element-wise arithmetic with a constant (``0.5 * x``, ``x / scale``): folded at L2.
SCALAR_ARITHMETIC = _names("""mul div add sub neg pow true_divide floor_divide remainder sqrt
                              rsqrt abs square reciprocal clamp clip""")

#: Sources that only create placeholder values (zero initial states, outside constants).
CONSTANT_OPS = _names("""zeros ones full empty zeros_like ones_like full_like empty_like
                         new_zeros new_ones new_full new_empty new_tensor tensor as_tensor arange
                         linspace eye constant""")


def _single_data_input(graph: Graph, node: Node):
    edges = graph.in_edges(node.id)
    return edges[0] if len(edges) == 1 and edges[0].kind is EdgeKind.DATA else None


class RemoveLayoutOps:
    """Bypass layout operations (``view``, ``permute``, ``getitem``…) that have one input and
    one output. A layout op without inputs that reads a buffer (``self.pe[:, :T]``) becomes a
    named tensor source (label ``pe``) instead."""

    name = "remove-layout-ops"

    def __init__(self, ops: Iterable[str] = LAYOUT_OPS) -> None:
        self.ops = frozenset(ops)

    def apply(self, graph: Graph) -> Graph:
        resolve: dict[str, tuple[str, int]] = {}
        renamed: dict[str, Node] = {}
        for n in graph.nodes:  # execution order: a chain resolves front to back
            if n.op not in self.ops or len(n.output_shapes) != 1:
                continue
            edge = _single_data_input(graph, n)
            if edge is not None:
                resolve[n.id] = resolve.get(edge.source, (edge.source, edge.source_port))
            elif not graph.in_edges(n.id) and n.params.get("weights"):
                label = n.label or ", ".join(n.params["weights"])
                renamed[n.id] = dataclasses.replace(n, category=OpCategory.TENSOR, label=label)
        if not resolve and not renamed:
            return graph
        edges = []
        for e in graph.edges:
            if e.target in resolve:
                continue
            if e.source in resolve:
                source, port = resolve[e.source]
                outputs = graph.node(source).output_shapes
                shape = outputs[port] if port < len(outputs) else None
                e = dataclasses.replace(e, source=source, source_port=port, shape=shape)
            edges.append(e)
        nodes = [renamed.get(n.id, n) for n in graph.nodes if n.id not in resolve]
        return rebuild(graph, nodes=nodes, edges=edges)


class FoldIntoPrevious:
    """Fold activations, regularisation and constant arithmetic into the operation before
    them, when that operation feeds nothing else: ``Linear → ReLU → Dropout`` becomes one
    ``Linear`` node with ``params["folded"] == ("ReLU", "Dropout")``.

    Categories in ``across`` (activations, by default) may join a layer of another module: an
    activation that opens a module (pix2pix's ``LeakyReLU → Conv``) joins the layer before it,
    which then carries the activated tensor to every consumer, as the trace did; a ``Dropout``
    opening a classifier stays in its module unless regularisation is in ``across``."""

    name = "fold-into-previous"

    def __init__(
        self,
        categories: Iterable[OpCategory] = (OpCategory.ACTIVATION, OpCategory.REGULARIZATION),
        *,
        arithmetic: bool = True,
        across: Iterable[OpCategory] = (OpCategory.ACTIVATION,),
    ) -> None:
        self.categories = frozenset(categories)
        self.arithmetic = arithmetic
        self.across = frozenset(across)  # categories that may join a layer of another module

    def _foldable(self, graph: Graph, n: Node) -> bool:
        if len(n.output_shapes) != 1 or _single_data_input(graph, n) is None:
            return False
        if n.category in self.categories:
            return True
        op = n.op.rstrip("_")
        return self.arithmetic and n.category is OpCategory.OTHER and op in SCALAR_ARITHMETIC

    def apply(self, graph: Graph) -> Graph:
        consumers = {n.id: {e.target for e in graph.out_edges(n.id)} for n in graph.nodes}
        into: dict[str, str] = {}
        folded: dict[str, list[str]] = defaultdict(list)

        def find(node_id: str) -> str:
            while node_id in into:
                node_id = into[node_id]
            return node_id

        for n in graph.nodes:
            if not self._foldable(graph, n):
                continue
            target = find(_single_data_input(graph, n).source)
            t = graph.node(target)
            if (
                t.category is OpCategory.IO
                or (t.group != n.group and n.category not in self.across)
                or len(t.output_shapes) != 1
                or consumers[target] != {n.id}
            ):
                continue
            into[n.id] = target
            folded[target].append(n.display_label)
            consumers[target] = consumers[n.id]
        if not into:
            return graph
        edges = []
        for e in graph.edges:
            if e.target in into:
                continue
            if e.source in into:
                source = find(e.source)
                e = dataclasses.replace(
                    e, source=source, source_port=0, shape=graph.node(source).output_shapes[0]
                )
            edges.append(e)
        nodes = []
        for n in graph.nodes:
            if n.id in into:
                continue
            if n.id in folded:
                previous = tuple(n.params.get("folded", ()))
                params = {**n.params, "folded": previous + tuple(folded[n.id])}
                n = dataclasses.replace(n, params=params)
            nodes.append(n)
        return rebuild(graph, nodes=nodes, edges=edges)


class RemoveConstants:
    """Drop placeholder sources such as zero initial states (``new_zeros``) and constants
    created outside the trace. Named buffers (positional encodings) are kept."""

    name = "remove-constants"

    def __init__(self, ops: Iterable[str] = CONSTANT_OPS) -> None:
        self.ops = frozenset(ops)

    def apply(self, graph: Graph) -> Graph:
        gone = {
            n.id
            for n in graph.nodes
            if n.category is OpCategory.TENSOR
            and n.op in self.ops
            and not graph.in_edges(n.id)
            and not n.params.get("weights")
        }
        if not gone:
            return graph
        return rebuild(
            graph,
            nodes=[n for n in graph.nodes if n.id not in gone],
            edges=[e for e in graph.edges if e.source not in gone],
        )


class PruneEmptyGroups:
    """Remove groups that contain no nodes and no non-empty sub-groups."""

    name = "prune-empty-groups"

    def apply(self, graph: Graph) -> Graph:
        occupied = {n.group for n in graph.nodes}
        keep = set()
        for g in graph.groups:
            if g.id in occupied:
                keep.add(g.id)
                keep.update(a.id for a in graph.ancestors(g.id))
        keep.add(graph.root.id)
        if len(keep) == len(graph.groups):
            return graph
        return rebuild(graph, groups=[g for g in graph.groups if g.id in keep])
