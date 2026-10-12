"""The volume view's marking (step 11c): which layers are drawn as tensor volumes."""

from __future__ import annotations

import dataclasses

from tensorgami.ir import VOLUME, Graph, Node
from tensorgami.transforms.base import rebuild
from tensorgami.vocabulary import OpCategory

_NOT = {OpCategory.MERGE}  # merges keep their circles


class MarkVolumes:
    """Mark every layer whose output is an image tensor ``(N, C, H, W)`` with
    ``params[VOLUME] = (C, H, W)``, and every flat vector ``(N, F)`` with ``(F,)``. The input
    image is a volume too; graph outputs and merges stay as they are."""

    name = "mark-volumes"

    def apply(self, graph: Graph) -> Graph:
        nodes = [self._mark(n) for n in graph.nodes]
        if all(a is b for a, b in zip(nodes, graph.nodes, strict=True)):
            return graph
        return rebuild(graph, nodes=nodes)

    @staticmethod
    def _mark(node: Node) -> Node:
        if node.is_output or node.category in _NOT or len(node.output_shapes) != 1:
            return node
        shape = node.output_shapes[0]
        if any(d is None for d in shape):
            return node
        if len(shape) == 4:
            volume = tuple(shape[1:])
        elif len(shape) == 2 and not node.is_input:
            volume = (shape[1],)
        else:
            return node
        return dataclasses.replace(node, params={**node.params, VOLUME: volume})
