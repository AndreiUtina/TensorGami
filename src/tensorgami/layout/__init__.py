"""Layout: IR graph → positioned :class:`~tensorgami.scene.Scene` (roadmap step 8, ADR-0004)."""

from __future__ import annotations

from tensorgami.ir import Graph
from tensorgami.layout import unrolled, ushape  # noqa: F401  (register block layouts)
from tensorgami.layout.base import Direction, LayoutEngine, LayoutOptions
from tensorgami.layout.labels import Labeler, NodeText
from tensorgami.layout.layered import LayeredLayout
from tensorgami.scene import Scene
from tensorgami.style import Theme


def layout(graph: Graph, theme: Theme, options: LayoutOptions | None = None) -> Scene:
    """Lay out ``graph`` with the default layered engine."""
    return LayeredLayout(options).layout(graph, theme)


__all__ = [
    "Direction",
    "LayeredLayout",
    "LayoutEngine",
    "LayoutOptions",
    "Labeler",
    "NodeText",
    "layout",
]
