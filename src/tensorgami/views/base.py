"""View contract: how a captured graph becomes a scene (ADR-0007 ``view=``)."""

from __future__ import annotations

from typing import Protocol

from tensorgami.ir import Graph
from tensorgami.layout import LayoutOptions
from tensorgami.scene import Scene
from tensorgami.style import Theme
from tensorgami.transforms import Level


class View(Protocol):
    """A way of drawing a model: which transforms run, which layout engine places the result.

    ``draw`` takes the graph as captured and returns the graph that is drawn (simplified to
    ``level``) together with its scene.
    """

    name: str

    def draw(
        self, graph: Graph, theme: Theme, *, level: Level, options: LayoutOptions
    ) -> tuple[Graph, Scene]: ...
