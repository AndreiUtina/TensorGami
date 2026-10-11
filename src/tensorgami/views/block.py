"""The block view and the automatic choice of a view."""

from __future__ import annotations

from tensorgami.ir import Graph
from tensorgami.layout import LayoutOptions, layout
from tensorgami.scene import Scene
from tensorgami.style import Theme
from tensorgami.transforms import Level, simplify
from tensorgami.views.base import View


class BlockView:
    """Boxes and wires: one box per stage, layer or operation (by level) in a layered layout.
    Fits every model family; recurrent cells are drawn once with a state loop."""

    name = "block"

    def draw(
        self, graph: Graph, theme: Theme, *, level: Level, options: LayoutOptions
    ) -> tuple[Graph, Scene]:
        shown = simplify(graph, level)
        return shown, layout(shown, theme, options)


class AutoView:
    """Picks the view that suits the captured graph, then draws with it.

    Every model gets the block view until the views of roadmap step 11 arrive (then, e.g.,
    recurrent models get the unrolled view).
    """

    name = "auto"

    def __init__(self, block: View) -> None:
        self._block = block

    def choose(self, graph: Graph) -> View:
        return self._block

    def draw(
        self, graph: Graph, theme: Theme, *, level: Level, options: LayoutOptions
    ) -> tuple[Graph, Scene]:
        return self.choose(graph).draw(graph, theme, level=level, options=options)
