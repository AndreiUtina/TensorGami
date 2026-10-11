"""The block and unrolled views, and the automatic choice between them."""

from __future__ import annotations

from tensorgami.ir import Graph, RepeatKind
from tensorgami.layout import LayoutOptions, layout
from tensorgami.scene import Scene
from tensorgami.style import Theme
from tensorgami.transforms import DetectRecurrence, Level, Pipeline, pipeline_for, simplify
from tensorgami.transforms.unroll import UnrollRecurrence
from tensorgami.transforms.ushape import FoldUShape
from tensorgami.views.base import View


class BlockView:
    """Boxes and wires: one box per stage, layer or operation (by level) in a layered layout.
    Fits every model family; recurrent cells are drawn once with a state loop, and nested
    encoder–decoder skips (U-Net) fold into a U unless ``options.u_shape`` is off."""

    name = "block"

    def draw(
        self, graph: Graph, theme: Theme, *, level: Level, options: LayoutOptions
    ) -> tuple[Graph, Scene]:
        shown = simplify(graph, level)
        if options.u_shape:
            shown = FoldUShape().apply(shown)
        return shown, layout(shown, theme, options)


class UnrolledView:
    """Recurrence unrolled over time: every recurrent cell drawn once per step, as a time ×
    layer grid with per-step inputs x_t (fused ``nn.LSTM``/``GRU``/``RNN`` modules too). A
    sequence longer than ``max_steps`` shows its first steps, "⋯" and the last one."""

    name = "unrolled"

    def __init__(self, max_steps: int = 4) -> None:
        self.max_steps = max_steps

    def draw(
        self, graph: Graph, theme: Theme, *, level: Level, options: LayoutOptions
    ) -> tuple[Graph, Scene]:
        steps = [pipeline_for(level, fold_time=False), UnrollRecurrence(self.max_steps)]
        shown = Pipeline(steps).apply(graph)
        return shown, layout(shown, theme, options)


class AutoView:
    """Picks the view that suits the captured graph, then draws with it (ADR-0007): the
    unrolled view when a module was called step after step with state passed between the
    calls (a recurrent loop), the block view otherwise."""

    name = "auto"

    def __init__(self, block: View, unrolled: View) -> None:
        self._block, self._unrolled = block, unrolled

    def choose(self, graph: Graph) -> View:
        loops = DetectRecurrence().apply(graph).repeats
        return self._unrolled if any(r.kind is RepeatKind.TIME for r in loops) else self._block

    def draw(
        self, graph: Graph, theme: Theme, *, level: Level, options: LayoutOptions
    ) -> tuple[Graph, Scene]:
        return self.choose(graph).draw(graph, theme, level=level, options=options)
