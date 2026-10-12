"""The block, unrolled, volume and cell views, and the automatic choice of a view."""

from __future__ import annotations

import dataclasses

from tensorgami.ir import Graph, RepeatKind
from tensorgami.layout import Direction, LayoutOptions, layout
from tensorgami.scene import Scene
from tensorgami.style import Theme
from tensorgami.transforms import (
    DetectRecurrence,
    FoldIntoPrevious,
    Level,
    Pipeline,
    pipeline_for,
    simplify,
)
from tensorgami.transforms.cells import cell_figure
from tensorgami.transforms.unroll import UnrollRecurrence
from tensorgami.transforms.ushape import FoldUShape
from tensorgami.transforms.volume import MarkVolumes
from tensorgami.views.base import View
from tensorgami.vocabulary import OpCategory


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
            shown = FoldUShape(max_levels=options.u_levels).apply(shown)
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


class VolumeView:
    """CNN layers as tensor volumes, as in the AlexNet and LeNet figures: every image tensor a
    box of C × H × W (the input image too), flat vectors thin bars, each named with its shape
    below. Normalisation, activations and dropout fold into the layer before them. Always
    drawn left → right, without shapes on the wires (the volumes carry them)."""

    name = "volume"

    def draw(
        self, graph: Graph, theme: Theme, *, level: Level, options: LayoutOptions
    ) -> tuple[Graph, Scene]:
        layer_parts = (OpCategory.ACTIVATION, OpCategory.REGULARIZATION, OpCategory.NORM)
        fold = FoldIntoPrevious(layer_parts, across=layer_parts)  # a volume is a layer's output
        steps = [pipeline_for(level), fold, MarkVolumes()]
        if options.u_shape:  # a U-Net's volumes descend and climb, as in the U-Net paper
            steps.append(FoldUShape(max_levels=options.u_levels))
        shown = Pipeline(steps).apply(graph)
        options = dataclasses.replace(
            options, direction=Direction.LEFT_RIGHT, shapes_on_edges=False
        )
        return shown, layout(shown, theme, options)


class CellView:
    """A model's recurrent cells drawn as their gates, one framed cell per distinct cell: the
    classic LSTM figure (cell state along the top, the four gates, the hidden state along the
    bottom) for fused ``nn.LSTM``/``LSTMCell`` and for custom cells recognised as LSTMs (e.g.
    ConvLSTM, whose gates are a convolution); the textbook gates of GRU and RNN cells; the traced
    operations of any other cell. ``level`` does not apply; always left → right."""

    name = "cell"

    def draw(
        self, graph: Graph, theme: Theme, *, level: Level, options: LayoutOptions
    ) -> tuple[Graph, Scene]:
        shown = cell_figure(graph)
        if shown is None:
            raise ValueError(
                f"{graph.name} has no recurrent cell to draw: view='cell' shows LSTM, GRU and "
                "RNN cells (PyTorch's or your own); use view='block' or 'auto'"
            )
        options = dataclasses.replace(
            options, direction=Direction.LEFT_RIGHT, shapes_on_edges=False
        )
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
