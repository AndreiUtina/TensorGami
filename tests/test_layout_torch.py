"""Layout invariants on real traces of the reference models."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from layout_checks import check_scene  # noqa: E402
from models import REFERENCE  # noqa: E402

from tensorgami.importers.torch import trace  # noqa: E402
from tensorgami.layout import Direction, LayeredLayout, LayoutOptions, layout  # noqa: E402
from tensorgami.render import render_svg  # noqa: E402
from tensorgami.transforms import simplify  # noqa: E402

TRACES = {}


def traced(name: str):
    if name not in TRACES:
        make, example = REFERENCE[name]
        TRACES[name] = trace(make(), example())
    return TRACES[name]


@pytest.mark.parametrize("name", REFERENCE)
@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("direction", list(Direction))
def test_reference_layouts_keep_the_invariants(theme, name, level, direction):
    graph = simplify(traced(name), level)
    options = LayoutOptions(direction=direction, wrap_width=None, wrap_height=None)
    scene = LayeredLayout(options).layout(graph, theme)
    check_scene(scene, graph, theme, monotonic="x" if direction is Direction.LEFT_RIGHT else "y")


@pytest.mark.parametrize("name", REFERENCE)
def test_default_figures_render(theme, name):
    graph = simplify(traced(name), 2)
    scene = layout(graph, theme)  # default options: wrapping on, legend on
    check_scene(scene, graph, theme)
    assert render_svg(scene, theme).startswith("<svg")


def test_unrolled_convlstm_lays_out_every_step(theme):
    graph = simplify(traced("convlstm"), 2, fold_time=False)
    scene = layout(graph, theme)
    check_scene(scene, graph, theme)
    assert sum(n.label == "ConvLSTMCell" for n in scene.nodes) == 6  # 2 layers × 3 steps
