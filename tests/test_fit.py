"""Figures made for a slide (roadmap step 13c): ``fit="slide"``."""

from __future__ import annotations

import io

import pytest
from layout_checks import check_scene

from tensorgami.api.figure import Figure
from tensorgami.api.fit import PAGES, check_fit, fit_to_page
from tensorgami.ir import GraphBuilder
from tensorgami.layout import LayoutOptions, layout
from tensorgami.render.mpl import PX_PER_INCH
from tensorgami.scene import Box, Edge, Node, Note, Point, Scene, SceneError
from tensorgami.style import TextRole, ThemeValidator, available_styles, load_style
from tensorgami.style.variants import presentation
from tensorgami.transforms import Level
from tensorgami.views import get_view
from tensorgami.vocabulary import OpCategory as C


def chain(length: int):
    b = GraphBuilder("chain")
    prev = b.input((1, 3, 32, 32))
    for k in range(length):
        n = b.node("Conv2d", category=C.CONV, module=f"conv{k}", outputs=[(1, 16, 32, 32)])
        b.edge(prev, n)
        prev = n
    b.output(prev)
    return b.build()


def _rows(scene) -> set[int]:
    """The rows of convolutions (the legend's sample box aside)."""
    return {
        round(n.box.cy)
        for n in scene.nodes
        if n.category is C.CONV and not n.id.startswith("legend:")
    }


def fitted(graph, page: str = "slide", **options):
    theme = load_style()
    return fit_to_page(
        PAGES[page],
        graph,
        theme,
        get_view("block"),
        level=Level.OPERATORS,
        options=LayoutOptions.for_style(theme, **options),
        fixed=set(options),
    )


@pytest.mark.parametrize(("page", "aspect"), [("slide", 16 / 9), ("slide-4:3", 4 / 3)])
def test_a_fitted_figure_has_the_pages_proportions(page, aspect):
    shown, scene, theme, info = fitted(chain(8), page)
    assert scene.width / scene.height == pytest.approx(aspect)
    assert info.page is PAGES[page]
    check_scene(scene, shown, theme)


def test_the_slide_layout_gives_larger_text_than_the_natural_one():
    """The figure's text, once it fills the slide: a long chain folded into rows beats one
    long row and beats the page-width rows of a document."""
    graph = chain(24)
    _, scene, theme, info = fitted(graph)
    page = PAGES["slide"]
    for options in (LayoutOptions(wrap_width=None), LayoutOptions()):  # one row; 1100 px rows
        natural = layout(graph, presentation(load_style()), options)
        scale = min(
            page.width_in * PX_PER_INCH / natural.width,
            page.height_in * PX_PER_INCH / natural.height,
        )
        label = theme.font(TextRole.LABEL).size * scale * 0.75
        assert info.label_pt >= label - 0.01
    rows = _rows(scene)
    assert len(rows) >= 3


@pytest.mark.filterwarnings("ignore:on a 16.9 slide")  # one long row: small text, on purpose
def test_a_wrap_length_given_by_the_caller_stays():
    _, scene, _, _ = fitted(chain(12), wrap_width=None)
    rows = _rows(scene)
    assert len(rows) == 1


def test_text_too_small_for_a_projector_is_reported():
    with pytest.warns(UserWarning, match=r"box names are \d+ pt.*level=\"stages\""):
        fitted(chain(150))


def test_the_slide_style_has_larger_text_and_lines_and_still_passes_the_checks():
    for name in available_styles():
        theme = load_style(name)
        slide = presentation(theme)
        assert slide.font(TextRole.LABEL).size == pytest.approx(
            1.35 * theme.font(TextRole.LABEL).size
        )
        assert slide.metrics.node_stroke == pytest.approx(1.5 * theme.metrics.node_stroke)
        assert ThemeValidator().validate(slide) == []


def test_a_slide_figure_is_saved_at_the_slides_size():
    shown, scene, theme, info = fitted(chain(8))
    fig = Figure(scene, theme, shown, fitted=info)
    assert fig.inches == pytest.approx((40 / 3, 7.5))
    buffer = io.BytesIO()
    fig.figure.savefig(buffer, format="png", dpi=144)
    from matplotlib.image import imread

    buffer.seek(0)
    assert imread(buffer).shape[:2] == (1080, 1920)
    assert "slide 16:9, box names" in repr(fig)


def test_unknown_fits_are_named():
    with pytest.raises(ValueError, match="unknown fit 'poster'; available: slide, slide-4:3"):
        check_fit("poster")


def test_a_scene_centred_on_a_larger_canvas():
    scene = Scene(100, 50, title="t")
    scene.add(
        Node("a", Box(10, 10, 20, 10), C.CONV, "A"),
        Node("b", Box(60, 10, 20, 10), C.CONV, "B"),
        Edge("a", "b", via=(Point(40, 15),), label="x", label_at=Point(45, 8)),
        Edge(Point(0, 40), "b"),
        Note(Point(50, 45), "note"),
    )
    centred = scene.centred_in(200, 150)
    assert (centred.width, centred.height, centred.title) == (200, 150, "t")
    assert centred.node("a").box == Box(60, 60, 20, 10)
    first, second = centred.edges
    assert first.via == (Point(90, 65),) and first.label_at == Point(95, 58)
    assert second.source == Point(50, 90) and second.target == "b"
    assert centred.notes[0].at == Point(100, 95)
    with pytest.raises(SceneError):
        scene.centred_in(50, 50)
