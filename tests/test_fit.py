"""Figures made for a slide (roadmap step 13c): ``fit="slide"``."""

from __future__ import annotations

import io

import pytest
from layout_checks import check_scene

from tensorgami.api.figure import Figure
from tensorgami.api.fit import (
    PAGE_HEIGHT_IN,
    PAGES,
    check_fit,
    document,
    fit_to_page,
    parse_length,
)
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


# --------------------------------------------------------------------------- a paper's column


@pytest.mark.parametrize(
    ("value", "inches"),
    [("3.5in", 3.5), ("8.89cm", 3.5), ("88.9mm", 3.5), ("252pt", 3.5), ("336px", 3.5), (3.5, 3.5)],
)
def test_lengths_in_any_unit(value, inches):
    assert parse_length(value) == pytest.approx(inches)


@pytest.mark.parametrize("value", ["3.5", "wide", "-1in", "0cm", 0, -2, True, None])
def test_lengths_that_are_not_lengths(value):
    with pytest.raises(ValueError, match="width"):
        parse_length(value)


def test_a_document_page_is_at_most_a_page_tall_unless_told():
    assert document("3.5in").height_in == PAGE_HEIGHT_IN == 9.0
    assert document("7in", "4in").height_in == 4.0
    assert document("88mm").name == "3.46 in wide"


def column(graph, width: str = "3.5in", height=None, **options):
    theme = load_style()
    return fit_to_page(
        document(width, height),
        graph,
        theme,
        get_view("block"),
        level=Level.OPERATORS,
        options=LayoutOptions.for_style(theme, **options),
        fixed=set(options),
    )


def test_a_small_network_fits_a_column_at_true_size_and_exactly_its_width():
    """Box names keep the style's 9 pt, and the canvas is exactly 3.5 in wide, the legend
    stacked to fit it."""
    shown, scene, theme, info = column(chain(4))
    assert info.label_pt == pytest.approx(theme.font(TextRole.LABEL).size * 0.75)  # 9 pt
    assert scene.width == pytest.approx(3.5 * PX_PER_INCH)  # true size: 1 px = 1/96 in
    legend = [n.box for n in scene.nodes if n.id.startswith("legend:")]
    assert max(b.right for b in legend) <= scene.width
    fig = Figure(scene, theme, shown, fitted=info)
    assert fig.inches[0] == pytest.approx(3.5)
    assert fig.inches[1] == pytest.approx(scene.height / PX_PER_INCH)
    assert fig.to_svg().startswith('<svg xmlns="http://www.w3.org/2000/svg" width="3.5in"')
    assert "3.5 in wide, box names 9 pt" in repr(fig)
    check_scene(scene, shown, theme)


def test_a_figure_is_never_taller_than_the_page_and_says_when_its_text_shrinks():
    with pytest.warns(UserWarning, match=r"at 3.5 in wide the box names are [\d.]+ pt"):
        _, scene, _, info = column(chain(60))
    assert info.label_pt < 7
    assert scene.height * 3.5 / (scene.width / PX_PER_INCH) / PX_PER_INCH <= 9.0 + 1e-6


@pytest.mark.filterwarnings("ignore:at 7 in wide")  # 16 boxes in 2 in: small, on purpose
def test_a_height_limit_is_kept():
    _, scene, _, _ = column(chain(16), "7in", "2in")
    assert scene.height / scene.width * 7 <= 2.0 + 1e-6


def test_text_never_grows_beyond_the_styles_size():
    _, _, theme, info = column(chain(2), "7in")
    assert info.label_pt == pytest.approx(theme.font(TextRole.LABEL).size * 0.75)
