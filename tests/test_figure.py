"""matplotlib canvas and ``tg.Figure`` (step 9, ADR-0005)."""

from __future__ import annotations

import io
import logging
import subprocess
import sys

import matplotlib.pyplot as plt
import pytest
from ir_fixtures import residual_cnn, transformer
from matplotlib.figure import Figure as MplFigure
from PIL import Image

import tensorgami as tg
from tensorgami.layout import LayoutOptions, layout
from tensorgami.render import Paint, render_svg
from tensorgami.render.mpl import (
    MatplotlibCanvas,
    font_families,
    font_weight,
    render_figure,
    to_mathtext,
)
from tensorgami.scene import Box, Point, Scene
from tensorgami.style import Color, TextRole
from tensorgami.transforms import simplify
from tensorgami.vocabulary import OpCategory


@pytest.fixture(scope="module")
def figure(theme):
    graph = simplify(residual_cnn(), 2)
    return tg.Figure(layout(graph, theme), theme, graph)


# --------------------------------------------------------------------------- text helpers


@pytest.mark.parametrize(
    ("text", "math"),
    [
        ("Conv 3×3", "Conv 3×3"),
        ("x_{t-1}", "x$_{t-1}$"),
        ("h^{2} and $5", r"h$^{2}$ and \$5"),
        ("a_{b_c}", r"a$_{b\_c}$"),
    ],
)
def test_markup_becomes_mathtext(text, math):
    assert to_mathtext(text) == math


def test_font_stack_keeps_installed_fonts_and_a_glyph_fallback(caplog):
    with caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"):
        families = font_families(("No Such Font", "DejaVu Sans", "sans-serif"))
    assert families == ("DejaVu Sans",)
    assert font_families(("No Such Font",)) == ("DejaVu Sans",)
    assert font_families(("monospace",))[-1] == "DejaVu Sans Mono"
    assert not caplog.records  # missing fonts are skipped silently


def test_font_weight_snaps_like_browsers():
    assert font_weight("DejaVu Sans", 400) == 400
    assert font_weight("DejaVu Sans", 600) == 700  # no semibold cut: the next heavier
    assert font_weight("No Such Font", 600) == 600


# --------------------------------------------------------------------------- canvas


def test_canvas_keeps_paint_order_and_units(theme):
    fig = MplFigure(figsize=(2, 1), dpi=96)
    ax = fig.add_axes((0, 0, 1, 1))
    canvas = MatplotlibCanvas(ax, 192, 96)
    canvas.rect(Box(10, 10, 50, 30), Paint(Color(255, 0, 0), Color(0, 0, 0), 2, (4, 2)), 6)
    canvas.polyline([Point(0, 0), Point(100, 50)], Paint(None, Color(0, 0, 255), 1.5, (6, 4)))
    canvas.circle(Point(150, 50), 10, Paint(Color(0, 255, 0), opacity=0.5))
    canvas.text(Point(40, 25), "x_{t}", theme.font(TextRole.LABEL))
    artists = [*ax.patches, *ax.lines, *ax.texts]
    orders = sorted(a.get_zorder() for a in artists)
    assert orders == [1, 2, 3, 4]  # strictly in call order, as in SVG
    rect, circle = ax.patches
    assert rect.get_linewidth() == pytest.approx(1.5)  # 2 px → 1.5 pt
    assert circle.get_facecolor()[3] == pytest.approx(0.5)  # translucent fill
    (line,) = ax.lines
    assert line.get_linewidth() == pytest.approx(1.125)
    (text,) = ax.texts
    assert text.get_text() == "x$_{t}$" and text.get_fontsize() == pytest.approx(9)
    assert ax.get_ylim()[0] > ax.get_ylim()[1]  # y points down, as in the scene


def test_render_figure_has_the_natural_size(theme):
    scene = Scene(480, 240)
    fig = render_figure(scene, theme)
    assert tuple(fig.get_size_inches()) == pytest.approx((5.0, 2.5))  # 96 px per inch


def test_svg_supports_translucent_fills():
    from tensorgami.render import SvgCanvas

    canvas = SvgCanvas(10, 10)
    canvas.rect(Box(0, 0, 5, 5), Paint(Color(1, 2, 3), opacity=0.4))
    assert 'fill-opacity="0.4"' in canvas.to_svg()
    with pytest.raises(ValueError):
        Paint(opacity=1.5)


# --------------------------------------------------------------------------- Figure


def test_png_matches_the_scene_geometry(figure, tmp_path):
    """Sample the PNG inside every box: the pixel has the box's fill colour."""
    path = figure.save(tmp_path / "cnn.png", dpi=192)
    image = Image.open(path).convert("RGB")
    w, h = figure.size
    assert abs(image.width - 2 * w) <= 1 and abs(image.height - 2 * h) <= 1
    checked = 0
    for node in figure.scene.nodes:
        if node.id.startswith("legend:") or node.category is OpCategory.MERGE:
            continue
        x, y = node.box.x + 10, node.box.bottom - 5  # inside, clear of text and border
        pixel = image.getpixel((round(2 * x), round(2 * y)))
        fill = figure.style.node_style(node.category).fill
        assert max(abs(a - b) for a, b in zip(pixel, (fill.r, fill.g, fill.b), strict=True)) < 20
        checked += 1
    assert checked >= 5


def test_save_formats(figure, tmp_path):
    assert figure.save(tmp_path / "a.svg").read_text(encoding="utf-8") == render_svg(
        figure.scene, figure.style
    )
    assert figure.save(tmp_path / "a.pdf").read_bytes().startswith(b"%PDF")
    assert figure.save(tmp_path / "a.jpg", dpi=96).read_bytes()[:2] == b"\xff\xd8"
    with pytest.raises(ValueError, match="use .png, .svg"):
        figure.save(tmp_path / "a.docx")


def test_notebook_display_is_sharp_but_natural_size(figure):
    data, meta = figure._repr_png_()
    assert data.startswith(b"\x89PNG")
    w, h = figure.size
    assert meta == {"width": round(w), "height": round(h)}
    assert Image.open(io.BytesIO(data)).width == pytest.approx(2 * w, abs=1)


def test_no_pyplot_figures_are_left_behind(figure):
    before = plt.get_fignums()
    figure.to_png(dpi=48)
    _ = figure.figure
    assert plt.get_fignums() == before  # notebooks would otherwise show the figure twice


def test_draw_into_a_subplot_scales_text(theme):
    graph = simplify(transformer(), 2)
    tg_fig = tg.Figure(layout(graph, theme, LayoutOptions(legend=False)), theme, graph)
    panels, (left, right) = plt.subplots(1, 2, figsize=(6, 3))
    try:
        tg_fig.draw_on(right)
        natural = theme.font(TextRole.LABEL).size * 0.75
        sizes = [t.get_fontsize() for t in right.texts]
        assert sizes and max(sizes) < natural  # the panel is smaller than the natural size
        assert not left.texts
    finally:
        plt.close(panels)


def test_saving_is_quiet(figure, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"):
        figure.save(tmp_path / "quiet.png", dpi=96)
    assert not caplog.records


def test_repr_and_accessors(figure):
    assert repr(figure).startswith("<tensorgami.Figure 'residual_cnn'")
    assert figure.ax in figure.figure.axes
    assert figure.graph is not None and figure.style.name == "default"


def test_import_tensorgami_stays_light():
    code = "import sys, tensorgami; print('matplotlib' in sys.modules); tensorgami.Figure"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"  # matplotlib loads only when a figure is used
