"""Engraved textures and the etching style (roadmap step 13d)."""

from __future__ import annotations

import dataclasses

import pytest
from ir_fixtures import residual_cnn
from layout_checks import check_scene

from tensorgami.layout import layout
from tensorgami.render import render_svg
from tensorgami.render.canvas import Paint
from tensorgami.render.painters import textured
from tensorgami.render.textures import engrave
from tensorgami.scene import Box, Node, Point
from tensorgami.style import Color, Metrics, Texture, ThemeValidator, load_style
from tensorgami.style.validation import DistinctTexturesRule
from tensorgami.transforms import simplify
from tensorgami.vocabulary import OpCategory as C


class Recorder:
    """A canvas that keeps what is drawn on it."""

    def __init__(self) -> None:
        self.lines: list[list[Point]] = []
        self.dots: list[tuple[Point, float]] = []
        self.rects: list[Box] = []

    def polyline(self, points, paint: Paint) -> None:
        self.lines.append(list(points))

    def circle(self, center: Point, radius: float, paint: Paint) -> None:
        self.dots.append((center, radius))

    def rect(self, box: Box, paint: Paint, radius: float = 0.0) -> None:
        self.rects.append(box)

    def polygon(self, points, paint: Paint) -> None:
        self.lines.append(list(points))

    def text(self, *args, **kwargs) -> None:
        pass


AREA = Box(10.3, 20.7, 12, 30)
INK = Color(20, 20, 20)


def drawn(texture: Texture) -> Recorder:
    canvas = Recorder()
    engrave(canvas, AREA, texture, INK, Metrics())
    return canvas


@pytest.mark.parametrize("texture", list(Texture))
def test_every_texture_stays_inside_its_area(texture):
    canvas = drawn(texture)
    assert canvas.lines or canvas.dots or canvas.rects
    for line in canvas.lines:
        for p in line:
            assert AREA.x - 1e-6 <= p.x <= AREA.right + 1e-6
            assert AREA.y - 1e-6 <= p.y <= AREA.bottom + 1e-6
    for centre, radius in canvas.dots:
        assert AREA.x <= centre.x - radius and centre.x + radius <= AREA.right
        assert AREA.y <= centre.y - radius and centre.y + radius <= AREA.bottom


def test_every_texture_looks_different_and_is_deterministic():
    def signature(texture):
        c = drawn(texture)
        lines = tuple(tuple((round(p.x, 3), round(p.y, 3)) for p in line) for line in c.lines)
        return lines, tuple(c.dots), tuple(c.rects)

    signatures = {t: signature(t) for t in Texture}
    assert len(set(signatures.values())) == len(Texture)
    assert all(signature(t) == s for t, s in signatures.items())


def test_line_textures_run_at_their_angle():
    for texture, (dx, dy) in [
        (Texture.HORIZONTAL, (1, 0)),
        (Texture.VERTICAL, (0, 1)),
        (Texture.DIAGONAL, (1, -1)),  # rising to the right (y points down)
        (Texture.BACKDIAGONAL, (1, 1)),
    ]:
        for a, b in drawn(texture).lines:
            cross = (b.x - a.x) * dy - (b.y - a.y) * dx
            assert abs(cross) < 1e-6


def test_the_etching_style_passes_every_rule_and_tells_operations_apart_by_texture():
    etching = load_style("etching")
    assert ThemeValidator().validate(etching) == []
    textured_ops = {c for c, s in etching.nodes.items() if s.texture is not None}
    assert {C.CONV, C.LINEAR, C.ATTENTION, C.NORM, C.ACTIVATION, C.RESAMPLE} <= textured_ops
    assert all(s.fill == Color(255, 255, 255) for s in etching.nodes.values())


def test_two_categories_sharing_a_texture_are_reported():
    etching = load_style("etching")
    nodes = dict(etching.nodes)
    nodes[C.LINEAR] = dataclasses.replace(nodes[C.LINEAR], texture=Texture.DIAGONAL)
    issues = list(DistinctTexturesRule().check(dataclasses.replace(etching, nodes=nodes)))
    assert len(issues) == 1 and "share the texture diagonal" in issues[0].message


def test_a_textured_box_keeps_its_text_off_the_strip():
    etching = load_style("etching")
    node = Node("c", Box(0, 0, 120, 40), C.CONV, "Conv 3×3", "64")
    rest = textured(Recorder(), node, etching, node.box)
    assert rest.x == pytest.approx(etching.metrics.texture_strip)
    assert rest.right == node.box.right
    plain = Node("r", Box(0, 0, 120, 40), C.RESHAPE, "Flatten")
    assert textured(Recorder(), plain, etching, plain.box) == plain.box


def test_the_layout_widens_textured_boxes_by_the_strip(theme):
    graph = simplify(residual_cnn(), 2)
    etching = load_style("etching")
    plain = dataclasses.replace(
        etching, nodes={c: dataclasses.replace(s, texture=None) for c, s in etching.nodes.items()}
    )
    a, b = layout(graph, etching), layout(graph, plain)
    conv = next(n.id for n in graph.nodes if n.category is C.CONV)
    assert a.node(conv).box.w >= b.node(conv).box.w + etching.metrics.texture_strip - 8
    check_scene(a, graph, etching)
    assert "<polyline" in render_svg(a, etching) or "<line" in render_svg(a, etching)
