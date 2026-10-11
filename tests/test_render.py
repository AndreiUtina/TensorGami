from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from tensorgami.render import (
    HeuristicTextMeasurer,
    Paint,
    PainterRegistry,
    SceneRenderer,
    SvgCanvas,
    add_legend,
    build_legend,
    default_painters,
    parse_markup,
    plain_text,
    render_svg,
)
from tensorgami.render.edges import EdgeRouter, _midpoint
from tensorgami.render.text import Run, Script
from tensorgami.scene import Box, Edge, Group, Node, Note, Point, Scene, Side
from tensorgami.style import Color, Shape, TextRole
from tensorgami.vocabulary import EdgeKind, OpCategory

SVG_NS = "{http://www.w3.org/2000/svg}"


def parse(svg: str) -> ET.Element:
    return ET.fromstring(svg)


# --------------------------------------------------------------------------- text


def test_markup_parses_braced_scripts_only():
    assert parse_markup("x_{t-1}") == [Run("x"), Run("t-1", Script.SUB)]
    assert parse_markup("h^{2}+b") == [Run("h"), Run("2", Script.SUPER), Run("+b")]
    assert parse_markup("batch_norm") == [Run("batch_norm")]
    assert plain_text("W_{q} x^{T}") == "Wq xT"


def test_measurer_scales_with_size_and_length(theme):
    m, font = HeuristicTextMeasurer(), theme.font(TextRole.LABEL)
    assert m.width("", font) == 0
    assert m.width("Conv 3×3", font) > m.width("Conv", font) > 0
    mono = theme.font(TextRole.ANNOTATION)
    assert m.width("iiii", mono) == pytest.approx(m.width("MMMM", mono))
    assert m.width("x_{t}", font) < m.width("xt", font)


# --------------------------------------------------------------------------- svg canvas


def test_svg_canvas_emits_well_formed_document(theme):
    canvas = SvgCanvas(120, 80, title="T & <title>", description="d")
    canvas.begin_group("tg-node tg-conv", data_id='a"b')
    canvas.rect(Box(1, 2, 30, 20), Paint(Color(255, 0, 0), Color(0, 0, 0), 1.5, (4, 2)), 5)
    canvas.circle(Point(50, 50), 10, Paint(None, Color(0, 0, 0)))
    canvas.polygon([Point(0, 0), Point(10, 0), Point(5, 5)], Paint(Color(0, 0, 255)))
    canvas.polyline([Point(0, 0), Point(10, 10)], Paint(None, Color(0, 0, 0), 2))
    canvas.text(Point(60, 40), "x_{t} < y & z", theme.font(TextRole.LABEL))
    canvas.end_group()
    root = parse(canvas.to_svg())
    assert root.get("viewBox") == "0 0 120 80"
    assert root.find(f"{SVG_NS}title").text == "T & <title>"
    group = root.find(f"{SVG_NS}g")
    assert group.get("data-id") == 'a"b'
    rect = group.find(f"{SVG_NS}rect")
    assert rect.get("stroke-dasharray") == "4 2" and rect.get("rx") == "5"
    text = group.find(f"{SVG_NS}text")
    assert "".join(text.itertext()) == "xt < y & z"
    assert text.find(f"{SVG_NS}tspan").get("font-size") == "9"


def test_svg_canvas_rejects_unbalanced_groups():
    canvas = SvgCanvas(10, 10)
    with pytest.raises(RuntimeError):
        canvas.end_group()
    canvas.begin_group("g")
    with pytest.raises(RuntimeError):
        canvas.to_svg()


def test_svg_numbers_are_compact_and_deterministic():
    canvas = SvgCanvas(10, 10)
    canvas.rect(Box(0.123456, -0.0, 3.0, 4.5), Paint(Color(0, 0, 0)))
    assert '<rect x="0.12" y="0" width="3" height="4.5" fill="#000000"/>' in canvas.to_svg()


# --------------------------------------------------------------------------- painters


@pytest.mark.parametrize("shape", list(Shape))
def test_every_shape_has_a_painter_with_anchors_on_its_outline(theme, shape):
    node = Node("n", Box(100, 100, 80, 40), OpCategory.CONV, "label", "detail", shape=shape)
    painter = default_painters().for_node(node, theme)
    anchors = {side: painter.anchor(node, side, theme) for side in Side}
    assert anchors[Side.LEFT].x < anchors[Side.RIGHT].x
    assert anchors[Side.TOP].y < anchors[Side.BOTTOM].y
    canvas = SvgCanvas(300, 300)
    painter.paint(canvas, node, theme)
    assert parse(canvas.to_svg()).iter()  # paints something well-formed


def test_trapezoid_and_circle_anchors_hug_the_outline(theme):
    registry = default_painters()
    narrow = Node("e", Box(0, 0, 100, 100), OpCategory.CONV, shape=Shape.TRAPEZOID_NARROW)
    top = registry.for_node(narrow, theme).anchor(narrow, Side.TOP, theme)
    assert top.y > 0  # below the box top, on the sloped edge
    merge = Node("m", Box(0, 0, 20, 30), OpCategory.MERGE, "+")
    right = registry.for_node(merge, theme).anchor(merge, Side.RIGHT, theme)
    assert right == Point(20, 15)  # radius = min(w, h) / 2


def test_registry_is_open_for_new_shapes(theme):
    class Dummy:
        def paint(self, canvas, node, theme):
            canvas.rect(node.box, Paint(Color(1, 2, 3)))

        def anchor(self, node, side, theme):
            return node.box.anchor(side)

    registry = PainterRegistry()
    node = Node("n", Box(0, 0, 10, 10), OpCategory.CONV)
    with pytest.raises(LookupError):
        registry.for_node(node, theme)
    registry.register(Shape.RECT, Dummy())
    assert isinstance(registry.for_node(node, theme), Dummy)


# --------------------------------------------------------------------------- routing


def _pair_scene(dy: float = 0, **edge_kwargs) -> tuple[Scene, Edge]:
    scene = Scene(400, 300).add(
        Node("a", Box(0, 100, 50, 20), OpCategory.CONV),
        Node("b", Box(200, 100 + dy, 50, 20), OpCategory.CONV),
    )
    edge = Edge("a", "b", **edge_kwargs)
    scene.add(edge)
    return scene, edge


def test_aligned_nodes_get_a_straight_edge(theme):
    scene, edge = _pair_scene()
    path = EdgeRouter(scene, theme, default_painters()).route(edge)
    assert path == [Point(50, 110), Point(200, 110)]


def test_misaligned_nodes_get_orthogonal_elbows(theme):
    scene, edge = _pair_scene(dy=80)
    path = EdgeRouter(scene, theme, default_painters()).route(edge)
    assert path[0] == Point(50, 110) and path[-1] == Point(200, 190)
    for a, b in zip(path, path[1:], strict=False):
        assert a.x == b.x or a.y == b.y  # every segment is horizontal or vertical


def test_explicit_sides_and_waypoints_are_respected(theme):
    via = (Point(25, 250), Point(225, 250))
    scene, edge = _pair_scene(exit=Side.BOTTOM, entry=Side.BOTTOM, via=via)
    path = EdgeRouter(scene, theme, default_painters()).route(edge)
    assert path == [Point(25, 120), *via, Point(225, 120)]


def test_midpoint_is_measured_along_the_path():
    assert _midpoint([Point(0, 0), Point(10, 0), Point(10, 10)]) == Point(10, 0)


# --------------------------------------------------------------------------- scene renderer


def _demo_scene() -> Scene:
    scene = Scene(500, 220, title="demo")
    scene.add(
        Group("blk", Box(90, 20, 260, 110), "Residual block", "×2"),
        Node("x", Box.centered(40, 75, 60, 30), OpCategory.IO, "x"),
        Node("c", Box.centered(160, 75, 90, 44), OpCategory.CONV, "Conv 3×3", "64"),
        Node("m", Box.centered(300, 75, 22, 22), OpCategory.MERGE, "+"),
        Node("z", Box.centered(430, 75, 60, 30), OpCategory.STOCHASTIC, "z_{t}"),
        Edge("x", "c", label="3×32×32"),
        Edge("c", "m"),
        Edge("m", "z", EdgeKind.STATE, delay=True),
        Edge("x", "m", exit=Side.BOTTOM, entry=Side.BOTTOM, via=(Point(40, 120), Point(300, 120))),
        Note(Point(250, 200), "time →"),
    )
    return scene


def test_render_svg_produces_semantic_groups(theme):
    root = parse(render_svg(_demo_scene(), theme))
    classes = [g.get("class") for g in root.iter(f"{SVG_NS}g")]
    assert {"tg-groups", "tg-edges", "tg-nodes", "tg-labels"} <= set(classes)
    assert "tg-node tg-conv" in classes and "tg-edge tg-edge-state" in classes
    ids = {g.get("data-id") for g in root.iter(f"{SVG_NS}g") if g.get("data-id")}
    assert ids == {"blk", "x", "c", "m", "z"}
    texts = {"".join(t.itertext()) for t in root.iter(f"{SVG_NS}text")}
    assert {"Conv 3×3", "64", "+", "zt", "3×32×32", "Residual block", "×2", "time →"} <= texts


def test_render_uses_theme_colours(theme):
    svg = render_svg(_demo_scene(), theme)
    assert theme.node_style(OpCategory.CONV).fill.hex in svg
    assert theme.edge_style(EdgeKind.STATE).color.hex in svg


def test_render_is_deterministic(theme):
    assert render_svg(_demo_scene(), theme) == render_svg(_demo_scene(), theme)


def test_renderer_accepts_injected_collaborators(theme):
    class FixedWidth:
        def width(self, text, font):
            return 10.0

    canvas = SvgCanvas(500, 220)
    SceneRenderer(theme, default_painters(), FixedWidth()).render(_demo_scene(), canvas)
    assert canvas.to_svg().startswith("<svg")


# --------------------------------------------------------------------------- legend


def test_legend_lists_used_categories_and_edge_kinds(theme):
    scene = _demo_scene()
    box = add_legend(scene, theme, Point(10, 150), columns=2)
    names = {n.text for n in scene.notes if n.role is TextRole.LEGEND}
    assert names == {
        theme.node_style(c).name
        for c in (OpCategory.IO, OpCategory.CONV, OpCategory.MERGE, OpCategory.STOCHASTIC)
    } | {theme.edge_style(EdgeKind.DATA).name, theme.edge_style(EdgeKind.STATE).name}
    assert box.x == 10 and box.w > 0 and box.h > 0
    render_svg(scene, theme)  # legend items are ordinary scene items


def test_legend_columns_trade_height_for_width(theme):
    cats = list(OpCategory)
    one = build_legend(theme, cats, columns=1)
    three = build_legend(theme, cats, columns=3)
    assert three.box.h < one.box.h and three.box.w > one.box.w
    assert build_legend(theme, []).items == ()
    four = build_legend(theme, cats[:4], columns=3)  # 2 rows → only 2 columns used
    two = build_legend(theme, cats[:4], columns=2)
    assert four.box.w == two.box.w
    with pytest.raises(ValueError):
        build_legend(theme, cats, columns=0)


# --------------------------------------------------------------------------- style capabilities


@pytest.fixture(scope="module")
def origami():
    from tensorgami.style import load_style

    return load_style("origami")


def test_folded_box_has_two_facets_a_crease_and_text_on_the_flat_face(origami):
    node = Node("n", Box(0, 0, 140, 46), OpCategory.CONV, "Conv 3×3", "64")
    canvas = SvgCanvas(200, 100)
    default_painters().for_node(node, origami).paint(canvas, node, origami)
    root = parse(canvas.to_svg())
    polygons = root.findall(f".//{SVG_NS}polygon")
    assert len(polygons) == 2 and all(p.get("fill-opacity") for p in polygons)
    (crease,) = root.findall(f".//{SVG_NS}polyline")
    crease_x = min(float(c.split(",")[0]) for c in crease.get("points").split())
    label = next(t for t in root.iter(f"{SVG_NS}text") if "".join(t.itertext()) == "Conv 3×3")
    assert float(label.get("x")) < 70 < crease_x  # centred left of the fold, never under it


def test_dot_terminals_and_frame_titles_drawn_over_wires(origami):
    scene = Scene(400, 200).add(
        Group("g", Box(150, 20, 200, 120), "block"),
        Node("a", Box.centered(60, 80, 80, 32), OpCategory.IO, "x"),
        Node("b", Box.centered(250, 80, 100, 40), OpCategory.CONV, "Conv"),
        Edge("a", "b"),
    )
    root = parse(render_svg(scene, origami))
    edge_group = next(
        g for g in root.iter(f"{SVG_NS}g") if g.get("class") == "tg-edge tg-edge-data"
    )
    (dot,) = edge_group.findall(f"{SVG_NS}circle")  # the source terminal
    assert float(dot.get("cx")) == pytest.approx(100)  # right side of node a
    frame = next(g for g in root.iter(f"{SVG_NS}g") if g.get("class") == "tg-group")
    assert frame.find(f"{SVG_NS}rect").get("fill-opacity") == "0"  # hairline, no fill
    labels = next(g for g in root.iter(f"{SVG_NS}g") if g.get("class") == "tg-labels")
    texts = ["".join(t.itertext()) for t in labels.iter(f"{SVG_NS}text")]
    assert "BLOCK" in texts  # caps font; painted in the label pass, above the wires


def test_end_dot_does_not_shorten_the_line(theme):
    import dataclasses

    from tensorgami.style import ArrowHead

    edges = dict(theme.edges)
    edges[EdgeKind.DATA] = dataclasses.replace(edges[EdgeKind.DATA], arrow=ArrowHead.DOT)
    dotted = dataclasses.replace(theme, edges=edges)
    scene = Scene(200, 60).add(Edge(Point(10, 30), Point(190, 30)))
    root = parse(render_svg(scene, dotted))
    line = next(root.iter(f"{SVG_NS}polyline"))
    assert line.get("points").endswith("190,30")
    assert any(c.get("cx") == "190" for c in root.iter(f"{SVG_NS}circle"))
