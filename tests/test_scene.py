from __future__ import annotations

import pytest

from tensorgami.scene import Box, Edge, Group, Node, Note, Point, Scene, SceneError, Side
from tensorgami.vocabulary import EdgeKind, OpCategory


def test_box_geometry():
    b = Box(10, 20, 100, 40)
    assert (b.cx, b.cy, b.right, b.bottom) == (60, 40, 110, 60)
    assert Box.centered(60, 40, 100, 40) == b
    assert b.anchor(Side.LEFT) == Point(10, 40)
    assert b.anchor(Side.BOTTOM) == Point(60, 60)
    assert b.inflate(5) == Box(5, 15, 110, 50)
    assert Box.union([Box(0, 0, 10, 10), Box(20, 5, 10, 10)]) == Box(0, 0, 30, 15)


def test_box_rejects_negative_size():
    with pytest.raises(ValueError):
        Box(0, 0, -1, 5)


@pytest.mark.parametrize(
    ("target", "side"),
    [
        ((300, 40), Side.RIGHT),
        ((-100, 40), Side.LEFT),
        ((60, 300), Side.BOTTOM),
        ((60, -100), Side.TOP),
    ],
)
def test_box_facing(target, side):
    assert Box(10, 20, 100, 40).facing(Point(*target)) is side


def test_facing_is_normalised_by_aspect_ratio():
    wide = Box.centered(0, 0, 200, 20)
    # 50 px right but 30 px down: relative to the half-extents, "down" dominates
    assert wide.facing(Point(50, 30)) is Side.BOTTOM


def _node(node_id: str, x: float = 0) -> Node:
    return Node(node_id, Box(x, 0, 50, 20), OpCategory.CONV, node_id)


def test_scene_add_and_lookup():
    scene = Scene(200, 100).add(_node("a"), _node("b", 100), Edge("a", "b"))
    assert [n.id for n in scene.nodes] == ["a", "b"]
    assert scene.node("b").box.x == 100
    assert len(scene.edges) == 1


def test_scene_rejects_duplicates_and_dangling_edges():
    scene = Scene(200, 100).add(_node("a"))
    with pytest.raises(SceneError, match="duplicate node"):
        scene.add(_node("a"))
    with pytest.raises(SceneError, match="unknown node 'zz'"):
        scene.add(Edge("a", "zz"))
    with pytest.raises(SceneError):
        scene.node("missing")
    with pytest.raises(SceneError):
        scene.add("not an item")  # type: ignore[arg-type]


def test_edges_may_use_free_points():
    scene = Scene(200, 100).add(_node("a"), Edge(Point(0, 50), "a"), Edge("a", Point(190, 50)))
    assert len(scene.edges) == 2


def test_scene_size_must_be_positive():
    with pytest.raises(SceneError):
        Scene(0, 10)


def test_categories_and_edge_kinds_in_first_appearance_order():
    scene = Scene(300, 100).add(
        Node("x", Box(0, 0, 10, 10), OpCategory.IO),
        Node("n", Box(20, 0, 10, 10), OpCategory.NORM),
        Node("c", Box(40, 0, 10, 10), OpCategory.CONV),
        Node("c2", Box(60, 0, 10, 10), OpCategory.NORM),
        Edge("x", "n", EdgeKind.STATE),
        Edge("n", "c"),
        Edge("c", "c2", EdgeKind.STATE),
    )
    assert scene.categories() == [OpCategory.IO, OpCategory.NORM, OpCategory.CONV]
    assert scene.edge_kinds() == [EdgeKind.STATE, EdgeKind.DATA]


def test_iteration_yields_paint_order():
    scene = Scene(100, 100).add(
        _node("a"), Note(Point(1, 1), "n"), Group("g", Box(0, 0, 90, 90)), Edge(Point(0, 0), "a")
    )
    kinds = [type(item).__name__ for item in scene]
    assert kinds == ["Group", "Edge", "Node", "Note"]
