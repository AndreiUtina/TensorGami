"""Geometric invariants every layout must satisfy (shared by the layout tests)."""

from __future__ import annotations

from tensorgami.ir import Graph
from tensorgami.render import default_painters
from tensorgami.render.edges import EdgeRouter
from tensorgami.scene import Box, Point, Scene
from tensorgami.style import Theme

EPS = 0.5


def overlap(a: Box, b: Box, pad: float = 0.0) -> bool:
    return (
        a.x < b.right - pad
        and b.x < a.right - pad
        and a.y < b.bottom - pad
        and b.y < a.bottom - pad
    )


def inside(inner: Box, outer: Box) -> bool:
    return (
        inner.x >= outer.x - EPS
        and inner.y >= outer.y - EPS
        and inner.right <= outer.right + EPS
        and inner.bottom <= outer.bottom + EPS
    )


def segment_hits_box(a: Point, b: Point, box: Box, pad: float = 1.0) -> bool:
    x0, x1 = sorted((a.x, b.x))
    y0, y1 = sorted((a.y, b.y))
    return x0 < box.right - pad and x1 > box.x + pad and y0 < box.bottom - pad and y1 > box.y + pad


def routes(scene: Scene, theme: Theme) -> list[tuple]:
    router = EdgeRouter(scene, theme, default_painters())
    return [(edge, router.route(edge)) for edge in scene.edges]


def check_scene(scene: Scene, graph: Graph, theme: Theme, *, monotonic: str | None = None):
    """Assert the layout invariants; ``monotonic`` ('x' or 'y') checks the flow direction of
    forward edges when rows are not wrapped."""
    nodes = {n.id: n for n in scene.nodes if not n.id.startswith("legend:")}
    assert set(nodes) == {n.id for n in graph.nodes}
    boxes = [n.box for n in nodes.values()]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert not overlap(a, b), f"overlapping boxes {a} {b}"

    frames = {g.id: g.box for g in scene.groups}
    for node in graph.nodes:
        chain = [node.group, *(a.id for a in graph.ancestors(node.group))]
        for gid in chain:
            if gid in frames:
                assert inside(nodes[node.id].box, frames[gid]), f"{node.id} outside frame {gid}"
    for gid, frame in frames.items():
        members = {n.id for n in graph.nodes_under(gid)}
        for nid, n in nodes.items():
            if nid not in members:
                assert not overlap(n.box, frame, pad=1), f"{nid} cuts frame {gid}"

    for x, y in ((scene.width, scene.height),):
        assert all(b.right <= x + EPS and b.bottom <= y + EPS for b in boxes)

    loops = sum(e.source == e.target for e in graph.edges)
    assert len(scene.edges) >= len(graph.edges) - loops  # legend adds sample edges
    for edge, path in routes(scene, theme):
        for a, b in zip(path, path[1:], strict=False):
            assert abs(a.x - b.x) < EPS or abs(a.y - b.y) < EPS, f"diagonal segment {a}→{b}"
        if not isinstance(edge.source, str):
            continue  # loops and legend samples
        own = {edge.source, edge.target}
        for nid, n in nodes.items():
            if nid in own:
                continue
            for a, b in zip(path, path[1:], strict=False):
                assert not segment_hits_box(a, b, n.box), (
                    f"edge {edge.source}→{edge.target} crosses node {nid}"
                )
        if monotonic:
            coords = [getattr(p, monotonic) for p in path]
            assert all(q >= p - EPS for p, q in zip(coords, coords[1:], strict=False)), (
                f"edge {edge.source}→{edge.target} runs backwards: {coords}"
            )
