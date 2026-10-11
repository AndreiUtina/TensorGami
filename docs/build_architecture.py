"""Draw Tensorgami's own pipeline (ADR-0001) with Tensorgami: docs/architecture.svg.

uv run python docs/build_architecture.py
"""

from __future__ import annotations

from pathlib import Path

from tensorgami.render import render_svg
from tensorgami.scene import Box, Edge, Group, Node, Note, Point, Scene, Side, TextAnchor
from tensorgami.style import TextRole, Theme, load_theme
from tensorgami.vocabulary import OpCategory as C

OUT = Path(__file__).parent / "architecture.svg"


def at(cx: float, cy: float, w: float, h: float) -> Box:
    return Box.centered(cx, cy, w, h)


def pipeline() -> Scene:
    s = Scene(1000, 330, "Tensorgami pipeline (ADR-0001)")
    r1, r2 = 70, 230
    s.add(
        Group("today", Box(232, 182, 528, 96), "exists today (step 2)"),
        Node("model", at(84, r1, 140, 44), C.IO, "nn.Module", "+ example input"),
        Node("imp", at(280, r1, 156, 50), C.MODULE, "ModelImporter", "torch tracing ✓ step 6"),
        Node("ir1", at(460, r1, 104, 44), C.TENSOR, "ir.Graph", "✓ step 5"),
        Node("tr", at(640, r1, 168, 50), C.MODULE, "GraphTransforms", "levels · ×N ✓ step 7"),
        Node("ir2", at(840, r1, 104, 34), C.TENSOR, "ir.Graph", "chosen level"),
        Node("lay", at(120, r2, 168, 50), C.MODULE, "LayoutEngine", "layered ✓ step 8"),
        Node("scene", at(320, r2, 104, 34), C.TENSOR, "Scene"),
        Node("ren", at(500, r2, 168, 50), C.MODULE, "SceneRenderer", "+ Theme · painters"),
        Node("can", at(680, r2, 132, 50), C.MODULE, "Canvas", "SVG · matplotlib ✓"),
        Node("fig", at(880, r2, 128, 44), C.IO, "tg.Figure", "Jupyter · PNG ✓"),
    )
    s.add(
        Edge("model", "imp"),
        Edge("imp", "ir1"),
        Edge("ir1", "tr"),
        Edge("tr", "ir2"),
        Edge(
            "ir2", "lay", exit=Side.BOTTOM, entry=Side.TOP, via=(Point(840, 150), Point(120, 150))
        ),
        Edge("lay", "scene"),
        Edge("scene", "ren"),
        Edge("ren", "can"),
        Edge("can", "fig"),
        Note(
            Point(500, 310),
            "a View (block · volume · unrolled · cell) chooses the transforms, "
            "the layout engine and the glyph mapping — step 11",
            TextAnchor.MIDDLE,
            TextRole.NOTE,
        ),
    )
    return s


def build(theme: Theme | None = None, out: Path = OUT) -> Path:
    theme = theme or load_theme()
    out.write_text(render_svg(pipeline(), theme), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
