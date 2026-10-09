"""Automatic legend (survey R6): only the categories and edge kinds actually used."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tensorgami.render.text import HeuristicTextMeasurer, TextMeasurer
from tensorgami.scene import Box, Edge, Node, Note, Point, Scene, SceneItem, TextAnchor
from tensorgami.style.glyphs import Shape, TextRole
from tensorgami.style.theme import Theme
from tensorgami.vocabulary import EdgeKind, OpCategory

SWATCH_W, SWATCH_H, ROW_H, TEXT_GAP, COLUMN_GAP = 30.0, 16.0, 24.0, 8.0, 24.0
ORIGIN = Point(0, 0)
_SYMBOL = {OpCategory.MERGE: "+"}


@dataclass(frozen=True)
class Legend:
    items: tuple[SceneItem, ...]
    box: Box


def build_legend(
    theme: Theme,
    categories: Sequence[OpCategory],
    edge_kinds: Sequence[EdgeKind] = (),
    origin: Point = ORIGIN,
    columns: int = 1,
    measurer: TextMeasurer | None = None,
) -> Legend:
    """Lay out legend entries in ``columns`` columns, top-left at ``origin``."""
    if columns < 1:
        raise ValueError("columns must be >= 1")
    measurer = measurer or HeuristicTextMeasurer()
    font = theme.font(TextRole.LEGEND)
    entries: list[tuple[str, OpCategory | EdgeKind]] = [
        (theme.node_style(c).name, c) for c in categories
    ] + [(theme.edge_style(k).name, k) for k in edge_kinds]
    if not entries:
        return Legend((), Box(origin.x, origin.y, 0, 0))

    rows = -(-len(entries) // columns)
    col_w = SWATCH_W + TEXT_GAP + max(measurer.width(n, font) for n, _ in entries) + COLUMN_GAP
    items: list[SceneItem] = []
    for i, (name, what) in enumerate(entries):
        col, row = divmod(i, rows)
        x, cy = origin.x + col * col_w, origin.y + row * ROW_H + ROW_H / 2
        items.extend(_swatch(theme, what, x, cy, i))
        items.append(
            Note(Point(x + SWATCH_W + TEXT_GAP, cy), name, TextAnchor.START, TextRole.LEGEND)
        )
    used_columns = -(-len(entries) // rows)
    width = used_columns * col_w - COLUMN_GAP
    return Legend(tuple(items), Box(origin.x, origin.y, width, rows * ROW_H))


def add_legend(scene: Scene, theme: Theme, origin: Point, columns: int = 1) -> Box:
    """Append a legend for everything currently in ``scene``; returns its bounding box."""
    legend = build_legend(theme, scene.categories(), scene.edge_kinds(), origin, columns)
    scene.add(*legend.items)
    return legend.box


def _swatch(
    theme: Theme, what: OpCategory | EdgeKind, x: float, cy: float, index: int
) -> list[SceneItem]:
    if isinstance(what, EdgeKind):
        return [Edge(Point(x, cy), Point(x + SWATCH_W, cy), what)]
    shape = theme.node_style(what).shape
    if shape is Shape.CIRCLE:
        box = Box.centered(x + SWATCH_W / 2, cy, SWATCH_H + 2, SWATCH_H + 2)
    else:
        box = Box(x, cy - SWATCH_H / 2, SWATCH_W, SWATCH_H)
    return [Node(f"legend:{index}:{what}", box, what, label=_SYMBOL.get(what, ""))]
