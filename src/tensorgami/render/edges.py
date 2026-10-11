"""Edge routing (where a line goes) and edge painting (how it looks), kept separate."""

from __future__ import annotations

import math
from collections.abc import Sequence

from tensorgami.render.canvas import Canvas, Paint
from tensorgami.render.painters import PainterRegistry
from tensorgami.render.text import TextMeasurer
from tensorgami.scene import Box, Edge, Endpoint, Point, Scene, Side, TextAnchor
from tensorgami.style.glyphs import ArrowHead, TextRole
from tensorgami.style.theme import Theme

_ALIGNED = 0.5  # px tolerance for treating two points as axis-aligned


class EdgeRouter:
    """Turns an :class:`Edge` into a poly-line between node outlines.

    Without explicit ``via`` points, misaligned ends get orthogonal elbows so every edge is
    made of horizontal and vertical segments.
    """

    def __init__(self, scene: Scene, theme: Theme, painters: PainterRegistry) -> None:
        self._scene, self._theme, self._painters = scene, theme, painters

    def route(self, edge: Edge) -> list[Point]:
        first_hop = edge.via[0] if edge.via else self._center(edge.target)
        last_hop = edge.via[-1] if edge.via else self._center(edge.source)
        start, exit_side = self._attach(edge.source, edge.exit, first_hop)
        end, entry_side = self._attach(edge.target, edge.entry, last_hop)
        if edge.via:
            return _distinct([start, *edge.via, end])
        return _distinct([start, *self._elbows(start, exit_side, end, entry_side), end])

    def _attach(self, end: Endpoint, side: Side | None, toward: Point) -> tuple[Point, Side | None]:
        if isinstance(end, Point):
            return end, side
        node = self._scene.node(end)
        side = side or node.box.facing(toward)
        painter = self._painters.for_node(node, self._theme)
        return painter.anchor(node, side, self._theme), side

    def _center(self, end: Endpoint) -> Point:
        return end if isinstance(end, Point) else self._scene.node(end).box.center

    @staticmethod
    def _elbows(
        start: Point, exit_side: Side | None, end: Point, entry_side: Side | None
    ) -> list[Point]:
        if abs(start.x - end.x) < _ALIGNED or abs(start.y - end.y) < _ALIGNED:
            return []
        leaves_horizontally = exit_side.is_horizontal if exit_side else True
        enters_horizontally = entry_side.is_horizontal if entry_side else leaves_horizontally
        if leaves_horizontally and enters_horizontally:
            mx = (start.x + end.x) / 2
            return [Point(mx, start.y), Point(mx, end.y)]
        if not leaves_horizontally and not enters_horizontally:
            my = (start.y + end.y) / 2
            return [Point(start.x, my), Point(end.x, my)]
        if leaves_horizontally:
            return [Point(end.x, start.y)]
        return [Point(start.x, end.y)]


class EdgePainter:
    """Draws a routed edge in the style of its :class:`EdgeKind`."""

    def __init__(self, theme: Theme, measurer: TextMeasurer) -> None:
        self._theme, self._measurer = theme, measurer

    def paint_line(self, canvas: Canvas, edge: Edge, path: Sequence[Point]) -> None:
        style = self._theme.edge_style(edge.kind)
        m = self._theme.metrics
        line = list(path)
        if style.arrow is ArrowHead.FILLED:
            line[-1] = _retreat(line[-2], line[-1], m.arrow_length * 0.9)
        canvas.polyline(line, Paint(None, style.color, style.width, style.dash))
        if style.arrow is ArrowHead.DOT:
            canvas.circle(path[-1], m.terminal_radius, Paint(style.color))
        elif style.arrow is not ArrowHead.NONE:
            self._arrow_head(canvas, path[-2], path[-1], style.arrow, style.color, style.width)
        if style.start is ArrowHead.DOT and not edge.delay:  # not on recurrent self-loops
            canvas.circle(path[0], m.terminal_radius, Paint(style.color))
        if edge.delay:
            mid = _midpoint(path)
            size = m.delay_marker
            canvas.rect(Box.centered(mid.x, mid.y, size, size), Paint(style.color))

    def paint_label(self, canvas: Canvas, edge: Edge, path: Sequence[Point]) -> None:
        if not edge.label:
            return
        font = self._theme.font(TextRole.ANNOTATION)
        width = self._measurer.width(edge.label, font)
        halo = self._theme.metrics.label_halo
        if edge.label_upright and edge.label_at:  # reading bottom → top, centred on label_at
            at = edge.label_at
            box = Box(at.x - font.size * 0.6, at.y - width / 2, font.size * 1.2, width)
            canvas.rect(box.inflate(halo), Paint(self._theme.background), radius=2)
            canvas.text(at, edge.label, font, TextAnchor.MIDDLE, rotation=90)
            return
        if edge.label_at:
            at, anchor = edge.label_at, TextAnchor.MIDDLE
        else:
            a, b = _longest_segment(path)
            mid = Point((a.x + b.x) / 2, (a.y + b.y) / 2)
            if abs(a.y - b.y) < _ALIGNED:  # horizontal: label above the line
                at, anchor = mid.offset(dy=-(font.size / 2 + 4)), TextAnchor.MIDDLE
            else:  # vertical or diagonal: label to the right
                at, anchor = mid.offset(dx=6), TextAnchor.START
        left = {
            TextAnchor.START: at.x,
            TextAnchor.MIDDLE: at.x - width / 2,
            TextAnchor.END: at.x - width,
        }[anchor]
        canvas.rect(
            Box(
                left - halo,
                at.y - font.size * 0.6 - halo,
                width + 2 * halo,
                font.size * 1.2 + 2 * halo,
            ),
            Paint(self._theme.background),
            radius=2,
        )
        canvas.text(at, edge.label, font, anchor)

    def _arrow_head(
        self, canvas: Canvas, before: Point, tip: Point, kind: ArrowHead, color, width: float
    ) -> None:
        m = self._theme.metrics
        ux, uy = _unit(before, tip)
        base = Point(tip.x - ux * m.arrow_length, tip.y - uy * m.arrow_length)
        px, py = -uy * m.arrow_width / 2, ux * m.arrow_width / 2
        left, right = base.offset(px, py), base.offset(-px, -py)
        if kind is ArrowHead.FILLED:
            canvas.polygon([tip, left, right], Paint(color, color, 0.5))
        else:
            canvas.polyline([left, tip, right], Paint(None, color, width))


def _distinct(points: list[Point]) -> list[Point]:
    """Drop repeated points: a zero-length final segment would leave the arrowhead without a
    direction (it would collapse and vanish)."""
    out = [points[0]]
    for p in points[1:]:
        if abs(p.x - out[-1].x) > 0.01 or abs(p.y - out[-1].y) > 0.01:
            out.append(p)
    return out if len(out) >= 2 else points[:1] + points[-1:]


def _unit(a: Point, b: Point) -> tuple[float, float]:
    length = math.hypot(b.x - a.x, b.y - a.y) or 1.0
    return (b.x - a.x) / length, (b.y - a.y) / length


def _retreat(before: Point, tip: Point, distance: float) -> Point:
    ux, uy = _unit(before, tip)
    return Point(tip.x - ux * distance, tip.y - uy * distance)


def _longest_segment(path: Sequence[Point]) -> tuple[Point, Point]:
    return max(
        zip(path, path[1:], strict=False),
        key=lambda s: math.hypot(s[1].x - s[0].x, s[1].y - s[0].y),
    )


def _midpoint(path: Sequence[Point]) -> Point:
    """Point halfway along the poly-line, measured by length."""
    segments = list(zip(path, path[1:], strict=False))
    lengths = [math.hypot(b.x - a.x, b.y - a.y) for a, b in segments]
    remaining = math.fsum(lengths) / 2
    for (a, b), length in zip(segments, lengths, strict=True):
        if remaining <= length and length > 0:
            t = remaining / length
            return Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
        remaining -= length
    return path[-1]
