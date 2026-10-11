"""Glyph painters: one object per :class:`Shape`, registered in a :class:`PainterRegistry`.

A painter draws a node and knows where edges attach to it, so routing works for any shape
(a new shape is supported by registering one more painter; nothing else changes).
"""

from __future__ import annotations

import math
from typing import Protocol

from tensorgami.render.canvas import Canvas, Paint
from tensorgami.scene import Node, Point, Side
from tensorgami.style.glyphs import NodeStyle, Shape, TextRole
from tensorgami.style.theme import Theme


class GlyphPainter(Protocol):
    def paint(self, canvas: Canvas, node: Node, theme: Theme) -> None: ...

    def anchor(self, node: Node, side: Side, theme: Theme) -> Point: ...


def node_style(node: Node, theme: Theme) -> NodeStyle:
    return theme.node_style(node.category)


def outline(style: NodeStyle, theme: Theme) -> Paint:
    return Paint(style.fill, style.stroke, theme.metrics.node_stroke, style.dash, style.opacity)


def draw_text_block(canvas: Canvas, node: Node, theme: Theme, center: Point) -> None:
    """Label line(s) and an optional detail line, vertically centred on ``center``.

    ``\\n`` in a label or detail starts a new line.
    """
    label_font = theme.node_font(node.category, TextRole.LABEL)
    detail_font = theme.node_font(node.category, TextRole.DETAIL)
    lines = [(t, label_font) for t in node.label.split("\n") if t]
    lines += [(t, detail_font) for t in node.detail.split("\n") if t]
    if not lines:
        return
    gap = theme.metrics.line_gap
    heights = [font.size * 1.15 for _, font in lines]
    y = center.y - (math.fsum(heights) + gap * (len(lines) - 1)) / 2
    for (text, font), h in zip(lines, heights, strict=True):
        canvas.text(Point(center.x, y + h / 2), text, font)
        y += h + gap


class RectPainter:
    """Rounded rectangle: operations and layers."""

    def paint(self, canvas: Canvas, node: Node, theme: Theme) -> None:
        style = node_style(node, theme)
        canvas.rect(node.box, outline(style, theme), theme.metrics.corner_radius)
        draw_text_block(canvas, node, theme, node.box.center)

    def anchor(self, node: Node, side: Side, theme: Theme) -> Point:
        return node.box.anchor(side)


class PillPainter(RectPainter):
    """Stadium shape: data (inputs, outputs, named tensors, sampled latents)."""

    def paint(self, canvas: Canvas, node: Node, theme: Theme) -> None:
        style = node_style(node, theme)
        canvas.rect(node.box, outline(style, theme), node.box.h / 2)
        draw_text_block(canvas, node, theme, node.box.center)


class CirclePainter:
    """Merge operation: a circle holding a symbol (+, ×, ‖); detail text centred above it.

    Edges usually enter and leave merges horizontally, so the detail goes where they don't.
    """

    def paint(self, canvas: Canvas, node: Node, theme: Theme) -> None:
        style = node_style(node, theme)
        r = self._radius(node)
        canvas.circle(node.box.center, r, outline(style, theme))
        if node.label:
            font = theme.node_font(node.category, TextRole.SYMBOL)
            canvas.text(node.box.center, node.label, font)
        if node.detail:
            font = theme.node_font(node.category, TextRole.DETAIL)
            at = Point(node.box.cx, node.box.cy - r - font.size * 0.8)
            canvas.text(at, node.detail, font)

    def anchor(self, node: Node, side: Side, theme: Theme) -> Point:
        r, c = self._radius(node), node.box.center
        return {
            Side.LEFT: Point(c.x - r, c.y),
            Side.RIGHT: Point(c.x + r, c.y),
            Side.TOP: Point(c.x, c.y - r),
            Side.BOTTOM: Point(c.x, c.y + r),
        }[side]

    @staticmethod
    def _radius(node: Node) -> float:
        return min(node.box.w, node.box.h) / 2


class TrapezoidPainter:
    """Encoder (narrowing) or decoder (widening) stage, read left to right."""

    def __init__(self, narrowing: bool) -> None:
        self._narrowing = narrowing

    def paint(self, canvas: Canvas, node: Node, theme: Theme) -> None:
        style = node_style(node, theme)
        canvas.polygon(self._corners(node, theme), outline(style, theme))
        draw_text_block(canvas, node, theme, node.box.center)

    def anchor(self, node: Node, side: Side, theme: Theme) -> Point:
        b = node.box
        if side.is_horizontal:
            return b.anchor(side)
        half = self._height_at_centre(node, theme) / 2
        return Point(b.cx, b.cy - half if side is Side.TOP else b.cy + half)

    def _corners(self, node: Node, theme: Theme) -> list[Point]:
        b = node.box
        short = b.h * theme.metrics.trapezoid_ratio
        left, right = (b.h, short) if self._narrowing else (short, b.h)
        return [
            Point(b.x, b.cy - left / 2),
            Point(b.right, b.cy - right / 2),
            Point(b.right, b.cy + right / 2),
            Point(b.x, b.cy + left / 2),
        ]

    @staticmethod
    def _height_at_centre(node: Node, theme: Theme) -> float:
        return node.box.h * (1 + theme.metrics.trapezoid_ratio) / 2


class CuboidPainter:
    """Tensor volume: ``box`` is the front face (H × C), ``depth`` the oblique W axis.

    Labels go *below* the glyph, as in the AlexNet / PlotNeuralNet convention.
    """

    def paint(self, canvas: Canvas, node: Node, theme: Theme) -> None:
        style = node_style(node, theme)
        b, (dx, dy) = node.box, self._offset(node, theme)
        width = theme.metrics.node_stroke
        top = [
            Point(b.x, b.y),
            Point(b.x + dx, b.y - dy),
            Point(b.right + dx, b.y - dy),
            Point(b.right, b.y),
        ]
        side = [
            Point(b.right, b.y),
            Point(b.right + dx, b.y - dy),
            Point(b.right + dx, b.bottom - dy),
            Point(b.right, b.bottom),
        ]
        canvas.polygon(side, Paint(style.fill.darken(0.14), style.stroke, width, style.dash))
        canvas.polygon(top, Paint(style.fill.lighten(0.45), style.stroke, width, style.dash))
        canvas.rect(b, outline(style, theme))
        label_font = theme.node_font(node.category, TextRole.LABEL)
        y = b.bottom + 6 + label_font.size / 2
        if node.label:
            canvas.text(Point(b.cx, y), node.label, label_font)
            y += label_font.size * 1.15 + theme.metrics.line_gap
        if node.detail:
            canvas.text(
                Point(b.cx, y), node.detail, theme.node_font(node.category, TextRole.DETAIL)
            )

    def anchor(self, node: Node, side: Side, theme: Theme) -> Point:
        # Anchors sit on the front face: edges are painted before nodes, so the part of an
        # edge behind the side/top face is hidden and arrows between cuboids stay straight.
        return node.box.anchor(side)

    @staticmethod
    def _offset(node: Node, theme: Theme) -> tuple[float, float]:
        depth = node.depth or min(node.box.w, node.box.h) * 0.5
        angle = math.radians(theme.metrics.cuboid_angle)
        return depth * math.cos(angle), depth * math.sin(angle)


class FoldedPainter(RectPainter):
    """A paper card with its right edge folded over: a slanted crease separates the flat
    face from a narrow, slightly darker strip. Sharp corners, hairline outline (origami).

    The text is centred on the flat face; the layout widens folded boxes by the strip's width
    (``Metrics.fold_width``), so a crease never crosses a label.
    """

    def paint(self, canvas: Canvas, node: Node, theme: Theme) -> None:
        style = node_style(node, theme)
        b, m = node.box, theme.metrics
        fold = min(m.fold_width, b.w / 3)
        top, bottom = b.right - fold * 0.55, b.right - fold  # the crease leans like a fold
        face = [Point(b.x, b.y), Point(top, b.y), Point(bottom, b.bottom), Point(b.x, b.bottom)]
        strip = [
            Point(top, b.y),
            Point(b.right, b.y),
            Point(b.right, b.bottom),
            Point(bottom, b.bottom),
        ]
        canvas.polygon(face, Paint(style.fill, opacity=style.opacity))
        canvas.polygon(strip, Paint(style.fill.darken(m.fold_shade), opacity=style.opacity))
        crease = Paint(None, style.stroke.lighten(0.35), m.node_stroke * 0.8)
        canvas.polyline([Point(top, b.y), Point(bottom, b.bottom)], crease)
        canvas.rect(b, Paint(None, style.stroke, m.node_stroke, style.dash))
        draw_text_block(canvas, node, theme, Point(b.x + (b.w - fold * 0.78) / 2, b.cy))


class PainterRegistry:
    """Maps shapes to painters; register a painter to support a new shape."""

    def __init__(self, painters: dict[Shape, GlyphPainter] | None = None) -> None:
        self._painters: dict[Shape, GlyphPainter] = dict(painters or {})

    def register(self, shape: Shape, painter: GlyphPainter) -> None:
        self._painters[shape] = painter

    def for_node(self, node: Node, theme: Theme) -> GlyphPainter:
        shape = node.shape or theme.node_style(node.category).shape
        try:
            return self._painters[shape]
        except KeyError:
            raise LookupError(f"no painter registered for shape '{shape}'") from None

    def shapes(self) -> set[Shape]:
        return set(self._painters)


def default_painters() -> PainterRegistry:
    return PainterRegistry(
        {
            Shape.RECT: RectPainter(),
            Shape.PILL: PillPainter(),
            Shape.CIRCLE: CirclePainter(),
            Shape.TRAPEZOID_NARROW: TrapezoidPainter(narrowing=True),
            Shape.TRAPEZOID_WIDEN: TrapezoidPainter(narrowing=False),
            Shape.CUBOID: CuboidPainter(),
            Shape.FOLDED: FoldedPainter(),
        }
    )
