"""Engraved textures (the etching style): categories told apart without colour.

A texture is drawn with plain lines, dots and rectangles clipped to its area, so the SVG and
matplotlib canvases render it identically and deterministically. Lines are anchored to the
page's origin, so the textures of neighbouring boxes line up. Orientations are 45° apart and
the ink density is about equal across textures (He et al., IEEE TVCG 2024).
"""

from __future__ import annotations

import math
from collections.abc import Iterator

from tensorgami.render.canvas import Canvas, Paint
from tensorgami.scene import Box, Point
from tensorgami.style import Color, Metrics, Texture

FC = tuple[float, float]


def engrave(canvas: Canvas, area: Box, texture: Texture, color: Color, metrics: Metrics) -> None:
    """Draw ``texture`` over ``area`` in ``color``."""
    if area.w <= 0.5 or area.h <= 0.5:
        return
    s, width = metrics.texture_spacing, metrics.texture_line
    ink = Paint(None, color, width)
    if texture is Texture.SOLID:
        canvas.rect(area, Paint(color))
        return
    if texture is Texture.DOTS:
        for centre in _dots(area, s, width * 1.1):
            canvas.circle(centre, width * 1.1, Paint(color))
        return
    if texture is Texture.ZIGZAG:
        for line in _zigzags(area, s):
            canvas.polyline(line, ink)
        return
    if texture is Texture.DASHES:
        for a, b in _dashes(area, s):
            canvas.polyline([a, b], ink)
        return
    angles = {
        Texture.DIAGONAL: (45,),  # rising to the right: ////
        Texture.BACKDIAGONAL: (135,),
        Texture.HORIZONTAL: (0,),
        Texture.VERTICAL: (90,),
        Texture.CROSSHATCH: (45, 135),
        Texture.GRID: (0, 90),
    }[texture]
    spacing = s * (1.5 if len(angles) > 1 else 1.0)  # two sets of lines: about the same ink
    for angle in angles:
        for a, b in _lines(area, angle, spacing):
            canvas.polyline([a, b], ink)


def _lines(area: Box, angle: float, spacing: float) -> Iterator[tuple[Point, Point]]:
    """Parallel lines at ``angle`` degrees (0: horizontal, 45: rising to the right, the y-axis
    pointing down), ``spacing`` apart, clipped to ``area``."""
    rad = math.radians(angle)
    d = (math.cos(rad), -math.sin(rad))  # along the line
    n = (-d[1], d[0])  # across the lines
    corners = [(area.x, area.y), (area.right, area.y), (area.x, area.bottom)]
    corners.append((area.right, area.bottom))
    offsets = [n[0] * x + n[1] * y for x, y in corners]
    first = math.ceil(min(offsets) / spacing)
    last = math.floor(max(offsets) / spacing)
    for k in range(first, last + 1):
        c = k * spacing
        segment = _clip((n[0] * c, n[1] * c), d, area)
        if segment is not None:
            yield segment


def _clip(origin: FC, d: FC, area: Box) -> tuple[Point, Point] | None:
    """The part of the line through ``origin`` along ``d`` inside ``area`` (Liang–Barsky)."""
    low, high = -math.inf, math.inf
    for p, q in (
        (-d[0], origin[0] - area.x),
        (d[0], area.right - origin[0]),
        (-d[1], origin[1] - area.y),
        (d[1], area.bottom - origin[1]),
    ):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            low = max(low, t)
        else:
            high = min(high, t)
    if high - low < 0.5:  # a sliver at a corner: leave it out
        return None
    a = Point(origin[0] + d[0] * low, origin[1] + d[1] * low)
    b = Point(origin[0] + d[0] * high, origin[1] + d[1] * high)
    return a, b


def _dots(area: Box, spacing: float, radius: float) -> Iterator[Point]:
    """Dots on a staggered grid, every one wholly inside ``area``."""
    row = math.ceil((area.y + radius) / spacing)
    while row * spacing <= area.bottom - radius:
        y = row * spacing
        shift = spacing / 2 if row % 2 else 0.0
        col = math.ceil((area.x + radius - shift) / spacing)
        while col * spacing + shift <= area.right - radius:
            yield Point(col * spacing + shift, y)
            col += 1
        row += 1


def _zigzags(area: Box, spacing: float) -> Iterator[list[Point]]:
    """Horizontal zigzag lines, 1.5 × ``spacing`` apart, each ``spacing`` / 2 tall."""
    pitch, amplitude = spacing * 1.5, spacing / 2
    row = math.ceil((area.y + amplitude) / pitch)
    while row * pitch + amplitude <= area.bottom:
        y = row * pitch
        points, x, up = [], math.floor(area.x / spacing) * spacing, row % 2 == 0
        while x <= area.right + spacing:
            points.append(Point(min(max(x, area.x), area.right), y - amplitude if up else y))
            x += spacing / 2
            up = not up
        yield points
        row += 1


def _dashes(area: Box, spacing: float) -> Iterator[tuple[Point, Point]]:
    """Short horizontal dashes in staggered rows, like brickwork."""
    row = math.ceil(area.y / spacing)
    while row * spacing <= area.bottom:
        y = row * spacing
        shift = spacing if row % 2 else 0.0
        x = math.floor((area.x - shift) / (2 * spacing)) * 2 * spacing + shift
        while x < area.right:
            a, b = max(x, area.x), min(x + spacing, area.right)
            if b - a > 0.5:
                yield Point(a, y), Point(b, y)
            x += 2 * spacing
        row += 1
