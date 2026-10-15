"""SVG implementation of :class:`~tensorgami.render.canvas.Canvas`."""

from __future__ import annotations

from collections.abc import Sequence
from xml.sax.saxutils import escape, quoteattr

from tensorgami.render.canvas import Paint
from tensorgami.render.text import (
    BASELINE_SHIFT,
    SCRIPT_SCALE,
    Script,
    display_text,
    parse_markup,
)
from tensorgami.scene import Box, Point, TextAnchor
from tensorgami.style.glyphs import Font

SCRIPT_SHIFT = {Script.SUB: 0.28, Script.SUPER: -0.42}  # baseline shift of scripts, in em


def _num(value: float) -> str:
    """Compact, deterministic number formatting (max two decimals)."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _points(points: Sequence[Point]) -> str:
    return " ".join(f"{_num(p.x)},{_num(p.y)}" for p in points)


class SvgCanvas:
    """Accumulates SVG elements; :meth:`to_svg` returns a standalone document."""

    def __init__(
        self,
        width: float,
        height: float,
        title: str = "",
        description: str = "",
        *,
        inches: tuple[float, float] | None = None,
    ):
        self.width = width
        self.height = height
        self.title = title
        self.description = description
        self.inches = inches  # the printed size, when it is not the natural one (1 px = 1/96 in)
        self._parts: list[str] = []
        self._depth = 0

    # ------------------------------------------------------------------ primitives

    def rect(self, box: Box, paint: Paint, radius: float = 0.0) -> None:
        r = min(radius, box.w / 2, box.h / 2)
        corner = f' rx="{_num(r)}"' if r > 0 else ""
        self._emit(
            f'<rect x="{_num(box.x)}" y="{_num(box.y)}" width="{_num(box.w)}" '
            f'height="{_num(box.h)}"{corner}{self._paint(paint)}/>'
        )

    def circle(self, center: Point, radius: float, paint: Paint) -> None:
        self._emit(
            f'<circle cx="{_num(center.x)}" cy="{_num(center.y)}" r="{_num(radius)}"'
            f"{self._paint(paint)}/>"
        )

    def polygon(self, points: Sequence[Point], paint: Paint) -> None:
        self._emit(f'<polygon points="{_points(points)}"{self._paint(paint)}/>')

    def polyline(self, points: Sequence[Point], paint: Paint) -> None:
        self._emit(f'<polyline points="{_points(points)}"{self._paint(paint)}/>')

    def text(
        self,
        at: Point,
        text: str,
        font: Font,
        anchor: TextAnchor = TextAnchor.MIDDLE,
        rotation: float = 0.0,
    ):
        runs = parse_markup(display_text(text, font))
        if not runs:
            return
        attrs = (
            f'x="{_num(at.x)}" y="{_num(at.y + BASELINE_SHIFT * font.size)}" '
            f'font-family={quoteattr(font.css_family)} font-size="{_num(font.size)}"'
        )
        if rotation:  # SVG angles turn clockwise
            attrs += f' transform="rotate({_num(-rotation)} {_num(at.x)} {_num(at.y)})"'
        if font.weight != 400:
            attrs += f' font-weight="{font.weight}"'
        if font.italic:
            attrs += ' font-style="italic"'
        if anchor is not TextAnchor.START:
            attrs += f' text-anchor="{anchor}"'
        attrs += f' fill="{font.color.hex}"'
        body, shift = [], 0.0
        for run in runs:
            target = SCRIPT_SHIFT.get(run.script, 0.0) * font.size
            dy = f' dy="{_num(target - shift)}"' if target != shift else ""
            shift = target
            if run.script is Script.NORMAL:
                body.append(f"<tspan{dy}>{escape(run.text)}</tspan>" if dy else escape(run.text))
            else:
                size = _num(font.size * SCRIPT_SCALE)
                body.append(f'<tspan{dy} font-size="{size}">{escape(run.text)}</tspan>')
        self._emit(f"<text {attrs}>{''.join(body)}</text>")

    def begin_group(self, css_class: str, data_id: str | None = None) -> None:
        ident = f" data-id={quoteattr(data_id)}" if data_id else ""
        self._emit(f"<g class={quoteattr(css_class)}{ident}>")
        self._depth += 1

    def end_group(self) -> None:
        if self._depth == 0:
            raise RuntimeError("end_group() without matching begin_group()")
        self._depth -= 1
        self._emit("</g>")

    # ------------------------------------------------------------------ output

    def to_svg(self) -> str:
        if self._depth:
            raise RuntimeError(f"{self._depth} group(s) left open")
        w, h = _num(self.width), _num(self.height)
        size_w, size_h = w, h
        if self.inches is not None:
            size_w, size_h = (f"{_num(round(v, 3))}in" for v in self.inches)
        head = [
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{size_w}" height="{size_h}" viewBox="0 0 {w} {h}" role="img">'
        ]
        if self.title:
            head.append(f"  <title>{escape(self.title)}</title>")
        if self.description:
            head.append(f"  <desc>{escape(self.description)}</desc>")
        return "\n".join([*head, *self._parts, "</svg>", ""])

    # ------------------------------------------------------------------ helpers

    def _emit(self, element: str) -> None:
        indent = "  " * (self._depth + 1)
        self._parts.append(indent + element)

    @staticmethod
    def _paint(paint: Paint) -> str:
        fill = paint.fill.hex if paint.fill else "none"
        out = f' fill="{fill}"'
        if paint.fill and paint.opacity < 1:
            out += f' fill-opacity="{_num(paint.opacity)}"'
        if paint.stroke:
            out += f' stroke="{paint.stroke.hex}" stroke-width="{_num(paint.width)}"'
            out += ' stroke-linejoin="round"'
            if paint.dash:
                out += f' stroke-dasharray="{" ".join(_num(d) for d in paint.dash)}"'
        return out
