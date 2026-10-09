"""The drawing surface abstraction every renderer backend implements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from tensorgami.scene import Box, Point, TextAnchor
from tensorgami.style.color import Color
from tensorgami.style.glyphs import Font


@dataclass(frozen=True, slots=True)
class Paint:
    fill: Color | None = None
    stroke: Color | None = None
    width: float = 1.0
    dash: tuple[float, ...] = ()


class Canvas(Protocol):
    """Minimal vector-drawing interface (SVG today; PDF, TikZ or matplotlib later).

    ``text`` is positioned by the vertical *centre* of the line, and supports the tiny markup
    ``x_{t-1}`` (subscript) and ``h^{2}`` (superscript).
    """

    def rect(self, box: Box, paint: Paint, radius: float = 0.0) -> None: ...

    def circle(self, center: Point, radius: float, paint: Paint) -> None: ...

    def polygon(self, points: Sequence[Point], paint: Paint) -> None: ...

    def polyline(self, points: Sequence[Point], paint: Paint) -> None: ...

    def text(
        self, at: Point, text: str, font: Font, anchor: TextAnchor = TextAnchor.MIDDLE
    ) -> None: ...

    def begin_group(self, css_class: str, data_id: str | None = None) -> None: ...

    def end_group(self) -> None: ...
