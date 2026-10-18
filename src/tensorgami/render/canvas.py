"""The drawing surface abstraction every renderer backend implements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from tensorgami.scene import Box, Point, TextAnchor
from tensorgami.style.color import Color
from tensorgami.style.glyphs import Font

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class Paint:
    fill: Color | None = None
    stroke: Color | None = None
    width: float = 1.0
    dash: tuple[float, ...] = ()
    opacity: float = 1.0  # of the fill only; translucent layers (e.g. the origami style)

    def __post_init__(self) -> None:
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"opacity must be within 0..1, got {self.opacity}")


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
        self,
        at: Point,
        text: str,
        font: Font,
        anchor: TextAnchor = TextAnchor.MIDDLE,
        rotation: float = 0.0,
    ) -> None:
        """One line whose visual centre line passes through ``at``; ``rotation`` turns it
        counter-clockwise about ``at`` (degrees; 90 reads bottom → top)."""
        ...

    def image(self, box: Box, pixels: np.ndarray, *, smooth: bool = False) -> None:
        """An RGB raster (``h × w × 3``, uint8) stretched over ``box``; its pixels drawn as
        squares unless ``smooth`` (a feature map shows its resolution)."""
        ...

    def begin_group(self, css_class: str, data_id: str | None = None) -> None: ...

    def end_group(self) -> None: ...
