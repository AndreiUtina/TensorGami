"""Immutable RGB colour value object with the colour maths the style rules need."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_HEX = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


@dataclass(frozen=True, slots=True)
class Color:
    """An sRGB colour with 8-bit channels."""

    r: int
    g: int
    b: int

    def __post_init__(self) -> None:
        for channel in (self.r, self.g, self.b):
            if not 0 <= channel <= 255:
                raise ValueError(f"colour channel out of range 0..255: {channel}")

    @classmethod
    def from_hex(cls, value: str) -> Color:
        """Parse ``#rgb`` or ``#rrggbb``."""
        if not isinstance(value, str) or not _HEX.match(value):
            raise ValueError(f"invalid hex colour: {value!r}")
        digits = value[1:]
        if len(digits) == 3:
            digits = "".join(c * 2 for c in digits)
        return cls(int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))

    @property
    def hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def __str__(self) -> str:
        return self.hex

    @property
    def rgb01(self) -> tuple[float, float, float]:
        """Channels as floats in 0..1 (matplotlib's convention)."""
        return self.r / 255, self.g / 255, self.b / 255

    # ------------------------------------------------------------------ mixing

    def mix(self, other: Color, amount: float) -> Color:
        """Linear blend in sRGB space; ``amount=0`` is ``self``, ``1`` is ``other``."""
        if not 0.0 <= amount <= 1.0:
            raise ValueError(f"mix amount must be within 0..1, got {amount}")

        def blend(a: int, b: int) -> int:
            return round(a + (b - a) * amount)

        return Color(blend(self.r, other.r), blend(self.g, other.g), blend(self.b, other.b))

    def lighten(self, amount: float) -> Color:
        return self.mix(WHITE, amount)

    def darken(self, amount: float) -> Color:
        return self.mix(BLACK, amount)

    # ------------------------------------------------------------------ perception

    @property
    def relative_luminance(self) -> float:
        """WCAG 2.x relative luminance (0 = black, 1 = white)."""
        r, g, b = (_linearise(c) for c in (self.r, self.g, self.b))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def contrast_ratio(self, other: Color) -> float:
        """WCAG contrast ratio, from 1 (identical) to 21 (black on white)."""
        lighter, darker = sorted((self.relative_luminance, other.relative_luminance), reverse=True)
        return (lighter + 0.05) / (darker + 0.05)

    def to_lab(self) -> tuple[float, float, float]:
        """CIE L*a*b* under the D65 illuminant."""
        r, g, b = (_linearise(c) for c in (self.r, self.g, self.b))
        x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
        fx, fy, fz = (_lab_f(t) for t in (x, y, z))
        return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)

    @property
    def chroma(self) -> float:
        """CIE L*C*h chroma: ~0 for greys, large for saturated colours."""
        _, a, b = self.to_lab()
        return math.hypot(a, b)

    def delta_e(self, other: Color) -> float:
        """CIE76 colour difference (≈2.3 is a just-noticeable difference)."""
        return math.dist(self.to_lab(), other.to_lab())


def _linearise(channel: int) -> float:
    c = channel / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lab_f(t: float) -> float:
    delta = 6 / 29
    return t ** (1 / 3) if t > delta**3 else t / (3 * delta**2) + 4 / 29


WHITE = Color(255, 255, 255)
BLACK = Color(0, 0, 0)
