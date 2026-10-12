"""Geometry of tensor volumes (the volume view, step 11c).

A volume ``(C, H, W)`` is a box whose front face is ``C`` thick and ``H`` tall, with ``W`` as
the oblique depth; a flat vector ``(F,)`` is a thin bar ``F`` tall. Sizes grow with the square
root of the dimension, so a 224 × 224 image and a 6 × 6 map are both legible.
"""

from __future__ import annotations

import math

from tensorgami.ir import format_shape


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def volume_geometry(volume: tuple[int, ...]) -> tuple[float, float, float]:
    """(front-face width, front-face height, oblique depth) in px."""
    if len(volume) == 3:
        c, h, w = volume
        return (
            _clamp(2.2 * math.sqrt(c), 6, 64),
            _clamp(10 * math.sqrt(h), 14, 160),
            _clamp(5 * math.sqrt(w), 6, 80),
        )
    (f,) = volume
    return 8.0, _clamp(2.4 * math.sqrt(f), 24, 170), 6.0


def volume_label(volume: tuple[int, ...]) -> str:
    """``(64, 55, 55)`` → ``"64×55×55"``."""
    return format_shape((1, *volume))
