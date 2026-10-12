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
            _clamp(2.0 * math.sqrt(c), 6, 56),
            _clamp(7 * math.sqrt(h), 12, 112),
            _clamp(3.5 * math.sqrt(w), 5, 56),
        )
    (f,) = volume
    return 8.0, _clamp(1.8 * math.sqrt(f), 24, 128), 5.0


def volume_label(volume: tuple[int, ...]) -> str:
    """``(64, 55, 55)`` → ``"64×55×55"``."""
    return format_shape((1, *volume))
