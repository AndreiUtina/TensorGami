"""Human-readable tensor shapes, following the style baseline (``64×56×56``, batch omitted)."""

from __future__ import annotations

from tensorgami.ir.model import Shape

TIMES = "×"


def format_shape(shape: Shape | None, *, drop_batch: bool = True) -> str:
    """``(1, 64, 56, 56)`` → ``"64×56×56"``. Unknown sizes print as ``?``.

    With ``drop_batch`` the leading (batch) dimension is omitted, except for 1-D shapes where
    nothing would be left.
    """
    if shape is None:
        return ""
    dims = shape[1:] if drop_batch and len(shape) > 1 else shape
    if not dims:
        return "scalar"
    return TIMES.join("?" if d is None else str(d) for d in dims)
