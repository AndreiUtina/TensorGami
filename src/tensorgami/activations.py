"""Real activations (roadmap step 14d): what each layer produced on the example input.

``tg.draw(model, x, activations=True)`` keeps, for every traced operation and module call, a
small summary of its output (:class:`Activation`): the value range, the share of zeros and a
preview to draw (the most active channels of a feature map, an input image, a sequence as a
heatmap, a vector's values). Summaries are made while the model runs, so no tensor is kept.
Which box of a figure shows which summary is :mod:`tensorgami.ir.activations`' business.

numpy only (it comes with matplotlib): nothing here needs PyTorch, the graph or the drawing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

MAX_CHANNELS = 8  # feature maps shown per layer: the most active
MAX_SIDE = 32  # a feature map's preview, at most this many pixels a side
MAX_IMAGE = 64  # an input image's preview
MAX_VALUES = 256  # a vector's preview
MAX_ROWS, MAX_COLUMNS = 32, 64  # a sequence's preview (positions × features)


class Kind(StrEnum):
    MAPS = "maps"  # C × H × W: feature maps, the most active channels shown
    IMAGE = "image"  # an input picture: RGB (or grey) as it is
    SEQUENCE = "sequence"  # T × D: positions × features, as a heatmap
    VECTOR = "vector"  # D values
    SCALAR = "scalar"  # one value (a loss)


@dataclass(frozen=True, eq=False)
class Activation:
    """A summary of one output on the example input (the batch's first sample).

    ``shape`` is the full shape, batch included; ``low``, ``high``, ``mean`` the range and
    mean of the values; ``zeros`` the share that is exactly zero (a ReLU's sparsity);
    ``preview`` a small float array to draw (see :class:`Kind`), with, for feature maps, the
    ``channels`` it shows."""

    shape: tuple[int, ...]
    kind: Kind
    low: float
    high: float
    mean: float
    zeros: float
    preview: np.ndarray = field(repr=False)
    channels: tuple[int, ...] = ()

    def __repr__(self) -> str:
        shape = "×".join(map(str, self.shape))
        return f"<Activation {self.kind} {shape}, {self.low:.3g} … {self.high:.3g}>"


def summarize(values: np.ndarray, *, image: bool = False) -> Activation | None:
    """The summary of one output (its batch's first sample); ``image``: an input that may be
    a picture (3 or 1 channels, drawn as it is). ``None`` when nothing finite is left."""
    full = tuple(int(d) for d in values.shape)
    x = np.abs(values) if np.iscomplexobj(values) else values
    x = np.asarray(x, dtype=np.float64)
    if x.ndim >= 2:
        x = x[0]  # the first sample of the batch
    while x.ndim > 3:
        x = x[0]  # a sequence of maps (T × C × H × W): its first step
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return None
    stats = {
        "low": float(finite.min()),
        "high": float(finite.max()),
        "mean": float(finite.mean()),
        "zeros": float(np.mean(finite == 0)),
    }
    x = np.nan_to_num(x, nan=0.0, posinf=stats["high"], neginf=stats["low"])
    if x.ndim == 3 and x.shape[1] * x.shape[2] == 1:
        x = x.reshape(-1)  # C × 1 × 1 (after global pooling): a vector
    if x.ndim == 3:
        c, h, w = x.shape
        if image and c in (1, 3) and min(h, w) >= 8:
            return Activation(full, Kind.IMAGE, preview=_shrink(x, MAX_IMAGE), **stats)
        energy = np.abs(x).mean(axis=(1, 2))
        top = np.argsort(-energy, kind="stable")[:MAX_CHANNELS]
        maps = _shrink(x[top], MAX_SIDE)
        return Activation(full, Kind.MAPS, preview=maps, channels=tuple(map(int, top)), **stats)
    if x.ndim == 2:
        if x.shape[0] == 1:
            x = x[0]
        else:
            preview = _bins(_bins(x, 0, MAX_ROWS), 1, MAX_COLUMNS)
            return Activation(full, Kind.SEQUENCE, preview=preview, **stats)
    if x.ndim == 1:
        return Activation(full, Kind.VECTOR, preview=_bins(x, 0, MAX_VALUES), **stats)
    return Activation(full, Kind.SCALAR, preview=np.reshape(x, (1,)), **stats)


def _shrink(x: np.ndarray, side: int) -> np.ndarray:
    """Maps (… × H × W) averaged down to at most ``side`` pixels a side."""
    return _bins(_bins(x, x.ndim - 2, side), x.ndim - 1, side)


def _bins(x: np.ndarray, axis: int, most: int) -> np.ndarray:
    """``x`` averaged along ``axis`` into at most ``most`` equal-ish bins."""
    n = x.shape[axis]
    if n <= most:
        return x.astype(np.float32)
    edges = np.linspace(0, n, most + 1).astype(int)
    sums = np.add.reduceat(x, edges[:-1], axis=axis)
    counts = np.diff(edges).reshape([-1 if i == axis else 1 for i in range(x.ndim)])
    return (sums / counts).astype(np.float32)
