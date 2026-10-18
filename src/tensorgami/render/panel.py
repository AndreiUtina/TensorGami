"""What each layer produced on the example input, shown under the figure as it runs (roadmap
step 14d).

The panel is an overlay of an animation: a band under the figure where, at each step, a card
per running layer shows its activation, framed in the layer's colour. A card holds the layer's
name and output shape, a picture of the values (the most active feature maps, an input image,
a sequence as a heatmap, a vector's values as cells) in a ramp from the background to the
layer's colour, and their range. Once the whole model has run, the cards show its outputs.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Hashable, Mapping, Sequence

import numpy as np

from tensorgami.activations import Activation, Kind
from tensorgami.render.canvas import Canvas, Paint
from tensorgami.render.scene_renderer import FrameState, Item, Layer
from tensorgami.render.text import HeuristicTextMeasurer
from tensorgami.scene import Box, Point, Scene, TextAnchor
from tensorgami.style.color import Color
from tensorgami.style.glyphs import TextRole
from tensorgami.style.theme import Theme

MARGIN = 16  # the figure's margin, kept around the panel
GAP = 12  # between the figure and the cards
PAD = 8  # inside a card
TILE, TILE_GAP, COLUMNS, ROWS = 38, 3, 4, 2  # feature maps: up to 4 × 2 tiles
PICTURE_W = COLUMNS * TILE + (COLUMNS - 1) * TILE_GAP
PICTURE_H = ROWS * TILE + (ROWS - 1) * TILE_GAP
CARD_GAP = 10
VECTOR_COLUMNS = 32


class ActivationPanel:
    """The panel for ``scene`` (a figure): ``activations`` by node id, one per pass;
    ``passes``: the folded block (group id) whose badge says which pass a node is in;
    ``outputs``: the nodes shown once the model has run."""

    def __init__(
        self,
        scene: Scene,
        theme: Theme,
        activations: Mapping[str, Sequence[Activation]],
        *,
        passes: Mapping[str, str] | None = None,
        outputs: Sequence[str] = (),
    ) -> None:
        self._theme = theme
        self._activations = {k: tuple(v) for k, v in activations.items() if v}
        self._passes = dict(passes or {})
        self._order = [n.id for n in scene.nodes if n.id in self._activations]
        self._outputs = [n for n in outputs if n in self._activations]
        self._nodes = {n.id: n for n in scene.nodes}
        self._measure = HeuristicTextMeasurer()
        self._title = dataclasses.replace(theme.font(TextRole.LEGEND), weight=600)
        self._small = theme.font(TextRole.ANNOTATION)
        self._top = scene.height + GAP
        self.card_height = (
            PAD + self._title.size * 1.3 + 4 + PICTURE_H + 4 + self._small.size * 1.3 + PAD
        )
        self.scene = _extended(scene, GAP + self.card_height + MARGIN)

    def layer(self) -> Layer:
        return ("tg-activations", [Item("tg-activation-panel", None, self._look, self._paint)])

    # ------------------------------------------------------------------ what shows when

    def _look(self, state: FrameState | None, emphases: object) -> Hashable | None:
        if state is None:
            nodes = self._outputs
        else:
            nodes = [n for n in self._order if n in state.active]
            if not state.active:
                nodes = self._outputs  # the model has run: what it produced
        return tuple((n, self._pass(n, state)) for n in nodes) or None

    def _pass(self, node_id: str, state: FrameState | None) -> int:
        group = self._passes.get(node_id)
        badge = state.badges.get(group) if state is not None and group else None
        count = len(self._activations[node_id])
        if not badge or "/" not in badge:
            return 0
        return min(count - 1, max(0, int(badge.split("/")[0]) - 1))

    # ------------------------------------------------------------------ painting

    def _paint(self, canvas: Canvas, look: Hashable) -> None:
        cards = [(n, self._activations[n][k]) for n, k in look]  # type: ignore[union-attr]
        x, right = MARGIN, self.scene.width - MARGIN
        for i, (node_id, activation) in enumerate(cards):
            width = self._width(node_id, activation)
            if x + width > right and i:
                more = f"+{len(cards) - i} more"
                canvas.text(
                    Point(x, self._top + self.card_height / 2), more, self._small, TextAnchor.START
                )
                return
            self._card(canvas, Box(x, self._top, width, self.card_height), node_id, activation)
            x += width + CARD_GAP

    def _texts(self, node_id: str, activation: Activation) -> tuple[str, str, str]:
        node = self._nodes[node_id]
        title = node.label or node_id
        shape = "×".join(map(str, activation.shape[1:] or activation.shape))  # no batch
        values = f"{_short(activation.low)} … {_short(activation.high)}"
        if activation.kind is Kind.SCALAR:
            values = _short(activation.mean)
        if activation.kind is Kind.MAPS:
            values = f"{len(activation.channels)} of {activation.shape[-3]} maps · {values}"
        if activation.zeros >= 0.05:
            values += f" · {activation.zeros:.0%} zero"
        return title, shape, values

    def _width(self, node_id: str, activation: Activation) -> float:
        title, shape, values = self._texts(node_id, activation)
        # an estimate, kept generous: "…" and "·" come from wider fallback fonts
        head = (
            self._measure.width(title, self._title) + 10 + self._measure.width(shape, self._small)
        )
        foot = self._measure.width(values, self._small)
        return math.ceil(max(PICTURE_W, 1.08 * head, 1.12 * foot) + 2 * PAD)

    def _card(self, canvas: Canvas, box: Box, node_id: str, activation: Activation) -> None:
        theme = self._theme
        node = self._nodes[node_id]
        ink = theme.node_style(node.category).stroke
        canvas.rect(
            box,
            Paint(theme.background, ink, theme.metrics.node_stroke),
            theme.metrics.corner_radius,
        )
        title, shape, values = self._texts(node_id, activation)
        line = box.y + PAD + self._title.size * 0.65
        canvas.text(Point(box.x + PAD, line), title, self._title, TextAnchor.START)
        canvas.text(Point(box.right - PAD, line), shape, self._small, TextAnchor.END)
        picture = Box(box.x + PAD, box.y + PAD + self._title.size * 1.3 + 4, PICTURE_W, PICTURE_H)
        self._picture(canvas, picture, activation, _ramp(theme.background, ink))
        foot = picture.bottom + 4 + self._small.size * 0.65
        canvas.text(Point(box.x + PAD, foot), values, self._small, TextAnchor.START)

    def _picture(self, canvas: Canvas, area: Box, a: Activation, ramp: np.ndarray) -> None:
        outline = Paint(None, self._theme.group.stroke, 0.5)
        if a.kind is Kind.MAPS:
            for i, channel in enumerate(a.preview):
                row, column = divmod(i, COLUMNS)
                tile = Box(
                    area.x + column * (TILE + TILE_GAP),
                    area.y + row * (TILE + TILE_GAP),
                    TILE,
                    TILE,
                )
                canvas.image(tile, ramp[_levels(channel)])
                canvas.rect(tile, outline)
        elif a.kind is Kind.IMAGE:
            c, h, w = a.preview.shape
            scale = min(area.w / w, area.h / h)
            box = Box(area.x, area.y, w * scale, h * scale)
            if c == 3:
                pixels = np.moveaxis(_levels(a.preview), 0, -1).astype(np.uint8)
            else:
                pixels = ramp[_levels(a.preview[0])]
            canvas.image(box, pixels, smooth=True)
            canvas.rect(box, outline)
        elif a.kind is Kind.SEQUENCE:
            canvas.image(area, ramp[_levels(a.preview)])
            canvas.rect(area, outline)
        elif a.kind is Kind.VECTOR:
            n = len(a.preview)
            columns = min(n, VECTOR_COLUMNS)
            rows = math.ceil(n / columns)
            cell = min(area.w / columns, area.h / rows)
            grid = np.full(rows * columns, np.nan)
            grid[:n] = a.preview
            levels = _levels(grid.reshape(rows, columns))
            pixels = ramp[levels]
            pixels[np.isnan(grid.reshape(rows, columns))] = ramp[0]
            box = Box(area.x, area.y, columns * cell, rows * cell)
            canvas.image(box, pixels)
            canvas.rect(box, outline)
        else:  # a scalar: its value, large
            font = dataclasses.replace(self._title, size=self._title.size * 1.6)
            canvas.text(area.center, _short(a.mean), font)


def _levels(values: np.ndarray) -> np.ndarray:
    """Values scaled to 0 … 255 between their own minimum and maximum (NaN as 0)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.intp)
    low, high = float(finite.min()), float(finite.max())
    span = high - low or 1.0
    scaled = np.nan_to_num((values - low) / span, nan=0.0)
    return np.clip(np.round(scaled * 255), 0, 255).astype(np.intp)


def _ramp(start: Color, end: Color) -> np.ndarray:
    """256 RGB colours from ``start`` (low values) to ``end`` (high)."""
    a = np.array([start.r, start.g, start.b], dtype=np.float64)
    b = np.array([end.r, end.g, end.b], dtype=np.float64)
    t = np.linspace(0, 1, 256)[:, None]
    return np.round(a + (b - a) * t).astype(np.uint8)


def _short(value: float) -> str:
    """A value in at most three significant digits, with a true minus sign."""
    text = f"{value:.3g}"
    return text.replace("-", "−")


def _extended(scene: Scene, extra: float) -> Scene:
    """``scene`` with ``extra`` px of room below it."""
    out = Scene(scene.width, scene.height + extra, scene.title, scene.description)
    out.add(*scene.nodes)
    out.add(*scene.groups)
    out.add(*scene.edges)
    out.add(*scene.notes)
    return out
