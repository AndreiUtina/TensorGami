"""Colour palettes (roadmap step 13): the operation categories recoloured, for any style.

A palette gives each kind of operation a base colour; the fill is a tint of it light enough for
the style's text to keep a contrast of at least 4.5 : 1, the outline the colour itself (darkened
where needed to show on the background, 3 : 1), and the coloured wires (recurrent state,
sampling, training) take their category's outline. Shapes, fonts and the neutral categories
(inputs, tensors, merges, reshapes) stay the style's. Palettes combine with variants:
``tg.draw(model, x, style="origami", palette="colorblind", variant="dark")``.

* ``colorblind``: Okabe & Ito's palette (2008), the usual recommendation for scientific figures
  read by people with colour-vision deficiency, completed from Paul Tol's colour-blind-safe
  sets (cyan, purple, sand, red, indigo); every pair stays apart in the style checks.
* ``tab10``: matplotlib's default categorical colours, so a figure matches the plots of the same
  paper.
* ``vivid``: saturated hues with stronger fills, for slides and posters, where pastels wash out
  on a projector.
* ``mono``: one hue. Layers with weights (convolution, linear, recurrent, attention, embedding)
  in a medium tint, the operations between them lighter, every outline the same slate: colour
  marks where the parameters are, the labels say the rest.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from tensorgami.style.color import BLACK, WHITE, Color
from tensorgami.style.glyphs import TextRole
from tensorgami.style.theme import Theme
from tensorgami.vocabulary import EdgeKind, OpCategory

C = OpCategory


@dataclass(frozen=True)
class Palette:
    name: str
    description: str
    colors: Mapping[OpCategory, str]  # base colour per category (hex)
    tint: Mapping[OpCategory, float] | float = 0.6  # the fill: at least this much white
    outline: str | None = None  # one outline for every category; else each its base colour

    def __post_init__(self) -> None:
        object.__setattr__(self, "colors", MappingProxyType(dict(self.colors)))
        if isinstance(self.tint, Mapping):
            object.__setattr__(self, "tint", MappingProxyType(dict(self.tint)))

    def tint_of(self, category: OpCategory) -> float:
        return self.tint[category] if isinstance(self.tint, Mapping) else self.tint


_WEIGHTS = (C.CONV, C.LINEAR, C.RECURRENT, C.ATTENTION, C.EMBEDDING)

PALETTES: dict[str, Palette] = {
    p.name: p
    for p in (
        Palette(
            "colorblind",
            "Okabe–Ito with Paul Tol's colour-blind-safe hues: safe for colour-vision deficiency",
            {
                C.CONV: "#0072B2",  # Okabe–Ito blue
                C.NORM: "#F0E442",  # yellow
                C.ACTIVATION: "#D55E00",  # vermillion
                C.ATTENTION: "#E69F00",  # orange
                C.RECURRENT: "#009E73",  # bluish green
                C.RESAMPLE: "#88CCEE",  # Tol cyan
                C.LINEAR: "#AA4499",  # Tol purple
                C.EMBEDDING: "#DDCC77",  # Tol sand
                C.LOSS: "#EE6677",  # Tol red
                C.STOCHASTIC: "#332288",  # Tol indigo
            },
            tint=0.62,
        ),
        Palette(
            "tab10",
            "matplotlib's default categorical colours: matches the plots of the same paper",
            {
                C.CONV: "#1f77b4",  # tab:blue
                C.ATTENTION: "#ff7f0e",  # tab:orange
                C.RECURRENT: "#2ca02c",  # tab:green
                C.LOSS: "#d62728",  # tab:red
                C.LINEAR: "#9467bd",  # tab:purple
                C.EMBEDDING: "#8c564b",  # tab:brown
                C.ACTIVATION: "#e377c2",  # tab:pink
                C.STOCHASTIC: "#7f7f7f",  # tab:gray
                C.NORM: "#bcbd22",  # tab:olive
                C.RESAMPLE: "#17becf",  # tab:cyan
            },
            tint=0.6,
        ),
        Palette(
            "vivid",
            "saturated hues with strong fills: for slides and posters",
            {
                C.CONV: "#1F6FEB",
                C.NORM: "#F4C20D",
                C.ACTIVATION: "#E8308C",
                C.RESAMPLE: "#00A6A6",
                C.LINEAR: "#8E44AD",
                C.ATTENTION: "#FB8500",
                C.RECURRENT: "#2DA44E",
                C.EMBEDDING: "#A0522D",
                C.LOSS: "#D7263D",
                C.STOCHASTIC: "#0EA5E9",
            },
            tint=0.42,
        ),
        Palette(
            "mono",
            "one hue: layers with weights darker, the operations between them lighter",
            dict.fromkeys(
                (*_WEIGHTS, C.NORM, C.ACTIVATION, C.RESAMPLE, C.LOSS, C.STOCHASTIC), "#1F5F9E"
            ),
            tint={
                **dict.fromkeys(_WEIGHTS, 0.62),
                C.LOSS: 0.72,
                C.STOCHASTIC: 0.78,
                C.NORM: 0.84,
                C.RESAMPLE: 0.84,
                C.ACTIVATION: 0.9,
            },
            outline="#3B4A5C",
        ),
    )
}

OUTLINE = 3.05  # outlines against the background: WCAG's 3 : 1, with room for rounding (greys)

#: coloured wires take the outline of their category
_WIRES = {EdgeKind.STATE: C.RECURRENT, EdgeKind.STOCHASTIC: C.STOCHASTIC, EdgeKind.TRAINING: C.LOSS}


def _darkened(color: Color, background: Color, ratio: float) -> Color:
    """``color`` mixed with as little black as reaches ``ratio`` against ``background``."""
    if color.contrast_ratio(background) >= ratio:
        return color
    low, high = 0.0, 1.0
    for _ in range(20):
        mid = (low + high) / 2
        if color.mix(BLACK, mid).contrast_ratio(background) >= ratio:
            high = mid
        else:
            low = mid
    return color.mix(BLACK, high)


def _fill(base: Color, tint: float, texts: list[Color], theme: Theme, opacity: float) -> Color:
    """The tint of ``base`` with at least ``tint`` white on which every text reaches 4.5 : 1,
    both as seen through a translucent card and opaque (as a dark variant makes it)."""
    amount = tint
    while amount < 0.97:
        fill = base.mix(WHITE, amount)
        seen = fill.mix(theme.background, 1 - opacity)
        if all(min(t.contrast_ratio(seen), t.contrast_ratio(fill)) >= 4.5 for t in texts):
            return fill
        amount += 0.01
    return base.mix(WHITE, 0.97)


def apply_palette(theme: Theme, name: str) -> Theme:
    """``theme`` with the operation categories in palette ``name``'s colours."""
    if name not in PALETTES:
        raise ValueError(f"unknown palette {name!r}; available: {', '.join(PALETTES)}")
    palette = PALETTES[name]
    detail = theme.font(TextRole.DETAIL).color
    nodes = dict(theme.nodes)
    for category, hex_color in palette.colors.items():
        style = theme.nodes[category]
        base = Color.from_hex(hex_color)
        texts = [style.text, detail]
        fill = _fill(base, palette.tint_of(category), texts, theme, style.opacity)
        outline = Color.from_hex(palette.outline) if palette.outline else base
        stroke = _darkened(outline, theme.background, OUTLINE)
        nodes[category] = dataclasses.replace(style, fill=fill, stroke=stroke)
    edges = dict(theme.edges)
    for kind, category in _WIRES.items():
        if category in palette.colors:
            color = Color.from_hex(palette.colors[category])
            edges[kind] = dataclasses.replace(
                edges[kind], color=_darkened(color, theme.background, OUTLINE)
            )
    return dataclasses.replace(theme, name=f"{theme.name}, {name}", nodes=nodes, edges=edges)


def available_palettes() -> list[str]:
    return list(PALETTES)


__all__ = ["PALETTES", "Palette", "apply_palette", "available_palettes"]
