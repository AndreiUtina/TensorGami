"""Variants of a style (roadmap step 13): the same style, recoloured for where the figure goes.

A variant is a function ``Theme → Theme``, so it applies to every style, the built-in ones and
your own ``.toml``: ``tg.draw(model, x, style="origami", variant="dark")``. Variants combine,
applied in order (``variant=["dark", "greyscale"]``). The result passes the same validation
rules as every style (text contrast, outlines against the background, distinct categories).

* ``dark``: for slides with a dark background. The cards keep their pastel fills and dark
  text; the background turns dark, and wires, outlines, frames and the text printed on the
  background (tensor shapes, titles, legend) are lightened just enough to stand out, so each
  keeps its hue.
* ``greyscale``: for black-and-white print. Every colour becomes the grey of the same
  luminance, so every contrast ratio stays exactly as it was; the category fills, pastels of
  nearly equal lightness, are spread over a wider range of greys so that they stay apart.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable

from tensorgami.style.color import WHITE, Color
from tensorgami.style.glyphs import TextRole
from tensorgami.style.theme import Theme

DARK_BACKGROUND = Color.from_hex("#0d1117")  # GitHub's dark page, as the logo banner

#: text printed on the background, and the contrast it gets on a dark one (WCAG AA is 4.5)
_ON_BACKGROUND = {
    TextRole.LABEL: 12.0,  # free notes such as ⋯ (labels on cards take their category's colour)
    TextRole.LEGEND: 12.0,  # near white
    TextRole.GROUP_TITLE: 9.0,
    TextRole.ANNOTATION: 6.5,  # muted, as on paper
    TextRole.NOTE: 6.5,
    TextRole.BADGE: 6.5,
}


def _visible(color: Color, background: Color, ratio: float) -> Color:
    """``color`` mixed with as little white as reaches ``ratio`` against ``background``."""
    if color.contrast_ratio(background) >= ratio:
        return color
    low, high = 0.0, 1.0
    for _ in range(20):  # bisection: deterministic, well below a 1/255 step
        mid = (low + high) / 2
        if color.mix(WHITE, mid).contrast_ratio(background) >= ratio:
            high = mid
        else:
            low = mid
    return color.mix(WHITE, high)


def dark(theme: Theme) -> Theme:
    """The style on a dark background (slides, dark-mode pages)."""
    bg = DARK_BACKGROUND
    nodes = {  # opaque: on a dark board translucent paper would only look dim
        category: dataclasses.replace(style, stroke=_visible(style.stroke, bg, 3.0), opacity=1.0)
        for category, style in theme.nodes.items()
    }
    edges = {
        kind: dataclasses.replace(style, color=_visible(style.color, bg, 4.5))
        for kind, style in theme.edges.items()
    }
    fonts = dict(theme.fonts)
    for role, ratio in _ON_BACKGROUND.items():
        fonts[role] = dataclasses.replace(fonts[role], color=_visible(fonts[role].color, bg, ratio))
    group = dataclasses.replace(
        theme.group,
        fill=bg.mix(WHITE, 0.035),  # a frame lifts only slightly off the board
        stroke=_visible(theme.group.stroke, bg, 3.0),
        badge_fill=bg.mix(WHITE, 0.1),
    )
    return dataclasses.replace(
        theme,
        name=f"{theme.name}, dark",
        background=bg,
        nodes=nodes,
        edges=edges,
        fonts=fonts,
        group=group,
    )


def _grey(color: Color) -> Color:
    """The grey of the same relative luminance (so every contrast ratio is kept)."""
    target = color.relative_luminance
    low, high = 0, 255
    while low < high:  # the smallest grey level at least as bright
        mid = (low + high) // 2
        if Color(mid, mid, mid).relative_luminance < target:
            low = mid + 1
        else:
            high = mid
    return Color(low, low, low)


def greyscale(theme: Theme) -> Theme:
    """The style in greys (black-and-white print)."""
    nodes = {
        c: dataclasses.replace(s, fill=_grey(s.fill), stroke=_grey(s.stroke), text=_grey(s.text))
        for c, s in theme.nodes.items()
    }
    # the coloured categories' pastels are nearly equally light: spread them over the
    # lightness range from the lightest one down, in the same order (darkest stays darkest)
    coloured = sorted(
        (c for c, s in theme.nodes.items() if s.fill.chroma >= 10),
        key=lambda c: (theme.nodes[c].fill.relative_luminance, c.value),
    )
    if len(coloured) > 1:
        top = max(nodes[c].fill.r for c in coloured)
        darkest = 150  # still ≥ 4.5 : 1 under near-black text
        for k, c in enumerate(coloured):
            level = round(darkest + (top - darkest) * k / (len(coloured) - 1))
            nodes[c] = dataclasses.replace(nodes[c], fill=Color(level, level, level))
    fonts = {r: dataclasses.replace(f, color=_grey(f.color)) for r, f in theme.fonts.items()}
    detail = fonts[TextRole.DETAIL]  # the detail line sits on the darkest fill too
    level = min(detail.color.r, 38)
    fonts[TextRole.DETAIL] = dataclasses.replace(detail, color=Color(level, level, level))
    background = _grey(theme.background)
    # kinds of wire told apart only by hue would merge: nudge a repeated grey away from the
    # background (more contrast, never less) until every kind looks different again
    step = 20 if background.relative_luminance < 0.5 else -20
    edges, seen = {}, set()
    for kind, style in theme.edges.items():
        style = dataclasses.replace(style, color=_grey(style.color))
        while (style.color, style.dash, style.arrow) in seen and 0 <= style.color.r + step <= 255:
            level = style.color.r + step
            style = dataclasses.replace(style, color=Color(level, level, level))
        seen.add((style.color, style.dash, style.arrow))
        edges[kind] = style
    group = dataclasses.replace(
        theme.group,
        fill=_grey(theme.group.fill),
        stroke=_grey(theme.group.stroke),
        badge_fill=_grey(theme.group.badge_fill),
    )
    return dataclasses.replace(
        theme,
        name=f"{theme.name}, greyscale",
        background=background,
        nodes=nodes,
        edges=edges,
        fonts=fonts,
        group=group,
    )


VARIANTS: dict[str, Callable[[Theme], Theme]] = {"dark": dark, "greyscale": greyscale}


def apply_variants(theme: Theme, variants: str | Iterable[str] | None) -> Theme:
    """``theme`` with the named variants applied, in order (``None`` or ``()``: unchanged)."""
    if variants is None:
        return theme
    names = [variants] if isinstance(variants, str) else list(variants)
    for name in names:
        if name not in VARIANTS:
            raise ValueError(f"unknown variant {name!r}; available: {', '.join(sorted(VARIANTS))}")
        theme = VARIANTS[name](theme)
    return theme


def available_variants() -> list[str]:
    return sorted(VARIANTS)
