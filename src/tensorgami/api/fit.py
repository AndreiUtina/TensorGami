"""Figures made for where they go (roadmap step 13c): ``tg.draw(..., fit="slide")``.

A slide is a fixed canvas, so the figure that serves it best is not the one at its natural
size but the one whose text is largest once it fills the slide. ``fit`` draws the model in the
slide style (larger text and lines, see :func:`~tensorgami.style.variants.presentation`),
tries rows (or, top → bottom, columns) of several lengths and keeps the layout that fills the
slide with the largest text, then centres it on a canvas of the slide's exact proportions.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Collection
from dataclasses import dataclass

from tensorgami.ir import Graph
from tensorgami.layout import LayoutOptions
from tensorgami.render.mpl import PT_PER_PX, PX_PER_INCH
from tensorgami.scene import Scene
from tensorgami.style import TextRole, Theme
from tensorgami.style.variants import presentation
from tensorgami.transforms import Level
from tensorgami.views import View


@dataclass(frozen=True)
class Page:
    """A canvas of fixed size, in inches."""

    name: str
    width_in: float
    height_in: float
    aspect: str


#: PowerPoint, Keynote and Google Slides: 13.33 × 7.5 in widescreen, 10 × 7.5 in standard
PAGES = {
    "slide": Page("slide", 40 / 3, 7.5, "16:9"),
    "slide-4:3": Page("slide-4:3", 10.0, 7.5, "4:3"),
}

#: row (or column) lengths tried, in px; ``None``: everything on one line
LENGTHS: tuple[float | None, ...] = (None, 2800, 2200, 1800, 1500, 1250, 1050, 900, 750, 600)

MIN_LABEL_PT = 10.0  # smaller box names are hard to read on a projected slide


@dataclass(frozen=True)
class Fitted:
    """How a figure was fitted: the page, and the size of the box names on it (pt)."""

    page: Page
    label_pt: float


def check_fit(fit: str) -> Page:
    if fit not in PAGES:
        raise ValueError(f"unknown fit {fit!r}; available: {', '.join(PAGES)}")
    return PAGES[fit]


def fit_to_page(
    page: Page,
    graph: Graph,
    theme: Theme,
    view: View,
    *,
    level: Level,
    options: LayoutOptions,
    fixed: Collection[str] = (),
) -> tuple[Graph, Scene, Theme, Fitted]:
    """Draw ``graph`` to fill ``page``: the slide style, the row length that gives the
    largest text (``fixed`` names the wrap options the caller set: those stay), centred on a
    canvas of the page's proportions."""
    theme = presentation(theme)
    chosen = bool({"wrap_width", "wrap_height"} & set(fixed))  # the caller's lengths stay
    best: tuple[float, Graph, Scene] | None = None
    for length in (None,) if chosen else LENGTHS:
        tried = options
        if not chosen:
            tried = dataclasses.replace(options, wrap_width=length, wrap_height=length)
        shown, scene = view.draw(graph, theme, level=level, options=tried)
        scale = min(
            page.width_in * PX_PER_INCH / scene.width, page.height_in * PX_PER_INCH / scene.height
        )
        if best is None or scale > best[0] * 1.01:  # on a near tie, the longer rows
            best = (scale, shown, scene)
    _, shown, scene = best
    width, height = scene.width, scene.height
    if width / height > page.width_in / page.height_in:
        height = width * page.height_in / page.width_in
    else:
        width = height * page.width_in / page.height_in
    scene = scene.centred_in(width, height)
    label_pt = theme.font(TextRole.LABEL).size * page.width_in * PX_PER_INCH / width * PT_PER_PX
    if label_pt < MIN_LABEL_PT:
        warnings.warn(
            f"on a {page.aspect} slide the box names are {label_pt:.0f} pt, small for a projector:"
            ' fewer boxes help, e.g. level="stages"',
            stacklevel=3,
        )
    return shown, scene, theme, Fitted(page, label_pt)
