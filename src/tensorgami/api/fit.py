"""Figures made for where they go (roadmap step 13): a slide, or a width in a document.

* ``tg.draw(..., fit="slide")`` (13c): a slide is a fixed canvas, so the figure that serves it
  best is the one whose text is largest once it fills the slide. The model is drawn in the
  slide style (larger text and lines, :func:`~tensorgami.style.variants.presentation`), rows
  (or, top → bottom, columns) of several lengths are tried, and the layout that fills the slide
  with the largest text is centred on a canvas of the slide's proportions.
* ``tg.draw(..., width="3.5in")`` (13b): a paper's column. The figure keeps the style's text
  size (box names 9 pt) and is laid out to fit the width, the shortest layout that fits at that
  size; only if none fits does it shrink, as little as it must. A narrower figure is centred on
  a canvas exactly that wide, so ``\\includegraphics{model.pdf}`` places it at true size.
  ``height=`` sets the most it may take (by default a page's text height, 9 in).
"""

from __future__ import annotations

import dataclasses
import math
import re
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
    """Where a figure goes. A *slide* is a fixed canvas the figure fills (its text may grow);
    otherwise a width in a document, with the most it may take in height (``None``: as tall as
    it needs), where text keeps the style's size at most."""

    name: str
    width_in: float
    height_in: float | None = None
    slide: bool = False
    aspect: str = ""


#: PowerPoint, Keynote and Google Slides: 13.33 × 7.5 in widescreen, 10 × 7.5 in standard
PAGES = {
    "slide": Page("slide", 40 / 3, 7.5, slide=True, aspect="16:9"),
    "slide-4:3": Page("slide-4:3", 10.0, 7.5, slide=True, aspect="4:3"),
}

#: row (or column) lengths tried, in px; ``None``: everything on one line
LENGTHS: tuple[float | None, ...] = (None, 2800, 2200, 1800, 1500, 1250, 1050, 900, 750, 600)

PAGE_HEIGHT_IN = 9.0  # a page's text height (Letter or A4 less margins): the tallest figure

MIN_SLIDE_PT = 10.0  # smaller box names are hard to read on a projected slide
MIN_PAPER_PT = 7.0  # smaller box names are hard to read in print

_UNITS = {"in": 1.0, "cm": 1 / 2.54, "mm": 1 / 25.4, "pt": 1 / 72, "px": 1 / PX_PER_INCH}


def parse_length(value: object, what: str = "width") -> float:
    """A length in inches: ``"3.5in"``, ``"8.9cm"``, ``"88mm"``, ``"252pt"``, ``"336px"``, or
    a number of inches."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        inches = float(value)
    elif isinstance(value, str) and (
        match := re.fullmatch(r"\s*(\d*\.?\d+)\s*(in|cm|mm|pt|px)\s*", value)
    ):
        inches = float(match.group(1)) * _UNITS[match.group(2)]
    else:
        raise ValueError(
            f"{what}: a length such as '3.5in', '8.9cm', '88mm' or '252pt' (or a number of "
            f"inches); got {value!r}"
        )
    if not math.isfinite(inches) or inches <= 0:
        raise ValueError(f"{what} must be positive, got {value!r}")
    return inches


def check_fit(fit: str) -> Page:
    if fit not in PAGES:
        raise ValueError(f"unknown fit {fit!r}; available: {', '.join(PAGES)}")
    return PAGES[fit]


def document(width: object, height: object = None) -> Page:
    """A width in a document (``width="3.5in"``), at most ``height`` tall: by default a
    page's text height (9 in), since a figure taller than the page cannot be printed."""
    w = parse_length(width, "width")
    h = parse_length(height, "height") if height is not None else PAGE_HEIGHT_IN
    return Page(f"{w:.3g} in wide", w, h)


@dataclass(frozen=True)
class Fitted:
    """How a figure was fitted: the page, and the size of the box names on it (pt)."""

    page: Page
    label_pt: float


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
    """Draw ``graph`` for ``page``: the row (or column) length that gives the largest text
    there (``fixed`` names the wrap options the caller set: those stay), on a canvas of the
    page's width (and, for a slide, its proportions)."""
    if page.slide:
        theme = presentation(theme)
    width_px = page.width_in * PX_PER_INCH
    height_px = page.height_in * PX_PER_INCH if page.height_in is not None else math.inf
    lengths: tuple[float | None, ...] = LENGTHS
    if not page.slide:  # rows as long as the width, as on a page at true size
        lengths = (width_px - 2 * options.margin, *LENGTHS)
    if not page.slide and "legend_width" not in fixed:  # the legend stacks to fit the width
        options = dataclasses.replace(options, legend_width=width_px - 2 * options.margin)
    chosen = bool({"wrap_width", "wrap_height"} & set(fixed))  # the caller's lengths stay
    best: tuple[float, float, Graph, Scene] | None = None
    for length in (None,) if chosen else lengths:
        tried = options
        if not chosen:
            tried = dataclasses.replace(options, wrap_width=length, wrap_height=length)
        shown, scene = view.draw(graph, theme, level=level, options=tried)
        scale = min(width_px / scene.width, height_px / scene.height)
        if not page.slide:
            scale = min(scale, 1.0)  # a document keeps the style's text size at most
        tall = scene.height * scale
        if best is None or scale > best[0] * 1.01:
            best = (scale, tall, shown, scene)
        elif not page.slide and scale > best[0] * 0.99 and tall < best[1] - 0.5:
            best = (scale, tall, shown, scene)  # as large, and shorter: less of the page
    scale, _, shown, scene = best
    if page.slide:  # the slide's proportions
        width, height = scene.width, scene.height
        if width / height > page.width_in / page.height_in:
            height = width * page.height_in / page.width_in
        else:
            width = height * page.width_in / page.height_in
    else:  # exactly the width asked for
        width, height = width_px / scale, scene.height
    scene = scene.centred_in(max(width, scene.width), height)
    label_pt = theme.font(TextRole.LABEL).size * width_px / scene.width * PT_PER_PX
    _warn_if_small(page, label_pt)
    return shown, scene, theme, Fitted(page, label_pt)


def _warn_if_small(page: Page, label_pt: float) -> None:
    if page.slide and label_pt < MIN_SLIDE_PT:
        warnings.warn(
            f"on a {page.aspect} slide the box names are {label_pt:.0f} pt, small for a projector:"
            ' fewer boxes help, e.g. level="stages"',
            stacklevel=4,
        )
    elif not page.slide and label_pt < MIN_PAPER_PT:
        warnings.warn(
            f"at {page.width_in:.3g} in wide the box names are {label_pt:.1f} pt, small in print:"
            ' try direction="TB" (top → bottom) or level="stages", or a wider figure',
            stacklevel=4,
        )
