"""matplotlib implementation of :class:`~tensorgami.render.canvas.Canvas` (ADR-0005).

Scene coordinates (px, y down) are used directly as data coordinates of an Axes whose y-axis is
inverted. 1 scene px = 0.75 pt (1/96 inch), so a figure created at its natural size has the
same proportions as the SVG and ``dpi`` alone sets the PNG resolution. When drawing into a
smaller Axes (a subplot), text and line widths are scaled with the drawing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

from tensorgami.render.canvas import Paint
from tensorgami.render.scene_renderer import SceneRenderer
from tensorgami.render.svg import BASELINE_SHIFT
from tensorgami.render.text import Script, parse_markup
from tensorgami.scene import Box, Point, Scene, TextAnchor
from tensorgami.style import Theme
from tensorgami.style.glyphs import Font

PX_PER_INCH = 96.0
PT_PER_PX = 0.75
_GENERIC = {"sans-serif": "DejaVu Sans", "serif": "DejaVu Serif", "monospace": "DejaVu Sans Mono"}
_ANCHOR = {TextAnchor.START: "left", TextAnchor.MIDDLE: "center", TextAnchor.END: "right"}
_MATH_SPECIAL = str.maketrans({c: "\\" + c for c in "\\{}$%#&_^"})

#: rcParams for every drawing: dash lengths in absolute points (as in SVG), and upright
#: sub/superscripts.
RC = {"lines.scale_dashes": False, "mathtext.default": "regular"}


@lru_cache(maxsize=64)
def font_families(family: tuple[str, ...]) -> tuple[str, ...]:
    """The installed fonts of a style's font stack, in order, plus DejaVu as the glyph
    fallback (for symbols such as ‖ ⊙ ε). Missing fonts are skipped silently instead of
    letting matplotlib warn about each one."""
    from matplotlib import font_manager

    found: list[str] = []
    for name in family:
        name = _GENERIC.get(name, name)
        if name in found:
            continue
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=name), fallback_to_default=False
            )
        except ValueError:
            continue
        found.append(name)
    fallback = "DejaVu Sans Mono" if "mono" in "".join(family).lower() else "DejaVu Sans"
    if fallback not in found:
        found.append(fallback)
    return tuple(found)


@lru_cache(maxsize=64)
def font_weight(family: str, weight: int) -> int:
    """The nearest weight the font really has, as browsers choose it (CSS font matching):
    600 on a font with only 400/500/700 becomes 700, which also keeps matplotlib quiet."""
    from matplotlib import font_manager

    available = sorted(
        {
            font_manager.weight_dict.get(f.weight, 400) if isinstance(f.weight, str) else f.weight
            for f in font_manager.fontManager.ttflist
            if f.name == family
        }
    )
    if not available or weight in available:
        return weight
    heavier = [w for w in available if w > weight]
    lighter = [w for w in reversed(available) if w < weight]
    order = heavier + lighter if weight >= 500 else lighter + heavier
    return order[0]


@contextmanager
def quiet_fonts() -> Iterator[None]:
    """Silence matplotlib's font fallback notices while Tensorgami draws."""
    logger = logging.getLogger("matplotlib.font_manager")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)


def to_mathtext(text: str) -> str:
    """``x_{t-1}`` → ``x$_{t-1}$`` (mathtext sub/superscripts); literal ``$`` is escaped."""
    parts = []
    for run in parse_markup(text):
        if run.script is Script.NORMAL:
            parts.append(run.text.replace("$", r"\$"))
        else:
            mark = "_" if run.script is Script.SUB else "^"
            parts.append(f"${mark}{{{run.text.translate(_MATH_SPECIAL)}}}$")
    return "".join(parts)


class MatplotlibCanvas:
    """Draws onto an Axes whose data units are scene pixels. ``scale`` converts scene pixels
    to on-page size: 1 at natural size, < 1 when drawing into a smaller subplot."""

    def __init__(self, ax: Axes, width: float, height: float, *, scale: float = 1.0) -> None:
        self.ax = ax
        self.scale = scale
        self._z = 0
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal")
        ax.set_axis_off()

    def _pt(self, px: float) -> float:
        return px * PT_PER_PX * self.scale

    def _z_next(self) -> int:
        self._z += 1
        return self._z  # explicit order: matplotlib would otherwise draw text over patches

    def _style(self, paint: Paint) -> dict:
        face = (*paint.fill.rgb01, paint.opacity) if paint.fill else "none"
        style = {"facecolor": face, "zorder": self._z_next()}
        if paint.stroke:
            style.update(
                edgecolor=paint.stroke.hex, linewidth=self._pt(paint.width), joinstyle="round"
            )
            if paint.dash:
                style["linestyle"] = (0, tuple(self._pt(d) for d in paint.dash))
        else:
            style.update(edgecolor="none", linewidth=0)
        return style

    # ------------------------------------------------------------------ primitives

    def rect(self, box: Box, paint: Paint, radius: float = 0.0) -> None:
        r = min(radius, box.w / 2, box.h / 2)
        if r > 0:
            patch = FancyBboxPatch(
                (box.x, box.y),
                box.w,
                box.h,
                boxstyle=f"round,pad=0,rounding_size={r}",
                **self._style(paint),
            )
        else:
            patch = Rectangle((box.x, box.y), box.w, box.h, **self._style(paint))
        self.ax.add_patch(patch)

    def circle(self, center: Point, radius: float, paint: Paint) -> None:
        self.ax.add_patch(Circle((center.x, center.y), radius, **self._style(paint)))

    def polygon(self, points: Sequence[Point], paint: Paint) -> None:
        self.ax.add_patch(Polygon([(p.x, p.y) for p in points], closed=True, **self._style(paint)))

    def polyline(self, points: Sequence[Point], paint: Paint) -> None:
        if paint.stroke is None:
            return
        line = Line2D(
            [p.x for p in points],
            [p.y for p in points],
            color=paint.stroke.hex,
            linewidth=self._pt(paint.width),
            solid_joinstyle="round",
            dash_joinstyle="round",
            solid_capstyle="butt",
            zorder=self._z_next(),
        )
        if paint.dash:
            line.set_linestyle((0, tuple(self._pt(d) for d in paint.dash)))
        self.ax.add_line(line)

    def text(self, at: Point, text: str, font: Font, anchor: TextAnchor = TextAnchor.MIDDLE):
        if not text:
            return
        families = font_families(font.family)
        self.ax.text(
            at.x,
            at.y + BASELINE_SHIFT * font.size,
            to_mathtext(text),
            fontsize=self._pt(font.size),
            fontfamily=list(families),
            fontweight=font_weight(families[0], font.weight),
            fontstyle="italic" if font.italic else "normal",
            color=font.color.hex,
            ha=_ANCHOR[anchor],
            va="baseline",
            zorder=self._z_next(),
        )

    def begin_group(self, css_class: str, data_id: str | None = None) -> None:
        """Groups only matter for SVG semantics; nothing to do in matplotlib."""

    def end_group(self) -> None:
        """See :meth:`begin_group`."""


# --------------------------------------------------------------------------- entry points


def render_figure(scene: Scene, theme: Theme, *, dpi: float = PX_PER_INCH) -> MplFigure:
    """A standalone matplotlib Figure at the scene's natural size (not registered with
    pyplot, so notebooks do not display it twice)."""
    figure = MplFigure(figsize=(scene.width / PX_PER_INCH, scene.height / PX_PER_INCH), dpi=dpi)
    ax = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    with mpl.rc_context(RC), quiet_fonts():
        SceneRenderer(theme).render(scene, MatplotlibCanvas(ax, scene.width, scene.height))
    return figure


def draw_scene(ax: Axes, scene: Scene, theme: Theme) -> None:
    """Draw ``scene`` into an existing Axes (e.g. one panel of ``plt.subplots``), scaling
    text and lines with the space the Axes offers."""
    figure = ax.figure
    box = ax.get_position()
    width_in = box.width * figure.get_figwidth()
    height_in = box.height * figure.get_figheight()
    scale = min(width_in * PX_PER_INCH / scene.width, height_in * PX_PER_INCH / scene.height)
    with mpl.rc_context(RC), quiet_fonts():
        canvas = MatplotlibCanvas(ax, scene.width, scene.height, scale=scale)
        SceneRenderer(theme).render(scene, canvas)
