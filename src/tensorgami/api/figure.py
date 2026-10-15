"""``tg.Figure``: what the user gets back — shown inline in Jupyter, saved only on request
(ADR-0005, ADR-0007)."""

from __future__ import annotations

import io
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from tensorgami.api.fit import Fitted
from tensorgami.ir import Graph
from tensorgami.render.mpl import PX_PER_INCH, draw_scene, quiet_fonts, render_figure
from tensorgami.render.scene_renderer import render_svg
from tensorgami.scene import Scene
from tensorgami.style import Theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure as MplFigure

#: formats written through matplotlib (SVG is written by Tensorgami's own SVG canvas)
RASTER_AND_PDF = {".png", ".pdf", ".jpg", ".jpeg", ".tif", ".tiff"}
SCREEN_DPI = 2 * PX_PER_INCH  # notebook display: rendered at 2× for sharp HiDPI screens


class Figure:
    """A drawn model. Nothing is written to disk until :meth:`save` is called.

    * in Jupyter, the figure renders inline (``_repr_png_``);
    * ``fig.save("model.png", dpi=300)`` — also ``.svg`` and ``.pdf``;
    * ``fig.show()`` opens a window from a plain script;
    * ``fig.figure`` / ``fig.ax`` are the underlying matplotlib objects (titles, annotations);
    * ``fig.draw_on(ax)`` draws into a panel of your own ``plt.subplots`` layout.
    """

    def __init__(
        self,
        scene: Scene,
        style: Theme,
        graph: Graph | None = None,
        *,
        fitted: Fitted | None = None,
    ) -> None:
        self._scene = scene
        self._style = style
        self._graph = graph
        self._fitted = fitted

    # ------------------------------------------------------------------ inspection

    @property
    def scene(self) -> Scene:
        return self._scene

    @property
    def style(self) -> Theme:
        return self._style

    @property
    def graph(self) -> Graph | None:
        return self._graph

    @property
    def size(self) -> tuple[float, float]:
        """Natural size in px (1 px = 1/96 inch)."""
        return self._scene.width, self._scene.height

    @property
    def fitted(self) -> Fitted | None:
        """How the figure was fitted (``fit="slide"``, ``width="3.5in"``): the page and the size of
        the box names on it, in pt; ``None`` at natural size."""
        return self._fitted

    @property
    def inches(self) -> tuple[float, float]:
        """Size when saved: a slide's (13.33 × 7.5), the width asked for (``width="3.5in"``,
        with the height that goes with it), or the natural size."""
        w, h = self.size
        if self._fitted is not None:
            width = self._fitted.page.width_in
            return width, width * h / w
        return w / PX_PER_INCH, h / PX_PER_INCH

    @cached_property
    def figure(self) -> MplFigure:
        return render_figure(self._scene, self._style, width_in=self.inches[0])

    @property
    def ax(self) -> Axes:
        return self.figure.axes[0]

    def __repr__(self) -> str:
        w, h = self.size
        title, style = self._scene.title, self._style.name
        page = ""
        if self._fitted is not None:
            fit = self._fitted
            where = f"{fit.page.name} {fit.page.aspect}".strip()
            page = f", {where}, box names {fit.label_pt:.0f} pt"
        return f"<tensorgami.Figure {title!r} {w:.0f}×{h:.0f} px, style {style!r}{page}>"

    # ------------------------------------------------------------------ output

    def save(self, path: str | Path, *, dpi: float = 300) -> Path:
        """Write the figure; the format follows the suffix: ``.png`` (default choice for
        reports), ``.svg`` (vector, editable), ``.pdf``, ``.jpg``, ``.tif``."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".svg":
            path.write_text(self.to_svg(), encoding="utf-8")
        elif suffix in RASTER_AND_PDF:
            with quiet_fonts():
                self.figure.savefig(path, dpi=dpi)
        else:
            raise ValueError(
                f"cannot save as {suffix or 'a file without suffix'}: use .png, .svg, .pdf, "
                ".jpg or .tif"
            )
        return path

    def to_png(self, *, dpi: float = SCREEN_DPI) -> bytes:
        buffer = io.BytesIO()
        with quiet_fonts():
            self.figure.savefig(buffer, format="png", dpi=dpi)
        return buffer.getvalue()

    def to_svg(self) -> str:
        """The SVG document; a fitted figure states its printed size (``width="3.5in"``)."""
        inches = self.inches if self._fitted is not None else None
        return render_svg(self._scene, self._style, inches=inches)

    def _repr_png_(self) -> tuple[bytes, dict[str, int]]:
        """Jupyter display: rendered at 2× but shown at the natural size."""
        w, h = self.size
        return self.to_png(), {"width": round(w), "height": round(h)}

    def draw_on(self, ax: Axes) -> Axes:
        """Draw into an existing Axes (e.g. a panel of ``plt.subplots``); returns it."""
        draw_scene(ax, self._scene, self._style)
        return ax

    def show(self) -> None:
        """Open the figure in a matplotlib window (from scripts)."""
        import matplotlib.pyplot as plt

        window = plt.figure(figsize=self.inches)
        self.draw_on(window.add_axes((0.0, 0.0, 1.0, 1.0)))
        with quiet_fonts():
            plt.show()
