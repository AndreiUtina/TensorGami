# ADR-0005: matplotlib for figures and PNG; SVG canvas for vector output

Status: Accepted · 2026-10-09

## Context

The user works in scripts and Jupyter notebooks. They want to see the figure inline, decide
afterwards whether to keep it, and save mainly as PNG (FR2, FR3). Figures go into reports and
presentations. They may need to combine a diagram with other plots, and later animate it
(FR13). The rendering layer already has a `Canvas` protocol and an `SvgCanvas` (step 2).

## Decision

Add a **`MatplotlibCanvas`** implementing the existing `Canvas` protocol. It is the default
backend for `tg.Figure`:

- inline display in Jupyter through matplotlib's own integration, plus `_repr_png_` on
  `tg.Figure`;
- `fig.save("x.png", dpi=300)` via the Agg renderer, with SVG/PDF through the same call;
- `fig.figure` / `fig.ax` expose the matplotlib objects (for multi-panel figures, titles, or
  `plt.subplots` layouts);
- animation later through `matplotlib.animation` (GIF/MP4) on the same canvas.

The **`SvgCanvas` stays** for exact vector output and the style specimens. Both canvases
render the same `Scene` through the same painters, so they cannot drift apart in content.

Mapping details for the matplotlib canvas:

- Scene units become points at 1 px = 0.75 pt, so `dpi` alone sets the PNG resolution.
- Text uses the theme's font stack with matplotlib's font fallback. The label markup
  `x_{t}` / `h^{2}` is translated to mathtext sub- and superscripts.
- Axes are hidden and the y-axis is inverted, so scene coordinates are used unchanged.

## Alternatives considered

- **SVG → PNG converters** (cairosvg needs the system cairo library; resvg-py and typst are
  extra binaries). Each adds a dependency without giving a notebook-native figure object.
- **Pillow drawing**. It has no vector output, weak text layout, and no matplotlib
  integration.
- **Plotly / Bokeh**. They are interactive and heavy, and their static export needs
  kaleido/selenium.

## Consequences

- matplotlib becomes the single runtime dependency (ADR-0006). It is standard in every
  scientific Python environment.
- Golden-file tests compare images with a tolerance (fonts differ slightly between
  machines); exact comparisons stay at the `Scene` and SVG level.
- Text metrics: the heuristic `TextMeasurer` can be replaced by one that asks matplotlib
  for real text extents, behind the same protocol.
