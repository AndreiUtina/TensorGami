"""Rendering: scenes → canvases (SVG today)."""

from __future__ import annotations

from tensorgami.render.canvas import Canvas, Paint
from tensorgami.render.edges import EdgePainter, EdgeRouter
from tensorgami.render.legend import Legend, add_legend, build_legend
from tensorgami.render.painters import GlyphPainter, PainterRegistry, default_painters
from tensorgami.render.scene_renderer import SceneRenderer, render_svg
from tensorgami.render.svg import SvgCanvas
from tensorgami.render.text import HeuristicTextMeasurer, TextMeasurer, parse_markup, plain_text

__all__ = [
    "Canvas",
    "EdgePainter",
    "EdgeRouter",
    "GlyphPainter",
    "HeuristicTextMeasurer",
    "Legend",
    "Paint",
    "PainterRegistry",
    "SceneRenderer",
    "SvgCanvas",
    "TextMeasurer",
    "add_legend",
    "build_legend",
    "default_painters",
    "parse_markup",
    "plain_text",
    "render_svg",
]
