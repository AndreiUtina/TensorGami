"""Tensorgami's visual style: colours, glyph styles, fonts and themes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from tensorgami.style.color import BLACK, WHITE, Color
from tensorgami.style.glyphs import (
    ArrowHead,
    BoxGeometry,
    EdgeStyle,
    Font,
    GroupStyle,
    Metrics,
    NodeStyle,
    Shape,
    TextLayout,
    TextRole,
)
from tensorgami.style.loader import available_themes, parse_theme, read_theme
from tensorgami.style.palettes import apply_palette, available_palettes
from tensorgami.style.theme import Theme, ThemeError
from tensorgami.style.validation import Issue, ThemeValidator
from tensorgami.style.variants import apply_variants, available_variants

DEFAULT_STYLE = "default"


def load_theme(source: str | Path = DEFAULT_STYLE, *, validate: bool = True) -> Theme:
    """Load a built-in style by name (``"default"`` unless given) or a ``.toml`` file by path.

    With ``validate=True`` (default) the style must pass the legibility and accessibility rules
    in :mod:`tensorgami.style.validation`, otherwise :class:`ThemeError` is raised.
    """
    theme = read_theme(source)
    return ThemeValidator().ensure_valid(theme) if validate else theme


def load_style(
    style: str | Path | Theme = DEFAULT_STYLE,
    variant: str | Sequence[str] | None = None,
    palette: str | None = None,
) -> Theme:
    """The user-facing name for :func:`load_theme`: ``load_style("origami")``, or
    ``load_style("path/to/my-style.toml")`` (ADR-0008); ``palette`` recolours its operations
    (``"colorblind"``, ``"tab10"``, ``"vivid"``, ``"mono"``), then ``variant`` recolours it for
    where the figure goes (``"dark"``, ``"greyscale"``; several apply in order). A loaded theme
    is passed through. The result is validated like every style."""
    theme = style if isinstance(style, Theme) else load_theme(style)
    if not variant and not palette:
        return theme
    if palette:
        theme = apply_palette(theme, palette)
    return ThemeValidator().ensure_valid(apply_variants(theme, variant))


def available_styles() -> list[str]:
    """Names of the built-in styles."""
    return available_themes()


__all__ = [
    "BLACK",
    "DEFAULT_STYLE",
    "WHITE",
    "ArrowHead",
    "BoxGeometry",
    "Color",
    "EdgeStyle",
    "Font",
    "GroupStyle",
    "Issue",
    "Metrics",
    "NodeStyle",
    "Shape",
    "TextLayout",
    "TextRole",
    "Theme",
    "ThemeError",
    "ThemeValidator",
    "apply_palette",
    "apply_variants",
    "available_palettes",
    "available_styles",
    "available_themes",
    "available_variants",
    "load_style",
    "load_theme",
    "parse_theme",
    "read_theme",
]
