"""Tensorgami's visual style: colours, glyph styles, fonts and themes."""

from __future__ import annotations

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
from tensorgami.style.theme import Theme, ThemeError
from tensorgami.style.validation import Issue, ThemeValidator

DEFAULT_STYLE = "default"


def load_theme(source: str | Path = DEFAULT_STYLE, *, validate: bool = True) -> Theme:
    """Load a built-in style by name (``"default"`` unless given) or a ``.toml`` file by path.

    With ``validate=True`` (default) the style must pass the legibility and accessibility rules
    in :mod:`tensorgami.style.validation`, otherwise :class:`ThemeError` is raised.
    """
    theme = read_theme(source)
    return ThemeValidator().ensure_valid(theme) if validate else theme


def load_style(style: str | Path = DEFAULT_STYLE) -> Theme:
    """The user-facing name for :func:`load_theme`: ``load_style("default")``, a future
    ``load_style("origami")``, or ``load_style("path/to/my-style.toml")`` (ADR-0008)."""
    return load_theme(style)


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
    "available_styles",
    "available_themes",
    "load_style",
    "load_theme",
    "parse_theme",
    "read_theme",
]
