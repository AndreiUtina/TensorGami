"""Tensorgami's visual style: colours, glyph styles, fonts and themes."""

from __future__ import annotations

from pathlib import Path

from tensorgami.style.color import BLACK, WHITE, Color
from tensorgami.style.glyphs import (
    ArrowHead,
    EdgeStyle,
    Font,
    GroupStyle,
    Metrics,
    NodeStyle,
    Shape,
    TextRole,
)
from tensorgami.style.loader import available_themes, parse_theme, read_theme
from tensorgami.style.theme import Theme, ThemeError
from tensorgami.style.validation import Issue, ThemeValidator


def load_theme(source: str | Path = "baseline", *, validate: bool = True) -> Theme:
    """Load a built-in theme by name (default ``"baseline"``) or a ``.toml`` file by path.

    With ``validate=True`` (default) the theme must pass the legibility and accessibility rules
    in :mod:`tensorgami.style.validation`, otherwise :class:`ThemeError` is raised.
    """
    theme = read_theme(source)
    return ThemeValidator().ensure_valid(theme) if validate else theme


__all__ = [
    "BLACK",
    "WHITE",
    "ArrowHead",
    "Color",
    "EdgeStyle",
    "Font",
    "GroupStyle",
    "Issue",
    "Metrics",
    "NodeStyle",
    "Shape",
    "TextRole",
    "Theme",
    "ThemeError",
    "ThemeValidator",
    "available_themes",
    "load_theme",
    "parse_theme",
    "read_theme",
]
