"""The :class:`Theme`: one immutable bundle of every visual decision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from tensorgami.style.color import Color
from tensorgami.style.glyphs import (
    BoxGeometry,
    EdgeStyle,
    Font,
    GroupStyle,
    Metrics,
    NodeStyle,
    TextRole,
)
from tensorgami.vocabulary import EdgeKind, OpCategory


class ThemeError(ValueError):
    """Raised when a theme is incomplete or malformed."""


@dataclass(frozen=True)
class Theme:
    name: str
    background: Color
    fonts: Mapping[TextRole, Font]
    nodes: Mapping[OpCategory, NodeStyle]
    edges: Mapping[EdgeKind, EdgeStyle]
    group: GroupStyle
    metrics: Metrics = field(default_factory=Metrics)
    description: str = ""
    layout: Mapping[str, object] = field(default_factory=dict)  # LayoutOptions overrides
    boxes: BoxGeometry = field(default_factory=BoxGeometry)

    def __post_init__(self) -> None:
        # Freeze the mappings so a theme can be shared safely.
        for attr in ("fonts", "nodes", "edges", "layout"):
            object.__setattr__(self, attr, MappingProxyType(dict(getattr(self, attr))))
        missing = [
            *(f"font '{r}'" for r in TextRole if r not in self.fonts),
            *(f"category '{c}'" for c in OpCategory if c not in self.nodes),
            *(f"edge kind '{k}'" for k in EdgeKind if k not in self.edges),
        ]
        if missing:
            raise ThemeError(f"theme '{self.name}' is missing: {', '.join(missing)}")

    def font(self, role: TextRole) -> Font:
        return self.fonts[role]

    def node_style(self, category: OpCategory) -> NodeStyle:
        return self.nodes[category]

    def edge_style(self, kind: EdgeKind) -> EdgeStyle:
        return self.edges[kind]

    def node_font(self, category: OpCategory, role: TextRole = TextRole.LABEL) -> Font:
        """Font for text drawn on a node: the role's font in the category's text colour."""
        font = self.fonts[role]
        if role is TextRole.DETAIL:
            return font
        return replace(font, color=self.nodes[category].text)

    def outside_font(self, category: OpCategory, role: TextRole = TextRole.LABEL) -> Font:
        """Font for a node's text drawn beside it, on the background (a volume's name and
        shape below it, the note beside a merge circle): the node's own font — unless that
        is too faint on the background (dark text on a dark variant), then in the colour of
        the text printed on the background (the legend's for names, the shapes' for details).
        """
        font = self.node_font(category, role)
        if font.color.contrast_ratio(self.background) >= 4.5:
            return font
        on_background = TextRole.LEGEND if role is TextRole.LABEL else TextRole.ANNOTATION
        return replace(font, color=self.fonts[on_background].color)
