"""The :class:`Theme`: one immutable bundle of every visual decision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from tensorgami.style.color import Color
from tensorgami.style.glyphs import EdgeStyle, Font, GroupStyle, Metrics, NodeStyle, TextRole
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

    def __post_init__(self) -> None:
        # Freeze the mappings so a theme can be shared safely.
        for attr in ("fonts", "nodes", "edges"):
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
