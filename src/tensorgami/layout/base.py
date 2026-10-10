"""Layout contract and options (ADR-0004)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Protocol

from tensorgami.ir import Graph
from tensorgami.scene import Scene
from tensorgami.style import Theme


class Direction(StrEnum):
    LEFT_RIGHT = "LR"  # main flow left → right (default, style baseline P4)
    TOP_BOTTOM = "TB"  # deep stacks top → bottom


@dataclass(frozen=True)
class LayoutOptions:
    direction: Direction = Direction.LEFT_RIGHT
    rank_gap: float = 48  # minimum space between consecutive ranks
    node_gap: float = 20  # minimum space between neighbours in a rank
    group_padding: float = 14
    title_height: float = 20  # space for a frame title
    margin: float = 24
    wrap_width: float | None = 1100  # LR only: wrap the top level into rows beyond this
    row_gap: float = 56
    legend: bool = True
    shapes_on_edges: bool = True
    loop_clearance: float = 14  # height of a state self-loop above its node

    @classmethod
    def for_style(cls, theme: Theme, **overrides: object) -> LayoutOptions:
        """Defaults, then the style's ``[layout]`` preferences, then explicit ``overrides``."""
        known = {f.name for f in fields(cls)}
        values = {**theme.layout, **overrides}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"style '{theme.name}': unknown layout option(s) {unknown}; known: {sorted(known)}"
            )
        if "direction" in values:
            values["direction"] = Direction(values["direction"])
        return cls(**values)


class LayoutEngine(Protocol):
    def layout(self, graph: Graph, theme: Theme) -> Scene: ...
