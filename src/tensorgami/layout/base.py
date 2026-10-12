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

    @classmethod
    def parse(cls, value: str) -> Direction:
        try:
            return cls(str(value).upper())
        except ValueError:
            raise ValueError(
                f"direction must be 'LR' (left → right) or 'TB' (top → bottom); got {value!r}"
            ) from None


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
    u_shape: bool = True  # fold encoder–decoder skips (U-Net) into a U (block view)
    u_levels: int | None = None  # a U shows at most this many levels (the middle: ⋮)

    def __post_init__(self) -> None:
        """Accept ``direction`` as a string (``"TB"``, ``"tb"``); check every other value, so
        a wrong option fails here rather than deep inside the layout."""
        object.__setattr__(self, "direction", Direction.parse(self.direction))
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type == "bool":
                if not isinstance(value, bool):
                    raise TypeError(
                        f"layout option {f.name}: expected True or False, got {value!r}"
                    )
            elif f.type.startswith("int"):
                if value is None and f.type.endswith("None"):
                    continue
                if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                    raise TypeError(
                        f"layout option {f.name}: expected a whole number ≥ 2, got {value!r}"
                    )
            elif f.type.startswith("float"):
                if value is None and f.type.endswith("None"):
                    continue
                if isinstance(value, bool) or not isinstance(value, int | float):
                    allowed = "a number or None" if f.type.endswith("None") else "a number"
                    raise TypeError(f"layout option {f.name}: expected {allowed}, got {value!r}")
                if value < 0:
                    raise ValueError(f"layout option {f.name}: must be ≥ 0, got {value!r}")

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
        return cls(**values)


class LayoutEngine(Protocol):
    def layout(self, graph: Graph, theme: Theme) -> Scene: ...
