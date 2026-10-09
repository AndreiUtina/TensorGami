"""Style records for the visual elements of a diagram (nodes, edges, groups, text)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tensorgami.style.color import Color


class Shape(StrEnum):
    """Outline of a node glyph. Painters are registered per shape."""

    RECT = "rect"  # operation / layer (rounded rectangle)
    PILL = "pill"  # data: inputs, outputs, named tensors
    CIRCLE = "circle"  # point-wise merge operation with a symbol (+, ×, ‖)
    TRAPEZOID_NARROW = "trapezoid-narrow"  # encoder: compresses left → right
    TRAPEZOID_WIDEN = "trapezoid-widen"  # decoder: expands left → right
    CUBOID = "cuboid"  # tensor volume (C × H × W) in the volume view


class ArrowHead(StrEnum):
    FILLED = "filled"
    OPEN = "open"
    NONE = "none"


class TextRole(StrEnum):
    """Every piece of text in a diagram has exactly one role, hence one font."""

    LABEL = "label"  # node name
    DETAIL = "detail"  # node hyper-parameters
    SYMBOL = "symbol"  # merge symbols and other single-glyph labels
    ANNOTATION = "annotation"  # tensor shapes on edges
    GROUP_TITLE = "group-title"
    BADGE = "badge"  # repeat badges such as "×6"
    NOTE = "note"  # free annotations (axes, captions inside the figure)
    LEGEND = "legend"


@dataclass(frozen=True, slots=True)
class Font:
    family: tuple[str, ...]
    size: float
    color: Color
    weight: int = 400
    italic: bool = False

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("font family list must not be empty")
        if self.size <= 0:
            raise ValueError(f"font size must be positive, got {self.size}")

    @property
    def css_family(self) -> str:
        generic = {"serif", "sans-serif", "monospace", "cursive", "fantasy", "system-ui"}
        return ", ".join(f if f in generic else f"'{f}'" for f in self.family)


@dataclass(frozen=True, slots=True)
class NodeStyle:
    name: str  # human-readable category name, used by the legend
    shape: Shape
    fill: Color
    stroke: Color
    text: Color
    dash: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class EdgeStyle:
    name: str
    color: Color
    width: float
    arrow: ArrowHead = ArrowHead.FILLED
    dash: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class GroupStyle:
    fill: Color
    stroke: Color
    width: float
    corner_radius: float
    padding: float
    dash: tuple[float, ...] = ()
    badge_fill: Color = Color(255, 255, 255)


@dataclass(frozen=True, slots=True)
class Metrics:
    """Sizes in px (SVG user units). The baseline uses an 8 px grid."""

    grid: float = 8
    corner_radius: float = 6
    node_stroke: float = 1.25
    arrow_length: float = 8
    arrow_width: float = 7
    label_halo: float = 2.5
    delay_marker: float = 7
    cuboid_angle: float = 35  # degrees of the oblique depth axis
    trapezoid_ratio: float = 0.45  # short side / long side
    line_gap: float = 3  # vertical gap between label and detail lines
