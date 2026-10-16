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
    FOLDED = "folded"  # folded paper: two facets split by a slanted crease (origami style)


class Texture(StrEnum):
    """An engraved texture telling a category apart without colour (the etching style): in a
    narrow strip at the start of a box, or over a volume's face. Orientations 45° apart and
    equal ink density, after He et al., *Design Characterization for Black-and-White Textures
    in Visualization* (IEEE TVCG 2024)."""

    DIAGONAL = "diagonal"  # ////
    BACKDIAGONAL = "backdiagonal"  # \\\\
    HORIZONTAL = "horizontal"  # ≡
    VERTICAL = "vertical"  # |||
    CROSSHATCH = "crosshatch"  # ×××
    GRID = "grid"  # +++
    DOTS = "dots"  # ···
    ZIGZAG = "zigzag"  # /\/\
    DASHES = "dashes"  # staggered short dashes
    SOLID = "solid"  # filled with ink


class TextLayout(StrEnum):
    """How a box arranges its label and detail."""

    STACKED = "stacked"  # label above detail, centred
    INLINE = "inline"  # one line: label on the left, detail on the right (block flowchart)
    ROTATED = "rotated"  # one line turned upright, reading bottom → top: a narrow, tall box

    @property
    def single_line(self) -> bool:
        return self is not TextLayout.STACKED


class ArrowHead(StrEnum):
    FILLED = "filled"
    OPEN = "open"
    DOT = "dot"  # a small round terminal (origami wires)
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
    caps: bool = False  # set in capitals (frame titles of the origami style)

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
    opacity: float = 1.0  # of the fill: translucent paper lets what lies beneath show through
    texture: Texture | None = None  # engraved in the outline's colour (etching)


@dataclass(frozen=True, slots=True)
class EdgeStyle:
    name: str
    color: Color
    width: float
    arrow: ArrowHead = ArrowHead.FILLED  # at the target
    dash: tuple[float, ...] = ()
    start: ArrowHead = ArrowHead.NONE  # at the source (e.g. a dot terminal)


@dataclass(frozen=True, slots=True)
class GroupStyle:
    fill: Color
    stroke: Color
    width: float
    corner_radius: float
    padding: float
    dash: tuple[float, ...] = ()
    badge_fill: Color = Color(255, 255, 255)
    fill_opacity: float = 1.0  # 0: hairline frames without a fill


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
    terminal_radius: float = 2.5  # dot terminals of edges
    fold_width: float = 18  # width of the folded strip on the right of a folded box (px)
    fold_shade: float = 0.07  # how much darker the folded strip is
    texture_strip: float = 14  # width of a box's textured strip (etching)
    texture_spacing: float = 4  # between the lines (or dots) of a texture
    texture_line: float = 0.8  # line width of a texture


@dataclass(frozen=True)
class BoxGeometry:
    """Size rules for box-shaped nodes (style table ``[boxes]``; px). Circles keep their own
    size; pills follow ``height`` and ``detail_height`` minus 4 px and never turn upright.

    The text arrangement can differ per flow direction (``text = { TB = "inline", LR =
    "rotated" }`` in TOML); boxes placed by hand, without a layout, keep stacked text.
    """

    text_lr: TextLayout = TextLayout.STACKED  # in a left → right flow
    text_tb: TextLayout = TextLayout.STACKED  # in a top → bottom flow
    padding: float = 32  # along the text, both ends together
    min_width: float = 96  # minimum length along the text
    height: float = 36  # a box with one line of text (its thickness, when rotated)
    detail_height: float = 46  # a box with a label and a detail line (stacked text)
    gap: float = 12  # between label and detail on one line
    uniform_width: bool = False  # single-line boxes all as long as the longest: a tidy column

    def text(self, *, vertical: bool) -> TextLayout:
        """The arrangement for a flow direction."""
        return self.text_tb if vertical else self.text_lr
