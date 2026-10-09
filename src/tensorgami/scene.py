"""A *scene* is a fully positioned diagram: nodes with boxes, routed edges, groups and notes.

Scenes are what the renderers draw. Today they are built by hand (style specimens); later the
layout engine will produce them from the model IR, so styling and rendering are reused as-is.
Coordinates are SVG user units with the origin top-left and y pointing down.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum

from tensorgami.style.glyphs import Shape, TextRole
from tensorgami.vocabulary import EdgeKind, OpCategory


class SceneError(ValueError):
    """Raised for inconsistent scenes (duplicate ids, dangling edges, ...)."""


# --------------------------------------------------------------------------- geometry


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def offset(self, dx: float = 0.0, dy: float = 0.0) -> Point:
        return Point(self.x + dx, self.y + dy)


class Side(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"

    @property
    def is_horizontal(self) -> bool:
        """True for sides whose edges leave horizontally (LEFT, RIGHT)."""
        return self in (Side.LEFT, Side.RIGHT)


class TextAnchor(StrEnum):
    START = "start"
    MIDDLE = "middle"
    END = "end"


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned rectangle given by its top-left corner and size."""

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if self.w < 0 or self.h < 0:
            raise ValueError(f"box size must be non-negative, got {self.w}×{self.h}")

    @classmethod
    def centered(cls, cx: float, cy: float, w: float, h: float) -> Box:
        return cls(cx - w / 2, cy - h / 2, w, h)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def center(self) -> Point:
        return Point(self.cx, self.cy)

    def anchor(self, side: Side) -> Point:
        """Midpoint of one side."""
        return {
            Side.LEFT: Point(self.x, self.cy),
            Side.RIGHT: Point(self.right, self.cy),
            Side.TOP: Point(self.cx, self.y),
            Side.BOTTOM: Point(self.cx, self.bottom),
        }[side]

    def facing(self, target: Point) -> Side:
        """The side of this box that faces ``target`` (normalised by the box aspect)."""
        dx = (target.x - self.cx) / max(self.w / 2, 1e-9)
        dy = (target.y - self.cy) / max(self.h / 2, 1e-9)
        if abs(dx) >= abs(dy):
            return Side.RIGHT if dx >= 0 else Side.LEFT
        return Side.BOTTOM if dy >= 0 else Side.TOP

    def inflate(self, d: float) -> Box:
        return Box(self.x - d, self.y - d, self.w + 2 * d, self.h + 2 * d)

    @staticmethod
    def union(boxes: Iterable[Box]) -> Box:
        boxes = list(boxes)
        if not boxes:
            raise ValueError("cannot take the union of zero boxes")
        x0, y0 = min(b.x for b in boxes), min(b.y for b in boxes)
        x1, y1 = max(b.right for b in boxes), max(b.bottom for b in boxes)
        return Box(x0, y0, x1 - x0, y1 - y0)


# --------------------------------------------------------------------------- scene items


@dataclass(frozen=True, slots=True)
class Node:
    """A glyph. ``label`` is the name, ``detail`` the hyper-parameters (second line).

    ``shape`` overrides the category's default shape (e.g. a CONV encoder drawn as a
    trapezoid); ``depth`` is the oblique depth of a cuboid in the volume view.
    """

    id: str
    box: Box
    category: OpCategory
    label: str = ""
    detail: str = ""
    shape: Shape | None = None
    depth: float = 0.0


Endpoint = str | Point  # a node id, or a free point (arrows entering from outside)


@dataclass(frozen=True, slots=True)
class Edge:
    """A routed connection. Without ``via`` points it is drawn straight or with one elbow.

    ``exit``/``entry`` pick the sides of the source/target nodes (default: facing each
    other); ``label`` is usually a tensor shape; ``delay`` draws the one-step-delay marker of
    folded recurrent diagrams.
    """

    source: Endpoint
    target: Endpoint
    kind: EdgeKind = EdgeKind.DATA
    exit: Side | None = None
    entry: Side | None = None
    via: tuple[Point, ...] = ()
    label: str = ""
    label_at: Point | None = None
    delay: bool = False


@dataclass(frozen=True, slots=True)
class Group:
    """A frame around related nodes, e.g. a block repeated ``badge="×6"`` times."""

    id: str
    box: Box
    title: str = ""
    badge: str = ""


@dataclass(frozen=True, slots=True)
class Note:
    """Free text: axis labels, captions inside the figure, legend entries."""

    at: Point
    text: str
    anchor: TextAnchor = TextAnchor.MIDDLE
    role: TextRole = TextRole.NOTE


SceneItem = Node | Edge | Group | Note


class Scene:
    """Mutable container of positioned items; validates references as items are added."""

    def __init__(self, width: float, height: float, title: str = "", description: str = ""):
        if width <= 0 or height <= 0:
            raise SceneError(f"scene size must be positive, got {width}×{height}")
        self.width = width
        self.height = height
        self.title = title
        self.description = description
        self._nodes: dict[str, Node] = {}
        self._groups: dict[str, Group] = {}
        self._edges: list[Edge] = []
        self._notes: list[Note] = []

    def add(self, *items: SceneItem) -> Scene:
        for item in items:
            match item:
                case Node():
                    self._add_unique(self._nodes, item, "node")
                case Group():
                    self._add_unique(self._groups, item, "group")
                case Edge():
                    for end in (item.source, item.target):
                        if isinstance(end, str) and end not in self._nodes:
                            raise SceneError(f"edge references unknown node '{end}'")
                    self._edges.append(item)
                case Note():
                    self._notes.append(item)
                case _:
                    raise SceneError(f"not a scene item: {item!r}")
        return self

    @staticmethod
    def _add_unique(registry: dict, item: Node | Group, what: str) -> None:
        if item.id in registry:
            raise SceneError(f"duplicate {what} id '{item.id}'")
        registry[item.id] = item

    def node(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise SceneError(f"unknown node '{node_id}'") from None

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes.values())

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(self._edges)

    @property
    def groups(self) -> tuple[Group, ...]:
        return tuple(self._groups.values())

    @property
    def notes(self) -> tuple[Note, ...]:
        return tuple(self._notes)

    def categories(self) -> list[OpCategory]:
        """Categories in order of first appearance (legend order)."""
        return list(dict.fromkeys(n.category for n in self._nodes.values()))

    def edge_kinds(self) -> list[EdgeKind]:
        return list(dict.fromkeys(e.kind for e in self._edges))

    def __iter__(self) -> Iterator[SceneItem]:
        yield from self.groups
        yield from self.edges
        yield from self.nodes
        yield from self.notes
