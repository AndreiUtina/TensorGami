"""Draws a :class:`Scene` onto any :class:`Canvas` using a :class:`Theme`."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from tensorgami.render.canvas import Canvas, Paint
from tensorgami.render.edges import EdgePainter, EdgeRouter
from tensorgami.render.groups import GroupPainter
from tensorgami.render.painters import PainterRegistry, default_painters
from tensorgami.render.svg import SvgCanvas
from tensorgami.render.text import HeuristicTextMeasurer, TextMeasurer
from tensorgami.scene import Box, Edge, Group, Node, Note, Point, Scene
from tensorgami.style.glyphs import Shape
from tensorgami.style.theme import Theme
from tensorgami.style.variants import faded


class Emphasis(StrEnum):
    """Where a part of the figure is in an animation (roadmap step 14)."""

    PENDING = "pending"  # not run yet: faded
    ACTIVE = "active"  # running now: highlighted
    DONE = "done"  # run: drawn as in the still figure


@dataclass(frozen=True)
class FrameState:
    """One frame of an animation: the nodes running now and those already done (every other
    node is still to come), and badges replaced for the frame (a repeated block's pass)."""

    active: frozenset[str] = frozenset()
    done: frozenset[str] = frozenset()
    badges: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "badges", MappingProxyType(dict(self.badges)))

    def of(self, node_id: str) -> Emphasis:
        if node_id in self.active:
            return Emphasis.ACTIVE
        if node_id in self.done or node_id.startswith("legend:"):
            return Emphasis.DONE
        return Emphasis.PENDING


@dataclass(frozen=True)
class Item:
    """One part of the figure as the renderer paints it: ``look(state, emphases)`` says how it
    looks in a frame of an animation (``None``: not drawn; ``state`` is ``None`` for the still
    figure), ``paint(canvas, look)`` paints it so. An animated SVG paints each look an item
    takes once and switches between them."""

    css_class: str | None  # its own ``<g>``, or painted straight into its layer
    data_id: str | None
    look: Callable[[FrameState | None, _Emphases | None], Hashable | None]
    paint: Callable[[Canvas, Hashable], None]


class SceneRenderer:
    """Paints back to front: background, groups, edge lines, nodes, edge labels, notes.

    Collaborators (painters, text measurer) are injected, so the renderer never changes when
    a shape or backend is added.
    """

    def __init__(
        self,
        theme: Theme,
        painters: PainterRegistry | None = None,
        measurer: TextMeasurer | None = None,
    ) -> None:
        self._theme = theme
        self._painters = painters or default_painters()
        measurer = measurer or HeuristicTextMeasurer()
        self._measurer = measurer
        self._edges = EdgePainter(theme, measurer)
        self._groups = GroupPainter(theme, measurer)
        self._faded: Theme | None = None

    @property
    def theme(self) -> Theme:
        return self._theme

    @property
    def painters(self) -> PainterRegistry:
        return self._painters

    def render(self, scene: Scene, canvas: Canvas, state: FrameState | None = None) -> None:
        """Paint ``scene``; with a ``state``, as one frame of an animation: what has not run
        yet faded, what runs now highlighted."""
        self.background(scene, canvas)
        emphases = _Emphases(scene, state) if state is not None else None
        for layer, items in self.layers(scene):
            canvas.begin_group(layer)
            for item in items:
                look = item.look(state, emphases)
                if look is None:
                    continue
                if item.css_class is not None:
                    canvas.begin_group(item.css_class, item.data_id)
                item.paint(canvas, look)
                if item.css_class is not None:
                    canvas.end_group()
            canvas.end_group()

    def background(self, scene: Scene, canvas: Canvas) -> None:
        canvas.rect(Box(0, 0, scene.width, scene.height), Paint(self._theme.background))

    def emphases(self, scene: Scene, state: FrameState) -> _Emphases:
        """How the wires and frames of ``scene`` follow the nodes in ``state``."""
        return _Emphases(scene, state)

    def layers(self, scene: Scene) -> list[tuple[str, list[Item]]]:
        """The figure's items, layer by layer, back to front."""
        router = EdgeRouter(scene, self._theme, self._painters)
        routes = [(edge, router.route(edge)) for edge in scene.edges]
        labels = [self._group_labels(g) for g in scene.groups]
        labels += [self._edge_label(edge, path) for edge, path in routes]
        labels += [self._note(note) for note in scene.notes]
        return [
            ("tg-groups", [self._group(g) for g in scene.groups]),
            ("tg-edges", [self._edge(edge, path) for edge, path in routes]),
            ("tg-nodes", [self._node(n) for n in scene.nodes]),
            ("tg-labels", labels),
        ]

    # ------------------------------------------------------------------ items

    def _dim(self) -> Theme:
        if self._faded is None:
            self._faded = faded(self._theme)
            self._dim_edges = EdgePainter(self._faded, self._measurer)
            self._dim_groups = GroupPainter(self._faded, self._measurer)
        return self._faded

    def _group_painter(self, look: Emphasis) -> GroupPainter:
        if look is Emphasis.PENDING:
            self._dim()
            return self._dim_groups
        return self._groups

    def _edge_painter(self, look: Emphasis) -> EdgePainter:
        if look is Emphasis.PENDING:
            self._dim()
            return self._dim_edges
        return self._edges

    def _group(self, group: Group) -> Item:
        def look(state: FrameState | None, emphases: _Emphases | None) -> Emphasis:
            return emphases.group(group) if emphases is not None else Emphasis.DONE

        def paint(canvas: Canvas, look: Emphasis) -> None:
            self._group_painter(look).paint(canvas, group)

        return Item("tg-group", group.id, look, paint)

    def _group_labels(self, group: Group) -> Item:
        def look(state: FrameState | None, emphases: _Emphases | None) -> tuple[Emphasis, str]:
            if state is None or emphases is None:
                return Emphasis.DONE, group.badge
            return emphases.group(group), state.badges.get(group.id, group.badge)

        def paint(canvas: Canvas, look: tuple[Emphasis, str]) -> None:
            emphasis, badge = look
            shown = dataclasses.replace(group, badge=badge) if badge != group.badge else group
            self._group_painter(emphasis).paint_labels(canvas, shown)

        return Item(None, None, look, paint)

    def _edge(self, edge: Edge, path: list[Point]) -> Item:
        def look(state: FrameState | None, emphases: _Emphases | None) -> Emphasis:
            if emphases is not None and emphases.edge(edge) is Emphasis.PENDING:
                return Emphasis.PENDING
            return Emphasis.DONE

        def paint(canvas: Canvas, look: Emphasis) -> None:
            self._edge_painter(look).paint_line(canvas, edge, path)

        return Item(f"tg-edge tg-edge-{edge.kind}", None, look, paint)

    def _edge_label(self, edge: Edge, path: list[Point]) -> Item:
        def look(state: FrameState | None, emphases: _Emphases | None) -> Emphasis | None:
            if emphases is not None and emphases.edge(edge) is Emphasis.PENDING:
                return None  # its shape appears as the data arrives
            return Emphasis.DONE

        def paint(canvas: Canvas, look: Emphasis) -> None:
            self._edges.paint_label(canvas, edge, path)

        return Item(None, None, look, paint)

    def _note(self, note: Note) -> Item:
        def paint(canvas: Canvas, look: Emphasis) -> None:
            canvas.text(note.at, note.text, self._theme.font(note.role), note.anchor)

        return Item(None, None, lambda state, emphases: Emphasis.DONE, paint)

    def _node(self, node: Node) -> Item:
        def look(state: FrameState | None, emphases: _Emphases | None) -> Emphasis:
            return state.of(node.id) if state is not None else Emphasis.DONE

        def paint(canvas: Canvas, look: Emphasis) -> None:
            if look is Emphasis.ACTIVE:
                self._halo(canvas, node)
            styled = self._dim() if look is Emphasis.PENDING else self._theme
            self._painters.for_node(node, styled).paint(canvas, node, styled)
            if look is Emphasis.ACTIVE:
                self._ring(canvas, node)

        return Item(f"tg-node tg-{node.category}", node.id, look, paint)

    def _outline(self, canvas: Canvas, node: Node, grow: float, paint: Paint) -> None:
        """The node's own outline, grown by ``grow``: round for a merge circle, a stadium for
        a pill, a rounded rectangle otherwise."""
        shape = node.shape or self._theme.node_style(node.category).shape
        box = node.box
        if shape is Shape.CIRCLE:
            canvas.circle(box.center, min(box.w, box.h) / 2 + grow, paint)
        elif shape is Shape.PILL:
            canvas.rect(box.inflate(grow), paint, box.h / 2 + grow)
        else:
            canvas.rect(box.inflate(grow), paint, self._theme.metrics.corner_radius + grow)

    def _halo(self, canvas: Canvas, node: Node) -> None:
        """Behind an active node: a soft glow in its own outline colour."""
        stroke = self._theme.node_style(node.category).stroke
        self._outline(canvas, node, 6, Paint(stroke, opacity=0.18))

    def _ring(self, canvas: Canvas, node: Node) -> None:
        """Around an active node: a bold outline."""
        stroke = self._theme.node_style(node.category).stroke
        self._outline(canvas, node, 2.5, Paint(None, stroke, self._theme.metrics.node_stroke * 2.2))


class _Emphases:
    """How wires and frames follow the nodes: a wire takes its target's state (its source's,
    when it ends at a free point: a training arrow, a state loop), a frame is pending until a
    node inside it has run or runs."""

    def __init__(self, scene: Scene, state: FrameState) -> None:
        self.scene, self.state = scene, state
        self.nodes = [n for n in scene.nodes if not n.id.startswith("legend:")]

    def _near(self, point: Point) -> Emphasis:
        for node in self.nodes:
            box = node.box.inflate(30)
            if box.x <= point.x <= box.right and box.y <= point.y <= box.bottom:
                return self.state.of(node.id)
        return Emphasis.DONE

    def edge(self, edge: Edge) -> Emphasis:
        if any(isinstance(e, str) and e.startswith("legend:") for e in (edge.source, edge.target)):
            return Emphasis.DONE
        if isinstance(edge.target, str):
            return self.state.of(edge.target)
        if isinstance(edge.source, str):
            return self.state.of(edge.source)
        return self._near(edge.source)

    def group(self, group: Group) -> Emphasis:
        b = group.box
        inside = [
            n
            for n in self.nodes
            if b.x <= n.box.x
            and n.box.right <= b.right
            and b.y <= n.box.y
            and n.box.bottom <= b.bottom
        ]
        looks = {self.state.of(n.id) for n in inside}
        return Emphasis.PENDING if looks <= {Emphasis.PENDING} else Emphasis.DONE


def render_svg(
    scene: Scene,
    theme: Theme | None = None,
    *,
    inches: tuple[float, float] | None = None,
    state: FrameState | None = None,
) -> str:
    """Render ``scene`` to an SVG document (the ``default`` style unless given); ``inches``:
    the printed size, when it is not the natural one (a slide, a paper's column); ``state``:
    one frame of an animation."""
    if theme is None:
        from tensorgami.style import load_theme

        theme = load_theme()
    canvas = SvgCanvas(scene.width, scene.height, scene.title, scene.description, inches=inches)
    SceneRenderer(theme).render(scene, canvas, state)
    return canvas.to_svg()
