"""Draws a :class:`Scene` onto any :class:`Canvas` using a :class:`Theme`."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from tensorgami.render.canvas import Canvas, Paint
from tensorgami.render.edges import EdgePainter, EdgeRouter
from tensorgami.render.groups import GroupPainter
from tensorgami.render.painters import PainterRegistry, default_painters
from tensorgami.render.svg import SvgCanvas
from tensorgami.render.text import HeuristicTextMeasurer, TextMeasurer
from tensorgami.scene import Box, Edge, Group, Node, Point, Scene
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

    def render(self, scene: Scene, canvas: Canvas, state: FrameState | None = None) -> None:
        """Paint ``scene``; with a ``state``, as one frame of an animation: what has not run
        yet faded, what runs now highlighted."""
        theme = self._theme
        canvas.rect(Box(0, 0, scene.width, scene.height), Paint(theme.background))
        if state is not None:
            fade = faded(theme)
            dim_edges = EdgePainter(fade, self._measurer)
            dim_groups = GroupPainter(fade, self._measurer)
            emphasis = _Emphases(scene, state)
        groups = [
            dataclasses.replace(g, badge=state.badges[g.id])
            if state is not None and g.id in state.badges
            else g
            for g in scene.groups
        ]

        def group_painter(group: Group) -> GroupPainter:
            dim = state is not None and emphasis.group(group) is Emphasis.PENDING
            return dim_groups if dim else self._groups

        def edge_painter(edge: Edge) -> EdgePainter:
            dim = state is not None and emphasis.edge(edge) is Emphasis.PENDING
            return dim_edges if dim else self._edges

        canvas.begin_group("tg-groups")
        for group in groups:
            canvas.begin_group("tg-group", group.id)
            group_painter(group).paint(canvas, group)
            canvas.end_group()
        canvas.end_group()

        router = EdgeRouter(scene, theme, self._painters)
        routes = [(edge, router.route(edge)) for edge in scene.edges]
        canvas.begin_group("tg-edges")
        for edge, path in routes:
            canvas.begin_group(f"tg-edge tg-edge-{edge.kind}")
            edge_painter(edge).paint_line(canvas, edge, path)
            canvas.end_group()
        canvas.end_group()

        canvas.begin_group("tg-nodes")
        for node in scene.nodes:
            canvas.begin_group(f"tg-node tg-{node.category}", node.id)
            look = state.of(node.id) if state is not None else Emphasis.DONE
            if look is Emphasis.ACTIVE:
                self._halo(canvas, node)
            styled = fade if look is Emphasis.PENDING else theme
            self._painters.for_node(node, styled).paint(canvas, node, styled)
            if look is Emphasis.ACTIVE:
                self._ring(canvas, node)
            canvas.end_group()
        canvas.end_group()

        canvas.begin_group("tg-labels")
        for group in groups:
            group_painter(group).paint_labels(canvas, group)
        for edge, path in routes:
            if state is not None and emphasis.edge(edge) is Emphasis.PENDING:
                continue  # its shape appears as the data arrives
            edge_painter(edge).paint_label(canvas, edge, path)
        for note in scene.notes:
            canvas.text(note.at, note.text, theme.font(note.role), note.anchor)
        canvas.end_group()

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
