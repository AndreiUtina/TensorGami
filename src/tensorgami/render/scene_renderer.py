"""Draws a :class:`Scene` onto any :class:`Canvas` using a :class:`Theme`."""

from __future__ import annotations

from tensorgami.render.canvas import Canvas, Paint
from tensorgami.render.edges import EdgePainter, EdgeRouter
from tensorgami.render.groups import GroupPainter
from tensorgami.render.painters import PainterRegistry, default_painters
from tensorgami.render.svg import SvgCanvas
from tensorgami.render.text import HeuristicTextMeasurer, TextMeasurer
from tensorgami.scene import Box, Scene
from tensorgami.style.theme import Theme


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
        self._edges = EdgePainter(theme, measurer)
        self._groups = GroupPainter(theme, measurer)

    def render(self, scene: Scene, canvas: Canvas) -> None:
        theme = self._theme
        canvas.rect(Box(0, 0, scene.width, scene.height), Paint(theme.background))

        canvas.begin_group("tg-groups")
        for group in scene.groups:
            canvas.begin_group("tg-group", group.id)
            self._groups.paint(canvas, group)
            canvas.end_group()
        canvas.end_group()

        router = EdgeRouter(scene, theme, self._painters)
        routes = [(edge, router.route(edge)) for edge in scene.edges]
        canvas.begin_group("tg-edges")
        for edge, path in routes:
            canvas.begin_group(f"tg-edge tg-edge-{edge.kind}")
            self._edges.paint_line(canvas, edge, path)
            canvas.end_group()
        canvas.end_group()

        canvas.begin_group("tg-nodes")
        for node in scene.nodes:
            canvas.begin_group(f"tg-node tg-{node.category}", node.id)
            self._painters.for_node(node, theme).paint(canvas, node, theme)
            canvas.end_group()
        canvas.end_group()

        canvas.begin_group("tg-labels")
        for group in scene.groups:
            self._groups.paint_labels(canvas, group)
        for edge, path in routes:
            self._edges.paint_label(canvas, edge, path)
        for note in scene.notes:
            canvas.text(note.at, note.text, theme.font(note.role), note.anchor)
        canvas.end_group()


def render_svg(scene: Scene, theme: Theme | None = None) -> str:
    """Render ``scene`` to an SVG document (the ``default`` style unless given)."""
    if theme is None:
        from tensorgami.style import load_theme

        theme = load_theme()
    canvas = SvgCanvas(scene.width, scene.height, scene.title, scene.description)
    SceneRenderer(theme).render(scene, canvas)
    return canvas.to_svg()
