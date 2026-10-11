"""Group frames: a titled, dashed frame with an optional repeat badge (e.g. ``×6``)."""

from __future__ import annotations

from tensorgami.render.canvas import Canvas, Paint
from tensorgami.render.text import TextMeasurer
from tensorgami.scene import Box, Group, Point, TextAnchor
from tensorgami.style.glyphs import TextRole
from tensorgami.style.theme import Theme


class GroupPainter:
    def __init__(self, theme: Theme, measurer: TextMeasurer) -> None:
        self._theme, self._measurer = theme, measurer

    def paint(self, canvas: Canvas, group: Group) -> None:
        """The frame itself (painted first, behind everything)."""
        style = self._theme.group
        paint = Paint(style.fill, style.stroke, style.width, style.dash, style.fill_opacity)
        canvas.rect(group.box, paint, style.corner_radius)

    def paint_labels(self, canvas: Canvas, group: Group) -> None:
        """Title and badge, painted *after* the edges on a small background patch: a wire that
        enters the frame where its title sits passes visibly under the title."""
        style = self._theme.group
        b = group.box
        if group.title:
            font = self._theme.font(TextRole.GROUP_TITLE)
            at = Point(b.x + style.corner_radius, b.y + style.padding * 0.5 + font.size / 2)
            halo = self._theme.metrics.label_halo
            width = self._measurer.width(group.title, font)
            behind = self._theme.background if style.fill_opacity < 1 else style.fill
            canvas.rect(
                Box(
                    at.x - halo,
                    at.y - font.size * 0.6 - halo,
                    width + 2 * halo,
                    font.size * 1.2 + 2 * halo,
                ),
                Paint(behind),
                radius=2,
            )
            canvas.text(at, group.title, font, TextAnchor.START)
        if group.badge:
            self._badge(canvas, group)

    def _badge(self, canvas: Canvas, group: Group) -> None:
        """A pill sitting on the top-right corner of the frame."""
        style, font = self._theme.group, self._theme.font(TextRole.BADGE)
        width = self._measurer.width(group.badge, font) + 14
        height = font.size + 8
        cx = group.box.right - style.corner_radius - width / 2
        pill = Box.centered(cx, group.box.y, width, height)
        canvas.rect(pill, Paint(style.badge_fill, style.stroke, style.width), height / 2)
        canvas.text(pill.center, group.badge, font)
