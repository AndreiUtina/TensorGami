"""An animation as one SVG document (roadmap step 14c).

Every part of the figure is drawn once for each look it takes during the animation (still to
come, running, run; a frame's badge for each pass), and SMIL ``<animate>`` elements on one
shared clock show the look of the moment. Pulses move along the wires with
``<animateMotion>``. The document is vector, so it stays sharp at any size; it plays in
browsers (also as an ``<img>``: GitHub's README, VS Code's preview), and a viewer that does not
animate shows the still figure.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from tensorgami.render.edges import EdgeRouter
from tensorgami.render.scene_renderer import FrameState, Item, Layer, SceneRenderer
from tensorgami.render.svg import SvgCanvas, _num
from tensorgami.scene import Point, Scene
from tensorgami.style.theme import Theme

#: the look of a pulse, shared by the GIF and the SVG: its radius (px of the figure), a ring
#: in the background's colour, and a tail of fading dots (alpha 0…255, size)
PULSE_RADIUS = 4.5
PULSE_RING = 0.35  # the ring's width, in radii
LOOP_PULSE = 0.7  # a pulse on a state loop, which is small, is this much smaller
TRAIL = [(190, 0.62), (140, 0.52), (100, 0.44), (65, 0.37), (35, 0.31)]
TRAIL_SPACING = 1.25  # between the tail's dots, in radii


@dataclass(frozen=True)
class Pulse:
    """Data travelling along ``scene.edges[edge]`` from ``begin`` to ``end`` seconds into the
    animation, from ``start`` of the way along the wire (0 … 1) to its end."""

    edge: int
    begin: float
    end: float
    start: float = 0.0


def render_animated_svg(
    scene: Scene,
    theme: Theme,
    frames: Sequence[tuple[FrameState, float]],
    pulses: Sequence[Pulse] = (),
    *,
    inches: tuple[float, float] | None = None,
    overlays: Sequence[Layer] = (),
) -> str:
    """``scene`` as an animated SVG: each frame's state for its duration (seconds), the pulses
    on top, looping. Without animation (or at the end of each loop) it is the last frame."""
    total = sum(duration for _, duration in frames)
    if not frames or total <= 0:
        raise ValueError("an animation needs at least one frame that lasts")
    starts = [sum(d for _, d in frames[:i]) for i in range(len(frames))]
    clock = _Clock(total, starts)
    renderer = SceneRenderer(theme, overlays=overlays)
    states = [state for state, _ in frames]
    emphases = [renderer.emphases(scene, state) for state in states]
    canvas = SvgCanvas(scene.width, scene.height, scene.title, scene.description, inches=inches)
    renderer.background(scene, canvas)
    for layer, items in renderer.layers(scene):
        canvas.begin_group(layer)
        for item in items:
            looks = [item.look(s, e) for s, e in zip(states, emphases, strict=True)]
            _paint_item(canvas, item, looks, clock)
        canvas.end_group()
    if pulses:
        router = EdgeRouter(scene, theme, renderer.painters)
        canvas.begin_group("tg-pulses")
        for pulse in pulses:
            edge = scene.edges[pulse.edge]
            radius = PULSE_RADIUS * (LOOP_PULSE if edge.delay else 1.0)
            color = theme.edge_style(edge.kind).color.hex
            _paint_pulse(canvas, router.route(edge), pulse, radius, color, theme, clock)
        canvas.end_group()
    return canvas.to_svg()


class _Clock:
    """Times as SMIL wants them: fractions of the loop's duration."""

    def __init__(self, total: float, starts: list[float]) -> None:
        self.total, self.starts = total, starts

    def dur(self) -> str:
        return f'dur="{_num(self.total)}s" repeatCount="indefinite"'

    def key(self, seconds: float) -> str:
        return _fraction(seconds / self.total)

    def switch(self, shown: Sequence[bool]) -> str:
        """An ``<animate>`` showing its element during the frames where ``shown``."""
        values, times = [], []
        for visible, at in zip(shown, self.starts, strict=True):
            value = "visible" if visible else "hidden"
            if not values or value != values[-1]:
                values.append(value)
                times.append(self.key(at))
        times[0] = "0"
        return (
            f'<animate attributeName="visibility" {self.dur()} calcMode="discrete" '
            f'values="{";".join(values)}" keyTimes="{";".join(times)}"/>'
        )


def _paint_item(canvas: SvgCanvas, item: Item, looks: list[Hashable | None], clock: _Clock) -> None:
    """The item in each look it takes, each shown while it has that look."""
    distinct = list(dict.fromkeys(look for look in looks if look is not None))
    if not distinct:
        return
    if item.css_class is not None:
        canvas.begin_group(item.css_class, item.data_id)
    if len(distinct) == 1 and None not in looks:
        item.paint(canvas, distinct[0])  # the same all along
    else:
        for look in distinct:
            sketch = SvgCanvas(1, 1)
            item.paint(sketch, look)
            drawn = sketch.elements()
            if not drawn:
                continue
            shown = [other == look for other in looks]
            still = "visible" if shown[-1] else "hidden"  # what a viewer without SMIL shows
            canvas.begin_group("", None, f' visibility="{still}"')
            canvas.raw(clock.switch(shown))
            for element in drawn:
                canvas.raw(element)
            canvas.end_group()
    if item.css_class is not None:
        canvas.end_group()


def _paint_pulse(
    canvas: SvgCanvas,
    path: list[Point],
    pulse: Pulse,
    radius: float,
    color: str,
    theme: Theme,
    clock: _Clock,
) -> None:
    """The head and its fading tail, each a dot moving along the wire during the pulse."""
    length = sum(math.dist((a.x, a.y), (b.x, b.y)) for a, b in zip(path, path[1:], strict=False))
    if length <= 0 or pulse.start >= 1:
        return
    route = "M" + " L".join(f"{_num(p.x)},{_num(p.y)}" for p in path)
    for k in range(len(TRAIL), 0, -1):  # the tail first, the head drawn over it
        behind = k * radius * TRAIL_SPACING / length  # of the wire's length
        if pulse.start + behind >= 1:
            continue
        alpha, size = TRAIL[k - 1]
        late = pulse.begin + (pulse.end - pulse.begin) * behind / (1 - pulse.start)
        paint = f'fill="{color}" fill-opacity="{_num(alpha / 255)}"'
        dot = _Dot(radius * size, paint, late, pulse.end, pulse.start, 1 - behind)
        canvas.raw(dot.markup(route, clock))
    ring = radius * PULSE_RING
    paint = f'fill="{color}" stroke="{theme.background.hex}" stroke-width="{_num(ring)}"'
    head = _Dot(radius + ring / 2, paint, pulse.begin, pulse.end, pulse.start, 1.0)
    canvas.raw(head.markup(route, clock))


@dataclass(frozen=True)
class _Dot:
    """A dot moving from ``start`` to ``stop`` of the way along a route between ``begin`` and
    ``end`` seconds, hidden before and after."""

    radius: float
    paint: str
    begin: float
    end: float
    start: float
    stop: float

    def markup(self, route: str, clock: _Clock) -> str:
        a, b = clock.key(self.begin), clock.key(self.end)
        s, e = _fraction(self.start), _fraction(self.stop)
        if b == "1":
            times, points = f"0;{a};1", f"{s};{s};{e}"
        else:
            times, points = f"0;{a};{b};1", f"{s};{s};{e};{e}"
        return (
            f'<circle r="{_num(self.radius)}" {self.paint} visibility="hidden">'
            f'<animateMotion {clock.dur()} calcMode="linear" keyTimes="{times}" '
            f'keyPoints="{points}" path="{route}"/>'
            f'<animate attributeName="visibility" {clock.dur()} calcMode="discrete" '
            f'values="hidden;visible;hidden" keyTimes="0;{a};{b}"/></circle>'
        )


def _fraction(value: float) -> str:
    """A fraction 0 … 1, to a millionth (a key time, a point along a wire)."""
    text = f"{min(1.0, max(0.0, value)):.6f}".rstrip("0").rstrip(".")
    return text or "0"
