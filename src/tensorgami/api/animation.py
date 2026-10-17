"""``fig.animate()``: the figure lighting up as the model runs (roadmap steps 14a, 14b).

The *timeline* orders the figure's nodes the way data reaches them: a node's step is one more
than the latest of its inputs, so parallel branches light up together and a merge waits for
every branch; buffers and constants light up just before the layer that reads them. A block
folded as ×N repeats its steps N times, its badge counting the passes (1/N, 2/N, …).

Every frame is the still figure in a state (:class:`~tensorgami.render.FrameState`): what has
not run yet faded, what runs now highlighted (a halo in its own colour and a bold outline),
what has run as in the still figure. Between two steps the data travels (14b): a pulse moves
along every wire bringing the next step its input, and a wire's shape appears once its data
has arrived; when a repeated block (a recurrent cell folded over time) starts another pass,
pulses go round its state loops. An animation is saved as a GIF (Pillow, which matplotlib
already needs), as an MP4 (with ffmpeg), or as one file per step (PNG, PDF, SVG) for slides
that build up step by step.
"""

from __future__ import annotations

import io
import math
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tensorgami.api.player import page_html, player_html
from tensorgami.ir import Graph
from tensorgami.render.animated import (
    LOOP_PULSE,
    PULSE_RADIUS,
    PULSE_RING,
    TRAIL,
    TRAIL_SPACING,
    Pulse,
    render_animated_svg,
)
from tensorgami.render.edges import EdgeRouter
from tensorgami.render.mpl import quiet_fonts, render_figure
from tensorgami.render.painters import default_painters
from tensorgami.render.scene_renderer import FrameState, render_svg
from tensorgami.scene import Box, Point, Scene
from tensorgami.vocabulary import EdgeKind

if TYPE_CHECKING:
    from PIL.Image import Image

    from tensorgami.api.figure import Figure


@dataclass(frozen=True)
class Step:
    """What runs at one step of the timeline, and the badges it shows (a block's pass)."""

    active: frozenset[str]
    badges: dict[str, str] = field(default_factory=dict)


def timeline(graph: Graph, scene: Scene) -> list[Step]:
    """The steps of ``graph`` as drawn in ``scene``, in the order data reaches its nodes."""
    shown = {n.id for n in scene.nodes if not n.id.startswith("legend:")}
    order = {n.id: i for i, n in enumerate(graph.nodes)}
    preds: dict[str, set[str]] = defaultdict(set)
    succs: dict[str, set[str]] = defaultdict(set)
    for e in graph.edges:
        if e.kind is EdgeKind.TRAINING or e.source == e.target:
            continue  # a training signal flows back; a loop is the state a cell keeps
        if order[e.source] >= order[e.target]:
            continue  # state fed back to an earlier node: not something it waits for
        preds[e.target].add(e.source)
        succs[e.source].add(e.target)
    depth: dict[str, int] = {}
    for node in graph.nodes:  # execution order: every predecessor comes first
        depth[node.id] = max((depth[p] + 1 for p in preds[node.id]), default=0)
    for node in graph.nodes:  # a buffer or constant: just before the layer that reads it
        if not preds[node.id] and not node.is_input and succs[node.id]:
            depth[node.id] = max(0, min(depth[s] for s in succs[node.id]) - 1)
    levels: dict[int, set[str]] = defaultdict(set)
    for node_id, d in depth.items():
        if node_id in shown:
            levels[d].add(node_id)
    steps = [Step(frozenset(levels[d])) for d in sorted(levels)]
    return _repeat_passes(graph, steps)


def _repeat_passes(graph: Graph, steps: list[Step]) -> list[Step]:
    """A run of steps inside a block folded ×N, played N times with its pass on the badge."""
    repeated = [g for g in graph.groups if g.count > 1]
    if not repeated:
        return steps
    under = {g.id: {n.id for n in graph.nodes_under(g.id)} for g in repeated}
    outermost = [g for g in repeated if not any(a.id in under for a in graph.ancestors(g.id))]

    def block(step: Step) -> str | None:
        return next((g.id for g in outermost if step.active <= under[g.id]), None)

    played: list[Step] = []
    i = 0
    while i < len(steps):
        gid = block(steps[i])
        if gid is None:
            played.append(steps[i])
            i += 1
            continue
        j = i
        while j < len(steps) and block(steps[j]) == gid:
            j += 1
        count = graph.group(gid).count
        for k in range(1, count + 1):
            played += [Step(s.active, {gid: f"{k}/{count}"}) for s in steps[i:j]]
        i = j
    return played


@dataclass(frozen=True)
class Frame:
    """One frame: the state of step ``step`` (an index into :attr:`Animation.steps`) and the
    data on its way to the next step: scene edge index → how far along its wire (0 … 1), and
    where along it each pulse set off (its tail never reaches behind that)."""

    step: int
    pulses: Mapping[int, float] = field(default_factory=dict)
    starts: Mapping[int, float] = field(default_factory=dict)


class Animation:
    """A figure lighting up as the model runs: ``fig.animate()``.

    * ``anim.save("model.gif")`` (or ``.mp4``, with ffmpeg installed);
    * ``anim.save_frames("frames/")``: one PNG (or PDF, SVG) per step, for slide builds;
    * ``anim.steps``: the state of each step; ``anim[i]``: frame ``i`` (a step's state and the
      pulses on their way); ``len(anim)``: the number of frames;
    * in Jupyter it plays inline.
    """

    def __init__(
        self,
        figure: Figure,
        *,
        step: float = 0.6,
        hold: float = 1.5,
        flow: bool = True,
        travel: float = 0.4,
        fps: float = 15,
    ) -> None:
        if figure.graph is None:
            raise ValueError("an animation follows the model's graph: draw it with tg.draw")
        if min(step, hold, travel, fps) <= 0:
            raise ValueError("step, hold, travel and fps must be greater than 0")
        if fps > 50:
            raise ValueError(f"fps: at most 50 (browsers slow faster GIFs down), got {fps}")
        self._figure = figure
        self._images: dict[tuple[int, float], Image] = {}
        self._paths: dict[int, list[Point]] = {}
        timed = timeline(figure.graph, figure.scene)
        everything = frozenset(n for s in timed for n in s.active)
        self._steps: list[FrameState] = []
        done: frozenset[str] = frozenset()
        for s in timed:
            self._steps.append(FrameState(s.active, done - s.active, s.badges))
            done |= s.active
        self._steps.append(FrameState(frozenset(), everything))  # the whole model, run
        self._frames: list[Frame] = []
        self._durations: list[float] = []
        between = max(1, round(travel * fps))
        tick = round(1 / fps, 2)  # a GIF counts time in hundredths of a second
        for k in range(len(timed)):
            carrying = self._incoming(timed, k) if flow and k > 0 else {}
            for i in range(between if carrying else 0):  # the data on its way
                t = (i + 0.5) / between
                pulses = {e: start + (1 - start) * t for e, start in carrying.items()}
                self._frames.append(Frame(k - 1, pulses, carrying))
                self._durations.append(tick)
            self._frames.append(Frame(k))
            self._durations.append(step)
        self._frames.append(Frame(len(timed)))
        self._durations.append(hold)

    def _incoming(self, timed: list[Step], k: int) -> dict[int, float]:
        """The scene edges carrying data into step ``k``, each with where along it its pulse
        starts: wires from what has run into what runs next, and, when a repeated block starts
        another pass, its state loops.

        A block repeated over time (with state loops) reads each pass's input from outside, so
        its pulses start at their source. A stack of layers (no loops) reads, from its second
        pass on, what its previous pass produced: the wires entering it carry the block's
        input, and their pulses start where they enter its frame."""
        scene = self._figure.scene
        before = frozenset(n for s in timed[:k] for n in s.active)
        boxes = {g.id: g.box for g in scene.groups}
        stacks = [
            boxes[gid]
            for gid, badge in timed[k].badges.items()
            if gid in boxes and not badge.startswith("1/") and not self._loops(boxes[gid])
        ]
        edges: dict[int, float] = {}
        for index, e in enumerate(scene.edges):
            if e.kind is EdgeKind.TRAINING or not isinstance(e.target, str):
                continue
            if e.target in timed[k].active and isinstance(e.source, str) and e.source in before:
                source = scene.node(e.source).box.center
                outside = [b for b in stacks if not _inside(source, b)]
                edges[index] = self._entry(index, outside[0]) if outside else 0.0
        for gid, badge in timed[k].badges.items():
            if (
                gid in boxes
                and badge != timed[k - 1].badges.get(gid)
                and not badge.startswith("1/")
            ):
                edges.update(dict.fromkeys(self._loops(boxes[gid]), 0.0))  # another pass
        return edges

    def _loops(self, box: Box) -> list[int]:
        """The state loops drawn inside ``box``."""
        return [
            i
            for i, e in enumerate(self._figure.scene.edges)
            if e.delay and isinstance(e.source, Point) and _inside(e.source, box)
        ]

    def _entry(self, edge: int, box: Box) -> float:
        """How far along ``edge`` (0 … 1) it enters ``box``."""
        path = self._path(edge)
        total, run = _length(path), 0.0
        for a, b in zip(path, path[1:], strict=False):
            length = math.dist((a.x, a.y), (b.x, b.y))
            if not _inside(a, box) and _inside(b, box):
                low, high = 0.0, 1.0  # where the segment crosses the frame
                for _ in range(30):
                    mid = (low + high) / 2
                    point = Point(a.x + (b.x - a.x) * mid, a.y + (b.y - a.y) * mid)
                    low, high = (low, mid) if _inside(point, box) else (mid, high)
                return (run + high * length) / total if total else 0.0
            run += length
        return 0.0

    # ------------------------------------------------------------------ inspection

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> Frame:
        return self._frames[index]

    @property
    def steps(self) -> list[FrameState]:
        """The state of each step, the whole model last: what ``save_frames`` writes."""
        return list(self._steps)

    @property
    def durations(self) -> list[float]:
        """How long each frame shows, in seconds."""
        return list(self._durations)

    @property
    def figure(self) -> Figure:
        return self._figure

    def __repr__(self) -> str:
        title = self._figure.scene.title
        seconds = sum(self._durations)
        return (
            f"<tensorgami.Animation {title!r} {len(self._steps)} steps, {len(self)} frames, "
            f"{seconds:.1f} s>"
        )

    # ------------------------------------------------------------------ frames

    def _render(self, step: int, dpi: float) -> bytes:
        fig = self._figure
        figure = render_figure(
            fig.scene, fig.style, dpi=dpi, width_in=fig.inches[0], state=self._steps[step]
        )
        buffer = io.BytesIO()
        with quiet_fonts():
            figure.savefig(buffer, format="png", dpi=dpi)
        return buffer.getvalue()

    def _image(self, index: int, dpi: float) -> Image:
        """Frame ``index`` as an RGB image: its step rendered once, the pulses drawn on."""
        from PIL import Image

        frame = self._frames[index]
        key = (frame.step, dpi)
        if key not in self._images:
            self._images[key] = Image.open(io.BytesIO(self._render(frame.step, dpi))).convert("RGB")
        image = self._images[key]
        if not frame.pulses:
            return image
        image = image.copy()
        for edge, along in frame.pulses.items():
            self._pulse(image, edge, along, frame.starts.get(edge, 0.0))
        return image

    def to_png(self, index: int, *, dpi: float = 96) -> bytes:
        """Frame ``index`` as PNG bytes."""
        buffer = io.BytesIO()
        self._image(index, dpi).save(buffer, format="PNG")
        return buffer.getvalue()

    def to_svg(self, step: int | None = None) -> str:
        """The animation as an animated SVG: looping, sharp at any size, playing in browsers
        (a viewer that does not animate shows the still figure); or, with ``step``, that step
        (see :attr:`steps`) as a still SVG."""
        fig = self._figure
        inches = fig.inches if fig.fitted is not None else None
        if step is not None:
            return render_svg(fig.scene, fig.style, inches=inches, state=self._steps[step])
        frames, pulses, _ = self._timed()
        return render_animated_svg(fig.scene, fig.style, frames, pulses, inches=inches)

    def to_html(self) -> str:
        """The player: the animated SVG with play/pause, step back and forward, a time slider
        and the speed (what Jupyter shows)."""
        _, _, starts = self._timed()
        total = sum(self._durations)
        return player_html(
            self.to_svg(), total=total, starts=starts, title=self._figure.scene.title
        )

    def _timed(self) -> tuple[list[tuple[FrameState, float]], list[Pulse], list[float]]:
        """The frames as (state, seconds), the pulses with their times, and when each step
        lights up (seconds into the animation)."""
        timed = list(zip(self._frames, self._durations, strict=True))
        frames = [(self._steps[frame.step], seconds) for frame, seconds in timed]
        pulses: list[Pulse] = []
        starts: list[float] = []
        now, travel = 0.0, None  # travel: (since when, from where along) the data under way
        for frame, seconds in timed:
            if frame.pulses and travel is None:
                travel = (now, frame.starts)
            elif not frame.pulses:
                if travel is not None:
                    pulses += [Pulse(e, travel[0], now, s) for e, s in travel[1].items()]
                    travel = None
                starts.append(now)
            now += seconds
        return frames, pulses, starts

    def _path(self, edge: int) -> list[Point]:
        if edge not in self._paths:
            fig = self._figure
            router = EdgeRouter(fig.scene, fig.style, default_painters())
            self._paths[edge] = router.route(fig.scene.edges[edge])
        return self._paths[edge]

    def _pulse(self, image: Image, edge: int, along: float, start: float = 0.0) -> None:
        """A dot ``along`` the way down the wire, in the wire's colour, ringed in the
        background's and trailing a short fading tail (a comet: the motion reads in a still)."""
        scene, style = self._figure.scene, self._figure.style
        path = self._path(edge)
        color = style.edge_style(scene.edges[edge].kind).color.hex
        radius = PULSE_RADIUS * (LOOP_PULSE if scene.edges[edge].delay else 1.0)
        scale = image.width / scene.width
        length = _length(path)
        head = along * length
        for k in range(len(TRAIL), 0, -1):  # the tail first, the head drawn over it
            behind = head - k * radius * TRAIL_SPACING
            if behind > start * length:
                alpha, size = TRAIL[k - 1]
                self._paste(image, _dot(color, radius * size * scale, alpha), path, behind)
        ring = style.background.hex
        self._paste(image, _head(color, ring, radius * scale), path, head)

    def _paste(self, image: Image, sprite: Image, path: list[Point], distance: float) -> None:
        scene = self._figure.scene
        point = _point_at(path, distance)
        x = round(point.x * image.width / scene.width - sprite.width / 2)
        y = round(point.y * image.height / scene.height - sprite.height / 2)
        image.paste(sprite, (x, y), sprite)

    # ------------------------------------------------------------------ output

    def save(self, path: str | Path, *, dpi: float = 96) -> Path:
        """Write the animation: ``.gif`` (looping), ``.svg`` (animated, sharp at any size),
        ``.html`` (a page with the player) or ``.mp4`` (needs ffmpeg)."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".gif":
            path.write_bytes(self._gif(dpi))
        elif suffix == ".svg":
            path.write_text(self.to_svg(), encoding="utf-8")
        elif suffix == ".html":
            title = self._figure.scene.title or "animation"
            path.write_text(page_html(self.to_html(), title), encoding="utf-8")
        elif suffix == ".mp4":
            self._mp4(path, dpi)
        else:
            raise ValueError(
                f"an animation is saved as .gif, .svg, .html or .mp4, not "
                f"{suffix or 'a file without suffix'}; for one image per step use "
                "anim.save_frames(folder)"
            )
        return path

    def save_frames(
        self, folder: str | Path, *, format: str = "png", dpi: float = 300
    ) -> list[Path]:
        """One file per step (``step-01.png`` …), the whole model last: e.g. for a slide that
        builds up step by step; ``format``: ``"png"``, ``"pdf"`` or ``"svg"``."""
        if format not in ("png", "pdf", "svg"):
            raise ValueError(f"format: 'png', 'pdf' or 'svg', got {format!r}")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        width = len(str(len(self._steps)))
        fig = self._figure
        paths = []
        for i in range(len(self._steps)):
            path = folder / f"step-{i + 1:0{width}d}.{format}"
            if format == "svg":
                path.write_text(self.to_svg(i), encoding="utf-8")
            elif format == "png":
                path.write_bytes(self._render(i, dpi))
            else:
                figure = render_figure(
                    fig.scene, fig.style, width_in=fig.inches[0], state=self._steps[i]
                )
                with quiet_fonts():
                    figure.savefig(path, format="pdf")
            paths.append(path)
        return paths

    def _gif(self, dpi: float) -> bytes:
        from PIL import Image

        frames = [self._image(i, dpi) for i in range(len(self))]
        # one palette for every frame (from the first, middle and last), so colours never flicker
        w, h = frames[0].size
        sample = Image.new("RGB", (w * 3, h))
        for k, i in enumerate((0, len(frames) // 2, len(frames) - 1)):
            sample.paste(frames[i], (k * w, 0))
        palette = sample.quantize(colors=256)
        indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        buffer = io.BytesIO()
        indexed[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=indexed[1:],
            duration=[round(d * 1000) for d in self._durations],
            loop=0,
            disposal=1,
            optimize=False,  # Pillow's per-frame optimiser: 30× slower for 15% smaller
        )
        return buffer.getvalue()

    def _mp4(self, path: Path, dpi: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "saving an MP4 needs ffmpeg (https://ffmpeg.org) on the PATH; install it, or "
                "save the animation as .gif"
            )
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            lines = []
            for i, seconds in enumerate(self._durations):
                frame = folder / f"{i:05d}.png"
                frame.write_bytes(self.to_png(i, dpi=dpi))
                lines += [f"file '{frame.name}'", f"duration {seconds}"]
            lines.append(f"file '{len(self) - 1:05d}.png'")  # the concat format's last frame
            (folder / "frames.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            command = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(folder / "frames.txt"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white,format=yuv420p",
                "-r",
                "30",
                "-c:v",
                "libx264",
                str(path.resolve()),
            ]
            subprocess.run(command, check=True, cwd=folder)

    # ------------------------------------------------------------------ notebooks

    def _repr_html_(self) -> str:
        """Jupyter: the player."""
        return self.to_html()


def _inside(point: Point, box: Box) -> bool:
    return box.x <= point.x <= box.right and box.y <= point.y <= box.bottom


def _length(path: list[Point]) -> float:
    return sum(math.dist((a.x, a.y), (b.x, b.y)) for a, b in zip(path, path[1:], strict=False))


def _point_at(path: list[Point], distance: float) -> Point:
    """The point ``distance`` px down a polyline (its ends beyond it)."""
    for a, b in zip(path, path[1:], strict=False):
        length = math.dist((a.x, a.y), (b.x, b.y))
        if distance <= length and length > 0:
            t = max(0.0, distance) / length
            return Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
        distance -= length
    return path[-1]


_SPRITES: dict[tuple, Image] = {}


def _disc(fill: tuple[int, ...], radius: float, ring: tuple[int, ...] | None = None) -> Image:
    """A smooth disc: drawn four times larger and scaled down."""
    from PIL import Image, ImageDraw

    key = (fill, round(radius, 2), ring)
    if key not in _SPRITES:
        size = math.ceil(2 * radius) + 3
        big = Image.new("RGBA", (size * 4, size * 4), (*fill[:3], 0))
        c, r = size * 2, radius * 4
        width = max(1, round(radius * 1.4)) if ring else 0
        ImageDraw.Draw(big).ellipse([c - r, c - r, c + r, c + r], fill, ring, width)
        _SPRITES[key] = big.resize((size, size), Image.Resampling.LANCZOS)
    return _SPRITES[key]


def _dot(color: str, radius: float, alpha: int) -> Image:
    from PIL import ImageColor

    return _disc((*ImageColor.getrgb(color)[:3], alpha), radius)


def _head(color: str, ring: str, radius: float) -> Image:
    from PIL import ImageColor

    rgb = ImageColor.getrgb(color)[:3]
    return _disc((*rgb, 255), radius * (1 + PULSE_RING), (*ImageColor.getrgb(ring)[:3], 255))
