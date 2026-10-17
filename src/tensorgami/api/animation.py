"""``fig.animate()``: the figure lighting up as the model runs (roadmap step 14a).

The *timeline* orders the figure's nodes the way data reaches them: a node's step is one more
than the latest of its inputs, so parallel branches light up together and a merge waits for
every branch; buffers and constants light up just before the layer that reads them. A block
folded as ×N repeats its steps N times, its badge counting the passes (1/N, 2/N, …).

Every frame is the still figure in a state (:class:`~tensorgami.render.FrameState`): what has
not run yet faded, what runs now highlighted (a halo in its own colour and a bold outline),
what has run as in the still figure. An animation is saved as a GIF (Pillow, which matplotlib
already needs), as an MP4 (with ffmpeg), or as one file per step (PNG, PDF, SVG) for slides
that build up step by step.
"""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tensorgami.ir import Graph
from tensorgami.render.mpl import quiet_fonts, render_figure
from tensorgami.render.scene_renderer import FrameState, render_svg
from tensorgami.scene import Scene
from tensorgami.vocabulary import EdgeKind

if TYPE_CHECKING:
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


class Animation:
    """A figure lighting up as the model runs: ``fig.animate()``.

    * ``anim.save("model.gif")`` (or ``.mp4``, with ffmpeg installed);
    * ``anim.save_frames("frames/")``: one PNG (or PDF, SVG) per step, for slide builds;
    * ``anim[i]`` is the state of frame ``i``; ``len(anim)`` the number of frames;
    * in Jupyter it plays inline.
    """

    def __init__(self, figure: Figure, *, step: float = 0.6, hold: float = 1.5) -> None:
        if figure.graph is None:
            raise ValueError("an animation follows the model's graph: draw it with tg.draw")
        if step <= 0 or hold <= 0:
            raise ValueError("step and hold are durations in seconds, greater than 0")
        self._figure = figure
        self._steps = timeline(figure.graph, figure.scene)
        everything = frozenset(n for s in self._steps for n in s.active)
        self._frames: list[FrameState] = []
        done: frozenset[str] = frozenset()
        for s in self._steps:
            self._frames.append(FrameState(s.active, done - s.active, s.badges))
            done |= s.active
        self._frames.append(FrameState(frozenset(), everything))  # the whole model, run
        self._durations = [step] * len(self._steps) + [hold]

    # ------------------------------------------------------------------ inspection

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> FrameState:
        return self._frames[index]

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
        return f"<tensorgami.Animation {title!r} {len(self)} frames, {seconds:.1f} s>"

    # ------------------------------------------------------------------ frames

    def to_png(self, index: int, *, dpi: float = 96) -> bytes:
        """Frame ``index`` as PNG bytes."""
        fig = self._figure
        figure = render_figure(
            fig.scene, fig.style, dpi=dpi, width_in=fig.inches[0], state=self._frames[index]
        )
        buffer = io.BytesIO()
        with quiet_fonts():
            figure.savefig(buffer, format="png", dpi=dpi)
        return buffer.getvalue()

    def to_svg(self, index: int) -> str:
        fig = self._figure
        inches = fig.inches if fig.fitted is not None else None
        return render_svg(fig.scene, fig.style, inches=inches, state=self._frames[index])

    # ------------------------------------------------------------------ output

    def save(self, path: str | Path, *, dpi: float = 96) -> Path:
        """Write the animation: ``.gif`` (looping) or ``.mp4`` (needs ffmpeg)."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".gif":
            path.write_bytes(self._gif(dpi))
        elif suffix == ".mp4":
            self._mp4(path, dpi)
        else:
            raise ValueError(
                f"an animation is saved as .gif or .mp4, not {suffix or 'a file without suffix'};"
                " for one image per step use anim.save_frames(folder)"
            )
        return path

    def save_frames(
        self, folder: str | Path, *, format: str = "png", dpi: float = 300
    ) -> list[Path]:
        """One file per frame (``step-01.png`` …), e.g. for a slide that builds up step by
        step; ``format``: ``"png"``, ``"pdf"`` or ``"svg"``."""
        if format not in ("png", "pdf", "svg"):
            raise ValueError(f"format: 'png', 'pdf' or 'svg', got {format!r}")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        width = len(str(len(self)))
        paths = []
        for i in range(len(self)):
            path = folder / f"step-{i + 1:0{width}d}.{format}"
            if format == "svg":
                path.write_text(self.to_svg(i), encoding="utf-8")
            elif format == "png":
                path.write_bytes(self.to_png(i, dpi=dpi))
            else:
                fig = self._figure
                figure = render_figure(
                    fig.scene, fig.style, width_in=fig.inches[0], state=self._frames[i]
                )
                with quiet_fonts():
                    figure.savefig(path, format="pdf")
            paths.append(path)
        return paths

    def _gif(self, dpi: float) -> bytes:
        from PIL import Image

        frames = [
            Image.open(io.BytesIO(self.to_png(i, dpi=dpi))).convert("RGB") for i in range(len(self))
        ]
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
        """Jupyter: the GIF, inline, at the figure's natural size."""
        data = base64.b64encode(self._gif(96)).decode("ascii")
        width = round(self._figure.size[0])
        return f'<img src="data:image/gif;base64,{data}" width="{width}" alt="animation">'
