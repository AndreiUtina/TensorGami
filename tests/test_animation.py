"""``fig.animate()``: the figure lighting up as the model runs (roadmap step 14a)."""

from __future__ import annotations

import shutil

import pytest
from ir_fixtures import transformer

from tensorgami.api.animation import Animation, timeline
from tensorgami.api.figure import Figure
from tensorgami.ir import GraphBuilder
from tensorgami.layout import layout
from tensorgami.render import Emphasis, FrameState, render_svg
from tensorgami.transforms import simplify
from tensorgami.vocabulary import EdgeKind
from tensorgami.vocabulary import OpCategory as C


def branches():
    """input → a → (b1 → b2 | c) → add → output, a buffer into add, a training loss on it."""
    b = GraphBuilder("branches")
    x = b.input((1, 8))
    a = b.node("Linear", category=C.LINEAR, module="a", outputs=[(1, 8)])
    b1 = b.node("Linear", category=C.LINEAR, module="b1", outputs=[(1, 8)])
    b2 = b.node("Linear", category=C.LINEAR, module="b2", outputs=[(1, 8)])
    c = b.node("Linear", category=C.LINEAR, module="c", outputs=[(1, 8)])
    buf = b.node("constant", category=C.TENSOR, label="bias", outputs=[(1, 8)])
    add = b.node("add", category=C.MERGE, outputs=[(1, 8)])
    b.chain([x, a, b1, b2, add])
    b.edge(a, c)
    b.edge(c, add)
    b.edge(buf, add)
    out = b.output(add)
    b.edge(add, a, EdgeKind.TRAINING)  # flows back: not something `a` waits for
    return b.build(), dict(x=x, a=a, b1=b1, b2=b2, c=c, buf=buf, add=add, out=out)


def figure(graph, theme):
    return Figure(layout(graph, theme), theme, graph)


def test_steps_follow_the_data(theme):
    graph, n = branches()
    steps = [s.active for s in timeline(graph, layout(graph, theme))]
    assert steps == [
        {n["x"]},
        {n["a"]},
        {n["b1"], n["c"]},  # parallel branches light up together
        {n["b2"], n["buf"]},  # the buffer just before the merge that reads it
        {n["add"]},  # the merge waits for the longer branch
        {n["out"]},
    ]


def test_a_repeated_block_plays_its_passes(theme):
    graph = simplify(transformer(layers=3), 2)
    (block,) = [g for g in graph.groups if g.count > 1]
    assert block.count == 3
    steps = timeline(graph, layout(graph, theme))
    passes = [s.badges[block.id] for s in steps if block.id in s.badges]
    each = len(passes) // 3
    assert each > 1 and passes == [f"{k}/3" for k in (1, 2, 3) for _ in range(each)]


def test_frames_build_up_and_end_with_the_still_figure(theme):
    graph, n = branches()
    fig = figure(graph, theme)
    anim = fig.animate(step=0.5, hold=2.0)
    assert len(anim) == 7 and anim.durations == [0.5] * 6 + [2.0]
    assert anim[0].active == {n["x"]} and not anim[0].done
    assert anim[3].done == {n["x"], n["a"], n["b1"], n["c"]}
    assert anim[3].of(n["add"]) is Emphasis.PENDING and anim[3].of(n["b2"]) is Emphasis.ACTIVE
    still = render_svg(fig.scene, fig.style)
    assert anim.to_svg(len(anim) - 1) == still  # the last frame is the still figure
    assert anim.to_svg(2) != still


def test_legend_entries_never_fade():
    state = FrameState(frozenset({"n1"}), frozenset())
    assert state.of("legend:0:conv") is Emphasis.DONE
    assert state.of("n7") is Emphasis.PENDING


def test_an_animation_is_saved_as_a_gif_and_as_frames(theme, tmp_path):
    from PIL import Image

    graph, _ = branches()
    anim = figure(graph, theme).animate(step=0.4, hold=1.2)
    gif = anim.save(tmp_path / "model.gif", dpi=40)
    with Image.open(gif) as image:
        assert image.n_frames == len(anim)
        durations = []
        for i in range(image.n_frames):
            image.seek(i)
            durations.append(image.info["duration"])
    assert durations == [400] * (len(anim) - 1) + [1200]
    for fmt in ("png", "svg", "pdf"):
        paths = anim.save_frames(tmp_path / fmt, format=fmt, dpi=40)
        assert [p.name for p in paths][:2] == [f"step-1.{fmt}", f"step-2.{fmt}"]
        assert len(paths) == len(anim) and all(p.stat().st_size > 0 for p in paths)
    assert anim._repr_html_().startswith('<img src="data:image/gif;base64,')


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_an_animation_is_saved_as_an_mp4(theme, tmp_path):
    graph, _ = branches()
    path = figure(graph, theme).animate().save(tmp_path / "model.mp4", dpi=40)
    assert path.stat().st_size > 0


def test_animation_errors(theme, tmp_path):
    graph, _ = branches()
    fig = figure(graph, theme)
    with pytest.raises(ValueError, match="saved as .gif or .mp4"):
        fig.animate().save(tmp_path / "model.avi")
    with pytest.raises(ValueError, match="greater than 0"):
        fig.animate(step=0)
    with pytest.raises(ValueError, match="draw it with tg.draw"):
        Animation(Figure(fig.scene, fig.style))
    with pytest.raises(ValueError, match="format"):
        fig.animate().save_frames(tmp_path, format="jpg")
