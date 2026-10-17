"""``fig.animate()``: the figure lighting up as the model runs, and the data travelling between
its steps (roadmap steps 14a, 14b)."""

from __future__ import annotations

import io
import shutil

import pytest
from ir_fixtures import convlstm, transformer

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


def test_steps_build_up_and_end_with_the_still_figure(theme):
    graph, n = branches()
    fig = figure(graph, theme)
    anim = fig.animate(step=0.5, hold=2.0, flow=False)  # one frame per step, as in 14a
    steps = anim.steps
    assert len(anim) == len(steps) == 7 and anim.durations == [0.5] * 6 + [2.0]
    assert [f.step for f in anim] == list(range(7)) and not any(f.pulses for f in anim)
    assert steps[0].active == {n["x"]} and not steps[0].done
    assert steps[3].done == {n["x"], n["a"], n["b1"], n["c"]}
    assert steps[3].of(n["add"]) is Emphasis.PENDING and steps[3].of(n["b2"]) is Emphasis.ACTIVE
    still = render_svg(fig.scene, fig.style)
    assert anim.to_svg(len(steps) - 1) == still  # the last step is the still figure
    assert anim.to_svg(2) != still


def carried(anim, frame):
    """(source, target) of the wires a frame's pulses travel along."""
    edges = anim.figure.scene.edges
    return {(edges[e].source, edges[e].target) for e in frame.pulses}


def test_data_travels_along_the_wires_into_the_next_step(theme):
    graph, n = branches()
    anim = figure(graph, theme).animate(step=0.5, hold=2.0, travel=0.4, fps=10)
    assert len(anim) == 7 + 5 * 4  # four frames of travel before each step of the model's
    # but the first
    assert anim.durations[:6] == [0.5, 0.1, 0.1, 0.1, 0.1, 0.5]
    travel = [f for f in anim if f.pulses]
    assert [f.step for f in travel] == [k for k in range(5) for _ in range(4)]
    assert carried(anim, travel[0]) == {(n["x"], n["a"])}
    assert carried(anim, travel[4]) == {(n["a"], n["b1"]), (n["a"], n["c"])}  # both branches
    into_add = carried(anim, travel[12])  # every input of the merge, the skip from c too
    assert into_add == {(n["b2"], n["add"]), (n["c"], n["add"]), (n["buf"], n["add"])}
    alongs = [f.pulses[next(iter(f.pulses))] for f in travel[:4]]
    assert alongs == sorted(alongs) and 0 < alongs[0] and alongs[-1] < 1
    edges = anim.figure.scene.edges
    assert not any(edges[e].kind is EdgeKind.TRAINING for f in travel for e in f.pulses)


def test_a_folded_recurrence_sends_its_state_round_the_loops(theme):
    graph = simplify(convlstm(steps=3), 2)
    anim = figure(graph, theme).animate()
    edges = anim.figure.scene.edges
    loops = {i for i, e in enumerate(edges) if e.delay}
    assert len(loops) == 2
    passes = {}  # the badge each loop pulse leads to → the loops it travels
    for f in anim:
        if loops & set(f.pulses):
            passes.setdefault(anim.steps[f.step + 1].badges["cell"], set()).update(f.pulses)
    assert set(passes) == {"2/3", "3/3"}  # not into the first pass: no state yet
    assert all(loops <= p for p in passes.values())
    assert all(len(p - loops) >= 1 for p in passes.values())  # with the next input, from outside
    assert all(s == 0 for f in anim for s in f.starts.values())


def test_a_stack_reads_its_previous_pass_where_its_input_enters_its_frame(theme):
    graph = simplify(transformer(layers=2), 2)
    anim = figure(graph, theme).animate()
    (frame,) = anim.figure.scene.groups
    starts = [
        (anim.steps[f.step + 1].badges.get(frame.id), s)
        for f in anim
        for s in f.starts.values()
        if s > 0
    ]
    assert starts and {badge for badge, _ in starts} == {"2/2"}  # only from the second pass
    for f in anim:
        for e, along in f.pulses.items():
            assert f.starts.get(e, 0) < along < 1


def test_a_pulse_is_drawn_on_its_wire_and_nowhere_else(theme):
    from PIL import Image, ImageChops

    graph, n = branches()
    anim = figure(graph, theme).animate()
    index = next(i for i, f in enumerate(anim) if f.pulses)
    frame = anim[index]
    (edge,) = frame.pulses
    plain = Image.open(io.BytesIO(anim.to_png(index - 1)))  # the step before: no pulse
    pulsed = Image.open(io.BytesIO(anim.to_png(index)))
    box = ImageChops.difference(plain.convert("RGB"), pulsed.convert("RGB")).getbbox()
    assert box is not None
    path = anim._path(edge)
    scale = plain.width / anim.figure.scene.width
    margin = 10 * scale
    assert min(p.x for p in path) * scale - margin <= box[0]
    assert box[2] <= max(p.x for p in path) * scale + margin
    assert min(p.y for p in path) * scale - margin <= box[1]
    assert box[3] <= max(p.y for p in path) * scale + margin


def test_a_wire_shows_its_shape_once_its_data_has_arrived(theme):
    graph = simplify(transformer(layers=2), 2)
    scene = layout(graph, theme)
    labelled = [e for e in scene.edges if e.label and isinstance(e.target, str)]
    assert labelled
    nothing = FrameState(frozenset(), frozenset())
    everything = FrameState(frozenset(), frozenset(n.id for n in scene.nodes))
    for e in labelled:
        assert e.label not in render_svg(scene, theme, state=nothing)
    assert all(e.label in render_svg(scene, theme, state=everything) for e in labelled)


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
    assert durations == [round(d * 1000) for d in anim.durations]
    assert durations[-1] == 1200 and durations.count(400) == len(anim.steps) - 1
    for fmt in ("png", "svg", "pdf"):
        paths = anim.save_frames(tmp_path / fmt, format=fmt, dpi=40)
        assert [p.name for p in paths][:2] == [f"step-1.{fmt}", f"step-2.{fmt}"]
        assert len(paths) == len(anim.steps) and all(p.stat().st_size > 0 for p in paths)
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
    with pytest.raises(ValueError, match="greater than 0"):
        fig.animate(fps=0)
    with pytest.raises(ValueError, match="at most 50"):
        fig.animate(fps=60)
    with pytest.raises(ValueError, match="draw it with tg.draw"):
        Animation(Figure(fig.scene, fig.style))
    with pytest.raises(ValueError, match="format"):
        fig.animate().save_frames(tmp_path, format="jpg")
