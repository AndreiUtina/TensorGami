"""The unrolled view: recurrence as a time × layer grid (step 11a; no torch needed)."""

from __future__ import annotations

import pytest
from ir_fixtures import convlstm, residual_cnn
from layout_checks import check_scene

from tensorgami.ir import LAYER, ROLE, STEP, UNROLLED, Graph, GraphBuilder
from tensorgami.layout import Direction, LayoutOptions
from tensorgami.layout.unrolled import UnrolledLayout
from tensorgami.transforms import pipeline_for
from tensorgami.transforms.unroll import UnrollRecurrence
from tensorgami.views import get_view
from tensorgami.vocabulary import EdgeKind
from tensorgami.vocabulary import OpCategory as C


def fused_lstm(steps: int = 5, layers: int = 2) -> Graph:
    """``nn.LSTM(8, 32, num_layers, batch_first=True)`` traced as one call, then a head on
    the last layer's final hidden state (``h_n[-1]``)."""
    b = GraphBuilder("classifier", root_type="LSTMClassifier")
    x = b.input((1, steps, 8))
    params = {"hidden_size": 32, "num_layers": layers, "batch_first": True}
    final = (layers, 1, 32)
    lstm = b.node(
        "LSTM",
        category=C.RECURRENT,
        module="lstm",
        params=params,
        outputs=[(1, steps, 32), final, final],
    )
    b.edge(x, lstm)
    head = b.node("Linear", category=C.LINEAR, module="head", outputs=[(1, 4)])
    b.edge(lstm, head, source_port=1)
    b.output(head)
    return b.build()


def unrolled(graph: Graph, max_steps: int = 4) -> Graph:
    return UnrollRecurrence(max_steps).apply(pipeline_for(2, fold_time=False).apply(graph))


def grid_nodes(graph: Graph, role: str = "cell"):
    return [n for n in graph.nodes if n.params.get(ROLE) == role]


# --------------------------------------------------------------------------- transform


def test_a_traced_loop_becomes_one_grid_with_per_step_inputs():
    g = unrolled(convlstm(steps=3))
    (grid,) = [x for x in g.groups if x.type_name == UNROLLED]
    cells = grid_nodes(g)
    assert sorted(n.params[STEP] for n in cells) == [0, 1, 2]
    assert {n.group for n in cells} == {grid.id}
    inputs = grid_nodes(g, "in")
    assert [n.label for n in inputs] == ["x_{1}", "x_{2}", "x_{3}"]
    assert inputs[0].output_shapes == ((1, 1, 16, 16),)  # the time axis dropped
    states = [e for e in g.edges if e.kind is EdgeKind.STATE]
    assert {(g.node(e.source).params[STEP], g.node(e.target).params[STEP]) for e in states} == {
        (0, 1),
        (1, 2),
    }


def test_a_fused_lstm_becomes_layers_times_steps_cells():
    g = unrolled(fused_lstm(steps=3, layers=2))
    cells = grid_nodes(g)
    assert len(cells) == 6
    assert {(n.params[LAYER], n.params[STEP]) for n in cells} == {
        (k, t) for k in range(2) for t in range(3)
    }
    assert {n.op for n in cells} == {"LSTMCell"} and cells[0].params["hidden_size"] == 32
    states = [e for e in g.edges if e.kind is EdgeKind.STATE]
    assert len(states) == 2 * 2 * 2  # h and c, 2 layers, 2 step transitions
    (head,) = [n for n in g.nodes if n.op == "Linear"]
    (into_head,) = g.in_edges(head.id)
    assert g.node(into_head.source).params[LAYER] == 1  # h_n[-1]: the top layer…
    assert g.node(into_head.source).params[STEP] == 2  # …at the last step


def test_long_sequences_keep_the_first_steps_and_the_last():
    g = unrolled(fused_lstm(steps=9), max_steps=4)
    assert sorted({n.params[STEP] for n in grid_nodes(g)}) == [0, 1, 2, 8]
    assert [n.label for n in grid_nodes(g, "in")] == ["x_{1}", "x_{2}", "x_{3}", "x_{9}"]
    bridges = [
        e for e in g.edges if e.kind is EdgeKind.STATE and g.node(e.target).params[STEP] == 8
    ]
    assert bridges and all(g.node(e.source).params[STEP] == 2 for e in bridges)


def test_graphs_without_recurrence_are_left_alone():
    g = pipeline_for(2).apply(residual_cnn())
    assert UnrollRecurrence().apply(g) == g


def test_max_steps_is_checked():
    with pytest.raises(ValueError, match="max_steps"):
        UnrollRecurrence(1)


# --------------------------------------------------------------------------- layout


def by_id(scene):
    return {n.id: n for n in scene.nodes}


@pytest.mark.parametrize("direction", list(Direction))
@pytest.mark.parametrize("make", [lambda: convlstm(steps=3), lambda: fused_lstm(steps=6)])
def test_grid_layout_keeps_the_invariants(theme, make, direction):
    g = unrolled(make())
    scene = UnrolledLayout(LayoutOptions(direction=direction, wrap_width=None)).layout(g, theme)
    check_scene(scene, g, theme)


def test_time_runs_along_rows_and_layers_stack_upwards(theme):
    g = unrolled(fused_lstm(steps=3))
    nodes = by_id(UnrolledLayout(LayoutOptions(wrap_width=None)).layout(g, theme))
    cells = {(n.params[LAYER], n.params[STEP]): nodes[n.id].box for n in grid_nodes(g)}
    for t in range(3):
        assert cells[0, t].cx == cells[1, t].cx  # a column per step
        assert cells[1, t].bottom < cells[0, t].y  # the first layer at the bottom
    for k in range(2):
        assert cells[k, 0].cy == cells[k, 1].cy == cells[k, 2].cy  # a row per layer
        assert cells[k, 0].right < cells[k, 1].x < cells[k, 2].x
    pills = {n.params[STEP]: nodes[n.id].box for n in grid_nodes(g, "in")}
    assert all(pills[t].cx == cells[0, t].cx and pills[t].y > cells[0, t].bottom for t in pills)


def test_top_to_bottom_turns_the_grid(theme):
    g = unrolled(fused_lstm(steps=3))
    options = LayoutOptions(direction="TB")
    nodes = by_id(UnrolledLayout(options).layout(g, theme))
    cells = {(n.params[LAYER], n.params[STEP]): nodes[n.id].box for n in grid_nodes(g)}
    assert cells[0, 0].cy == cells[1, 0].cy and cells[0, 0].right < cells[1, 0].x  # layers →
    assert cells[0, 0].bottom < cells[0, 1].y  # time ↓


def test_hidden_steps_are_marked(theme):
    g = unrolled(fused_lstm(steps=9))
    scene = UnrolledLayout(LayoutOptions(wrap_width=None)).layout(g, theme)
    assert [n.text for n in scene.notes].count("⋯") == 3  # two layers and the inputs
    short = UnrolledLayout(LayoutOptions(wrap_width=None)).layout(unrolled(fused_lstm(3)), theme)
    assert not any(n.text == "⋯" for n in short.notes)


# --------------------------------------------------------------------------- views


def test_auto_unrolls_traced_loops_only():
    auto = get_view("auto")
    assert auto.choose(convlstm(steps=3)) is get_view("unrolled")
    assert auto.choose(fused_lstm()) is get_view("block")  # one call: the block view
    assert auto.choose(residual_cnn()) is get_view("block")


def test_unrolled_view_draws_a_grid(theme):
    shown, scene = get_view("unrolled").draw(
        fused_lstm(steps=3), theme, level=2, options=LayoutOptions(wrap_width=None)
    )
    assert any(g.type_name == UNROLLED for g in shown.groups)
    assert not [g for g in scene.groups if "unrolled" in g.id]  # a grid needs no frame
