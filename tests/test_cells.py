"""The cell view: recurrent cells drawn as their gates (step 11d; no torch needed)."""

from __future__ import annotations

import pytest
from ir_fixtures import convlstm, residual_cnn
from layout_checks import check_scene

from tensorgami.ir import CAPTION, CELL_ROLE, LSTM_CELL, GraphBuilder
from tensorgami.layout import LayoutOptions, layout
from tensorgami.transforms.cells import cell_figure
from tensorgami.views import get_view
from tensorgami.vocabulary import EdgeKind
from tensorgami.vocabulary import OpCategory as C

ROLES = {"c_prev", "h_prev", "x", "concat", *"figo", "mul_f", "mul_ig", "add_c", "tanh_c"}
ROLES |= {"mul_o", "c", "h"}


def fused(op: str, layers: int = 2):
    """``nn.LSTM`` / ``nn.GRU`` (8 → 32) over 5 steps, traced as one call, then a head."""
    b = GraphBuilder("seq", root_type="SeqModel")
    x = b.input((1, 5, 8))
    params = {"input_size": 8, "hidden_size": 32, "num_layers": layers, "batch_first": True}
    rnn = b.node(op, category=C.RECURRENT, module="rnn", params=params, outputs=[(1, 5, 32)])
    b.edge(x, rnn)
    head = b.node("Linear", category=C.LINEAR, module="head", outputs=[(1, 5, 4)])
    b.edge(rnn, head)
    b.output(head)
    return b.build()


def roles(graph):
    return {n.params[CELL_ROLE]: n for n in graph.nodes if CELL_ROLE in n.params}


def test_a_fused_lstm_becomes_the_classic_cell():
    g = cell_figure(fused("LSTM"))
    (cell,) = g.groups[1:]
    assert cell.type_name == LSTM_CELL and cell.label == "LSTM cell · 8 → 32"
    parts = roles(g)
    assert set(parts) == ROLES
    assert {parts[r].category for r in "figo"} == {C.LINEAR}
    assert [parts[r].params[CAPTION] for r in "figo"] == ["forget", "input", "candidate", "output"]
    assert parts["x"].params[CAPTION] == "8" and parts["h"].params[CAPTION] == "32"
    state = {(e.source, e.target) for e in g.edges if e.kind is EdgeKind.STATE}
    assert (parts["c_prev"].id, parts["mul_f"].id) in state  # the cell state runs on top


def test_a_traced_lstm_cell_is_recognised_from_its_data_flow():
    g = cell_figure(convlstm(steps=3))
    (cell,) = g.groups[1:]  # three calls of one cell type: drawn once
    assert cell.type_name == LSTM_CELL and cell.label == "ConvLSTMCell · gates: Conv2d"
    parts = roles(g)
    assert {parts[r].category for r in "figo"} == {C.CONV}  # the gates are a convolution
    assert parts["x"].params[CAPTION] == "1×16×16" and parts["c"].params[CAPTION] == "8×16×16"


def test_gru_and_rnn_cells_get_their_textbook_gates():
    g = cell_figure(fused("GRU"))
    captions = {n.params.get(CAPTION) for n in g.nodes}
    assert {"reset", "update", "candidate"} <= captions
    assert g.groups[1].label == "GRU cell · 8 → 32" and g.groups[1].type_name == "cell"
    assert "W [h, x] + b" in {n.params.get(CAPTION) for n in cell_figure(fused("RNN")).nodes}


def test_any_other_cell_shows_its_operations():
    b = GraphBuilder("m", root_type="M")
    x = b.input((1, 8))
    h = b.node("zeros", category=C.TENSOR, outputs=[(1, 8)])
    cell = b.group("cell", "MyCell", module="cell", category=C.RECURRENT)
    add = b.node("add", category=C.MERGE, group=cell, outputs=[(1, 8)])
    b.edge(x, add)
    b.edge(h, add)
    act = b.node("tanh", category=C.ACTIVATION, group=cell, outputs=[(1, 8)])
    b.edge(add, act)
    b.output(act)
    g = cell_figure(b.build())
    assert g.groups[1].label == "MyCell"
    assert {n.label for n in g.nodes if n.category is C.TENSOR} == {"in 1", "in 2", "out 1"}
    assert {n.op for n in g.nodes} >= {"add", "tanh"}


def test_models_without_cells_are_refused(theme):
    assert cell_figure(residual_cnn()) is None
    with pytest.raises(ValueError, match="no recurrent cell to draw"):
        get_view("cell").draw(residual_cnn(), theme, level=2, options=LayoutOptions())


@pytest.mark.parametrize("graph", [fused("LSTM"), fused("GRU"), convlstm(steps=2)])
def test_cell_figures_keep_the_invariants(theme, graph):
    shown, scene = get_view("cell").draw(graph, theme, level=2, options=LayoutOptions())
    check_scene(scene, shown, theme)
    assert [g.title for g in scene.groups]  # each cell is framed and named


def test_the_lstm_figure_has_the_classic_rows(theme):
    g = cell_figure(fused("LSTM"))
    boxes = {n.id: n.box for n in layout(g, theme, LayoutOptions()).nodes}
    at = {role: boxes[node.id] for role, node in roles(g).items()}
    assert at["c_prev"].cy == at["mul_f"].cy == at["add_c"].cy == at["c"].cy  # state on top
    assert at["f"].cy == at["i"].cy == at["g"].cy == at["o"].cy > at["c"].cy  # gates below
    assert at["h_prev"].cy == at["concat"].cy == at["h"].cy > at["f"].cy  # hidden at the bottom
    assert at["f"].cx < at["i"].cx < at["g"].cx < at["o"].cx < at["tanh_c"].cx < at["h"].cx
    assert at["mul_f"].cx == at["f"].cx and at["x"].cx == at["concat"].cx
