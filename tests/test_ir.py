"""IR model, builder and queries."""

from __future__ import annotations

import pytest
from ir_fixtures import ALL, convlstm, residual_cnn, transformer, vae

from tensorgami.ir import (
    ROOT,
    Edge,
    Graph,
    GraphBuilder,
    GraphError,
    Node,
    RepeatKind,
    format_shape,
    invocation_id,
)
from tensorgami.vocabulary import EdgeKind, OpCategory


@pytest.mark.parametrize("make", ALL.values(), ids=ALL.keys())
def test_reference_graphs_are_valid(make):
    graph = make()
    assert isinstance(graph, Graph)
    assert graph.inputs and graph.outputs


def test_nodes_keep_execution_order_and_auto_ids():
    g = residual_cnn()
    assert [n.id for n in g.nodes[:3]] == ["n0", "n1", "n2"]
    assert [n.op for n in g.nodes][:4] == ["input", "Conv2d", "BatchNorm2d", "relu"]


def test_edges_get_shapes_and_ports_automatically():
    g = residual_cnn()
    add = next(n for n in g.nodes if n.op == "add")
    ins = g.in_edges(add.id)
    assert [e.target_port for e in ins] == [0, 1]
    assert all(e.shape == (1, 16, 32, 32) for e in ins)
    assert {p.op for p in g.predecessors(add.id)} == {"relu", "Conv2d"}


def test_graph_outputs_are_io_nodes_fed_by_their_source():
    g = vae()
    labels = [n.label for n in g.outputs]
    assert labels == ["reconstruction", "mu", "logvar"]
    assert all(n.category is OpCategory.IO for n in g.outputs)
    assert g.in_edges(g.outputs[1].id)[0].shape == (1, 16)


def test_stochastic_source_has_no_data_input():
    """ADR-0002: randn_like depends on μ only for its shape, so it gets no data edge."""
    g = vae()
    eps = next(n for n in g.nodes if n.op == "randn_like")
    assert g.in_edges(eps.id) == ()
    assert [e.kind for e in g.out_edges(eps.id)] == [EdgeKind.STOCHASTIC]


def test_group_queries_on_unrolled_recurrence():
    g = convlstm(steps=3)
    steps = g.children(ROOT)
    assert [s.id for s in steps] == ["cell", "cell#1", "cell#2"]
    assert [s.call_index for s in steps] == [0, 1, 2]
    assert all(s.module == "cell" and s.category is OpCategory.RECURRENT for s in steps)
    conv = next(n for n in g.members("cell#1") if n.op == "Conv2d")
    assert conv.module == "cell.conv" and conv.call_index == 1
    assert len(g.nodes_under("cell")) == len(g.members("cell")) == 12
    assert g.depth("cell#2") == 1 and g.depth(ROOT) == 0
    assert g.ancestors("cell#2") == (g.root,)
    (rep,) = g.repeats
    assert rep.kind is RepeatKind.TIME and rep.count == 3


def test_state_flows_from_one_step_to_the_next():
    g = convlstm(steps=2)
    first_out = [n for n in g.members("cell") if n.op == "mul"][-1]  # h of step 0
    consumers = {n.group for n in g.successors(first_out.id)}
    assert "cell#1" in consumers


def test_layers_repeat_and_descendants():
    g = transformer(layers=3)
    (rep,) = g.repeats
    assert rep.kind is RepeatKind.LAYERS and rep.groups == ("layers.0", "layers.1", "layers.2")
    assert [d.id for d in g.descendants(ROOT)] == ["layers.0", "layers.1", "layers.2"]
    assert len(g.nodes_under(ROOT)) == len(g.nodes)


def test_lookups_fail_clearly():
    g = residual_cnn()
    with pytest.raises(KeyError, match="unknown node 'nope'"):
        g.node("nope")
    with pytest.raises(KeyError, match="unknown group"):
        g.group("nope")


def test_graph_is_immutable_and_hashable():
    g = residual_cnn()
    with pytest.raises(AttributeError):
        g.name = "x"  # type: ignore[misc]
    with pytest.raises(TypeError):
        g.nodes[1].params["kernel_size"] = 5  # type: ignore[index]
    assert hash(g) == hash(residual_cnn())
    assert len({*g.nodes}) == len(g.nodes)


def test_categories_in_first_appearance_order():
    assert residual_cnn().categories()[:3] == [OpCategory.IO, OpCategory.CONV, OpCategory.NORM]


def test_builder_rejects_bad_references():
    b = GraphBuilder()
    with pytest.raises(GraphError, match="unknown parent"):
        b.group("a.b", "Block", parent="a")
    b.group("a", "Block")
    with pytest.raises(GraphError, match="duplicate group"):
        b.group("a", "Block")
    n = b.node("relu", group="a")
    with pytest.raises(GraphError, match="duplicate node"):
        b.node("relu", node_id=n)
    with pytest.raises(GraphError, match="unknown group"):
        b.node("relu", group="zzz")
    with pytest.raises(GraphError, match="unknown node"):
        b.edge(n, "zzz")


def test_builder_explicit_shape_and_port_override():
    b = GraphBuilder()
    x = b.input((1, 4))
    y = b.node("add", outputs=[(1, 4)])
    e = b.edge(x, y, target_port=1, shape=None)
    assert e.shape is None and e.target_port == 1
    assert b.edge(x, y).target_port == 2  # continues after the highest port used


def test_invocation_ids():
    assert invocation_id("cell", 0) == "cell"
    assert invocation_id("cell", 3) == "cell#3"


@pytest.mark.parametrize(
    ("shape", "drop_batch", "text"),
    [
        ((1, 64, 56, 56), True, "64×56×56"),
        ((1, 64, 56, 56), False, "1×64×56×56"),
        ((8, None, 512), True, "?×512"),
        ((10,), True, "10"),
        ((), True, "scalar"),
        (None, True, ""),
    ],
)
def test_format_shape(shape, drop_batch, text):
    assert format_shape(shape, drop_batch=drop_batch) == text


def test_display_labels_fall_back_to_op_and_type():
    assert Node("a", "Conv2d").display_label == "Conv2d"
    assert Node("a", "Conv2d", label="conv1").display_label == "conv1"
    assert convlstm().group("cell").display_label == "ConvLSTMCell"
    assert Edge("a", "b").kind is EdgeKind.DATA
