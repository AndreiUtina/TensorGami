"""Each IR rule catches its violation; the reference graphs pass all of them."""

from __future__ import annotations

import dataclasses

import pytest
from ir_fixtures import ALL, convlstm, residual_cnn, transformer

from tensorgami.ir import ROOT, Edge, Graph, GraphError, GraphValidator, Group, Node, Repeat
from tensorgami.ir.model import RepeatKind
from tensorgami.ir.validation import (
    AcyclicDataFlowRule,
    EdgeEndpointsRule,
    GroupTreeRule,
    InputOutputRule,
    RepeatRule,
    ShapeConsistencyRule,
    UniqueIdsRule,
)
from tensorgami.vocabulary import EdgeKind


def with_(graph: Graph, **changes) -> Graph:
    return dataclasses.replace(graph, **changes)


def messages(rule, graph: Graph) -> list[str]:
    return [i.message for i in rule.check(graph)]


@pytest.mark.parametrize("make", ALL.values(), ids=ALL.keys())
def test_reference_graphs_pass_every_rule(make):
    assert GraphValidator().validate(make()) == []


def test_unique_ids():
    g = residual_cnn()
    bad = with_(g, nodes=(*g.nodes, g.nodes[1]))
    assert any("node id 'n1' is used 2 times" in m for m in messages(UniqueIdsRule(), bad))


def test_group_tree_needs_one_root_and_known_parents():
    g = transformer()
    no_root = with_(g, groups=g.groups[1:])
    assert any("one root group" in m for m in messages(GroupTreeRule(), no_root))
    orphan = with_(g, groups=(*g.groups, Group("x", "Block", parent="missing")))
    assert any("unknown parent 'missing'" in m for m in messages(GroupTreeRule(), orphan))
    lost = with_(g, nodes=(*g.nodes, Node("zz", "relu", group="nowhere")))
    assert any("unknown group 'nowhere'" in m for m in messages(GroupTreeRule(), lost))


def test_group_tree_detects_parent_cycles():
    groups = (Group(ROOT, "M", parent=None), Group("a", "A", parent="b"), Group("b", "B", "a"))
    g = Graph(nodes=(), edges=(), groups=groups)
    assert any("cycle" in m for m in messages(GroupTreeRule(), g))


def test_edge_endpoints_and_ports():
    g = residual_cnn()
    dangling = with_(g, edges=(*g.edges, Edge("n1", "ghost")))
    assert any("unknown node 'ghost'" in m for m in messages(EdgeEndpointsRule(), dangling))
    bad_port = with_(g, edges=(*g.edges, Edge("n1", "n2", source_port=3)))
    assert any("source port 3" in m for m in messages(EdgeEndpointsRule(), bad_port))


def test_shape_consistency():
    g = residual_cnn()
    e = g.edges[1]
    wrong = dataclasses.replace(e, shape=(1, 99))
    bad = with_(g, edges=(wrong, *g.edges[2:]))
    assert any("differs from source output" in m for m in messages(ShapeConsistencyRule(), bad))


def test_data_cycles_are_rejected_but_state_loops_allowed():
    g = residual_cnn()
    loop_data = with_(g, edges=(*g.edges, Edge("n4", "n1")))
    assert messages(AcyclicDataFlowRule(), loop_data)
    loop_state = with_(g, edges=(*g.edges, Edge("n4", "n1", EdgeKind.STATE)))
    assert messages(AcyclicDataFlowRule(), loop_state) == []


def test_acyclic_check_handles_deep_graphs_without_recursion_errors():
    from tensorgami.ir import GraphBuilder

    b = GraphBuilder()
    ids = [b.input((1,))] + [b.node("relu", outputs=[(1,)]) for _ in range(5000)]
    b.chain(ids)
    assert len(b.build().nodes) == 5001


def test_inputs_receive_no_data():
    g = residual_cnn()
    x = g.inputs[0].id
    bad = with_(g, edges=(*g.edges, Edge("n2", x)))
    assert any(f"input '{x}'" in m for m in messages(InputOutputRule(), bad))


def test_repeats_need_two_known_sibling_groups():
    g = convlstm()
    single = with_(g, repeats=(Repeat("r", RepeatKind.TIME, ("cell",)),))
    assert any("needs ≥ 2" in m for m in messages(RepeatRule(), single))
    unknown = with_(g, repeats=(Repeat("r", RepeatKind.TIME, ("cell", "nope")),))
    assert any("unknown groups" in m for m in messages(RepeatRule(), unknown))
    twice = with_(g, repeats=(*g.repeats, Repeat("r2", RepeatKind.TIME, ("cell", "cell#1"))))
    assert any("belongs to 2 repeats" in m for m in messages(RepeatRule(), twice))


def test_validator_raises_with_all_issues_and_accepts_custom_rules():
    g = residual_cnn()
    bad = with_(g, edges=(*g.edges, Edge("n1", "ghost"), Edge("n4", "n1")))
    with pytest.raises(GraphError) as info:
        GraphValidator().ensure_valid(bad)
    assert "ghost" in str(info.value) and "cycle" in str(info.value)
    assert GraphValidator([]).validate(bad) == []
