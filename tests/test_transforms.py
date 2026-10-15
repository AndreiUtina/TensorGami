"""Graph transforms on small hand-built graphs (no torch needed)."""

from __future__ import annotations

import dataclasses

import pytest
from ir_fixtures import ALL, convlstm, residual_cnn, transformer, vae

from tensorgami.ir import Edge, Graph, GraphBuilder, RepeatKind, invocation_id, to_json
from tensorgami.transforms import (
    CollapseCells,
    CollapseGroups,
    CollapseToSystem,
    DetectLayerRepeats,
    DetectRecurrence,
    FoldIntoPrevious,
    FoldRepeats,
    Level,
    Pipeline,
    PruneEmptyGroups,
    RemoveConstants,
    RemoveLayoutOps,
    pipeline_for,
    simplify,
)
from tensorgami.transforms.base import normalize_edges
from tensorgami.vocabulary import EdgeKind
from tensorgami.vocabulary import OpCategory as C


def no_repeats(graph: Graph) -> Graph:
    return dataclasses.replace(graph, repeats=())


def ops(graph: Graph) -> list[str]:
    return [n.op for n in graph.nodes]


def by_op(graph: Graph, op: str):
    return [n for n in graph.nodes if n.op == op]


# --------------------------------------------------------------------------- composition


def test_pipeline_applies_in_order_and_nests():
    calls = []

    class Mark:
        def __init__(self, tag):
            self.name = tag

        def apply(self, graph):
            calls.append(self.name)
            return graph

    inner = Pipeline([Mark("b"), Mark("c")])
    Pipeline([Mark("a"), inner, Mark("d")]).apply(residual_cnn())
    assert calls == ["a", "b", "c", "d"]
    assert "Pipeline([a, pipeline, d])" == repr(Pipeline([Mark("a"), inner, Mark("d")]))


def test_normalize_edges_dedupes_and_renumbers_ports():
    edges = [Edge("a", "c", target_port=4), Edge("b", "c", target_port=7), Edge("a", "c")]
    out = normalize_edges(edges)
    assert [(e.source, e.target_port) for e in out] == [("a", 0), ("b", 1)]


def test_transforms_never_modify_their_input():
    graph = convlstm()
    before = to_json(graph)
    simplify(graph, Level.LAYERS)
    assert to_json(graph) == before


# --------------------------------------------------------------------------- recurrence


def test_state_edges_and_time_repeat_are_detected():
    g = DetectRecurrence().apply(no_repeats(convlstm(steps=3)))
    (rep,) = g.repeats
    assert rep.kind is RepeatKind.TIME and rep.groups == ("cell", "cell#1", "cell#2")
    state = [e for e in g.edges if e.kind is EdgeKind.STATE]
    assert state and all(g.node(e.source).group != g.node(e.target).group for e in state)
    assert len(state) == 2 * 2  # h and c, between 3 steps


def test_module_called_twice_without_state_is_not_recurrent():
    b = GraphBuilder()
    x, y = b.input((1, 4)), b.input((1, 4))
    outs = []
    for call, source in enumerate((x, y)):  # a shared encoder applied to two inputs (Siamese)
        g = b.group(invocation_id("encoder", call), "Encoder", module="encoder", call_index=call)
        n = b.node("Linear", category=C.LINEAR, group=g, call_index=call, outputs=[(1, 4)])
        b.edge(source, n)
        outs.append(n)
    merge = b.node("sub", category=C.MERGE, outputs=[(1, 4)])
    b.edge(outs[0], merge)
    b.edge(outs[1], merge)
    b.output(merge)
    g = DetectRecurrence().apply(b.build())
    assert g.repeats == () and all(e.kind is EdgeKind.DATA for e in g.edges)


# --------------------------------------------------------------------------- layer repeats


def test_identical_chained_layers_are_detected():
    g = DetectLayerRepeats().apply(no_repeats(transformer(layers=3)))
    (rep,) = g.repeats
    assert rep.kind is RepeatKind.LAYERS
    assert rep.groups == ("layers.0", "layers.1", "layers.2")


def _branches(chained: bool, same_shape: bool = True) -> Graph:
    b = GraphBuilder()
    x = b.input((1, 4))
    prev = x
    for k in range(2):
        g = b.group(f"block{k}", "Block")
        width = 4 if same_shape or k == 0 else 8
        n = b.node("Linear", category=C.LINEAR, group=g, outputs=[(1, width)])
        b.edge(prev if chained else x, n)
        prev = n
    b.output(prev)
    return b.build()


@pytest.mark.parametrize(("chained", "same_shape"), [(False, True), (True, False)])
def test_parallel_or_different_layers_are_not_a_stack(chained, same_shape):
    assert DetectLayerRepeats().apply(_branches(chained, same_shape)).repeats == ()


# --------------------------------------------------------------------------- cleanup


def test_layout_chains_are_bypassed_with_source_shapes():
    b = GraphBuilder()
    x = b.input((1, 8, 4))
    view = b.node("view", category=C.RESHAPE, outputs=[(1, 32)])
    cont = b.node("contiguous", category=C.RESHAPE, outputs=[(1, 32)])
    fc = b.node("Linear", category=C.LINEAR, outputs=[(1, 2)])
    b.chain([x, view, cont, fc])
    b.output(fc)
    g = RemoveLayoutOps().apply(b.build())
    assert ops(g) == ["input", "Linear", "output"]
    (edge,) = g.in_edges(by_op(g, "Linear")[0].id)
    assert edge.source == "n0" and edge.shape == (1, 8, 4)


def test_multi_output_layout_ops_and_buffer_sources():
    b = GraphBuilder()
    x = b.input((1, 8))
    split = b.node("split", category=C.RESHAPE, outputs=[(1, 4), (1, 4)])
    pe = b.node("getitem", category=C.RESHAPE, params={"weights": ("pe",)}, outputs=[(1, 4)])
    add = b.node("add", category=C.MERGE, outputs=[(1, 4)])
    b.edge(x, split)
    b.edge(split, add)
    b.edge(pe, add)
    b.output(add)
    g = RemoveLayoutOps().apply(b.build())
    assert "split" in ops(g)  # several outputs: structural, kept
    (buffer,) = by_op(g, "getitem")
    assert buffer.category is C.TENSOR and buffer.display_label == "pe"


@pytest.mark.parametrize(
    ("after", "kept"),
    [
        ((1, 64, 392, 392), True),  # U-Net's "copy and crop": only height and width shrink
        ((1, 64, 568), False),  # an index (x[..., 0]): fewer dimensions
        ((1, 32, 568, 568), False),  # a slice of the channels
        ((1, 64, 568, 568), False),  # everything: nothing changes
    ],
)
def test_a_crop_stays_other_indexing_is_bypassed(after, kept):
    b = GraphBuilder()
    x = b.input((1, 1, 572, 572))
    conv = b.node("Conv2d", category=C.CONV, outputs=[(1, 64, 568, 568)])
    item = b.node("getitem", category=C.RESHAPE, outputs=[after])
    b.chain([x, conv, item])
    b.output(item)
    g = RemoveLayoutOps().apply(b.build())
    assert ("getitem" in ops(g)) is kept
    if kept:
        (crop,) = by_op(g, "getitem")
        assert crop.label == "Crop" and crop.params["caption"] == "→ 392×392"


def test_activation_and_dropout_fold_into_the_layer():
    b = GraphBuilder()
    x = b.input((1, 4))
    fc = b.node("Linear", category=C.LINEAR, outputs=[(1, 8)])
    relu = b.node("ReLU", category=C.ACTIVATION, outputs=[(1, 8)])
    drop = b.node("Dropout", category=C.REGULARIZATION, outputs=[(1, 8)])
    head = b.node("Linear", category=C.LINEAR, outputs=[(1, 2)])
    b.chain([x, fc, relu, drop, head])
    b.output(head)
    g = FoldIntoPrevious().apply(b.build())
    assert ops(g) == ["input", "Linear", "Linear", "output"]
    assert g.nodes[1].params["folded"] == ("ReLU", "Dropout")
    assert g.predecessors(g.nodes[2].id) == (g.nodes[1],)


def test_folding_respects_branches_groups_and_inputs():
    g = FoldIntoPrevious().apply(vae())
    logvar = next(n for n in g.nodes if n.op == "Linear" and n.id == "n4")
    assert "folded" not in logvar.params  # logvar also feeds an output: not folded into it
    g = FoldIntoPrevious().apply(residual_cnn())
    bn = by_op(g, "BatchNorm2d")[0]
    assert bn.params["folded"] == ("relu",)
    (add,) = by_op(g, "add")
    assert bn in g.predecessors(add.id)  # the skip path now starts at the folded node

    b = GraphBuilder()
    x = b.input((1, 4))
    relu = b.node("relu", category=C.ACTIVATION, outputs=[(1, 4)])
    b.chain([x, relu])
    b.output(relu)
    assert "relu" in ops(FoldIntoPrevious().apply(b.build()))  # never into an input


def test_constants_removed_but_noise_and_buffers_kept():
    g = RemoveConstants().apply(convlstm())
    assert not by_op(g, "new_zeros") and not by_op(g, "zeros_like")
    assert by_op(RemoveConstants().apply(vae()), "randn_like")


def test_prune_empty_groups():
    b = GraphBuilder()
    b.group("empty", "Block")
    b.group("outer", "Block")
    b.group("outer.inner", "Block", parent="outer")
    x = b.input((1,))
    n = b.node("relu", group="outer.inner", outputs=[(1,)])
    b.edge(x, n)
    g = PruneEmptyGroups().apply(b.build())
    assert [grp.id for grp in g.groups] == ["", "outer", "outer.inner"]


# --------------------------------------------------------------------------- collapse


def test_recurrent_cells_collapse_to_one_node_per_step():
    g = CollapseCells().apply(DetectRecurrence().apply(no_repeats(convlstm(steps=3))))
    cells = by_op(g, "ConvLSTMCell")
    assert [c.group for c in cells] == ["cell", "cell#1", "cell#2"]
    assert all(c.category is C.RECURRENT for c in cells)
    assert cells[0].output_shapes == ((1, 8, 16, 16), (1, 8, 16, 16))  # h and c leave the cell
    assert cells[0].params["collapsed"] == 12
    links = [e for e in g.edges if e.source == cells[0].id and e.target == cells[1].id]
    assert {e.kind for e in links} == {EdgeKind.STATE} and len(links) == 2


def test_collapse_groups_at_depth_one():
    g = CollapseGroups(depth=1).apply(transformer(layers=2))
    assert ops(g) == ["input", "TransformerEncoderLayer", "TransformerEncoderLayer", "output"]
    with pytest.raises(ValueError):
        CollapseGroups(depth=0)


def test_collapse_to_system_keeps_inputs_and_outputs():
    g = CollapseToSystem().apply(vae())
    assert ops(g) == ["input", "VAE", "output", "output", "output"]
    model = by_op(g, "VAE")[0]
    assert len(model.output_shapes) == 3 and [grp.id for grp in g.groups] == [""]


# --------------------------------------------------------------------------- folding repeats


def test_layer_stack_folds_into_one_counted_copy():
    g = FoldRepeats([RepeatKind.LAYERS]).apply(transformer(layers=3))
    assert [grp.id for grp in g.groups] == ["", "layers.0"]
    assert g.group("layers.0").count == 3 and g.repeats == ()
    out = g.outputs[0]
    assert g.predecessors(out.id)[0].group == "layers.0"  # re-attached to the kept copy


def test_time_steps_fold_into_a_state_loop():
    g = simplify(convlstm(steps=4), Level.LAYERS)
    assert ops(g) == ["input", "ConvLSTMCell", "Conv2d", "output"]
    cell = by_op(g, "ConvLSTMCell")[0]
    assert g.group(cell.group).count == 4
    loops = [e for e in g.out_edges(cell.id) if e.target == cell.id]
    assert len(loops) == 2 and all(e.kind is EdgeKind.STATE for e in loops)


def test_unrolled_time_keeps_every_step():
    g = simplify(convlstm(steps=4), Level.LAYERS, fold_time=False)
    assert len(by_op(g, "ConvLSTMCell")) == 4


def test_leaf_cells_fold_too():
    b = GraphBuilder()
    x = b.input((1, 3, 4))
    prev = None
    for t in range(3):
        cell = b.node(
            "GRUCell", category=C.RECURRENT, module="cell", call_index=t, outputs=[(1, 4)]
        )
        b.edge(x, cell)
        if prev:
            b.edge(prev, cell)
        prev = cell
    b.output(prev)
    g = FoldRepeats().apply(DetectRecurrence().apply(b.build()))
    (cell,) = by_op(g, "GRUCell")
    assert cell.count == 3
    assert any(e.target == cell.id and e.kind is EdgeKind.STATE for e in g.out_edges(cell.id))


def test_copies_with_different_structure_are_not_folded():
    g = transformer(layers=2)
    extra = dataclasses.replace(g.nodes[-2], id="extra", group="layers.1")
    changed = dataclasses.replace(g, nodes=(*g.nodes, extra))
    assert FoldRepeats().apply(changed).group("layers.1")  # left alone


# --------------------------------------------------------------------------- levels


@pytest.mark.parametrize("make", ALL.values(), ids=ALL.keys())
def test_every_level_gives_a_valid_smaller_graph(make):
    graph = make()
    sizes = [len(simplify(graph, level).nodes) for level in (3, 2, 1, 0)]
    assert sizes == sorted(sizes, reverse=True)
    assert all(len(simplify(graph, lv).inputs) == len(graph.inputs) for lv in range(4))
    assert all(len(simplify(graph, lv).outputs) == len(graph.outputs) for lv in range(4))


def test_level_presets_are_deterministic_and_inspectable():
    assert to_json(simplify(convlstm(), 2)) == to_json(simplify(convlstm(), 2))
    assert "fold-repeats" in repr(pipeline_for(Level.LAYERS))
    assert repr(pipeline_for(Level.SYSTEM)) == "Pipeline([collapse-to-system])"
    assert "fold-repeats" not in repr(pipeline_for(2, fold_layers=False, fold_time=False))


def test_collapsed_blocks_take_the_colour_of_their_contents():
    from tensorgami.transforms.collapse import content_category

    b = GraphBuilder()
    x = b.input((1, 3, 8, 8))
    feats = b.group("features", "Sequential")
    head = b.group("head", "Sequential")
    mixed = b.group("mixed", "Block")
    prev = x
    for op, cat, grp in [
        ("Conv2d", C.CONV, feats),
        ("MaxPool2d", C.RESAMPLE, feats),
        ("Conv2d", C.CONV, feats),
        ("Linear", C.LINEAR, head),
        ("Conv2d", C.CONV, mixed),
        ("Linear", C.LINEAR, mixed),
    ]:
        n = b.node(op, category=cat, group=grp, outputs=[(1, 4)])
        b.edge(prev, n)
        prev = n
    b.output(prev)
    g = b.build()
    assert content_category(g, "features") is C.CONV  # 2 of 3 compute ops
    assert content_category(g, "head") is C.LINEAR
    assert content_category(g, "mixed") is C.MODULE  # no majority
    stages = CollapseGroups(depth=1).apply(g)
    assert [n.category for n in stages.nodes if n.op != "input" and n.op != "output"] == [
        C.CONV,
        C.LINEAR,
        C.MODULE,
    ]


@pytest.mark.parametrize(
    ("value", "level"),
    [(2, Level.LAYERS), ("2", Level.LAYERS), ("stages", Level.STAGES), ("System", Level.SYSTEM)],
)
def test_levels_by_number_or_name(value, level):
    assert Level.parse(value) is level


@pytest.mark.parametrize("value", [5, -1, 2.5, True, "deep", None])
def test_unknown_levels_are_explained(value):
    with pytest.raises(ValueError, match="level must be one of 0 system, 1 stages, 2 layers"):
        simplify(residual_cnn(), value)
