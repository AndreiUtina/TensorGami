"""U-shaped encoder–decoders and activations that open a module (step 11b; no torch)."""

from __future__ import annotations

import dataclasses

import pytest
from ir_fixtures import ALL, residual_cnn, unet
from layout_checks import check_scene

from tensorgami.ir import U_LEVEL, U_SIDE, USHAPE, GraphBuilder
from tensorgami.layout import Direction, LayoutOptions, layout
from tensorgami.scene import Side
from tensorgami.transforms import FoldIntoPrevious, simplify
from tensorgami.transforms.ushape import FoldUShape
from tensorgami.views import get_view
from tensorgami.vocabulary import OpCategory as C

NO_WRAP = LayoutOptions(wrap_width=None)


def folded(levels: int = 3):
    return FoldUShape().apply(simplify(unet(levels), 2))


def place(graph):
    return {n.id: (n.params[U_LEVEL], n.params[U_SIDE]) for n in graph.nodes if U_LEVEL in n.params}


# --------------------------------------------------------------------------- folding


def test_an_activation_opening_a_module_joins_the_layer_before():
    b = GraphBuilder("m", root_type="M")
    x = b.input((1, 8))
    block = b.group("block", "Block", module="block")
    linear = b.node("Linear", category=C.LINEAR, module="fc", outputs=[(1, 8)])
    b.edge(x, linear)
    act = b.node("LeakyReLU", category=C.ACTIVATION, group=block, outputs=[(1, 8)])
    b.edge(linear, act)
    drop = b.node("Dropout", category=C.REGULARIZATION, group=block, outputs=[(1, 8)])
    b.edge(act, drop)
    b.output(drop)
    g = FoldIntoPrevious().apply(b.build())
    ops = {n.op: n for n in g.nodes}
    assert "LeakyReLU" not in ops and ops["Linear"].params["folded"] == ("LeakyReLU",)
    assert ops["Dropout"].group == "block"  # regularisation stays in its module


# --------------------------------------------------------------------------- the U


def test_nested_skips_become_levels_with_two_sides_and_a_bottleneck():
    g = folded(3)
    (u,) = [x for x in g.groups if x.type_name == USHAPE]
    where = place(g)
    assert {n for n in where} == {n.id for n in g.nodes if n.group == u.id}
    by_op = {(g.node(n).op, level, side) for n, (level, side) in where.items()}
    assert ("cat", 0, "dec") in by_op and ("cat", 2, "dec") in by_op
    assert {side for level, side in where.values() if level == 3} == {"bottom"}
    assert all(g.node(n).op != "input" for n in where)  # the ends stay outside the U
    head = next(n for n in g.nodes if n.op == "Conv2d" and n.module == "head")
    assert head.group != u.id


@pytest.mark.parametrize("make", [residual_cnn, ALL["transformer"], ALL["vae"]])
def test_models_without_nested_skips_are_untouched(make):
    g = simplify(make(), 2)
    assert FoldUShape().apply(g) == g


def test_two_nested_skips_are_not_enough():
    g = simplify(unet(2), 2)
    assert FoldUShape(min_levels=3).apply(g) == g
    assert FoldUShape(min_levels=2).apply(g) != g


@pytest.mark.parametrize("direction", list(Direction))
def test_u_layout_keeps_the_invariants(theme, direction):
    g = folded(4)
    check_scene(layout(g, theme, LayoutOptions(direction=direction, wrap_width=None)), g, theme)


def test_levels_are_rows_with_the_skip_across_the_middle(theme):
    g = folded(3)
    nodes = {n.id: n for n in layout(g, theme, NO_WRAP).nodes}
    where = place(g)
    for level in range(3):
        enc = [nodes[n].box for n, p in where.items() if p == (level, "enc")]
        dec = [nodes[n].box for n, p in where.items() if p == (level, "dec")]
        assert len({b.cy for b in enc + dec}) == 1  # one row per level
        assert max(b.right for b in enc) < min(b.x for b in dec)  # encoder left, decoder right
    rows = [nodes[next(n for n, p in where.items() if p == (k, "enc"))].box.cy for k in range(3)]
    assert rows == sorted(rows)  # deeper levels lower
    bottom = [nodes[n].box for n, p in where.items() if p[1] == "bottom"]
    assert min(b.y for b in bottom) > rows[-1]


def test_the_u_is_much_narrower_than_the_line(theme):
    g = simplify(unet(4), 2)
    line = layout(g, theme, NO_WRAP)
    u = layout(FoldUShape().apply(g), theme, NO_WRAP)
    assert u.width < 0.6 * line.width


def test_block_view_folds_unless_asked_not_to(theme):
    view = get_view("block")
    shown, _ = view.draw(unet(3), theme, level=2, options=NO_WRAP)
    assert any(g.type_name == USHAPE for g in shown.groups)
    flat = dataclasses.replace(NO_WRAP, u_shape=False)
    shown, _ = view.draw(unet(3), theme, level=2, options=flat)
    assert not any(g.type_name == USHAPE for g in shown.groups)


# --------------------------------------------------------------------------- merge notes


def test_merge_notes_sit_where_no_wire_is(theme):
    b = GraphBuilder("m", root_type="M")
    x = b.input((1, 8))
    a = b.node("Linear", category=C.LINEAR, module="a", outputs=[(1, 8)])
    c = b.node("Linear", category=C.LINEAR, module="c", outputs=[(1, 8)])
    b.edge(x, a)
    b.edge(x, c)
    add = b.node("add", category=C.MERGE, outputs=[(1, 8)])
    b.edge(a, add)
    b.edge(c, add)
    act = b.node("relu", category=C.ACTIVATION, outputs=[(1, 8)])
    b.edge(add, act)
    b.output(act)
    g = simplify(b.build(), 2)
    for direction in Direction:
        scene = layout(g, theme, LayoutOptions(direction=direction, wrap_width=None))
        (merge,) = [n for n in scene.nodes if n.category is C.MERGE and ":" not in n.id]
        used = {e.entry for e in scene.edges if e.target == merge.id}
        used |= {e.exit for e in scene.edges if e.source == merge.id}
        assert merge.detail == "ReLU" and merge.note_side not in used
        assert merge.note_side in (
            (Side.TOP, Side.BOTTOM) if direction == "LR" else (Side.RIGHT, Side.LEFT)
        )
