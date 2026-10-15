"""Layout engine and labels on hand-built graphs (no torch needed)."""

from __future__ import annotations

import dataclasses

import pytest
from ir_fixtures import ALL, convlstm, residual_cnn, transformer
from layout_checks import check_scene, routes

from tensorgami.ir import GraphBuilder, Node
from tensorgami.layout import Direction, Labeler, LayeredLayout, LayoutOptions, layout
from tensorgami.render import render_svg
from tensorgami.style import Shape
from tensorgami.transforms import simplify
from tensorgami.vocabulary import EdgeKind
from tensorgami.vocabulary import OpCategory as C

NO_WRAP = LayoutOptions(wrap_width=None, wrap_height=None)


# --------------------------------------------------------------------------- labels


@pytest.mark.parametrize(
    ("node", "label", "detail"),
    [
        (
            Node(
                "n",
                "Conv2d",
                C.CONV,
                params={"kernel_size": (3, 3), "out_channels": 64, "stride": (2, 2)},
            ),
            "Conv 3×3",
            "64, s2",
        ),
        (
            Node(
                "n",
                "ConvTranspose2d",
                C.CONV,
                params={"kernel_size": (4, 4), "out_channels": 1, "stride": (1, 1)},
            ),
            "ConvT 4×4",
            "1",
        ),
        (
            Node(
                "n",
                "Linear",
                C.LINEAR,
                params={"in_features": 4096, "out_features": 10, "folded": ("relu", "Dropout")},
            ),
            "Linear",
            "4096→10, ReLU, Dropout",
        ),
        (
            Node("n", "MaxPool2d", C.RESAMPLE, params={"kernel_size": 2, "stride": 2}),
            "MaxPool 2×2",
            "",
        ),
        (
            Node("n", "AdaptiveAvgPool2d", C.RESAMPLE, params={"output_size": 1}),
            "AdaptiveAvgPool",
            "→ 1×1",
        ),
        (
            Node(
                "n",
                "LSTM",
                C.RECURRENT,
                params={"hidden_size": 32, "num_layers": 2, "bidirectional": True},
            ),
            "LSTM",
            "32 hidden, 2 layers, bidirectional",
        ),
        (
            Node("n", "MultiheadAttention", C.ATTENTION, params={"num_heads": 8}),
            "Multi-head attention",
            "8 heads",
        ),
        (Node("n", "add", C.MERGE, params={"folded": ("relu",)}), "+", "ReLU"),
        (Node("n", "cat", C.MERGE), "‖", ""),
        (Node("n", "mul", C.OTHER, params={"args": (0.5,), "folded": ("exp",)}), "× 0.5", "exp"),
        (Node("n", "randn_like", C.STOCHASTIC), "ε ~ N(0, I)", ""),
        (Node("n", "input", C.IO, label="image"), "image", ""),
        (Node("n", "Dropout", C.REGULARIZATION, params={"p": 0.5}), "Dropout", "p = 0.5"),
        (
            Node(
                "n", "Embedding", C.EMBEDDING, params={"num_embeddings": 100, "embedding_dim": 32}
            ),
            "Embedding",
            "100→32",
        ),
        (
            Node("n", "GRUCell", C.RECURRENT, params={"hidden_size": 8}, count=3),
            "GRUCell",
            "8 hidden, ×3 steps",
        ),
        (Node("n", "BatchNorm2d", C.NORM, params={"folded": ("ReLU",)}), "BatchNorm", "ReLU"),
        (Node("n", "fft_rfft", C.OTHER), "fft_rfft", ""),
        # a label from overrides names the box; the computed detail stays
        (
            Node(
                "n", "Linear", C.LINEAR, label="head", params={"in_features": 8, "out_features": 2}
            ),
            "head",
            "8→2",
        ),
        (Node("n", "Conv2d", C.CONV, label="stem", params={"kernel_size": 3}), "stem", ""),
    ],
)
def test_node_labels(node, label, detail):
    text = Labeler().describe(node)
    assert (text.label, text.detail) == (label, detail)


def test_shapes_are_labelled_only_where_they_change():
    g = residual_cnn()
    labeler = Labeler()
    labelled = {(e.source, e.target): labeler.edge_label(g, e) for e in g.edges}
    assert labelled[("n0", "n1")] == "3×32×32"  # from the input
    assert labelled[("n6", "n7")] == "16×16×16"  # the pool halves the resolution
    assert labelled[("n7", "n8")] == "4096"  # flatten
    assert labelled[("n2", "n3")] == ""  # BatchNorm keeps the shape


# --------------------------------------------------------------------------- invariants


@pytest.mark.parametrize("make", ALL.values(), ids=ALL.keys())
@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("direction", list(Direction))
def test_layout_invariants(theme, make, level, direction):
    graph = simplify(make(), level)
    options = LayoutOptions(direction=direction, wrap_width=None, wrap_height=None)
    scene = LayeredLayout(options).layout(graph, theme)
    check_scene(scene, graph, theme, monotonic="x" if direction is Direction.LEFT_RIGHT else "y")


@pytest.mark.parametrize("make", ALL.values(), ids=ALL.keys())
def test_wrapped_layouts_keep_the_invariants(theme, make):
    graph = simplify(make(), 3)
    scene = LayeredLayout(LayoutOptions(wrap_width=400)).layout(graph, theme)
    check_scene(scene, graph, theme)


def test_layout_is_deterministic(theme):
    a = render_svg(layout(simplify(transformer(), 2), theme), theme)
    b = render_svg(layout(simplify(transformer(), 2), theme), theme)
    assert a == b


# --------------------------------------------------------------------------- features


def test_residual_skip_is_routed_around_the_block(theme):
    """The skip into a block's merge enters through a port and runs in its own lane."""
    b = GraphBuilder()
    x = b.input((1, 8))
    block = b.group("block", "ResidualBlock")
    prev = x
    chain = []
    for _ in range(3):
        n = b.node("Linear", category=C.LINEAR, group=block, outputs=[(1, 8)])
        b.edge(prev, n)
        prev = n
        chain.append(n)
    add = b.node("add", category=C.MERGE, group=block, outputs=[(1, 8)])
    b.edge(prev, add)
    b.edge(x, add)
    b.output(add)
    graph = b.build()
    scene = layout(graph, theme, NO_WRAP)
    check_scene(scene, graph, theme)  # in particular: the skip crosses no Linear box
    skip = next(
        path for edge, path in routes(scene, theme) if edge.source == x and edge.target == add
    )
    assert len(skip) > 2  # bent around the chain, not straight through it


def test_folded_recurrence_draws_state_loops_and_badges(theme):
    graph = simplify(convlstm(steps=3), 2)
    scene = layout(graph, theme)
    loops = [e for e in scene.edges if e.kind is EdgeKind.STATE and e.delay]
    assert len(loops) == 2  # h and c (the legend's sample state arrow has no delay marker)
    (frame,) = [g for g in scene.groups if g.id == "cell"]
    assert frame.badge == "×3" and frame.title == ""


def test_frames_titles_and_layer_badges(theme):
    scene = layout(simplify(transformer(layers=3), 2), theme)
    (frame,) = scene.groups
    assert (frame.title, frame.badge) == ("TransformerEncoderLayer", "×3")
    labels = {n.label for n in scene.nodes}
    assert {"tokens", "Multi-head attention", "LayerNorm", "encoded"} <= labels


def test_generic_containers_are_titled_by_attribute_name(theme):
    b = GraphBuilder()
    x = b.input((1, 4))
    g = b.group("features", "Sequential")
    first = b.node("Linear", category=C.LINEAR, group=g, outputs=[(1, 4)])
    second = b.node("ReLU", category=C.ACTIVATION, group=g, outputs=[(1, 4)])
    b.chain([x, first, second])
    b.output(second)
    scene = layout(b.build(), theme)
    assert [grp.title for grp in scene.groups] == ["features"]


def test_long_chains_wrap_into_rows(theme):
    b = GraphBuilder()
    prev = b.input((1, 8))
    for _ in range(14):
        n = b.node("Linear", category=C.LINEAR, outputs=[(1, 8)])
        b.edge(prev, n)
        prev = n
    b.output(prev)
    graph = b.build()
    wrapped = layout(graph, theme, LayoutOptions(wrap_width=900))
    flat = layout(graph, theme, NO_WRAP)

    def rows(scene):
        return {round(n.box.cy) for n in scene.nodes if not n.id.startswith("legend:")}

    assert len(rows(wrapped)) >= 2 and wrapped.width < flat.width
    assert len(rows(flat)) == 1
    check_scene(wrapped, graph, theme)


def _chain(length: int):
    b = GraphBuilder()
    prev = b.input((1, 8))
    for _ in range(length):
        n = b.node("Linear", category=C.LINEAR, outputs=[(1, 8)])
        b.edge(prev, n)
        prev = n
    b.output(prev)
    return b.build()


def test_a_tall_network_stands_in_columns(theme):
    """Top → bottom, a network taller than ``wrap_height`` wraps into columns side by side,
    like a newspaper's; ``wrap_height=None`` keeps one column."""
    graph = _chain(20)
    tall = layout(graph, theme, LayoutOptions(direction="TB", wrap_height=None))
    cols = layout(graph, theme, LayoutOptions(direction="TB", wrap_height=600))

    def columns(scene):
        return {round(n.box.cx) for n in scene.nodes if not n.id.startswith("legend:")}

    assert len(columns(tall)) == 1
    assert len(columns(cols)) >= 2 and cols.height < tall.height / 2
    check_scene(cols, graph, theme)


def test_a_short_last_row_is_centred_under_the_long_one(theme):
    """A long block fills the first row and the rest is short (AlexNet's classifier after its
    features): the wire comes back to the middle and the short row is centred under the long
    one, sharing the white space on both sides."""
    b = GraphBuilder()
    x = b.input((1, 8))
    block = b.group("block", "ResidualBlock")  # its skip makes it one unbreakable unit
    prev = x
    for _ in range(5):
        n = b.node("Linear", category=C.LINEAR, group=block, outputs=[(1, 8)])
        b.edge(prev, n)
        prev = n
    add = b.node("add", category=C.MERGE, group=block, outputs=[(1, 8)])
    b.edge(prev, add)
    b.edge(x, add)
    head = b.node("Linear", category=C.LINEAR, outputs=[(1, 4)])
    b.edge(add, head)
    b.output(head)
    graph = b.build()
    scene = layout(graph, theme, LayoutOptions(wrap_width=800, legend=False))
    boxes = {n.id: n.box for n in scene.nodes}
    (frame,) = [g.box for g in scene.groups]
    output = next(n.box for n in scene.nodes if n.category is C.IO and n.box.x > boxes[x].x)
    assert boxes[head].y > frame.bottom  # a second row: the head and the output
    first = (boxes[x].x + frame.right) / 2
    second = (boxes[head].x + output.right) / 2
    assert abs(first - second) < 1 and boxes[head].x > boxes[x].x + 100
    check_scene(scene, graph, theme)


def _framed_chain(length: int, *, skip: bool = False):
    """input → a framed group of ``length`` layers (optionally with a skip from its input to
    a final ``+``, as in a residual block) → output."""
    b = GraphBuilder()
    x = b.input((1, 8))
    group = b.group("features", "Sequential")
    prev = x
    for _ in range(length):
        n = b.node("Linear", category=C.LINEAR, group=group, outputs=[(1, 8)])
        b.edge(prev, n)
        prev = n
    if skip:
        add = b.node("add", category=C.MERGE, group=group, outputs=[(1, 8)])
        b.edge(prev, add)
        b.edge(x, add)
        prev = add
    b.output(prev)
    return b.build()


def _rows(scene, ids) -> set[int]:
    return {round(n.box.cy) for n in scene.nodes if n.id in ids}


def test_a_frame_wider_than_the_page_wraps_inside(theme):
    graph = _framed_chain(14)
    layers = {n.id for n in graph.members("features")}
    flat = layout(graph, theme, NO_WRAP)
    wrapped = layout(graph, theme, LayoutOptions(wrap_width=900))
    assert len(_rows(flat, layers)) == 1
    assert len(_rows(wrapped, layers)) >= 2  # rows inside the frame
    (frame,) = wrapped.groups
    assert frame.box.w <= 900 < flat.width
    check_scene(wrapped, graph, theme)  # every layer inside the frame, no wire through a box


def test_frames_inside_frames_wrap_to_fit_the_outer_padding(theme):
    b = GraphBuilder()
    prev = b.input((1, 8))
    outer = b.group("model", "Model")
    inner = b.group("model.block", "Block", parent=outer)
    for _ in range(14):
        n = b.node("Linear", category=C.LINEAR, group=inner, outputs=[(1, 8)])
        b.edge(prev, n)
        prev = n
    head = b.node("Linear", category=C.LINEAR, group=outer, outputs=[(1, 8)])
    b.edge(prev, head)
    b.output(head)
    graph = b.build()
    scene = layout(graph, theme, LayoutOptions(wrap_width=900, legend=False))
    frames = {g.id: g.box for g in scene.groups}
    assert len(_rows(scene, {n.id for n in graph.members("model.block")})) >= 2
    assert frames["model.block"].w <= 900 - 2 * 14  # the outer frame's padding comes off
    check_scene(scene, graph, theme)


def test_a_residual_block_is_never_split_across_rows(theme):
    """Its skip runs alongside every layer: no place where a single wire continues."""
    graph = _framed_chain(14, skip=True)
    scene = layout(graph, theme, LayoutOptions(wrap_width=900))
    assert len(_rows(scene, {n.id for n in graph.nodes if n.op == "Linear"})) == 1
    check_scene(scene, graph, theme)


def test_a_pill_stays_beside_a_frame_that_nearly_fills_the_width():
    from tensorgami.layout.layered import _best_cuts

    # AlexNet, every operation: input, `features` wrapped to just under the width, avgpool,
    # flatten, `classifier`, output — the pills join the frames (15 % over at most)
    levels = [0, 1, 2, 3, 4, 5]
    sizes = {0: 64.0, 1: 1080.0, 2: 150.0, 3: 120.0, 4: 1000.0, 5: 72.0}
    gaps = dict.fromkeys(levels, 48.0)
    assert _best_cuts(levels, levels[:-1], sizes, gaps, 1100) == [1, 3]  # not [0, 1, 3, 4]


def _encoder_decoder():
    """Two inputs, an encoder of 3 layers and a decoder of 4 whose third layer also takes the
    encoder's output (the Transformer's cross-attention), each in a frame."""
    b = GraphBuilder("seq2seq", root_type="Seq2Seq")
    source, target = b.input((1, 8), label="source"), b.input((1, 8), label="target")
    encoder = b.group("encoder", "Encoder", module="encoder")
    decoder = b.group("decoder", "Decoder", module="decoder")
    prev = source
    for _ in range(3):
        n = b.node("Linear", category=C.LINEAR, group=encoder, outputs=[(1, 8)])
        b.edge(prev, n)
        prev = n
    memory, prev = prev, target
    for k in range(4):
        n = b.node(
            "MultiheadAttention" if k == 2 else "Linear",
            category=C.ATTENTION if k == 2 else C.LINEAR,
            group=decoder,
            outputs=[(1, 8)],
        )
        b.edge(prev, n)
        if k == 2:
            b.edge(memory, n)
        prev = n
    b.output(prev)
    return b.build()


@pytest.mark.parametrize("direction", list(Direction))
def test_an_encoder_and_its_decoder_stand_side_by_side(theme, direction):
    """The decoder takes the encoder's output partway along it: both stacks run in parallel,
    starting level, and the encoder's output enters the decoder from the side."""
    graph = _encoder_decoder()
    scene = layout(
        graph, theme, LayoutOptions(direction=direction, wrap_width=None, wrap_height=None)
    )
    frames = {g.id: g.box for g in scene.groups}
    enc, dec = frames["encoder"], frames["decoder"]
    if direction is Direction.TOP_BOTTOM:
        assert enc.right < dec.x and enc.y == dec.y  # beside it, from the same height
    else:
        assert enc.bottom < dec.y and enc.x == dec.x
    check_scene(scene, graph, theme)


def test_a_residual_skip_is_not_a_side_input(theme):
    from tensorgami.layout.labels import Labeler
    from tensorgami.layout.layered import _Run
    from tensorgami.render.text import HeuristicTextMeasurer

    for graph in (simplify(residual_cnn(), 2), _encoder_decoder()):
        run = _Run(graph, theme, LayoutOptions(), Labeler(), HeuristicTextMeasurer())
        sides = [graph.edges[i] for i in run.side_inputs]
        if graph.name == "seq2seq":
            (edge,) = sides
            assert graph.node(edge.target).op == "MultiheadAttention"
        else:
            assert sides == []  # the skip leaves from the block's own input


def test_top_bottom_direction_stacks_vertically(theme):
    graph = simplify(transformer(), 2)
    scene = layout(graph, theme, LayoutOptions(direction=Direction.TOP_BOTTOM))
    ys = [n.box.cy for n in scene.nodes if n.category is C.IO]
    assert ys[0] < ys[-1] and scene.height > scene.width / 2


def test_legend_is_optional(theme):
    graph = simplify(residual_cnn(), 2)
    with_legend = layout(graph, theme)
    without = layout(graph, theme, LayoutOptions(legend=False))
    assert without.height < with_legend.height
    assert not any(n.id.startswith("legend:") for n in without.nodes)


def test_custom_labeler_is_used(theme):
    class Upper(Labeler):
        def describe(self, node):
            text = super().describe(node)
            return dataclasses.replace(text, label=text.label.upper())

    scene = LayeredLayout(labeler=Upper()).layout(simplify(residual_cnn(), 2), theme)
    labels = {n.label for n in scene.nodes}
    assert "CONV 3×3" in labels and "Conv 3×3" not in labels


def test_containers_are_named_after_their_attribute_when_collapsed():
    text = Labeler().describe(
        Node("<features>", "Sequential", C.CONV, module="features", params={"collapsed": 8})
    )
    assert (text.label, text.detail) == ("features", "8 layers")


def test_wrapping_never_leaves_a_lone_pill_and_balances_rows():
    from tensorgami.layout.layered import _best_cuts

    levels = [0, 1, 2, 3, 4, 5]  # input, a wide block, 3 small layers, output
    sizes = {0: 90.0, 1: 1900.0, 2: 200.0, 3: 120.0, 4: 900.0, 5: 90.0}
    gaps = dict.fromkeys(levels, 48.0)
    assert _best_cuts(levels, levels[:-1], sizes, gaps, 1100) == [1]  # [input, block] | rest
    even = dict.fromkeys(levels, 300.0)
    assert _best_cuts(levels, levels[:-1], even, gaps, 1100) == [2]  # 3 + 3, not 4 + 2


def test_no_wrap_when_it_barely_narrows_the_figure():
    """input → one frame wider than the page → output (e.g. a PatchGAN discriminator): a
    second row would only strand the input pill and save < 20 % of the width."""
    from tensorgami.layout.layered import _best_cuts

    levels = [0, 1, 2]
    sizes = {0: 64.0, 1: 1279.0, 2: 72.0}
    gaps = {0: 75.0, 1: 64.0, 2: 48.0}
    assert _best_cuts(levels, [0, 1], sizes, gaps, 1100) == []


def test_wrapping_cost_stays_bounded():
    import time

    from tensorgami.layout.layered import _best_cuts

    levels = list(range(200))
    started = time.perf_counter()
    _best_cuts(levels, levels[:-1], dict.fromkeys(levels, 100.0), dict.fromkeys(levels, 48.0), 1100)
    assert time.perf_counter() - started < 2.0  # exhaustive search would never finish


def test_style_layout_preferences_apply_unless_options_are_explicit(theme):
    graph = simplify(residual_cnn(), 2)
    roomy = dataclasses.replace(theme, layout={"rank_gap": 160, "wrap_width": None})
    plain = LayoutOptions.for_style(theme)
    assert plain == LayoutOptions()  # the default style keeps the defaults
    assert LayoutOptions.for_style(roomy).rank_gap == 160
    wide = LayeredLayout().layout(graph, roomy)
    narrow = LayeredLayout(LayoutOptions(wrap_width=None, wrap_height=None)).layout(
        graph, roomy
    )  # explicit wins
    assert wide.width > narrow.width
    vertical = dataclasses.replace(theme, layout={"direction": "TB"})
    assert LayoutOptions.for_style(vertical).direction is Direction.TOP_BOTTOM
    with pytest.raises(ValueError, match="unknown layout option"):
        LayoutOptions.for_style(dataclasses.replace(theme, layout={"rank_spacing": 3}))


# --------------------------------------------------------------------------- origami


@pytest.fixture(scope="module")
def origami():
    from tensorgami.style import load_style

    return load_style("origami")


@pytest.mark.parametrize("make", ALL.values(), ids=ALL.keys())
@pytest.mark.parametrize("direction", list(Direction))
def test_origami_layouts_keep_the_invariants(origami, make, direction):
    graph = simplify(make(), 2)
    options = LayoutOptions.for_style(
        origami, direction=direction, wrap_width=None, wrap_height=None
    )
    scene = LayeredLayout(options).layout(graph, origami)
    check_scene(scene, graph, origami, monotonic="x" if direction is Direction.LEFT_RIGHT else "y")


@pytest.mark.parametrize(("direction", "along"), [("LR", "h"), ("TB", "w")])
def test_folded_boxes_are_lengthened_by_their_fold(origami, direction, along):
    """Along the text: upright cards (left → right) grow in height, flat ones in width."""
    each = dataclasses.replace(
        origami, boxes=dataclasses.replace(origami.boxes, uniform_width=False)
    )
    conv_style = dataclasses.replace(each.node_style(C.CONV), shape=Shape.RECT)
    unfolded = dataclasses.replace(each, nodes={**each.nodes, C.CONV: conv_style})
    graph = simplify(residual_cnn(), 2)
    options = LayoutOptions.for_style(origami, direction=direction)
    conv = next(n.id for n in graph.nodes if n.op == "Conv2d")
    folded = next(n.box for n in layout(graph, each, options).nodes if n.id == conv)
    plain = next(n.box for n in layout(graph, unfolded, options).nodes if n.id == conv)
    assert getattr(folded, along) >= getattr(plain, along) + origami.metrics.fold_width - 8


@pytest.mark.parametrize("value", ["TB", "tb", Direction.TOP_BOTTOM])
def test_direction_accepts_strings(value):
    assert LayoutOptions(direction=value).direction is Direction.TOP_BOTTOM


@pytest.mark.parametrize(
    ("option", "error", "message"),
    [
        ({"direction": "XY"}, ValueError, "direction must be 'LR'"),
        ({"wrap_width": "wide"}, TypeError, "wrap_width: expected a number or None"),
        ({"rank_gap": None}, TypeError, "rank_gap: expected a number"),
        ({"legend": "no"}, TypeError, "legend: expected True or False"),
        ({"node_gap": True}, TypeError, "node_gap: expected a number"),
        ({"margin": -4}, ValueError, "margin: must be ≥ 0"),
    ],
)
def test_layout_options_are_checked(option, error, message):
    with pytest.raises(error, match=message):
        LayoutOptions(**option)


def test_layout_options_accept_numbers_and_none():
    options = LayoutOptions(
        rank_gap=60, node_gap=22.5, wrap_width=None, wrap_height=None, legend=False
    )
    assert (options.rank_gap, options.wrap_width, options.legend) == (60, None, False)


def test_flat_boxes_with_inline_text_and_uniform_width(theme):
    from tensorgami.style import BoxGeometry, TextLayout

    flat = dataclasses.replace(
        theme,
        boxes=BoxGeometry(
            text_lr=TextLayout.INLINE, text_tb=TextLayout.INLINE, height=26, uniform_width=True
        ),
    )
    graph = simplify(residual_cnn(), 2)
    scene = layout(graph, flat, NO_WRAP)
    check_scene(scene, graph, flat)
    boxes = [n for n in scene.nodes if n.category not in (C.IO, C.MERGE) and ":" not in n.id]
    assert len({n.box.w for n in boxes}) == 1  # one width
    assert {n.box.h for n in boxes} == {26}  # one line, whether or not there is a detail
    stacked = layout(simplify(residual_cnn(), 2), theme, NO_WRAP)
    assert {n.box.h for n in stacked.nodes if n.detail and n.category is C.CONV} == {46}


def _boxes(scene):
    return [n for n in scene.nodes if n.category not in (C.IO, C.MERGE) and ":" not in n.id]


@pytest.mark.parametrize(
    ("direction", "text", "widths"), [("TB", "inline", 1), ("LR", "stacked", None)]
)
def test_each_flow_direction_gets_its_own_arrangement(theme, direction, text, widths):
    from tensorgami.style import BoxGeometry, TextLayout

    styled = dataclasses.replace(
        theme,
        boxes=BoxGeometry(
            text_tb=TextLayout.INLINE, min_width=48, height=26, detail_height=40, uniform_width=True
        ),
    )
    graph = simplify(residual_cnn(), 2)
    scene = layout(
        graph, styled, LayoutOptions(direction=direction, wrap_width=None, wrap_height=None)
    )
    check_scene(scene, graph, styled)
    boxes = _boxes(scene)
    assert {n.text.value for n in boxes} == {text}  # recorded for the painter
    assert (len({n.box.w for n in boxes}) == 1) is (widths == 1)  # equal widths: one line only
    if text == "stacked":
        assert {n.box.h for n in boxes if n.detail} == {40}


def test_upright_boxes_and_labels_in_a_horizontal_flow(theme):
    from tensorgami.style import BoxGeometry, TextLayout

    upright = dataclasses.replace(
        theme, boxes=BoxGeometry(text_lr=TextLayout.ROTATED, height=26, uniform_width=True)
    )
    graph = simplify(residual_cnn(), 2)
    scene = layout(graph, upright, NO_WRAP)
    check_scene(scene, graph, upright)
    boxes = _boxes(scene)
    assert {n.box.w for n in boxes} == {26}  # thin along the flow
    assert len({n.box.h for n in boxes}) == 1 and boxes[0].box.h > 26  # one length, upright
    pills = [n for n in scene.nodes if n.category is C.IO and ":" not in n.id]
    assert all(p.text is TextLayout.INLINE and p.box.w > p.box.h for p in pills)  # level
    labelled = [e for e in scene.edges if e.label]
    assert labelled and all(e.label_upright for e in labelled)
    level = layout(graph, theme, NO_WRAP)
    assert scene.width < level.width  # the point of it: much narrower
    assert not any(e.label_upright for e in level.edges)


def test_no_second_row_just_for_a_lone_pill():
    """input → a block that fits the page → output: wrapping would save < 20 %."""
    from tensorgami.layout.layered import _best_cuts

    sizes = {0: 64.0, 1: 1000.0, 2: 72.0}
    gaps = {0: 48.0, 1: 48.0, 2: 48.0}
    assert _best_cuts([0, 1, 2], [0, 1], sizes, gaps, 1100) == []
