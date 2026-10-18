"""Real activations (roadmap step 14d): summarised as the model runs, matched to the boxes of
the figure, and shown under it as each layer runs."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from models import REFERENCE  # noqa: E402
from test_animation import shown  # noqa: E402
from torch import nn  # noqa: E402

import tensorgami as tg  # noqa: E402
from tensorgami.activations import Kind, summarize  # noqa: E402
from tensorgami.importers.torch import TorchImporter, trace  # noqa: E402

# --------------------------------------------------------------------------- summaries


def test_feature_maps_keep_their_most_active_channels_small():
    x = np.zeros((1, 16, 100, 60))
    x[0, 5] = 3.0
    x[0, 9] = 1.0
    x[0, 2, :10] = 7.0  # strong, but over a tenth of the map: on average 0.7
    a = summarize(x)
    assert a.kind is Kind.MAPS and a.shape == (1, 16, 100, 60)
    assert a.channels[:3] == (5, 9, 2) and len(a.channels) == 8
    assert a.preview.shape == (8, 32, 32)
    assert (a.low, a.high) == (0.0, 7.0) and a.zeros == pytest.approx(
        1 - (6000 + 6000 + 600) / 96000
    )


def test_each_kind_of_output_gets_its_preview():
    picture = np.random.default_rng(0).random((1, 3, 224, 224))
    assert summarize(picture, image=True).kind is Kind.IMAGE
    assert summarize(picture, image=True).preview.shape == (3, 64, 64)
    assert summarize(picture).kind is Kind.MAPS  # a layer's 3 channels are maps
    tokens = summarize(np.ones((1, 6, 32)))
    assert tokens.kind is Kind.SEQUENCE and tokens.preview.shape == (6, 32)
    assert summarize(np.ones((1, 100, 512))).preview.shape == (32, 64)
    logits = summarize(np.arange(1000.0)[None])
    assert logits.kind is Kind.VECTOR and logits.preview.shape == (256,)
    assert logits.preview[0] == pytest.approx(np.arange(4.0).mean(), abs=0.5)
    assert summarize(np.ones((1, 64, 1, 1))).kind is Kind.VECTOR  # after global pooling
    assert summarize(np.array(2.5)).kind is Kind.SCALAR
    frames = summarize(np.ones((1, 4, 2, 16, 16)))  # a sequence of maps: its first step
    assert frames.kind is Kind.MAPS and frames.shape == (1, 4, 2, 16, 16)


def test_values_that_are_not_finite_are_left_out():
    x = np.array([[1.0, np.nan, -2.0, np.inf]])
    a = summarize(x)
    assert (a.low, a.high) == (-2.0, 1.0)
    assert summarize(np.full((1, 4), np.nan)) is None


# --------------------------------------------------------------------------- capture


class Residual(nn.Module):
    """conv → in-place ReLU → conv, added in place to its input: in-place operations would
    change what an earlier operation seemed to produce if summaries waited."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 4))

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))
        out += x
        return self.head(out)


def model_and_input():
    torch.manual_seed(0)
    return Residual(), torch.randn(1, 3, 12, 12)


def test_capturing_leaves_the_traced_graph_as_it_is():
    runs = [
        model_and_input(),
        *((make(), example()) for make, example in (REFERENCE["cnn"], REFERENCE["convlstm"])),
    ]
    for model, x in runs:
        plain = trace(model, x)
        graph, store = TorchImporter().import_with_activations(model, x)
        assert [(n.id, n.op) for n in graph.nodes] == [(n.id, n.op) for n in plain.nodes]
        assert graph.edges == plain.edges
        assert set(store.nodes) <= {n.id for n in graph.nodes}


def test_each_output_is_summarised_as_it_is_produced():
    model, x = model_and_input()
    graph, store = TorchImporter().import_with_activations(model, x)
    by_op = {n.op: store.nodes[n.id] for n in graph.nodes if n.id in store.nodes}
    assert by_op["Conv2d"].low < 0  # before the in-place ReLU changed its tensor
    assert by_op["ReLU"].low == 0
    assert store.groups[""].shape == (1, 4)  # the whole model's call
    assert store.groups["head"].shape == (1, 4)
    assert by_op["input"].kind is Kind.IMAGE


# --------------------------------------------------------------------------- the figure


def test_every_box_shows_what_it_outputs():
    model, x = model_and_input()
    fig = tg.draw(model, x, activations=True)
    boxes = [n for n in fig.scene.nodes if not n.id.startswith("legend:")]
    assert {n.id for n in boxes} == set(fig.activations)
    conv = next(n for n in boxes if n.label.startswith("Conv") and "ReLU" in n.detail)
    assert fig.activations[conv.id][0].low == 0  # its ReLU folded in: the maps after it


def test_the_still_figure_is_unchanged():
    model, x = model_and_input()
    assert tg.draw(model, x, activations=True).to_svg() == tg.draw(model, x).to_svg()
    assert tg.draw(model, x).activations == {}


def test_a_block_folded_n_times_has_one_activation_per_pass():
    make, example = REFERENCE["transformer"]
    torch.manual_seed(0)
    fig = tg.draw(make(), example(), activations=True)
    (block,) = [g for g in fig.graph.groups if g.count > 1]
    inside = [n.id for n in fig.graph.nodes_under(block.id)]
    assert all(len(fig.activations[n]) == block.count for n in inside)
    first, second = fig.activations[inside[0]][:2]
    assert (first.low, first.high) != (second.low, second.high)  # each copy its own
    make, example = REFERENCE["convlstm"]
    fig = tg.draw(make(), example(), view="block", activations=True)
    cells = [k for k, v in fig.activations.items() if k.startswith("<cells.")]
    assert cells and all(len(fig.activations[k]) == 3 for k in cells)


def test_an_unrolled_recurrence_shows_each_step():
    make, example = REFERENCE["convlstm"]
    fig = tg.draw(make(), example(), activations=True)  # unrolled: a time × layer grid
    boxes = [n.id for n in fig.scene.nodes if not n.id.startswith("legend:")]
    assert set(boxes) == set(fig.activations)  # the steps' inputs x_t and every cell call


def test_several_models_resolve_by_name():
    (g, gx), (d, dx) = REFERENCE["cgan_generator"], REFERENCE["cgan_discriminator"]
    fig = tg.draw(
        {"generator": g(), "discriminator": d()},
        {"generator": gx(), "discriminator": dx()},
        links=[("generator", "discriminator.input 0")],
        activations=True,
    )
    boxes = [n.id for n in fig.scene.nodes if not n.id.startswith("legend:")]
    assert set(boxes) == set(fig.activations)


def test_a_model_on_the_meta_device_has_nothing_to_show():
    make, example = REFERENCE["cnn"]
    with torch.device("meta"):
        model = make()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = tg.draw(model, example().to("meta"), activations=True)
    assert fig.activations == {}
    assert any("no values to show" in str(w.message) for w in caught)


# --------------------------------------------------------------------------- the animation


def panel_look(anim, state):
    (_, items) = anim._overlays[0]
    return items[0].look(state, None)


def test_the_panel_shows_the_running_layers_then_the_outputs():
    model, x = model_and_input()
    fig = tg.draw(model, x, activations=True)
    anim = fig.animate()
    assert anim.scene.width == fig.scene.width and anim.scene.height > fig.scene.height
    for state in anim.steps[:-1]:
        shown_ids = {node for node, _ in panel_look(anim, state)}
        assert shown_ids == {n for n in state.active if n in fig.activations}
    outputs = {n.id for n in fig.graph.nodes if n.is_output}
    assert {node for node, _ in panel_look(anim, anim.steps[-1])} == outputs
    plain = fig.animate(activations=False)
    assert plain.scene is fig.scene and plain.to_svg() == tg.draw(model, x).animate().to_svg()
    with pytest.raises(ValueError, match="activations=True"):
        tg.draw(model, x).animate(activations=True)


def test_each_pass_of_a_folded_block_shows_its_own_values():
    make, example = REFERENCE["convlstm"]
    fig = tg.draw(make(), example(), view="block", activations=True)
    anim = fig.animate()
    passes = {
        (node, k)
        for state in anim.steps
        for node, k in (panel_look(anim, state) or ())
        if node.startswith("<cells.0")
    }
    assert passes == {("<cells.0>", 0), ("<cells.0>", 1), ("<cells.0>", 2)}


def test_the_animated_svg_with_activations_shows_each_step_as_its_still(tmp_path):
    model, x = model_and_input()
    anim = tg.draw(model, x, activations=True).animate()
    animated, total = anim.to_svg(), sum(anim.durations)
    _, _, starts = anim._timed()
    lit = [d for f, d in zip(anim, anim.durations, strict=True) if not f.pulses]
    for k, (start, seconds) in enumerate(zip(starts, lit, strict=True)):
        assert shown(animated, (start + seconds / 2) / total) == shown(anim.to_svg(k), None)
    assert animated.count("data:image/png;base64,") > 0
    assert anim.to_png(0)[:8] == b"\x89PNG\r\n\x1a\n"  # matplotlib draws the images too
    paths = anim.save_frames(tmp_path, format="pdf", dpi=40)
    assert all(p.stat().st_size > 0 for p in paths)


@pytest.mark.filterwarnings("ignore:at 3.5 in wide")
def test_a_fitted_animation_keeps_its_width_and_grows_in_height():
    model, x = model_and_input()
    anim = tg.draw(model, x, width="3.5in", activations=True).animate()
    svg = anim.to_svg()
    assert 'width="3.5in"' in svg
    height = float(svg.split('height="', 1)[1].split("in", 1)[0])
    assert height == pytest.approx(3.5 * anim.scene.height / anim.scene.width, abs=0.01)
