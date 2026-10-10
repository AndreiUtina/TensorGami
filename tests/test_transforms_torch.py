"""Levels of detail on real traces of the reference models."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from models import REFERENCE  # noqa: E402

from tensorgami.importers.torch import trace  # noqa: E402
from tensorgami.transforms import Level, simplify  # noqa: E402
from tensorgami.vocabulary import EdgeKind  # noqa: E402

TRACES = {}


def traced(name: str):
    if name not in TRACES:
        make, example = REFERENCE[name]
        TRACES[name] = trace(make(), example())
    return TRACES[name]


@pytest.mark.parametrize("name", REFERENCE)
@pytest.mark.parametrize("level", list(Level))
def test_all_levels_keep_inputs_and_outputs(name, level):
    raw = traced(name)
    g = simplify(raw, level)
    assert len(g.inputs) == len(raw.inputs) and len(g.outputs) == len(raw.outputs)


def test_convlstm_layer_view():
    g = simplify(traced("convlstm"), Level.LAYERS)
    assert [n.op for n in g.nodes] == ["input", "ConvLSTMCell", "ConvLSTMCell", "Conv2d", "output"]
    assert [g.group(n.group).count for n in g.nodes if n.op == "ConvLSTMCell"] == [3, 3]
    assert sum(e.kind is EdgeKind.STATE for e in g.edges) == 4  # h and c loops per layer


def test_cnn_residual_blocks_fold_to_two():
    g = simplify(traced("cnn"), Level.LAYERS)
    assert g.group("blocks.0").count == 2
    assert not g.has_group("blocks.1")
    assert len([n for n in g.nodes if n.op == "add"]) == 1


def test_transformer_layers_fold_and_buffer_is_named():
    g = simplify(traced("transformer"), Level.LAYERS)
    assert g.group("encoder.layers.0").count == 2
    assert any(n.display_label == "pe" for n in g.nodes)


def test_stages_view_of_the_cnn():
    g = simplify(traced("cnn"), Level.STAGES)
    assert [n.op for n in g.nodes] == [
        "input",
        "Sequential",
        "Sequential",
        "MaxPool2d",
        "Sequential",
        "output",
    ]
