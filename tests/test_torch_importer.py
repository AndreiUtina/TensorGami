"""PyTorch importer (ADR-0003) on the reference models and on targeted edge cases."""

from __future__ import annotations

import subprocess
import sys
import warnings

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from models import REFERENCE  # noqa: E402
from models.convlstm import ConvLSTMCell, ConvLSTMForecaster  # noqa: E402

from tensorgami.importers import ImporterError, UnclassifiedOpWarning  # noqa: E402
from tensorgami.importers.torch import TorchImporter, trace  # noqa: E402
from tensorgami.ir import GraphValidator, to_json  # noqa: E402
from tensorgami.vocabulary import EdgeKind, OpCategory  # noqa: E402


def run(name: str, **options):
    make, example = REFERENCE[name]
    model = make()
    return model, trace(model, example(), **options)


def ops(graph, op: str):
    return [n for n in graph.nodes if n.op == op]


# --------------------------------------------------------------------------- all families


@pytest.mark.parametrize("name", REFERENCE)
def test_reference_models_trace_cleanly(name):
    make, example = REFERENCE[name]
    x = example()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnclassifiedOpWarning)  # every op is catalogued
        graph = trace(make(), x)
    assert GraphValidator().validate(graph) == []
    assert len(graph.inputs) == (len(x) if isinstance(x, tuple) else 1)
    assert graph.outputs
    assert graph.metadata["framework"] == "torch"
    assert graph.name == type(make()).__name__


@pytest.mark.parametrize("name", REFERENCE)
def test_tracing_leaves_the_model_untouched(name):
    make, example = REFERENCE[name]
    model = make()
    model.train()
    for module in list(model.modules())[1::2]:  # a mix of train and eval submodules
        module.eval()
    flags = [m.training for m in model.modules()]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    x = example()  # created first: building it consumes random numbers itself
    rng = torch.get_rng_state()

    trace(model, x)

    assert [m.training for m in model.modules()] == flags
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng)  # e.g. the VAE's randn_like
    hooks = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
    assert hooks == 0


def test_tracing_is_deterministic():
    assert to_json(run("cnn")[1]) == to_json(run("cnn")[1])


# --------------------------------------------------------------------------- per family


def test_cnn_residual_merges_and_groups():
    _, g = run("cnn")
    assert [grp.id for grp in g.groups] == [
        "",
        "stem",
        "blocks",
        "blocks.0",
        "blocks.1",
        "classifier",
    ]
    for block in ("blocks.0", "blocks.1"):
        (add,) = [n for n in g.members(block) if n.op == "add"]
        assert add.category is OpCategory.MERGE
        sources = {p.module for p in g.predecessors(add.id)}
        assert f"{block}.bn2" in sources and len(sources) == 2  # skip path + main path
    conv = ops(g, "Conv2d")[0]
    assert conv.params["kernel_size"] == (3, 3) and conv.params["out_channels"] == 16


def test_lstm_and_recovered_index():
    _, g = run("lstm")
    (lstm,) = ops(g, "LSTM")
    assert lstm.category is OpCategory.RECURRENT
    assert lstm.params["hidden_size"] == 32 and lstm.params["num_layers"] == 2
    (view,) = ops(g, "getitem")  # h[-1] is invisible to __torch_function__ in torch 2.2
    assert view.params["recovered"] is True
    (edge,) = g.in_edges(view.id)
    assert edge.source == lstm.id and edge.source_port == 1  # h, not the sequence output
    assert view.output_shapes == ((1, 32),)


def test_cgan_generator_has_two_inputs_merged_with_the_condition():
    _, g = run("cgan_generator")
    assert [n.label for n in g.inputs] == ["input 0", "input 1"]
    (cat,) = ops(g, "cat")
    sources = {p.op for p in g.predecessors(cat.id)}
    assert sources == {"input", "Embedding"}
    assert ops(g, "ConvTranspose2d")[0].category is OpCategory.CONV


def test_vae_sampling_is_a_source_and_outputs_are_labelled():
    _, g = run("vae")
    (noise,) = ops(g, "randn_like")
    assert noise.category is OpCategory.STOCHASTIC
    assert g.in_edges(noise.id) == ()
    assert [e.kind for e in g.out_edges(noise.id)] == [EdgeKind.STOCHASTIC]
    scale = ops(g, "mul")[0]  # 0.5 * logvar: arithmetic with a number, not a merge
    assert scale.category is OpCategory.OTHER
    assert [n.label for n in g.outputs] == ["output 0", "output 1", "output 2"]


def test_convlstm_steps_become_separate_invocations():
    _, g = run("convlstm")
    ids = [grp.id for grp in g.groups[1:]]
    assert ids == ["cells.0", "cells.1", "cells.0#1", "cells.1#1", "cells.0#2", "cells.1#2"]
    assert all(g.group(i).category is OpCategory.RECURRENT for i in ids)  # from the class name
    convs = [n for n in ops(g, "Conv2d") if n.module == "cells.0.gates"]
    assert [n.call_index for n in convs] == [0, 1, 2]
    crossing = [
        e
        for e in g.edges
        if g.node(e.source).group == "cells.0" and g.node(e.target).group == "cells.0#1"
    ]
    assert crossing  # hidden/cell state flows from step 0 to step 1
    assert all(g.in_edges(n.id) == () for n in ops(g, "new_zeros"))  # initial states


def test_transformer_layers_attention_and_buffer_source():
    _, g = run("transformer")
    assert {"encoder.layers.0", "encoder.layers.1"} <= {grp.id for grp in g.groups}
    attention = ops(g, "MultiheadAttention")
    assert len(attention) == 2 and attention[0].params["num_heads"] == 4
    assert len(g.in_edges(attention[0].id)) == 1  # self-attention (x, x, x): one edge
    positions = ops(g, "getitem")[0]
    assert positions.params["weights"] == ("pe",) and g.in_edges(positions.id) == ()
    assert not any(n.op.startswith("_transformer") for n in g.nodes)  # no fused fast path


# --------------------------------------------------------------------------- options


def test_overrides_by_class_name_and_path():
    model = ConvLSTMForecaster()
    g = trace(
        model,
        torch.randn(1, 2, 1, 8, 8),
        overrides={
            "ConvLSTMCell": {"label": "ConvLSTM cell"},
            "cells.1": {"category": "other", "label": "upper cell"},
            "head": {"label": "1×1 conv"},
        },
    )
    assert g.group("cells.0").label == "ConvLSTM cell"
    assert g.group("cells.1#1").label == "upper cell"  # path wins over class name
    assert g.group("cells.1").category is OpCategory.OTHER
    assert ops(g, "Conv2d")[-1].label == "1×1 conv"


@pytest.mark.parametrize("option", ["override", "leaf_types"])
def test_custom_module_as_single_operation(option):
    model = ConvLSTMForecaster()
    kwargs = (
        {"overrides": {"ConvLSTMCell": {"leaf": True}}}
        if option == "override"
        else {"leaf_types": (ConvLSTMCell,)}
    )
    g = trace(model, torch.randn(1, 3, 1, 8, 8), **kwargs)
    cells = ops(g, "ConvLSTMCell")
    assert len(cells) == 6 and len(g.groups) == 1  # 2 layers × 3 steps, no inner groups
    assert all(c.category is OpCategory.RECURRENT for c in cells)
    assert [c.call_index for c in cells if c.module == "cells.0"] == [0, 1, 2]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"x": {"colour": "red"}}, "unknown keys"),
        ({"x": {"category": "quantum"}}, "unknown category"),
        ({"x": {"leaf": "yes"}}, "True or False"),
        ({"x": "conv"}, "must be a dict"),
    ],
)
def test_invalid_overrides_are_explained(overrides, message):
    with pytest.raises(ImporterError, match=message):
        TorchImporter(overrides=overrides)


def test_keyword_inputs_use_their_names():
    class Scaled(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x, scale):
            return self.fc(x) * scale

    g = trace(Scaled(), {"x": torch.randn(1, 4), "scale": torch.randn(1, 2)})
    assert [n.label for n in g.inputs] == ["x", "scale"]
    assert ops(g, "mul")[0].category is OpCategory.MERGE  # two tensors: a merge


def test_bare_leaf_module_and_in_place_ops():
    g = trace(nn.Conv2d(3, 4, 3), torch.randn(1, 3, 8, 8))
    assert [n.op for n in g.nodes] == ["input", "Conv2d", "output"]

    class InPlace(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 4)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            y = self.a(x)
            y += self.b(x)
            return self.act(y)

    g = trace(InPlace(), torch.randn(1, 4))
    (add,) = ops(g, "add_")
    assert {p.module for p in g.predecessors(add.id)} == {"a", "b"}
    assert g.predecessors(ops(g, "ReLU")[0].id) == (add,)


def test_untraced_output_view_and_outside_constant():
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)
            self.scale = torch.full((3,), 2.0)  # plain attribute, not a buffer

        def forward(self, x):
            return (self.fc(x) * self.scale)[0]  # integer index on the result

    g = trace(Odd(), torch.randn(2, 4))
    (const,) = ops(g, "constant")
    assert const.category is OpCategory.TENSOR and g.in_edges(const.id) == ()
    (view,) = ops(g, "getitem")
    assert g.successors(view.id)[0].op == "output"
    assert view.output_shapes == ((3,),)


def test_unknown_operations_warn_but_do_not_fail():
    class Spectral(nn.Module):
        def forward(self, x):
            return torch.fft.rfft(x).abs()

    with pytest.warns(UnclassifiedOpWarning, match="fft_rfft"):
        g = trace(Spectral(), torch.randn(1, 8))
    assert ops(g, "fft_rfft")[0].category is OpCategory.OTHER


def test_model_errors_are_explained_and_cleaned_up():
    model, (_, example) = REFERENCE["cnn"][0](), REFERENCE["cnn"]
    model.train()
    with pytest.raises(ImporterError) as info:
        trace(model, torch.randn(3, 32, 32))  # missing batch dimension
    text = str(info.value)
    assert "SmallCNN" in text and "(3, 32, 32)" in text and "batch dimension" in text
    assert "expected 4D input" in str(info.value.__cause__)  # original error kept as cause
    assert model.training
    assert sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()) == 0
    trace(model, example())  # still usable afterwards


def test_rejects_non_modules():
    with pytest.raises(ImporterError, match="torch.nn.Module"):
        trace(lambda x: x, torch.randn(1))  # type: ignore[arg-type]


def test_summary_lists_operations():
    text = run("lstm")[1].summary()
    assert text.splitlines()[0] == "LSTMClassifier: 5 nodes, 4 edges, 1 groups"
    assert "LSTM" in text and "1×5×32 (+2)" in text


def test_core_package_does_not_import_torch():
    """ADR-0006: only tensorgami.importers.torch may load torch."""
    code = (
        "import sys, tensorgami, tensorgami.ir, tensorgami.importers, tensorgami.render; "
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
