"""The ten most-cited architectures of ``examples/classics.ipynb``: each model is the one its
paper describes, and draws cleanly in the view the notebook uses. Large models are built on
PyTorch's ``meta`` device: their shapes are traced without allocating any weights."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

import classics as C  # noqa: E402
import resnet  # noqa: E402
from layout_checks import check_scene  # noqa: E402

import tensorgami as tg  # noqa: E402
from tensorgami.ir import USHAPE  # noqa: E402


def meta(make):
    def build():
        with torch.device("meta"):
            return make()

    return build


def tokens(vocab: int, n: int):
    return torch.randint(0, vocab, (1, n))


#: name: (model, example input, tg.draw options as in the notebook)
CASES = {
    "resnet34": (meta(resnet.resnet34), lambda: torch.randn(1, 3, 224, 224), {}),
    "transformer": (
        meta(C.Transformer),
        lambda: (tokens(37000, 10), tokens(37000, 9)),
        {"direction": "TB"},
    ),
    "alexnet": (meta(C.AlexNet2012), lambda: torch.randn(1, 3, 224, 224), {"view": "volume"}),
    "bert": (
        meta(C.Bert),
        lambda: (tokens(30522, 12), torch.zeros(1, 12, dtype=torch.long)),
        {},
    ),
    "lstm": (C.LSTMNet, lambda: torch.randn(1, 10, 8), {"view": "cell"}),
    "unet": (meta(C.UNet), lambda: torch.randn(1, 1, 572, 572), {"view": "volume"}),
    "faster-rcnn": (  # proposals are picked from real scores: not on the meta device
        C.FasterRCNN,
        lambda: torch.randn(1, 3, 224, 224),
        {"level": "stages", "overrides": C.FASTER_RCNN_OVERRIDES},
    ),
    "vit": (meta(C.VisionTransformer), lambda: torch.randn(1, 3, 224, 224), {}),
    "lenet": (
        C.LeNet5,
        lambda: torch.randn(1, 1, 32, 32),
        {"view": "volume", "overrides": C.LENET_OVERRIDES},
    ),
    "yolo": (meta(C.YOLO), lambda: torch.randn(1, 3, 448, 448), {"view": "volume"}),
}


@pytest.mark.parametrize("name", CASES)
def test_every_classic_draws_cleanly(name):
    make, example, options = CASES[name]
    fig = tg.draw(make(), example(), **options)
    check_scene(fig.scene, fig.graph, fig.style)


@pytest.mark.parametrize(
    ("make", "millions", "paper"),
    [
        (C.AlexNet2012, 61.0, "60 million parameters"),
        (C.Transformer, 63.1, "65 M (base model, Table 3)"),
        (C.Bert, 110.1, "110M (BERT-base)"),
        (C.UNet, 31.0, None),
        (C.VisionTransformer, 86.6, "86M (ViT-Base, Table 1)"),
        (C.LeNet5, 0.06, "60,000 trainable parameters"),
    ],
)
def test_the_models_are_the_papers_sizes(make, millions, paper):
    with torch.device("meta"):
        model = make()
    assert sum(p.numel() for p in model.parameters()) / 1e6 == pytest.approx(millions, abs=0.1)


def test_unet_maps_a_572_tile_to_a_388_map_of_two_classes():
    with torch.device("meta"):
        out = C.UNet()(torch.empty(1, 1, 572, 572))
    assert out.shape == (1, 2, 388, 388)


def test_unet_is_folded_whole_into_its_u():
    with torch.device("meta"):
        model = C.UNet()
    fig = tg.draw(model, torch.randn(1, 1, 572, 572))
    (u,) = [g for g in fig.graph.groups if g.type_name == USHAPE]
    layers = [n for n in fig.graph.nodes if not (n.is_input or n.is_output)]
    assert all(n.group == u.id for n in layers)  # first convolutions and 1×1 head included


def test_yolo_predicts_a_7_by_7_grid_of_30_values():
    with torch.device("meta"):
        out = C.YOLO()(torch.empty(1, 3, 448, 448))
    assert out.shape == (1, 7, 7, 30)


@pytest.mark.parametrize("direction", ["TB", "LR"])
def test_the_transformers_encoder_and_decoder_run_in_parallel(direction):
    """As in the paper's Fig. 1: the decoder stands beside the encoder, not after it."""
    make, example, _ = CASES["transformer"]
    fig = tg.draw(make(), example(), direction=direction)
    frames = {g.title: g.box for g in fig.scene.groups}
    enc, dec = frames["TransformerEncoderLayer"], frames["TransformerDecoderLayer"]
    if direction == "TB":
        assert enc.right < dec.x and abs(enc.y - dec.y) < 1
    else:
        assert enc.bottom < dec.y and abs(enc.x - dec.x) < 1


def test_the_transformer_shows_no_mask_check():
    """PyTorch compares a decoder mask with a causal one unless told it is causal."""
    make, example, options = CASES["transformer"]
    fig = tg.draw(make(), example(), **options)
    assert not {"triu", "eq", "all"} & {n.op for n in fig.graph.nodes}
