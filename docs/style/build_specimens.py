"""Reference specimens of the Tensorgami ``default`` style.

Each specimen is a hand-positioned :class:`~tensorgami.scene.Scene` (the layout engine comes
in a later roadmap step) rendered with the default style. Rebuild with::

    uv run python docs/style/build_specimens.py [OUT_DIR]
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from tensorgami.render import add_legend, build_legend, render_svg
from tensorgami.scene import Box, Edge, Group, Node, Note, Point, Scene, Side, TextAnchor
from tensorgami.style import Shape, TextRole, Theme, load_theme
from tensorgami.vocabulary import EdgeKind, OpCategory

OUT_DIR = Path(__file__).parent / "specimens"

C = OpCategory
K = EdgeKind


def at(cx: float, cy: float, w: float, h: float) -> Box:
    return Box.centered(cx, cy, w, h)


def title(x: float, y: float, text: str) -> Note:
    return Note(Point(x, y), text, TextAnchor.START, TextRole.GROUP_TITLE)


def caption(x: float, y: float, text: str, anchor: TextAnchor = TextAnchor.MIDDLE) -> Note:
    return Note(Point(x, y), text, anchor, TextRole.NOTE)


# --------------------------------------------------------------------------- vocabulary


def vocabulary(theme: Theme) -> Scene:
    s = Scene(900, 760, "Tensorgami default style: visual vocabulary")
    s.add(title(24, 24, "Categories — fill colour encodes the operation category, nothing else"))
    legend = build_legend(theme, list(OpCategory), origin=Point(24, 40), columns=3)
    s.add(*legend.items)

    y = legend.box.bottom + 40
    s.add(title(24, y, "Shapes — outline encodes the kind of element"))
    cy = y + 70
    s.add(
        Node("op", at(90, cy, 124, 48), C.CONV, "Conv 3×3", "64, stride 1"),
        Node("data", at(240, cy, 96, 32), C.IO, "image"),
        Node("add", at(340, cy, 24, 24), C.MERGE, "+"),
        Node("mul", at(380, cy, 24, 24), C.MERGE, "×"),
        Node("cat", at(420, cy, 24, 24), C.MERGE, "‖"),
        Node("enc", at(530, cy, 110, 84), C.CONV, "Encoder", shape=Shape.TRAPEZOID_NARROW),
        Node("dec", at(660, cy, 110, 84), C.CONV, "Decoder", shape=Shape.TRAPEZOID_WIDEN),
        Node("vol", at(785, cy - 6, 26, 64), C.CONV, "", "64×56×56", shape=Shape.CUBOID, depth=40),
    )
    cap_y = cy + 72
    for x, text in [
        (90, "operation / layer"),
        (240, "data (I/O, tensor)"),
        (380, "merge: add · multiply · concat"),
        (530, "encoder (compress)"),
        (660, "decoder (expand)"),
        (800, "tensor volume"),
    ]:
        s.add(caption(x, cap_y, text))

    y = cap_y + 44
    s.add(title(24, y, "Edges — one solid style for data flow, distinct styles for other meanings"))
    edge_legend = build_legend(theme, [], list(EdgeKind), origin=Point(24, y + 14))
    s.add(*edge_legend.items)
    ey = y + 40
    s.add(
        Node("a", at(470, ey, 90, 36), C.CONV, "Conv"),
        Node("b", at(680, ey, 90, 36), C.NORM, "BatchNorm"),
        Edge("a", "b", label="64×56×56"),
        caption(575, ey + 32, "tensor shape on the edge (C×H×W, batch omitted)"),
    )
    ry = ey + 84
    s.add(
        Node("rnn", at(520, ry, 104, 44), C.RECURRENT, "LSTM", "hidden 256"),
        Edge(
            Point(572, ry - 8),
            Point(572, ry + 8),
            K.STATE,
            via=(Point(612, ry - 8), Point(612, ry + 8)),
            delay=True,
        ),
        caption(640, ry, "folded recurrence: ■ marks the one-step delay", TextAnchor.START),
    )

    y = ry + 64
    s.add(title(24, y, "Grouping — a frame is one module; the badge counts its repetitions"))
    gy = y + 20
    s.add(
        Group("layer", Box(24, gy, 360, 92), "Encoder layer", "×6"),
        Node("att", at(120, gy + 56, 140, 36), C.ATTENTION, "Self-attention"),
        Node("ffn", at(292, gy + 56, 128, 36), C.LINEAR, "Feed-forward"),
        Edge("att", "ffn"),
        caption(410, gy + 46, 'label = specific name ("BatchNorm", not "layer")', TextAnchor.START),
        caption(410, gy + 68, "detail = key hyper-parameters only", TextAnchor.START),
    )
    return s


# --------------------------------------------------------------------------- CNN (flat)


def cnn(theme: Theme) -> Scene:
    """Small image classifier at level L2, block view, left to right in two rows."""
    s = Scene(840, 330, "CNN classifier — block view (L2)")
    xs = [76, 244, 412, 580, 748]
    r1, r2 = 70, 190
    s.add(
        Node("x", at(xs[0], r1, 96, 32), C.IO, "image"),
        Node("c1", at(xs[1], r1, 110, 46), C.CONV, "Conv 3×3", "32, ReLU"),
        Node("p1", at(xs[2], r1, 110, 46), C.RESAMPLE, "MaxPool 2×2", "stride 2"),
        Node("c2", at(xs[3], r1, 110, 46), C.CONV, "Conv 3×3", "64, ReLU"),
        Node("p2", at(xs[4], r1, 110, 46), C.RESAMPLE, "MaxPool 2×2", "stride 2"),
        Node("fl", at(xs[0], r2, 96, 36), C.RESHAPE, "Flatten"),
        Node("l1", at(xs[1], r2, 110, 46), C.LINEAR, "Linear", "4096→128, ReLU"),
        Node("dr", at(xs[2], r2, 110, 46), C.REGULARIZATION, "Dropout", "p = 0.5"),
        Node("l2", at(xs[3], r2, 110, 46), C.LINEAR, "Linear", "128→10"),
        Node("y", at(xs[4], r2, 96, 32), C.IO, "class logits"),
    )
    s.add(
        Edge("x", "c1", label="3×32×32"),
        Edge("c1", "p1", label="32×32×32"),
        Edge("p1", "c2", label="32×16×16"),
        Edge("c2", "p2", label="64×16×16"),
        Edge(
            "p2",
            "fl",
            exit=Side.BOTTOM,
            entry=Side.TOP,
            via=(Point(xs[4], 130), Point(xs[0], 130)),
            label="64×8×8",
        ),
        Edge("fl", "l1", label="4096"),
        Edge("l1", "dr", label="128"),
        Edge("dr", "l2", label="128"),
        Edge("l2", "y", label="10"),
    )
    add_legend(s, theme, Point(24, 250), columns=4)
    return s


# --------------------------------------------------------------------------- CNN (volume)


def cnn_volume(theme: Theme) -> Scene:
    """The same CNN, tensor-centric: nodes are tensors, operations sit on the arrows."""
    s = Scene(1000, 300, "CNN classifier — tensor-volume view")
    cy = 120
    # (id, category, channels-width, spatial-height, depth, shape label)
    tensors = [
        ("t0", C.IO, 6, 120, 60, "3×32×32"),
        ("t1", C.CONV, 16, 120, 60, "32×32×32"),
        ("t2", C.RESAMPLE, 16, 64, 32, "32×16×16"),
        ("t3", C.CONV, 28, 64, 32, "64×16×16"),
        ("t4", C.RESAMPLE, 28, 34, 17, "64×8×8"),
        ("t5", C.RESHAPE, 10, 170, 6, "4096"),
        ("t6", C.LINEAR, 10, 70, 6, "128"),
        ("t7", C.IO, 10, 26, 6, "10"),
    ]
    ops = ["conv 3×3", "max-pool", "conv 3×3", "max-pool", "flatten", "linear", "linear"]
    x = 30.0
    for tid, cat, w, h, d, label in tensors:
        s.add(Node(tid, Box(x, cy - h / 2, w, h), cat, "", label, shape=Shape.CUBOID, depth=d))
        x += w + d * 0.82 + 78
    for (a, *_), (b, *_), op in zip(tensors, tensors[1:], ops, strict=False):
        s.add(Edge(a, b, label=op))
    s.add(
        caption(
            500,
            232,
            "glyph height ∝ spatial size · thickness ∝ log(channels) · "
            "operations on arrows (activations folded into the operation)",
        )
    )
    add_legend(s, theme, Point(24, 254), columns=5)
    return s


# --------------------------------------------------------------------------- VAE


def vae(theme: Theme) -> Scene:
    s = Scene(880, 440, "Variational autoencoder (L1 with reparameterisation)")
    cy = 150
    s.add(
        Node("x", at(50, cy, 56, 32), C.IO, "x"),
        Node(
            "enc",
            at(170, cy, 124, 132),
            C.CONV,
            "Encoder",
            "4× conv, s2",
            shape=Shape.TRAPEZOID_NARROW,
        ),
        Node("mu", at(312, 112, 52, 30), C.TENSOR, "μ"),
        Node("sigma", at(312, 188, 52, 30), C.TENSOR, "σ"),
        Node("eps", at(400, 262, 104, 30), C.STOCHASTIC, "ε ~ N(0, I)"),
        Node("mul", at(400, 188, 24, 24), C.MERGE, "×"),
        Node("add", at(470, cy, 24, 24), C.MERGE, "+"),
        Node("z", at(548, cy, 52, 30), C.TENSOR, "z"),
        Node(
            "dec",
            at(686, cy, 124, 132),
            C.CONV,
            "Decoder",
            "4× deconv, s2",
            shape=Shape.TRAPEZOID_WIDEN,
        ),
        Node("xr", at(818, cy, 56, 32), C.IO, "x̂"),
        Node("kl", at(470, 46, 132, 44), C.LOSS, "KL divergence", "KL(q(z|x) ‖ N(0, I))"),
        Node("rec", at(818, 300, 120, 44), C.LOSS, "Reconstruction", "‖x − x̂‖²"),
    )
    s.add(
        Edge("x", "enc"),
        Edge("enc", "mu"),
        Edge("enc", "sigma"),
        Edge("mu", "add", exit=Side.RIGHT, entry=Side.TOP),
        Edge("sigma", "mul"),
        Edge("eps", "mul", K.STOCHASTIC),
        Edge("mul", "add", exit=Side.RIGHT, entry=Side.BOTTOM),
        Edge("add", "z"),
        Edge("z", "dec", label="32"),
        Edge("dec", "xr"),
        Edge("mu", "kl", K.TRAINING, exit=Side.TOP, entry=Side.LEFT),
        Edge("xr", "rec", K.TRAINING),
        Edge("x", "rec", K.TRAINING, exit=Side.BOTTOM, via=(Point(50, 300),)),
        caption(500, 196, "z = μ + σ ⊙ ε", TextAnchor.START),
    )
    add_legend(s, theme, Point(24, 346), columns=4)
    return s


# --------------------------------------------------------------------------- RNN


def rnn(theme: Theme) -> Scene:
    s = Scene(760, 380, "Recurrent network — folded and unrolled")
    cy = 150
    s.add(
        Node("cell", at(100, cy, 96, 44), C.RECURRENT, "RNN cell", "tanh, 128"),
        Node("x", at(100, 250, 56, 30), C.IO, "x_{t}"),
        Node("h", at(100, 50, 56, 30), C.TENSOR, "h_{t}"),
        Edge("x", "cell"),
        Edge("cell", "h"),
        Edge(
            Point(148, cy - 8),
            Point(148, cy + 8),
            K.STATE,
            via=(Point(184, cy - 8), Point(184, cy + 8)),
            delay=True,
        ),
        Note(Point(232, cy), "=", role=TextRole.SYMBOL),
    )
    xs = [330, 470, 610]
    subs = ["t-1", "t", "t+1"]
    for i, (x, sub) in enumerate(zip(xs, subs, strict=True)):
        s.add(
            Node(f"c{i}", at(x, cy, 96, 44), C.RECURRENT, "RNN cell"),
            Node(f"x{i}", at(x, 250, 64, 30), C.IO, f"x_{{{sub}}}"),
            Node(f"h{i}", at(x, 50, 64, 30), C.TENSOR, f"h_{{{sub}}}"),
            Edge(f"x{i}", f"c{i}"),
            Edge(f"c{i}", f"h{i}"),
        )
    s.add(
        Edge(Point(262, cy), "c0", K.STATE),
        Edge("c0", "c1", K.STATE, label="h_{t-1}"),
        Edge("c1", "c2", K.STATE, label="h_{t}"),
        Edge("c2", Point(700, cy), K.STATE),
        caption(470, 300, "time →  (one cell, shared weights, unrolled over time)"),
    )
    add_legend(s, theme, Point(24, 326), columns=3)
    return s


# --------------------------------------------------------------------------- LSTM cell


def lstm_cell(theme: Theme) -> Scene:
    """Gate-level view (L4) in the Olah layout, drawn with the baseline vocabulary."""
    s = Scene(960, 450, "LSTM cell — gate-level view (L4)")
    top, bus, gate_y = 84, 262, 200
    s.add(Group("cell", Box(60, 40, 590, 250), "LSTM cell"))
    s.add(
        Node("mf", at(170, top, 24, 24), C.MERGE, "×"),
        Node("add", at(290, top, 24, 24), C.MERGE, "+"),
        Node("cat", at(120, bus, 24, 24), C.MERGE, "‖"),
        Node("x", at(120, 340, 56, 30), C.IO, "x_{t}"),
        Node("f", at(170, gate_y, 44, 28), C.LINEAR, "σ"),
        Node("i", at(250, gate_y, 44, 28), C.LINEAR, "σ"),
        Node("g", at(330, gate_y, 52, 28), C.LINEAR, "tanh"),
        Node("o", at(450, gate_y, 44, 28), C.LINEAR, "σ"),
        Node("mi", at(290, 140, 24, 24), C.MERGE, "×"),
        Node("tc", at(540, 136, 52, 28), C.ACTIVATION, "tanh"),
        Node("mo", at(540, gate_y, 24, 24), C.MERGE, "×"),
    )
    for gate in "figo":
        s.add(Edge("cat", gate, exit=Side.RIGHT, entry=Side.BOTTOM))
    s.add(
        Edge(Point(20, top), "mf", K.STATE),
        Edge("mf", "add", K.STATE),
        Edge("add", Point(690, top), K.STATE),
        Edge(Point(20, bus), "cat", K.STATE),
        Edge("x", "cat"),
        Edge("f", "mf"),
        Edge("i", "mi", exit=Side.TOP, entry=Side.LEFT),
        Edge("g", "mi", exit=Side.TOP, entry=Side.RIGHT),
        Edge("mi", "add"),
        Edge(Point(540, top), "tc"),
        Edge("tc", "mo"),
        Edge("o", "mo"),
        Edge("mo", Point(690, bus), K.STATE, exit=Side.BOTTOM, via=(Point(540, bus),)),
        Edge("mo", Point(610, 20), exit=Side.RIGHT, via=(Point(610, gate_y),)),
    )
    for gate, name in [("f", "f_{t}"), ("i", "i_{t}"), ("g", "g_{t}"), ("o", "o_{t}")]:
        s.add(caption(s.node(gate).box.x - 5, gate_y, name, TextAnchor.END))
    s.add(
        caption(20, top - 14, "c_{t-1}", TextAnchor.START),
        caption(690, top - 14, "c_{t}", TextAnchor.END),
        caption(20, bus - 14, "h_{t-1}", TextAnchor.START),
        caption(690, bus - 14, "h_{t}", TextAnchor.END),
        caption(620, 26, "h_{t}", TextAnchor.START),
    )
    equations = [
        "f_{t} = σ(W_{f} [h_{t-1}, x_{t}] + b_{f})",
        "i_{t} = σ(W_{i} [h_{t-1}, x_{t}] + b_{i})",
        "g_{t} = tanh(W_{g} [h_{t-1}, x_{t}] + b_{g})",
        "o_{t} = σ(W_{o} [h_{t-1}, x_{t}] + b_{o})",
        "c_{t} = f_{t} ⊙ c_{t-1} + i_{t} ⊙ g_{t}",
        "h_{t} = o_{t} ⊙ tanh(c_{t})",
    ]
    for i, eq in enumerate(equations):
        s.add(caption(720, 100 + i * 28, eq, TextAnchor.START))
    add_legend(s, theme, Point(24, 384), columns=3)
    return s


# --------------------------------------------------------------------------- ConvLSTM


def convlstm(theme: Theme) -> Scene:
    """Encoding–forecasting ConvLSTM (Shi et al. 2015): time × depth grid with 3-D inputs."""
    s = Scene(840, 470, "ConvLSTM encoder–forecaster")
    enc_x, fc_x = [110, 250, 390], [590, 730]
    l1, l2 = 270, 180
    s.add(
        Group("enc", Box(44, 128, 412, 170), "Encoding network"),
        Group("fc", Box(524, 128, 272, 170), "Forecasting network"),
    )
    for t, x in enumerate(enc_x, start=1):
        s.add(
            Node(
                f"X{t}", Box(x - 9, 336, 18, 40), C.IO, f"X_{{{t}}}", shape=Shape.CUBOID, depth=26
            ),
            Node(f"e1_{t}", at(x, l1, 100, 44), C.RECURRENT, "ConvLSTM", "64 ch, 3×3"),
            Node(f"e2_{t}", at(x, l2, 100, 44), C.RECURRENT, "ConvLSTM", "64 ch, 3×3"),
            Edge(f"X{t}", f"e1_{t}", label="1×64×64" if t == 1 else ""),
            Edge(f"e1_{t}", f"e2_{t}"),
        )
    for t, x in enumerate(fc_x, start=len(enc_x) + 1):
        s.add(
            Node(f"f1_{t}", at(x, l1, 100, 44), C.RECURRENT, "ConvLSTM", "64 ch, 3×3"),
            Node(f"f2_{t}", at(x, l2, 100, 44), C.RECURRENT, "ConvLSTM", "64 ch, 3×3"),
            Node(f"o{t}", at(x, 92, 112, 40), C.CONV, "Conv 1×1", "on ‖ of all states"),
            Node(f"Y{t}", at(x, 30, 56, 30), C.IO, f"Ŷ_{{{t}}}"),
            Edge(f"f1_{t}", f"f2_{t}"),
            Edge(f"f2_{t}", f"o{t}"),
            Edge(f"o{t}", f"Y{t}"),
        )
    for layer in ("1", "2"):
        for a, b in zip(enc_x, enc_x[1:], strict=False):
            ta, tb = enc_x.index(a) + 1, enc_x.index(b) + 1
            s.add(Edge(f"e{layer}_{ta}", f"e{layer}_{tb}", K.STATE))
        s.add(Edge(f"e{layer}_3", f"f{layer}_4", K.STATE, label="copy" if layer == "2" else ""))
        s.add(Edge(f"f{layer}_4", f"f{layer}_5", K.STATE))
    s.add(caption(660, 356, "time →"))
    add_legend(s, theme, Point(24, 420), columns=5)
    return s


# --------------------------------------------------------------------------- Transformer


def transformer(theme: Theme) -> Scene:
    """Transformer encoder, vertical stack (top to bottom) with residual connections."""
    s = Scene(700, 720, "Transformer encoder")
    cx, rx = 200, 360  # main axis and residual lane
    s.add(Group("layer", Box(40, 196, 360, 418), "Encoder layer", "×6"))
    s.add(
        Node("tok", at(cx, 30, 96, 30), C.IO, "tokens"),
        Node("emb", at(cx, 96, 170, 44), C.EMBEDDING, "Token embedding", "d = 512"),
        Node("pos", at(rx + 120, 156, 150, 44), C.EMBEDDING, "Positional enc.", "sinusoidal"),
        Node("a0", at(cx, 156, 24, 24), C.MERGE, "+"),
        Node("mha", at(cx, 270, 190, 46), C.ATTENTION, "Multi-head attention", "8 heads"),
        Node("a1", at(cx, 336, 24, 24), C.MERGE, "+"),
        Node("n1", at(cx, 392, 150, 34), C.NORM, "LayerNorm"),
        Node("ffn", at(cx, 466, 190, 46), C.LINEAR, "Feed-forward", "512→2048→512, ReLU"),
        Node("a2", at(cx, 532, 24, 24), C.MERGE, "+"),
        Node("n2", at(cx, 584, 150, 34), C.NORM, "LayerNorm"),
        Node("out", at(cx, 676, 140, 30), C.IO, "encoded tokens"),
    )
    s.add(
        Edge("tok", "emb", label="n"),
        Edge("emb", "a0"),
        Edge("pos", "a0"),
        Edge("a0", "mha", label="n×512", label_at=Point(cx + 34, 183)),
        Edge("mha", "a1"),
        Edge(Point(cx, 222), "a1", via=(Point(rx, 222), Point(rx, 336)), entry=Side.RIGHT),
        Edge("a1", "n1"),
        Edge("n1", "ffn"),
        Edge("ffn", "a2"),
        Edge(Point(cx, 425), "a2", via=(Point(rx, 425), Point(rx, 532)), entry=Side.RIGHT),
        Edge("a2", "n2"),
        Edge("n2", "out", label="n×512"),
        caption(rx + 8, 300, "residual", TextAnchor.START),
        caption(rx + 8, 496, "residual", TextAnchor.START),
    )
    add_legend(s, theme, Point(440, 300))
    return s


SPECIMENS: dict[str, Callable[[Theme], Scene]] = {
    "vocabulary": vocabulary,
    "cnn": cnn,
    "cnn-volume": cnn_volume,
    "vae": vae,
    "rnn": rnn,
    "lstm-cell": lstm_cell,
    "convlstm": convlstm,
    "transformer": transformer,
}


def build(out_dir: Path = OUT_DIR, theme: Theme | None = None) -> list[Path]:
    theme = theme or load_theme("default")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, make in SPECIMENS.items():
        path = out_dir / f"{name}.svg"
        path.write_text(render_svg(make(theme), theme), encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR
    for p in build(target):
        print(p)
