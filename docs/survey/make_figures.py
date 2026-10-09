"""Schematic re-drawings of published diagram styles for the literature survey.

The figures are *sketches* of each style (not copies of the original artwork) so the report can
show the visual conventions side by side. Run via ``build.py`` or directly::

    uv run python docs/survey/make_figures.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

OUT_DIR = Path(__file__).parent / "figures"

INK = "#2d3748"
MUTED = "#718096"
FS = 6.5  # base font size (pt) inside panels

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": FS,
        "svg.fonttype": "path",  # text as outlines -> renders identically everywhere
        "axes.titlesize": 7.5,
        "axes.titleweight": "bold",
    }
)


# --------------------------------------------------------------------------- primitives


def panel(ax, title: str, xlim=(0, 10), ylim=(0, 7)) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", pad=3, color=INK)


def box(ax, x, y, w, h, text="", fc="white", ec=INK, fs=FS, lw=0.8, round_=0.15, tc=INK, **kw):
    """Box with lower-left corner at (x, y)."""
    style = f"round,pad=0,rounding_size={round_}" if round_ else "square,pad=0"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, **kw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, p0, p1, color=INK, lw=0.8, ls="-", rad=0.0, head=5, shrink=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=f"-|>,head_length={head / 10},head_width={head / 18}",
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            lw=lw,
            ls=ls,
            shrinkA=shrink,
            shrinkB=shrink,
            mutation_scale=10,
        )
    )


def poly_arrow(ax, pts, color=INK, lw=0.8, ls="-"):
    """Orthogonal poly-line ending in an arrow head."""
    xs, ys = zip(*pts, strict=True)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls)


def circ(ax, x, y, r, text="", fc="white", ec=INK, fs=FS, lw=0.8, tc=INK):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc)


def shade(hex_color: str, factor: float) -> str:
    """Darken (<1) or lighten (>1) a hex colour."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    if factor < 1:
        r, g, b = (int(c * factor) for c in (r, g, b))
    else:
        r, g, b = (int(c + (255 - c) * (factor - 1)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def cuboid(ax, x, y, w, h, d, fc, ec=INK, lw=0.5, angle=40):
    """3-D box: front face (x, y, w, h) plus an oblique depth ``d``."""
    dx, dy = d * math.cos(math.radians(angle)), d * math.sin(math.radians(angle))
    front = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    top = [(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)]
    side = [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)]
    ax.add_patch(Polygon(side, closed=True, fc=shade(fc, 0.78), ec=ec, lw=lw))
    ax.add_patch(Polygon(top, closed=True, fc=shade(fc, 1.25), ec=ec, lw=lw))
    ax.add_patch(Polygon(front, closed=True, fc=fc, ec=ec, lw=lw))


def trapezoid(ax, x, y, w, h_left, h_right, fc, ec=INK, text="", fs=FS, lw=0.8):
    cy = y
    pts = [
        (x, cy - h_left / 2),
        (x + w, cy - h_right / 2),
        (x + w, cy + h_right / 2),
        (x, cy + h_left / 2),
    ]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, cy, text, ha="center", va="center", fontsize=fs, color=INK)


# --------------------------------------------------------------------------- figure 1 panels


def neuron_graph(ax):
    panel(ax, "(a) Neuron graph")
    layers = [3, 4, 4, 2]
    colors = ["#9ae6b4", "#90cdf4", "#90cdf4", "#feb2b2"]
    xs = [1.5, 4.0, 6.5, 9.0]
    pos = []
    for n, x in zip(layers, xs, strict=True):
        span = (n - 1) * 1.3
        pos.append([(x, 3.4 - span / 2 + i * 1.3) for i in range(n)])
    for a, b in zip(pos, pos[1:], strict=False):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#a0aec0", lw=0.4, zorder=1)
    for layer, col in zip(pos, colors, strict=True):
        for x, y in layer:
            ax.add_patch(Circle((x, y), 0.42, fc=col, ec=INK, lw=0.6, zorder=2))
    for x, t in zip(xs, ["input", "hidden", "hidden", "output"], strict=True):
        ax.text(x, 0.2, t, ha="center", fontsize=FS - 0.5, color=MUTED)


def stacked_planes(ax):
    panel(ax, "(b) Feature-map planes (LeNet)")
    specs = [  # x, size, count, label
        (0.1, 2.3, 1, "input"),
        (2.7, 1.9, 4, "C1"),
        (5.4, 1.25, 4, "S2"),
        (7.4, 0.8, 5, "C3"),
    ]
    for x, s, n, label in specs:
        for i in range(n):
            off = i * 0.18
            ax.add_patch(
                Rectangle((x + off, 2.2 + off), s, s, fc="#ebf4ff", ec=INK, lw=0.5, zorder=2 + i)
            )
        ax.text(x + s / 2 + (n - 1) * 0.09, 1.3, label, ha="center", fontsize=FS - 0.5)
    for i in range(6):  # fully connected column
        circ(ax, 9.65, 1.9 + i * 0.55, 0.2, fc="#feebc8", lw=0.5)
    ax.text(9.65, 1.3 - 0.0, "FC", ha="center", fontsize=FS - 0.5)
    for x0, y0, x1 in [(2.4, 3.3, 2.7), (4.6, 3.3, 5.4), (6.9, 3.0, 7.4)]:
        ax.plot([x0, x1], [y0, y0 + 0.2], color=MUTED, lw=0.4, zorder=1)


def tensor_volumes(ax):
    panel(ax, "(c) Tensor volumes (AlexNet)")
    blue, orange = "#90cdf4", "#fbd38d"
    specs = [  # x, w(depth on page), h, d(oblique), colour, label
        (0.3, 0.25, 3.4, 1.8, "#e2e8f0", "3×224²"),
        (2.2, 0.6, 2.8, 1.5, blue, "64×112²"),
        (4.1, 1.0, 2.1, 1.1, blue, "128×56²"),
        (6.2, 1.4, 1.4, 0.75, blue, "256×28²"),
        (8.9, 0.35, 3.0, 0.0, orange, "FC"),
    ]
    for x, w, h, d, col, label in specs:
        y = 3.4 - h / 2
        if d:
            cuboid(ax, x, y, w, h, d, col)
        else:
            box(ax, x, y, w, h, fc=col, round_=0, lw=0.5)
        dx = d * math.cos(math.radians(40))
        ax.text(x + (w + dx) / 2, 0.6, label, ha="center", fontsize=FS - 1.2)


def block_flowchart(ax):
    panel(ax, "(d) Block flowchart (Transformer)")
    frame = FancyBboxPatch(
        (1.6, 0.55), 5.6, 5.85, boxstyle="round,pad=0,rounding_size=0.3", fc="#f7fafc", ec=MUTED,
        lw=0.6,
    )
    ax.add_patch(frame)
    ax.text(0.9, 3.5, "N×", ha="center", va="center", fontsize=FS + 1, fontweight="bold")
    rows = [  # y, h, text, colour
        (0.95, 1.05, "Multi-Head\nAttention", "#fbd38d"),
        (2.5, 0.7, "Add & Norm", "#faf089"),
        (3.65, 0.7, "Feed Forward", "#90cdf4"),
        (5.15, 0.7, "Add & Norm", "#faf089"),
    ]
    for y, h, t, c in rows:
        box(ax, 2.4, y, 4.0, h, t, fc=c, fs=FS - 0.8)
    arrow(ax, (4.4, 0.0), (4.4, 0.95))
    arrow(ax, (4.4, 2.0), (4.4, 2.5))
    arrow(ax, (4.4, 3.2), (4.4, 3.65))
    arrow(ax, (4.4, 4.35), (4.4, 5.15))
    arrow(ax, (4.4, 5.85), (4.4, 7.0))
    poly_arrow(ax, [(4.4, 0.75), (6.9, 0.75), (6.9, 2.85), (6.4, 2.85)])
    poly_arrow(ax, [(4.4, 3.42), (6.9, 3.42), (6.9, 5.5), (6.4, 5.5)])


def op_graph(ax):
    panel(ax, "(e) Operator graph (Netron)")

    def node(x, y, name, attrs, head):
        w = 3.0
        h = 0.55 + 0.42 * len(attrs)
        box(ax, x, y, w, h, fc="white", ec=INK, round_=0.12, lw=0.6)
        ax.add_patch(
            FancyBboxPatch(
                (x, y + h - 0.55), w, 0.55, boxstyle="round,pad=0,rounding_size=0.12", fc=head,
                ec=INK, lw=0.6,
            )
        )
        ax.text(x + 0.15, y + h - 0.28, name, va="center", fontsize=FS - 0.5, color="white",
                fontweight="bold")
        for i, a in enumerate(attrs):
            ax.text(x + 0.15, y + h - 0.78 - 0.42 * i, a, va="center", fontsize=FS - 1.5,
                    color=INK, family="DejaVu Sans Mono")
        return x + w / 2, y, y + h

    cx, b1, t1 = node(0.6, 4.55, "Conv", ["W <64×3×3×3>", "B <64>"], "#3182ce")
    _, b2, t2 = node(0.6, 2.4, "BatchNorm", ["scale <64>"], "#4a5568")
    _, b3, t3 = node(0.6, 0.4, "Relu", [], "#c53030")
    _, b4, t4 = node(5.9, 0.4, "Add", [], "#2d3748")
    arrow(ax, (cx, 7.0), (cx, t1))
    arrow(ax, (cx, b1), (cx, t2))
    arrow(ax, (cx, b2), (cx, t3))
    arrow(ax, (3.6, 0.68), (5.9, 0.68))
    poly_arrow(ax, [(cx, 6.6), (7.4, 6.6), (7.4, t4)])
    ax.text(2.2, 4.25, "1×64×56×56", fontsize=FS - 1.8, color=MUTED, family="DejaVu Sans Mono")
    ax.text(7.55, 4.0, "1×64×56×56", fontsize=FS - 1.8, color=MUTED, rotation=90,
            family="DejaVu Sans Mono")


def lstm_cell(ax):
    panel(ax, "(f) Cell internals (LSTM)")
    box(ax, 0.9, 0.7, 8.2, 5.4, fc="#f7fafc", ec=MUTED, round_=0.4, lw=0.6)
    yellow, pink = "#faf089", "#fbb6ce"
    top, bot = 5.4, 1.5
    ax.plot([0, 10], [top, top], color=INK, lw=1.0)  # cell state highway
    ax.text(0.05, top + 0.3, "c$_{t-1}$", fontsize=FS)
    ax.text(9.3, top + 0.3, "c$_t$", fontsize=FS)
    ax.plot([0, 1.8], [bot, bot], color=INK, lw=0.8)
    ax.text(0.05, bot - 0.55, "h$_{t-1}$", fontsize=FS)
    ax.plot([1.8, 6.3], [bot, bot], color=INK, lw=0.8)
    gates = [(1.7, "σ"), (3.0, "σ"), (4.3, "tanh"), (5.9, "σ")]
    for x, t in gates:
        box(ax, x - 0.5, 2.25, 1.0, 0.75, t, fc=yellow, fs=FS - 0.3, round_=0.1)
        ax.plot([x, x], [bot, 2.25], color=INK, lw=0.6)
    circ(ax, 1.7, top, 0.33, "×", fc=pink)
    circ(ax, 4.3, top, 0.33, "+", fc=pink)
    circ(ax, 3.65, 3.85, 0.33, "×", fc=pink)
    circ(ax, 7.4, 2.6, 0.33, "×", fc=pink)
    box(ax, 6.85, 3.75, 1.1, 0.6, "tanh", fc=pink, fs=FS - 1, round_=0.25)
    arrow(ax, (1.7, 3.0), (1.7, top - 0.33))  # forget
    poly_arrow(ax, [(3.0, 3.0), (3.0, 3.85), (3.32, 3.85)])  # input
    poly_arrow(ax, [(4.3, 3.0), (4.3, 3.6), (3.95, 3.75)])
    arrow(ax, (3.65, 4.18), (4.15, top - 0.3))
    arrow(ax, (6.4, 2.6), (7.07, 2.6))  # output gate
    arrow(ax, (7.4, top), (7.4, 4.35))
    arrow(ax, (7.4, 3.75), (7.4, 2.93))
    poly_arrow(ax, [(7.73, 2.6), (8.5, 2.6), (8.5, 1.5), (10, 1.5)])
    arrow(ax, (8.5, 2.6), (8.5, 7.0))
    ax.text(8.65, 6.3, "h$_t$", fontsize=FS)
    ax.text(2.0, 0.15, "x$_t$", fontsize=FS)
    arrow(ax, (1.75, 0.0), (1.75, bot - 0.02))


def unrolled_rnn(ax):
    panel(ax, "(g) Folded / unrolled recurrence")
    green = "#c6f6d5"
    box(ax, 0.8, 2.8, 1.3, 1.2, "A", fc=green)
    arrow(ax, (1.45, 1.4), (1.45, 2.8))
    arrow(ax, (1.45, 4.0), (1.45, 5.4))
    ax.text(1.45, 0.95, "x", ha="center", fontsize=FS)
    ax.text(1.45, 5.6, "h", ha="center", fontsize=FS)
    loop = [(2.1, 3.4), (2.7, 3.4), (2.7, 4.6), (0.3, 4.6), (0.3, 3.4)]
    ax.plot(*zip(*loop, strict=True), color=INK, lw=0.8)
    arrow(ax, (0.3, 3.4), (0.8, 3.4))
    ax.add_patch(Rectangle((2.55, 3.85), 0.3, 0.3, fc=INK, ec=INK))  # one-step delay
    ax.text(3.45, 3.4, "=", fontsize=FS + 4, ha="center", va="center")
    for i, x in enumerate([4.2, 6.1, 8.0]):
        box(ax, x, 2.8, 1.3, 1.2, "A", fc=green)
        arrow(ax, (x + 0.65, 1.4), (x + 0.65, 2.8))
        arrow(ax, (x + 0.65, 4.0), (x + 0.65, 5.4))
        sub = ["t−1", "t", "t+1"][i]
        ax.text(x + 0.65, 0.95, f"x$_{{{sub}}}$", ha="center", fontsize=FS)
        ax.text(x + 0.65, 5.6, f"h$_{{{sub}}}$", ha="center", fontsize=FS)
        if i < 2:
            arrow(ax, (x + 1.3, 3.4), (x + 1.9, 3.4))


def hourglass(ax):
    panel(ax, "(h) Encoder–latent–decoder (VAE)")
    trapezoid(ax, 1.3, 3.4, 2.6, 4.4, 1.6, "#bee3f8", text="Encoder")
    trapezoid(ax, 6.1, 3.4, 2.6, 1.6, 4.4, "#c3dafe", text="Decoder")
    box(ax, 4.3, 3.7, 0.55, 0.5, "μ", fc="white", round_=0.08)
    box(ax, 4.3, 2.6, 0.55, 0.5, "σ", fc="white", round_=0.08)
    circ(ax, 5.55, 3.4, 0.38, "z", fc="#e9d8fd")
    ax.text(0.5, 3.4, "x", ha="center", va="center", fontsize=FS + 1)
    ax.text(9.5, 3.4, "x̂", ha="center", va="center", fontsize=FS + 1)
    arrow(ax, (0.7, 3.4), (1.3, 3.4))
    arrow(ax, (8.7, 3.4), (9.3, 3.4))
    arrow(ax, (3.9, 3.95), (4.3, 3.95))
    arrow(ax, (3.9, 2.85), (4.3, 2.85))
    arrow(ax, (4.85, 3.9), (5.25, 3.55))
    arrow(ax, (4.85, 2.9), (5.25, 3.25))
    arrow(ax, (5.93, 3.4), (6.1, 3.4))
    ax.text(5.55, 1.75, "ε ~ N(0, I)", ha="center", fontsize=FS - 1, color=MUTED)
    arrow(ax, (5.55, 2.0), (5.55, 3.02), ls=":", color=MUTED)


def circuit_diagram(ax):
    panel(ax, "(i) Tensor-axis circuit (Abbott)")
    ax.text(5.0, 6.2, "y = softmax(x W)", ha="center", fontsize=FS - 0.5, style="italic",
            color=MUTED)
    wire = dict(color=INK, lw=1.0)
    ax.plot([0.3, 9.7], [4.6, 4.6], **wire)  # batch axis passes through (broadcast)
    ax.plot([0.3, 2.6], [2.4, 2.4], **wire)
    ax.plot([4.2, 6.0], [2.4, 2.4], **wire)
    ax.plot([8.0, 9.7], [2.4, 2.4], **wire)
    box(ax, 2.6, 1.5, 1.6, 1.8, "W", fc="#c3dafe", round_=0.05)
    box(ax, 6.0, 1.7, 2.0, 1.4, "softmax", fc="#fed7e2", fs=FS - 0.8, round_=0.05)
    for x, y, lab in [(0.4, 4.8, "b"), (0.4, 2.6, "m"), (4.5, 2.6, "n"), (9.2, 2.6, "n"),
                      (9.2, 4.8, "b")]:
        ax.text(x, y, lab, fontsize=FS, color=MUTED, style="italic")
    ax.plot([0.2, 9.8], [3.6, 3.6], color=MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.text(5.0, 0.6, "wires = tensor axes, boxes = operations", ha="center", fontsize=FS - 1.3,
            color=MUTED)


def figure_styles(path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(6.9, 5.3))
    for fn, ax in zip(
        [neuron_graph, stacked_planes, tensor_volumes, block_flowchart, op_graph, lstm_cell,
         unrolled_rnn, hourglass, circuit_diagram],
        axes.flat,
        strict=True,
    ):
        fn(ax)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, wspace=0.08, hspace=0.22)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- figure 2 panels


def vae_detail(ax):
    panel(ax, "(a) VAE with reparameterisation", ylim=(0, 6))
    enc, dec = "#bee3f8", "#c3dafe"
    trapezoid(ax, 0.9, 3.4, 2.0, 3.2, 1.4, enc, text="q(z|x)", fs=FS - 0.5)
    box(ax, 3.4, 3.85, 0.7, 0.55, "μ", round_=0.08)
    box(ax, 3.4, 2.4, 0.7, 0.55, "σ", round_=0.08)
    circ(ax, 5.0, 3.4, 0.36, "+", fc="#fbb6ce")
    circ(ax, 4.6, 2.0, 0.3, "×", fc="#fbb6ce", fs=FS - 0.5)
    ax.text(4.6, 0.75, "ε ~ N(0,I)", ha="center", fontsize=FS - 1.2, color=MUTED)
    arrow(ax, (4.6, 1.0), (4.6, 1.7), color=MUTED, lw=0.6)
    circ(ax, 6.05, 3.4, 0.36, "z", fc="#e9d8fd")
    trapezoid(ax, 6.8, 3.4, 2.0, 1.4, 3.2, dec, text="p(x|z)", fs=FS - 0.5)
    ax.text(0.35, 3.4, "x", ha="center", va="center")
    ax.text(9.55, 3.4, "x̂", ha="center", va="center")
    arrow(ax, (0.55, 3.4), (0.9, 3.4))
    arrow(ax, (2.9, 4.1), (3.4, 4.1))
    arrow(ax, (2.9, 2.7), (3.4, 2.7))
    poly_arrow(ax, [(4.1, 4.1), (5.0, 4.1), (5.0, 3.76)])
    poly_arrow(ax, [(4.1, 2.7), (4.6, 2.7), (4.6, 2.3)])
    poly_arrow(ax, [(4.9, 2.0), (5.0, 2.0), (5.0, 3.04)])
    arrow(ax, (5.36, 3.4), (5.69, 3.4))
    arrow(ax, (6.41, 3.4), (6.8, 3.4))
    arrow(ax, (8.8, 3.4), (9.35, 3.4))
    ax.text(3.75, 5.35, "KL(q‖p)", ha="center", fontsize=FS - 1, color="#c53030")
    ax.plot([3.75, 3.75], [4.4, 5.1], color="#c53030", lw=0.6, ls="--")
    ax.text(9.0, 5.35, "‖x − x̂‖", ha="center", fontsize=FS - 1, color="#c53030")
    ax.plot([9.3, 9.1], [3.6, 5.1], color="#c53030", lw=0.6, ls="--")


def gan(ax):
    panel(ax, "(b) GAN", ylim=(0, 6))
    ax.text(0.4, 4.5, "z", ha="center", va="center")
    box(ax, 1.0, 4.0, 1.8, 1.0, "G", fc="#c6f6d5")
    box(ax, 5.3, 2.5, 1.8, 1.0, "D", fc="#fed7d7")
    box(ax, 3.2, 4.15, 1.4, 0.7, "x$_{fake}$", fc="white", fs=FS - 0.5)
    box(ax, 3.2, 1.15, 1.4, 0.7, "x$_{real}$", fc="white", fs=FS - 0.5)
    arrow(ax, (0.6, 4.5), (1.0, 4.5))
    arrow(ax, (2.8, 4.5), (3.2, 4.5))
    poly_arrow(ax, [(4.6, 4.5), (5.0, 4.5), (5.0, 3.2), (5.3, 3.2)])
    poly_arrow(ax, [(4.6, 1.5), (5.0, 1.5), (5.0, 2.8), (5.3, 2.8)])
    arrow(ax, (7.1, 3.0), (7.8, 3.0))
    ax.text(8.8, 3.0, "real /\nfake", ha="center", va="center", fontsize=FS - 0.5)
    poly_arrow(ax, [(8.8, 3.6), (8.8, 5.6), (1.9, 5.6), (1.9, 5.0)], color="#c53030", ls="--")
    ax.text(5.3, 5.75, "∇ adversarial loss", ha="center", fontsize=FS - 1.3, color="#c53030")


def unet(ax):
    panel(ax, "(c) U-Net", ylim=(0, 6))
    blue, green = "#90cdf4", "#9ae6b4"
    levels = [(5.0, 0.3, 1.1), (3.75, 0.45, 0.8), (2.5, 0.65, 0.55)]  # y, half-width, height
    for i, (y, w, h) in enumerate(levels):
        xl, xr = 0.5 + i * 1.0, 9.5 - i * 1.0 - 2 * w
        box(ax, xl, y - h / 2, 2 * w, h, fc=blue, round_=0.02, lw=0.5)
        box(ax, xr, y - h / 2, 2 * w, h, fc=green, round_=0.02, lw=0.5)
        arrow(ax, (xl + 2 * w, y), (xr, y), color=MUTED, ls="--", lw=0.6)
        if i < len(levels) - 1:
            ny, nw, nh = levels[i + 1]
            arrow(ax, (xl + w, y - h / 2), (0.5 + (i + 1) * 1.0 + nw, ny + nh / 2),
                  color="#c53030")
            arrow(ax, (9.5 - (i + 1) * 1.0 - nw, ny + nh / 2), (xr + w, y - h / 2),
                  color="#2f855a")
    bx, bw, bh, by = 3.9, 2.2, 0.4, 1.0  # bottleneck
    box(ax, bx, by, bw, bh, fc=blue, round_=0.02, lw=0.5)
    y, w, h = levels[-1]
    arrow(ax, (2.5 + w, y - h / 2), (bx + 0.4, by + bh), color="#c53030")
    arrow(ax, (bx + bw - 0.4, by + bh), (7.5 - w, y - h / 2), color="#2f855a")
    ax.text(5.0, 5.55, "copy & concatenate", ha="center", fontsize=FS - 1.3, color=MUTED)
    ax.text(0.4, 0.3, "↓ max-pool", fontsize=FS - 1.3, color="#c53030")
    ax.text(7.6, 0.3, "↑ up-conv", fontsize=FS - 1.3, color="#2f855a")


def convlstm(ax):
    panel(ax, "(d) ConvLSTM encoder–forecaster", ylim=(0, 6))
    green = "#c6f6d5"
    for t, x in enumerate([0.4, 2.3]):
        cuboid(ax, x, 0.35, 0.9, 0.9, 0.5, "#e2e8f0", lw=0.4)
        ax.text(x + 0.5, 0.0, f"X$_{t + 1}$", ha="center", fontsize=FS - 1.3)
    for row, y in enumerate([1.9, 3.6]):
        for col, x in enumerate([0.4, 2.3, 5.6, 7.5]):
            cuboid(ax, x, y, 0.9, 0.9, 0.5, green, lw=0.4)
            if col in (0, 2):
                arrow(ax, (x + 1.25, y + 0.55), (x + 1.9, y + 0.55), color="#2f855a")
            if row == 0 and col < 2:
                arrow(ax, (x + 0.5, 1.55), (x + 0.5, 1.9))
            if row == 0:
                arrow(ax, (x + 0.5, 3.1), (x + 0.5, 3.6))
    arrow(ax, (3.65, 2.45), (5.6, 2.45), color=MUTED, ls="--", lw=0.6)
    arrow(ax, (3.65, 4.15), (5.6, 4.15), color=MUTED, ls="--", lw=0.6)
    ax.text(4.6, 3.1, "copy\nstates", ha="center", va="center", fontsize=FS - 1.5, color=MUTED)
    for t, x in enumerate([5.6, 7.5]):
        arrow(ax, (x + 0.5, 4.95), (x + 0.5, 5.4))
        ax.text(x + 0.5, 5.55, f"Ŷ$_{t + 3}$", ha="center", fontsize=FS - 1.3)
    ax.text(1.85, 5.3, "encoding", ha="center", fontsize=FS - 0.8, color=INK)
    ax.text(9.65, 3.0, "forecasting", ha="center", va="center", fontsize=FS - 0.8, rotation=90,
            color=INK)


def resnet_block(ax):
    panel(ax, "(e) Residual block", ylim=(0, 6))
    box(ax, 3.0, 4.2, 3.2, 0.7, "3×3 conv, 64", fc="#90cdf4")
    box(ax, 3.0, 2.6, 3.2, 0.7, "3×3 conv, 64", fc="#90cdf4")
    circ(ax, 4.6, 1.3, 0.35, "+", fc="white")
    arrow(ax, (4.6, 5.9), (4.6, 4.9))
    arrow(ax, (4.6, 4.2), (4.6, 3.3))
    ax.text(4.75, 3.65, "ReLU", fontsize=FS - 1.3, color=MUTED)
    arrow(ax, (4.6, 2.6), (4.6, 1.65))
    arrow(ax, (4.6, 0.95), (4.6, 0.1))
    ax.text(4.75, 0.35, "ReLU", fontsize=FS - 1.3, color=MUTED)
    poly_arrow(ax, [(4.6, 5.5), (7.6, 5.5), (7.6, 1.3), (4.95, 1.3)])
    ax.text(7.75, 3.4, "identity", fontsize=FS - 1, color=INK, rotation=90, va="center")
    ax.text(2.1, 5.55, "x", fontsize=FS)
    ax.text(1.0, 0.35, "F(x) + x", fontsize=FS)


def seq2seq(ax):
    panel(ax, "(f) Seq2seq with attention", ylim=(0, 6))
    green, blue = "#c6f6d5", "#c3dafe"
    enc_x = [0.4, 1.7, 3.0]
    dec_x = [5.8, 7.1, 8.4]
    for x in enc_x:
        box(ax, x, 1.2, 1.0, 0.9, "", fc=green)
        arrow(ax, (x + 0.5, 0.45), (x + 0.5, 1.2))
    for x in dec_x:
        box(ax, x, 1.2, 1.0, 0.9, "", fc=blue)
        arrow(ax, (x + 0.5, 2.1), (x + 0.5, 2.8))
    for xs in (enc_x, dec_x):
        for a, b in zip(xs, xs[1:], strict=False):
            arrow(ax, (a + 1.0, 1.65), (b, 1.65))
    arrow(ax, (4.0, 1.65), (5.8, 1.65), color=MUTED, ls="--")
    for x, t in zip(enc_x, ["je", "suis", "là"], strict=True):
        ax.text(x + 0.5, 0.15, t, ha="center", fontsize=FS - 1)
    for x, t in zip(dec_x, ["I", "am", "here"], strict=True):
        ax.text(x + 0.5, 3.0, t, ha="center", fontsize=FS - 1)
    box(ax, 3.2, 4.3, 2.2, 0.75, "attention", fc="#fbd38d", fs=FS - 0.5)
    for x in enc_x:
        ax.plot([x + 0.5, 3.9], [2.1, 4.3], color="#dd6b20", lw=0.5, alpha=0.8)
    poly_arrow(ax, [(5.4, 4.67), (7.2, 4.67), (7.2, 2.1)], color="#dd6b20")
    ax.text(6.8, 4.85, "context c$_t$", fontsize=FS - 1.3, color="#dd6b20")


def figure_families(path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.15))
    for fn, ax in zip([vae_detail, gan, unet, convlstm, resnet_block, seq2seq], axes.flat,
                      strict=True):
        fn(ax)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.08, hspace=0.25)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- figure 3


def figure_levels(path: Path) -> None:
    """One residual network shown at four levels of detail (L0-L3)."""
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.75), gridspec_kw={"width_ratios": [1, 1.25,
                                                                                       1.25, 1.4]})
    conv, norm, act, io = "#90cdf4", "#faf089", "#fbb6ce", "#e2e8f0"

    ax = axes[0]
    panel(ax, "L0  system", xlim=(0, 6), ylim=(0, 6))
    box(ax, 0.9, 4.8, 4.2, 0.8, "image", fc=io)
    box(ax, 0.9, 2.6, 4.2, 1.2, "ResNet-18", fc="#edf2f7", fs=FS + 0.5)
    box(ax, 0.9, 0.4, 4.2, 0.8, "class scores", fc=io)
    arrow(ax, (3.0, 4.8), (3.0, 3.8))
    arrow(ax, (3.0, 2.6), (3.0, 1.2))

    ax = axes[1]
    panel(ax, "L1  stages", xlim=(0, 7.5), ylim=(0, 6))
    items = ["stem", "stage 1", "stage 2", "stage 3", "stage 4", "head"]
    for i, t in enumerate(items):
        y = 5.2 - i * 0.98
        box(ax, 1.4, y, 4.6, 0.62, t, fc=conv if "stage" in t else "#edf2f7", fs=FS - 0.5)
        if i:
            arrow(ax, (3.7, y + 0.98), (3.7, y + 0.62))
        if "stage" in t:
            ax.text(6.2, y + 0.31, "×2", va="center", fontsize=FS - 0.5, fontweight="bold")

    ax = axes[2]
    panel(ax, "L2  layers", xlim=(0, 7.5), ylim=(0, 6))
    rows = [("conv 3×3, 64", conv), ("batch norm", norm), ("ReLU", act), ("conv 3×3, 64", conv),
            ("batch norm", norm)]
    for i, (t, c) in enumerate(rows):
        y = 5.2 - i * 0.98
        box(ax, 1.2, y, 4.3, 0.62, t, fc=c, fs=FS - 0.8)
        if i:
            arrow(ax, (3.35, y + 0.98), (3.35, y + 0.62))
    circ(ax, 3.35, 0.35, 0.28, "+", fs=FS - 0.5)
    arrow(ax, (3.35, 1.28), (3.35, 0.63))
    poly_arrow(ax, [(3.35, 5.95), (6.6, 5.95), (6.6, 0.35), (3.63, 0.35)])

    ax = axes[3]
    panel(ax, "L3  operators", xlim=(0, 8.5), ylim=(0, 6))
    ops = ["conv2d", "sub", "div", "mul", "add", "relu"]
    for i, t in enumerate(ops):
        y = 5.25 - i * 0.98
        c = conv if t == "conv2d" else (act if t == "relu" else norm)
        box(ax, 0.5, y, 2.6, 0.58, t, fc=c, fs=FS - 1, round_=0.1)
        if i:
            arrow(ax, (1.8, y + 0.98), (1.8, y + 0.58))
    params = [("W  64×64×3×3", 5.25), ("μ  64", 4.27), ("σ  64", 3.29), ("γ  64", 2.31),
              ("β  64", 1.33)]
    for t, y in params:
        ax.text(3.6, y + 0.29, t, va="center", fontsize=FS - 1.6, family="DejaVu Sans Mono",
                color=MUTED)
        ax.plot([3.1, 3.5], [y + 0.29, y + 0.29], color=MUTED, lw=0.5, ls=":")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.08)
    fig.savefig(path)
    plt.close(fig)


FIGURES = {
    "fig1-styles": figure_styles,
    "fig2-families": figure_families,
    "fig3-levels": figure_levels,
}


def main(out_dir: Path = OUT_DIR, fmt: str = "svg") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["savefig.dpi"] = 200
    paths = []
    for name, draw in FIGURES.items():
        path = out_dir / f"{name}.{fmt}"
        draw(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR
    fmt = sys.argv[2] if len(sys.argv) > 2 else "svg"
    for p in main(target, fmt):
        print(p)
