"""Hand-built IR graphs mirroring real traces (see the ADR-0003 spike).

They are shared test fixtures for the IR, and later for transforms and layout.
"""

from __future__ import annotations

from tensorgami.ir import Graph, GraphBuilder, RepeatKind, invocation_id
from tensorgami.vocabulary import EdgeKind
from tensorgami.vocabulary import OpCategory as C


def residual_cnn() -> Graph:
    """conv → bn → relu → conv, residual add, pool, flatten, linear."""
    b = GraphBuilder("residual_cnn", root_type="CNN")
    x = b.input((1, 3, 32, 32), label="image")
    s = (1, 16, 32, 32)
    conv1 = b.node(
        "Conv2d",
        category=C.CONV,
        module="conv1",
        outputs=[s],
        params={"kernel_size": (3, 3), "out_channels": 16},
    )
    bn = b.node("BatchNorm2d", category=C.NORM, module="bn", outputs=[s])
    relu = b.node("relu", category=C.ACTIVATION, outputs=[s])
    conv2 = b.node("Conv2d", category=C.CONV, module="conv2", outputs=[s])
    add = b.node("add", category=C.MERGE, outputs=[s])
    pool = b.node("MaxPool2d", category=C.RESAMPLE, module="pool", outputs=[(1, 16, 16, 16)])
    flat = b.node("flatten", category=C.RESHAPE, outputs=[(1, 4096)])
    fc = b.node("Linear", category=C.LINEAR, module="fc", outputs=[(1, 10)])
    b.chain([x, conv1, bn, relu, conv2])
    b.edge(relu, add)
    b.edge(conv2, add)
    b.chain([add, pool, flat, fc])
    b.output(fc, label="logits")
    return b.build()


def convlstm(steps: int = 3) -> Graph:
    """A ConvLSTM: one custom cell called once per time step (unrolled trace)."""
    b = GraphBuilder("convlstm", root_type="ConvLSTM")
    x = b.input((1, steps, 1, 16, 16), label="frames")
    state = (1, 8, 16, 16)
    h = b.node("new_zeros", category=C.TENSOR, label="h0", outputs=[state])
    c = b.node("zeros_like", category=C.TENSOR, label="c0", outputs=[state])
    for t in range(steps):
        cell = b.group(
            invocation_id("cell", t),
            "ConvLSTMCell",
            module="cell",
            call_index=t,
            category=C.RECURRENT,
        )
        frame = b.node("getitem", category=C.RESHAPE, outputs=[(1, 1, 16, 16)])
        b.edge(x, frame)
        cat = b.node("cat", category=C.MERGE, group=cell, call_index=t, outputs=[(1, 9, 16, 16)])
        b.edge(frame, cat)
        b.edge(h, cat)
        conv = b.node(
            "Conv2d",
            category=C.CONV,
            group=cell,
            call_index=t,
            module="cell.conv",
            outputs=[(1, 32, 16, 16)],
        )
        b.edge(cat, conv)
        split = b.node("split", category=C.RESHAPE, group=cell, call_index=t, outputs=[state] * 4)
        b.edge(conv, split)
        i, f, o = (
            b.node("sigmoid", category=C.ACTIVATION, group=cell, call_index=t, outputs=[state])
            for _ in range(3)
        )
        g = b.node("tanh", category=C.ACTIVATION, group=cell, call_index=t, outputs=[state])
        for port, gate in enumerate([i, f, o, g]):
            b.edge(split, gate, source_port=port)
        fc = b.node("mul", category=C.MERGE, group=cell, call_index=t, outputs=[state])
        b.edge(f, fc)
        b.edge(c, fc)
        ig = b.node("mul", category=C.MERGE, group=cell, call_index=t, outputs=[state])
        b.edge(i, ig)
        b.edge(g, ig)
        c = b.node("add", category=C.MERGE, group=cell, call_index=t, outputs=[state])
        b.edge(fc, c)
        b.edge(ig, c)
        tc = b.node("tanh", category=C.ACTIVATION, group=cell, call_index=t, outputs=[state])
        b.edge(c, tc)
        h = b.node("mul", category=C.MERGE, group=cell, call_index=t, outputs=[state])
        b.edge(o, h)
        b.edge(tc, h)
    head = b.node("Conv2d", category=C.CONV, module="head", outputs=[(1, 1, 16, 16)])
    b.edge(h, head)
    b.output(head, label="prediction")
    b.repeat(
        [invocation_id("cell", t) for t in range(steps)], RepeatKind.TIME, repeat_id="cell-steps"
    )
    return b.build()


def vae() -> Graph:
    """VAE with explicit reparameterisation; the noise is a source, not fed by μ."""
    b = GraphBuilder("vae", root_type="VAE")
    x = b.input((1, 784), label="x")
    enc = b.node("Linear", category=C.LINEAR, module="enc", outputs=[(1, 64)])
    relu = b.node("relu", category=C.ACTIVATION, outputs=[(1, 64)])
    mu = b.node("Linear", category=C.LINEAR, module="mu", outputs=[(1, 16)])
    logvar = b.node("Linear", category=C.LINEAR, module="logvar", outputs=[(1, 16)])
    half = b.node("mul", category=C.MERGE, outputs=[(1, 16)])
    std = b.node("exp", category=C.ACTIVATION, outputs=[(1, 16)])
    eps = b.node("randn_like", category=C.STOCHASTIC, label="ε ~ N(0, I)", outputs=[(1, 16)])
    scaled = b.node("mul", category=C.MERGE, outputs=[(1, 16)])
    z = b.node("add", category=C.MERGE, outputs=[(1, 16)])
    dec = b.node("Linear", category=C.LINEAR, module="dec", outputs=[(1, 784)])
    sig = b.node("sigmoid", category=C.ACTIVATION, outputs=[(1, 784)])
    b.chain([x, enc, relu, mu])
    b.edge(relu, logvar)
    b.chain([logvar, half, std, scaled])
    b.edge(eps, scaled, EdgeKind.STOCHASTIC)
    b.edge(mu, z)
    b.edge(scaled, z)
    b.chain([z, dec, sig])
    b.output(sig, label="reconstruction")
    b.output(mu, label="mu")
    b.output(logvar, label="logvar")
    return b.build()


def transformer(layers: int = 2) -> Graph:
    """Transformer encoder: identical layers recognised as a LAYERS repeat."""
    b = GraphBuilder("transformer", root_type="TransformerEncoder")
    s = (1, 6, 32)
    prev = b.input(s, label="tokens")
    groups = []
    for k in range(layers):
        g = b.group(f"layers.{k}", "TransformerEncoderLayer")
        groups.append(g)
        attn = b.node(
            "MultiheadAttention",
            category=C.ATTENTION,
            group=g,
            module=f"layers.{k}.self_attn",
            outputs=[s],
        )
        add1 = b.node("add", category=C.MERGE, group=g, outputs=[s])
        n1 = b.node("LayerNorm", category=C.NORM, group=g, module=f"layers.{k}.norm1", outputs=[s])
        l1 = b.node(
            "Linear", category=C.LINEAR, group=g, module=f"layers.{k}.linear1", outputs=[(1, 6, 64)]
        )
        l2 = b.node("Linear", category=C.LINEAR, group=g, module=f"layers.{k}.linear2", outputs=[s])
        add2 = b.node("add", category=C.MERGE, group=g, outputs=[s])
        n2 = b.node("LayerNorm", category=C.NORM, group=g, module=f"layers.{k}.norm2", outputs=[s])
        b.edge(prev, attn)
        b.edge(prev, add1)
        b.chain([attn, add1, n1, l1, l2])
        b.edge(n1, add2)
        b.chain([l2, add2, n2])
        prev = n2
    b.output(prev, label="encoded")
    b.repeat(groups, RepeatKind.LAYERS, repeat_id="encoder-layers")
    return b.build()


def unet(levels: int = 3, *, stem: bool = False) -> Graph:
    """A small U-Net: per level, encoder Conv → ReLU → MaxPool (the skip leaves after the ReLU);
    a bottleneck Conv; per level, decoder ConvTranspose → cat with the skip → Conv; a 1×1 head.
    ``stem``: a Conv before the first level, ahead of the outermost skip."""
    b = GraphBuilder("unet", root_type="UNet")
    x = b.input((1, 3, 64, 64), label="image")
    if stem:
        first = b.node("Conv2d", category=C.CONV, module="stem", outputs=[(1, 8, 64, 64)])
        b.edge(x, first)
        x = first
    size, channels, skips = 64, 16, []
    for k in range(levels):
        conv = b.node(
            "Conv2d", category=C.CONV, module=f"down{k}.conv", outputs=[(1, channels, size, size)]
        )
        b.edge(x, conv)
        act = b.node("relu", category=C.ACTIVATION, outputs=[(1, channels, size, size)])
        b.edge(conv, act)
        skips.append((act, channels, size))
        size //= 2
        x = b.node(
            "MaxPool2d",
            category=C.RESAMPLE,
            module=f"down{k}.pool",
            outputs=[(1, channels, size, size)],
        )
        b.edge(act, x)
        channels *= 2
    x_ = b.node("Conv2d", category=C.CONV, module="bottleneck", outputs=[(1, channels, size, size)])
    b.edge(x, x_)
    x = x_
    for k, (skip, ch, sz) in reversed(list(enumerate(skips))):
        up = b.node(
            "ConvTranspose2d", category=C.CONV, module=f"up{k}.up", outputs=[(1, ch, sz, sz)]
        )
        b.edge(x, up)
        cat = b.node("cat", category=C.MERGE, outputs=[(1, 2 * ch, sz, sz)])
        b.edge(skip, cat)
        b.edge(up, cat)
        x = b.node("Conv2d", category=C.CONV, module=f"up{k}.conv", outputs=[(1, ch, sz, sz)])
        b.edge(cat, x)
    head = b.node("Conv2d", category=C.CONV, module="head", outputs=[(1, 1, 64, 64)])
    b.edge(x, head)
    b.output(head, label="mask")
    return b.build()


ALL = {
    "residual_cnn": residual_cnn,
    "convlstm": convlstm,
    "vae": vae,
    "transformer": transformer,
    "unet": unet,
}
