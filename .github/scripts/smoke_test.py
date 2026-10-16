"""Smoke test of an installed Tensorgami, without torch: the packaged styles load, a small
hand-built graph is laid out and saved in every style and format. Run against a fresh
environment holding only the built wheel, so a file missing from the package fails CI."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import tensorgami as tg
from tensorgami.ir import GraphBuilder
from tensorgami.layout import layout
from tensorgami.style import available_styles, load_style
from tensorgami.vocabulary import OpCategory as C

b = GraphBuilder("smoke", root_type="Smoke")
x = b.input((1, 3, 32, 32))
conv = b.node(
    "Conv2d",
    category=C.CONV,
    module="conv",
    params={"kernel_size": 3, "out_channels": 8},
    outputs=[(1, 8, 32, 32)],
)
b.edge(x, conv)
b.output(conv)
graph = b.build()

styles = available_styles()
assert styles == ["default", "etching", "origami", "origami-extended"], styles
folder = Path(tempfile.mkdtemp())
for name in styles:
    style = load_style(name)
    fig = tg.Figure(layout(graph, style), style, graph)
    for suffix in (".png", ".svg", ".pdf"):
        path = fig.save(folder / f"smoke-{name}{suffix}")
        assert path.stat().st_size > 0, path
print(f"tensorgami {tg.__version__} on Python {sys.version.split()[0]}: {', '.join(styles)} ok")
