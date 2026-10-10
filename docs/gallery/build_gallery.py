"""Draw the reference models automatically: trace → simplify → layout → SVG.

A preview gallery of the full pipeline; snapshot tests over it come in roadmap step 12.

    uv run python docs/gallery/build_gallery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "tests"))  # the reference models live with the tests

from models import REFERENCE  # noqa: E402

from tensorgami.importers.torch import trace  # noqa: E402
from tensorgami.layout import Direction, LayoutOptions, layout  # noqa: E402
from tensorgami.render import render_svg  # noqa: E402
from tensorgami.style import load_theme  # noqa: E402
from tensorgami.transforms import simplify  # noqa: E402

FIGURES = {  # file name: (reference model, level, layout options, extra simplify options)
    "cnn": ("cnn", 2, LayoutOptions(), {}),
    "cnn-stages": ("cnn", 1, LayoutOptions(), {}),
    "lstm": ("lstm", 2, LayoutOptions(), {}),
    "cgan-generator": ("cgan_generator", 2, LayoutOptions(), {}),
    "cgan-discriminator": ("cgan_discriminator", 2, LayoutOptions(), {}),
    "vae": ("vae", 2, LayoutOptions(), {}),
    "convlstm": ("convlstm", 2, LayoutOptions(), {}),
    "transformer": ("transformer", 2, LayoutOptions(direction=Direction.TOP_BOTTOM), {}),
}


def build(out_dir: Path = HERE) -> list[Path]:
    theme = load_theme()
    paths = []
    for name, (model, level, options, extra) in FIGURES.items():
        make, example = REFERENCE[model]
        graph = simplify(trace(make(), example()), level, **extra)
        path = out_dir / f"{name}.svg"
        path.write_text(render_svg(layout(graph, theme, options), theme), encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in build():
        print(p)
