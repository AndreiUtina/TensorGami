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
from tensorgami.layout import LayoutOptions, layout  # noqa: E402
from tensorgami.render import render_svg  # noqa: E402
from tensorgami.style import available_styles, load_style  # noqa: E402
from tensorgami.transforms import simplify  # noqa: E402

FIGURES = {  # file name: (reference model, level, layout overrides, extra simplify options)
    "cnn": ("cnn", 2, {}, {}),
    "cnn-stages": ("cnn", 1, {}, {}),
    "lstm": ("lstm", 2, {}, {}),
    "cgan-generator": ("cgan_generator", 2, {}, {}),
    "cgan-discriminator": ("cgan_discriminator", 2, {}, {}),
    "vae": ("vae", 2, {}, {}),
    "convlstm": ("convlstm", 2, {}, {}),
    "transformer": ("transformer", 2, {"direction": "TB"}, {}),
}


def gallery_dir(style: str) -> Path:
    """``default`` keeps the original folder; every other style gets a sub-folder."""
    return HERE if style == "default" else HERE / style


def build(style: str = "default") -> list[Path]:
    theme = load_style(style)
    out_dir = gallery_dir(style)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (model, level, overrides, extra) in FIGURES.items():
        make, example = REFERENCE[model]
        graph = simplify(trace(make(), example()), level, **extra)
        options = LayoutOptions.for_style(theme, **overrides)  # the style's spacing + overrides
        path = out_dir / f"{name}.svg"
        path.write_text(render_svg(layout(graph, theme, options), theme), encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for style in sys.argv[1:] or available_styles():
        for p in build(style):
            print(p)
