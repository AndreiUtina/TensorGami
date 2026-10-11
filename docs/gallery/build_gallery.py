"""Draw the reference models with ``tg.draw``, once per built-in style.

A preview gallery of the full pipeline; snapshot tests over it come in roadmap step 12.

    uv run python docs/gallery/build_gallery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "tests"))  # the reference models live with the tests

from models import REFERENCE  # noqa: E402

import tensorgami as tg  # noqa: E402
from tensorgami.style import available_styles  # noqa: E402

FIGURES = {  # file name: (reference model, tg.draw options besides the style)
    "cnn": ("cnn", {"level": 2}),
    "cnn-stages": ("cnn", {"level": 1}),
    "lstm": ("lstm", {"level": 2}),
    "lstm-unrolled": ("lstm", {"level": 2, "view": "unrolled"}),
    "cgan-generator": ("cgan_generator", {"level": 2}),
    "cgan-discriminator": ("cgan_discriminator", {"level": 2}),
    "vae": ("vae", {"level": 2}),
    "convlstm": ("convlstm", {"level": 2}),  # view="auto": unrolled (a loop over time)
    "convlstm-folded": ("convlstm", {"level": 2, "view": "block"}),
    "transformer": ("transformer", {"level": 2, "direction": "TB"}),
}


def gallery_dir(style: str) -> Path:
    """``default`` keeps the original folder; every other style gets a sub-folder."""
    return HERE if style == "default" else HERE / style


def build(style: str = "default") -> list[Path]:
    out_dir = gallery_dir(style)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (model, options) in FIGURES.items():
        make, example = REFERENCE[model]
        fig = tg.draw(make(), example(), style=style, **options)
        paths.append(fig.save(out_dir / f"{name}.svg"))
    return paths


if __name__ == "__main__":
    for style in sys.argv[1:] or available_styles():
        for p in build(style):
            print(p)
