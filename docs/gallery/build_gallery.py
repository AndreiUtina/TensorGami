"""The acceptance gallery (roadmap step 12): the reference models, AlexNet, ResNet and the composed
figures (a cGAN, a VAE with its losses, pix2pix whole), drawn with ``tg.draw`` in every view
that suits them, once per built-in style; and a few of them running, as GIFs (step 14).

``tests/test_gallery.py`` redraws every figure and checks it against the committed SVG (and
against the layout invariants), so a change to the drawing shows up as a failing test. After
an intended change, rebuild and review the figures::

    uv run python docs/gallery/build_gallery.py               # every style and the GIFs
    uv run python docs/gallery/build_gallery.py origami       # one style
    uv run python docs/gallery/build_gallery.py --animations  # only the GIFs
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "examples")]  # reference models, examples

import torch  # noqa: E402
from alexnet import AlexNet  # noqa: E402
from images import sample_image  # noqa: E402
from models import REFERENCE  # noqa: E402
from models.pix2pix import ConditionalDiscriminator  # noqa: E402
from resnet import resnet18  # noqa: E402

import tensorgami as tg  # noqa: E402
from tensorgami.style import available_styles  # noqa: E402

MODELS = {
    **REFERENCE,
    "alexnet": (AlexNet, lambda: torch.randn(1, 3, 224, 224)),
    "resnet18": (resnet18, lambda: torch.randn(1, 3, 224, 224)),
}

FIGURES = {  # file name: (model, tg.draw options besides the style)
    "cnn": ("cnn", {"level": 2}),
    "cnn-stages": ("cnn", {"level": 1}),
    "lstm": ("lstm", {"level": 2}),
    "lstm-unrolled": ("lstm", {"level": 2, "view": "unrolled"}),
    "cgan-generator": ("cgan_generator", {"level": 2}),
    "cgan-discriminator": ("cgan_discriminator", {"level": 2}),
    "vae": ("vae", {"level": 2}),
    "convlstm": ("convlstm", {"level": 2}),  # view="auto": unrolled (a loop over time)
    "convlstm-folded": ("convlstm", {"level": 2, "view": "block"}),
    "convlstm-cell": ("convlstm", {"view": "cell"}),  # the cell, as its gates
    "lstm-cell": ("lstm", {"view": "cell"}),
    "transformer": ("transformer", {"level": 2, "direction": "TB"}),
    "pix2pix-generator": ("pix2pix_generator", {"level": 2}),  # nested skips: a U
    "pix2pix-discriminator": ("pix2pix_discriminator", {"level": 2}),
    "alexnet": ("alexnet", {"level": 2}),  # the style's own direction
    "alexnet-horizontal": ("alexnet", {"level": 2, "direction": "LR"}),
    "alexnet-vertical": ("alexnet", {"level": 2, "direction": "TB"}),
    "alexnet-stages": ("alexnet", {"level": 1}),
    "alexnet-volume": ("alexnet", {"level": 2, "view": "volume"}),
    "resnet18": ("resnet18", {"level": 2}),  # frames wrapped into rows inside frames
    # variants (step 13): recoloured for slides and for black-and-white print
    "alexnet-dark": ("alexnet", {"level": 2, "variant": "dark"}),
    "alexnet-greyscale": ("alexnet", {"level": 2, "variant": "greyscale"}),
    "alexnet-volume-dark": ("alexnet", {"level": 2, "view": "volume", "variant": "dark"}),
    "pix2pix-generator-dark": ("pix2pix_generator", {"level": 2, "variant": "dark"}),
    "alexnet-slide": ("alexnet", {"level": 2, "fit": "slide"}),  # step 13c: a 16:9 slide
    "alexnet-paper": ("alexnet", {"level": 2, "width": "7in"}),  # step 13b: a paper's width
    # palettes (step 13): the operations recoloured, for any style
    "cnn-colorblind": ("cnn", {"level": 2, "palette": "colorblind"}),
    "cnn-tab10": ("cnn", {"level": 2, "palette": "tab10"}),
    "cnn-vivid": ("cnn", {"level": 2, "palette": "vivid"}),
    "cnn-mono": ("cnn", {"level": 2, "palette": "mono"}),
}


def gallery_dir(style: str) -> Path:
    """``default`` keeps the original folder; every other style gets a sub-folder."""
    return HERE if style == "default" else HERE / style


def cgan(**options: object) -> tg.Figure:
    """Generator and discriminator in one figure, with the adversarial loop (step 11e)."""
    (make_g, example_g), (make_d, example_d) = (
        MODELS["cgan_generator"],
        MODELS["cgan_discriminator"],
    )
    (z, labels), (image, _) = example_g(), example_d()
    return tg.draw(
        {"generator": make_g(), "discriminator": make_d()},
        {
            "generator": {"z": z, "labels": labels},
            "discriminator": {"image": image, "labels": labels},
        },
        links=[
            ("generator", "discriminator.image", "fake image"),
            ("generator.labels", "discriminator.labels"),
        ],
        annotations=[
            tg.Input("real image", to="discriminator.image"),
            tg.Loss("adversarial loss", "discriminator", trains=["generator", "discriminator"]),
        ],
        **options,
    )


def vae_losses(**options: object) -> tg.Figure:
    """The VAE with its two loss terms (step 11e)."""
    make, example = MODELS["vae"]
    return tg.draw(
        make(),
        example(),
        overrides={"mu": {"label": "μ"}, "logvar": {"label": "log σ²"}},
        annotations=[
            tg.Loss("reconstruction", "output 0", "input"),
            tg.Loss("KL divergence", "mu", "logvar"),
        ],
        **options,
    )


def pix2pix_system(**options: object) -> tg.Figure:
    """pix2pix whole: the U-Net generator, the PatchGAN judging (A, real or fake B), the GAN
    and L1 losses (step 11e); small sizes (5 levels, 32 × 32 images)."""
    (make_g, example_g), (make_d, _) = MODELS["pix2pix_generator"], MODELS["pix2pix_discriminator"]
    a, b = example_g(), example_g()
    return tg.draw(
        {"generator": make_g(), "discriminator": ConditionalDiscriminator(make_d())},
        {"generator": a, "discriminator": {"A": a, "B": b}},
        links=[
            ("generator.input", "discriminator.A", "A"),
            ("generator", "discriminator.B", "fake B"),
        ],
        annotations=[
            tg.Input("real B", to="discriminator.B"),
            tg.Loss("GAN loss", "discriminator", trains=["generator", "discriminator"]),
            tg.Loss("L1 loss", "generator", "real B", trains=["generator"]),
        ],
        **options,
    )


COMPOSED = {  # file name: (figure, tg.draw options besides the style)
    "cgan": (cgan, {"level": 0}),  # the overview: each model one box
    "cgan-layers": (cgan, {"direction": "TB"}),
    "vae-losses": (vae_losses, {}),
    "pix2pix-system": (pix2pix_system, {"view": "volume"}),
    "pix2pix-system-dark": (pix2pix_system, {"view": "volume", "variant": "dark"}),
}


#: figures also shown running (``fig.animate()``), saved in ``animations/`` as an animated SVG
#: (checked by the tests like every SVG here) and as a GIF (not checked: its pixels depend on
#: the platform's fonts)
ANIMATED = [
    "alexnet",
    "alexnet-dark",
    "transformer",
    "convlstm-folded",
    "lstm-unrolled",
    "pix2pix-generator",
]


def figure(name: str, style: str = "default") -> tg.Figure:
    """The gallery figure ``name`` (its file name without ``.svg``) in one style."""
    if name in COMPOSED:
        compose, options = COMPOSED[name]
        return compose(style=style, **options)
    model, options = FIGURES[name]
    make, example = MODELS[model]
    return tg.draw(make(), example(), style=style, **options)


def figures(style: str = "default") -> Iterator[tuple[str, tg.Figure]]:
    """Every gallery figure in one style, by file name (without ``.svg``)."""
    for name in [*FIGURES, *COMPOSED]:
        yield name, figure(name, style)


def build(style: str = "default") -> list[Path]:
    out_dir = gallery_dir(style)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [fig.save(out_dir / f"{name}.svg") for name, fig in figures(style)]


#: animations showing what each layer produces (step 14d), as animated SVGs and GIFs; not
#: compared byte for byte (the feature maps are PNGs, whose bytes depend on the zlib build)
WITH_ACTIVATIONS = ["alexnet-activations", "convlstm-folded-activations"]


def with_activations(name: str) -> tg.Figure:
    torch.manual_seed(0)  # the same untrained weights every time
    if name == "alexnet-activations":  # a picture, so that the feature maps show something
        return tg.draw(AlexNet(), sample_image(), activations=True)
    make, example = MODELS["convlstm"]
    return tg.draw(make(), example(), view="block", activations=True)


def build_animations() -> list[Path]:
    out_dir = HERE / "animations"
    out_dir.mkdir(exist_ok=True)
    paths = []
    for name in ANIMATED:
        anim = figure(name).animate()
        paths += [anim.save(out_dir / f"{name}.svg"), anim.save(out_dir / f"{name}.gif")]
    for name in WITH_ACTIVATIONS:
        anim = with_activations(name).animate()
        paths += [anim.save(out_dir / f"{name}.svg"), anim.save(out_dir / f"{name}.gif")]
    return paths


if __name__ == "__main__":
    styles = [a for a in sys.argv[1:] if not a.startswith("--")]
    animations = "--animations" in sys.argv or not sys.argv[1:]
    if styles or not animations:
        for style in styles or available_styles():
            for p in build(style):
                print(p)
    if animations:
        for p in build_animations():
            print(p)
