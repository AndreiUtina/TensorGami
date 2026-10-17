![TensorGami](docs/brand/logo-banner.png)

# Tensorgami

[![CI](https://github.com/AndreiUtina/TensorGami/actions/workflows/ci.yml/badge.svg)](https://github.com/AndreiUtina/TensorGami/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tensorgami)](https://pypi.org/project/tensorgami/)
[![Python](https://img.shields.io/pypi/pyversions/tensorgami)](https://pypi.org/project/tensorgami/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Draw the architecture and information flow of PyTorch models, locally, in Python: figures
for papers, reports and slides. Call `tg.draw` on a loaded model in a script or a notebook,
see the figure inline in Jupyter, and save it as PNG, SVG or PDF.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/gallery/origami/alexnet-dark.svg">
  <img alt="AlexNet in the origami style" src="docs/gallery/origami/alexnet.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/gallery/pix2pix-system-dark.svg">
  <img alt="pix2pix: U-Net generator and PatchGAN discriminator as tensor volumes" src="docs/gallery/pix2pix-system.svg">
</picture>

## Install

```sh
pip install tensorgami
```

Tensorgami needs Python ≥ 3.11 and matplotlib. To draw a model it runs the model once with
PyTorch (≥ 2.2), which it does not install itself, so your CPU or CUDA build stays as it is.
If PyTorch isn't installed yet, install it first
([pytorch.org](https://pytorch.org/get-started/locally/)), or let pip pick the default build
with `pip install "tensorgami[torch]"`.

The latest development version: `pip install "git+https://github.com/AndreiUtina/TensorGami"`.

## Quick start

```python
import torch
from torch import nn
import tensorgami as tg

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
    nn.Flatten(), nn.Linear(32, 10),
)

fig = tg.draw(model, torch.randn(1, 3, 32, 32))  # runs the model once, leaves it untouched
fig                                              # shows inline in Jupyter
fig.save("model.png", dpi=300)                   # or .svg / .pdf; nothing is saved before
```

The example input is what the model is called with: a tensor, a tuple of positional
arguments, or a dict of keyword arguments, including the batch dimension.

## What you can draw

```python
tg.draw(model, x, level="stages")                  # less detail: 0 system … 3 operators
tg.draw(model, x, style="origami")                 # or "origami-extended", "etching", or a TOML
tg.draw(model, x, variant="dark")                  # for dark slides; "greyscale" for print
tg.draw(model, x, palette="colorblind")            # or "tab10", "vivid", "mono"
tg.draw(model, x, fit="slide")                     # a 16:9 slide: larger text, best layout
tg.draw(model, x, width="3.5in")                   # a paper's column, at true text size
tg.draw(model, x).animate().save("model.gif")      # lights up as the model runs (or .mp4)
tg.draw(model, x, direction="TB", legend=False)    # top → bottom, no legend
tg.draw(model, x, wrap_width=1600)                 # fold into rows beyond 1600 px (None: never)
tg.draw(model, x, direction="TB", wrap_height=900) # top → bottom: columns beyond 900 px
tg.draw(model, x, view="unrolled")                 # recurrence as a time × layer grid
tg.draw(model, x, view="volume")                   # CNN layers as tensor volumes
tg.draw(model, x, view="cell")                     # an LSTM / GRU cell as its gates
tg.draw(unet, x, u_shape=False)                    # keep a U-Net on one line (default: a U)
tg.draw(model, x, overrides={"classifier.6": {"label": "logits"}})       # rename a box
tg.draw(vae, x, annotations=[tg.Loss("KL divergence", "mu", "logvar")])  # a loss term
tg.draw({"generator": G, "discriminator": D}, {"generator": (z, y), "discriminator": (x, y)},
        links=[("generator", "discriminator.input 0")],     # several models, wired together
        annotations=[tg.Loss("adversarial", "discriminator", trains=["generator", "discriminator"])])
```

- **Levels of detail**: the whole system, stages, layers or single operators. Repeated blocks
  fold into one frame marked ×N, activations fold into the layer before them, and shapes
  appear on the wires where they change.
- **Views**: block diagrams (`view="block"`), recurrent networks unrolled over time
  (`"unrolled"`), CNNs as 3-D tensor volumes (`"volume"`) and recurrent cells as their gates
  (`"cell"`). Encoder–decoders with nested skips (U-Net, pix2pix) fold into a U.
- **Styles**: [`default`](docs/style/style-baseline.md), [`etching`](docs/style/etching.md)
  (black ink, operations told apart by engraved textures: for black-and-white print),
  and [`origami` and `origami-extended`](docs/style/origami.md)
  (narrow boxes in the style of the survey figures). A style is a TOML file; your own can be
  passed by path. Any style comes in a `dark` variant for slides and a `greyscale` one for
  black-and-white print, and in four palettes: `colorblind` (safe for colour-vision
  deficiency), `tab10` (matplotlib's colours), `vivid` (for slides and posters) and `mono` (one
  hue).
- **Slides**: `fit="slide"` draws a figure for a 16:9 slide, with text sized for a projector
  and the layout chosen to make it as large as possible; saved at the slide's size.
- **Animations**: `fig.animate()` lights the figure up as the model runs, layer after layer
  in the order data reaches them; saved as a GIF, an MP4 or one image per step for slides.
- **Papers**: `width="3.5in"` (or `"8.9cm"`, `"252pt"`) lays a figure out for a column or text
  width with its text at true size (9 pt), and saves it exactly that wide, so LaTeX includes
  it without scaling.
- **Several models in one figure**: GANs, VAEs and other training set-ups, with links between
  the models, extra inputs, loss terms with dashed training arrows, and notes.

Model families covered by the tests: CNN (AlexNet, ResNet), LSTM, ConvLSTM, cGAN, VAE,
Transformer and pix2pix. The [acceptance gallery](docs/gallery/README.md)
shows each of them in every view and style. Every figure there is drawn automatically, and
the tests check it.

## Examples

Notebooks with every option, already run, are in
[`examples/`](examples):

- [`alexnet.ipynb`](examples/alexnet.ipynb):
  levels, views, styles, layout options, saving, panels;
- [`resnet.ipynb`](examples/resnet.ipynb):
  ResNet-18/34/50, how to read the figure, wide figures folded into rows;
- [`pix2pix.ipynb`](examples/pix2pix.ipynb):
  a U-Net generator, a PatchGAN discriminator and both together;
- [`tour.ipynb`](examples/tour.ipynb):
  one model per family;
- [`classics.ipynb`](examples/classics.ipynb):
  the ten most-cited deep-learning papers with an architecture figure (ResNet, Transformer,
  AlexNet, BERT, LSTM, U-Net, Faster R-CNN, ViT, LeNet-5, YOLO), each drawn by Tensorgami
  next to the paper's own figure.

The pipeline's stages can also be used one by one, e.g. to inspect what was captured:

```python
from tensorgami.importers.torch import trace
from tensorgami.transforms import simplify

graph = trace(model, torch.randn(1, 3, 32, 32))
print(graph.summary())                     # every operation, its module and shape
print(simplify(graph, level=2).summary())  # layers: activations folded, repeats ×N
```

Background: the
[literature survey](docs/survey/tensorgami-literature-survey.pdf)
on how neural-network architectures are drawn, and the
[architecture](docs/architecture.md)
of Tensorgami itself.

## Development

```sh
git clone https://github.com/AndreiUtina/TensorGami && cd TensorGami
uv sync                                        # the environment, with CPU PyTorch
uv run pytest                                  # the tests, including the gallery
uv run ruff check . && uv run ruff format .    # lint / format
uv run python docs/gallery/build_gallery.py    # accept changed drawings: rebuild the gallery
cd examples && uv run --with nbclient --with nbformat --with ipykernel jupyter execute --inplace *.ipynb
```

Releases are described in [docs/RELEASING.md](docs/RELEASING.md),
changes in the [changelog](CHANGELOG.md).

## License

MIT, see [LICENSE](LICENSE). The example
and test models adapted from torchvision and pytorch-CycleGAN-and-pix2pix keep their BSD
licenses, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
To cite Tensorgami, use [CITATION.cff](CITATION.cff)
(GitHub's "Cite this repository").
