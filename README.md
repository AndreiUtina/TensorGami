![TensorGami](docs/brand/logo-banner.png)

# Tensorgami

Draw the architecture and information flow of deep-learning / machine-learning models,
locally, in Python.

Status: in development; the pipeline works end to end (see [roadmap.md](roadmap.md)).

- Literature survey: [docs/survey/tensorgami-literature-survey.pdf](docs/survey/tensorgami-literature-survey.pdf)
- Drawing styles: [`default`](docs/style/style-baseline.md), [`origami` and `origami-extended`](docs/style/origami.md)

![Visual vocabulary](docs/style/specimens/vocabulary.svg)

## Development

```sh
uv sync                                        # create the environment
uv run pytest                                  # run the tests
uv run ruff check . && uv run ruff format .    # lint / format
uv run python docs/style/build_specimens.py    # rebuild the style specimens
uv run python docs/survey/build.py             # rebuild the survey PDF
```

## Drawing a PyTorch model

In a script or a notebook, after the model is loaded:

```python
import torch
import tensorgami as tg

fig = tg.draw(model, torch.randn(1, 3, 224, 224))  # runs the model once, leaves it untouched
fig                                                # shows inline in Jupyter
fig.save("model.png", dpi=300)                     # or .svg / .pdf; nothing is saved before

tg.draw(model, x, level="stages")                  # less detail: 0 system … 3 operators
tg.draw(model, x, style="origami")                 # or "origami-extended", or a path
tg.draw(model, x, direction="TB", legend=False)    # top → bottom, no legend
tg.draw(model, x, view="unrolled")                 # recurrence as a time × layer grid
tg.draw(model, x, view="volume")                   # CNN layers as tensor volumes
tg.draw(model, x, view="cell")                     # an LSTM / GRU cell as its gates
tg.draw(unet, x, u_shape=False)                    # keep a U-Net on one line (default: a U)
tg.draw(model, x, overrides={"classifier.6": {"label": "logits"}})  # rename a box
tg.draw(vae, x, annotations=[tg.Loss("KL divergence", "mu", "logvar")])  # a loss term
tg.draw({"generator": G, "discriminator": D}, {"generator": (z, y), "discriminator": (x, y)},
        links=[("generator", "discriminator.input 0")],     # several models, wired together
        annotations=[tg.Loss("adversarial", "discriminator", trains=["generator", "discriminator"])])
```

The example input is what the model is called with: a tensor, a tuple of positional
arguments, or a dict of keyword arguments, including the batch dimension.

Notebooks with every option, already run, in [`examples/`](examples/):
[`alexnet.ipynb`](examples/alexnet.ipynb) (levels, views, styles, layout options, saving,
panels), [`pix2pix.ipynb`](examples/pix2pix.ipynb) (a U-Net generator and a PatchGAN) and
[`tour.ipynb`](examples/tour.ipynb) (one model per family: CNN, LSTM, ConvLSTM, VAE, cGAN,
Transformer). To run them again:
`cd examples && uv run --with nbclient --with nbformat jupyter execute --inplace *.ipynb`.

The stages are also usable one by one, e.g. to inspect what was captured:

```python
from tensorgami.importers.torch import trace
from tensorgami.transforms import simplify

graph = trace(model, torch.randn(1, 3, 32, 32))
print(graph.summary())                    # every operation, its module and shape
print(simplify(graph, level=2).summary())  # layers: activations folded, repeats ×N
```

See the automatically drawn [gallery](docs/gallery/README.md).

## Drawing a scene today

Automatic layout is a later roadmap step; today a hand-positioned scene can be rendered in
the baseline style:

```python
from tensorgami.render import add_legend, render_svg
from tensorgami.scene import Box, Edge, Node, Point, Scene
from tensorgami.style import load_theme
from tensorgami.vocabulary import OpCategory

theme = load_theme()  # the baseline theme, validated
scene = Scene(420, 140)
scene.add(
    Node("x", Box.centered(50, 40, 70, 32), OpCategory.IO, "image"),
    Node("c", Box.centered(190, 40, 120, 46), OpCategory.CONV, "Conv 3×3", "64, ReLU"),
    Node("y", Box.centered(350, 40, 90, 32), OpCategory.IO, "features"),
    Edge("x", "c"),
    Edge("c", "y", label="64×32×32"),
)
add_legend(scene, theme, Point(16, 84), columns=3)
open("conv.svg", "w").write(render_svg(scene, theme))
```
