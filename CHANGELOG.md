# Changelog

All notable changes to Tensorgami. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (0.x: the API may still change between minor
versions).

## [Unreleased]

## [0.1.0] — first public release

The first release: a PyTorch model to a figure in one call.

### Drawing

- `tg.draw(model, example_input, …)` traces a model once (no gradients, eval mode; weights,
  training flags and the random state are left as they were) and returns a `tg.Figure`: shown
  inline in Jupyter, saved on request as PNG, SVG, PDF, JPG or TIFF (`fig.save`), drawn into
  a panel of your own matplotlib figure (`fig.draw_on(ax)`).
- Levels of detail: `level=0` (system) … `3` (every operation), by number or name.
- Views: `block` (boxes and wires), `unrolled` (recurrence as a time × layer grid),
  `volume` (CNN layers as tensor volumes), `cell` (an LSTM / GRU cell as its gates), and
  `auto` (unrolled for traced loops over time, block otherwise).
- Encoder–decoders with nested skips (U-Net, pix2pix) fold into a U; `u_levels` limits how
  many levels are drawn.
- Figures wider than `wrap_width` (1100 px by default) fold into rows that read like lines of
  text, frames included: a module too wide for the page wraps inside its frame. A residual
  block is never split.
- Several models in one figure (`tg.draw({"generator": G, "discriminator": D}, …,
  links=[…])`) and annotations: `tg.Loss` (with training arrows), `tg.Input`, `tg.Note`.
- `overrides` set the label, category or leaf-ness of a module; clear errors before the
  model runs (misspelt options, unknown styles, levels, views, references).

### Styles

- `default`, `origami` (upright cards left → right) and `origami-extended` (flat cards, top →
  bottom); a style is one TOML file, and your own can be passed by path.

### Examples

- Notebooks, already run: AlexNet, ResNet-18/34/50, pix2pix, and a tour of one model per
  family; `examples/alexnet.py` and `examples/resnet.py` define the models as torchvision
  does, without needing it.

### Under the hood

- Importer (PyTorch tracing), framework-neutral IR, transforms, layered layout with block
  layouts (unrolled grid, U-shape, LSTM cell), SVG and matplotlib renderers.
- An acceptance gallery (`docs/gallery/`) of every model family in every view and style,
  checked against committed SVGs and layout invariants by the test suite.
