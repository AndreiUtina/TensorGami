# Changelog

All notable changes to Tensorgami. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (0.x: the API may still change between minor
versions).

## [Unreleased]

### Added

- Style variants: `tg.draw(model, x, variant="dark")` for slides with a dark background and
  `variant="greyscale"` for black-and-white print, for every style (also
  `load_style("origami", "dark")`); several apply in order. The dark board is GitHub's dark
  page colour, so dark figures blend into a dark-mode README.
- The README's figures follow the reader's GitHub theme (light or dark versions).

- `examples/classics.ipynb`: the ten most-cited deep-learning papers with an architecture
  figure, each model rebuilt as its paper describes it (`examples/classics.py`) and drawn
  next to a link to the paper's figure, with what is the same and what differs.

- Slides: `tg.draw(model, x, fit="slide")` makes a figure for a 16:9 slide (or
  `"slide-4:3"`): larger text and lines, the row length whose text is largest once the figure
  fills the slide, centred on the slide's proportions and saved at 13.33 × 7.5 in
  (`dpi=144`: 1920 × 1080). The figure reports the size of its box names on the slide
  (`fig.fitted`) and warns when they are too small for a projector.
- Animations: `fig.animate()` lights the figure up as the model runs: layers in the order
  data reaches them (parallel branches together, a merge after all its inputs, buffers just
  before their reader), a block folded ×N once per pass with its badge counting (1/N, 2/N…),
  the whole model at the end. What has not run is faded, what runs is highlighted (a halo in
  its own colour, a bold outline). `anim.save("model.gif")` (or `.mp4` with ffmpeg),
  `anim.save_frames(folder)` (PNG, PDF or SVG per step, for slide builds); inline in Jupyter.
- Data in motion: between two steps of an animation a pulse travels along every wire into the
  next layer, and a wire's tensor shape appears once its data has arrived. A recurrent cell
  folded over time sends its state round its loops into each new pass; a stack of layers
  (×N) takes each new pass's input where its wire enters the frame. `fig.animate(flow=False)`
  keeps one frame per step; `travel=` and `fps=` set how long the data travels and how
  smoothly. `anim.steps` holds the state of each step, `anim[i]` the frames.
- Playback: in Jupyter an animation plays in a player (play/pause, step back and forward, a
  time slider, the step shown, the speed; space and the arrow keys), with no widget library
  or kernel needed. `anim.save("model.svg")` writes an animated SVG: vector, sharp at any
  size, a few kB compressed, playing in browsers, GitHub READMEs and VS Code (a viewer that
  does not animate shows the still figure); `anim.save("model.html")` a page with the player.
  `anim.to_svg()` returns the animated SVG, `anim.to_svg(step)` one step as a still.
- Animations in the gallery (animated SVGs, and GIFs) and the README.
- Real activations: `tg.draw(model, x, activations=True)` keeps what every layer produces on
  the example input, and `fig.animate()` shows it under the figure as each layer runs: a card
  in the layer's colour with its output shape, its 8 most active feature maps (an input
  picture as it is, a sequence as a heatmap, a vector's values), their range and share of
  zeros; the model's outputs at the end. Inside a block folded ×N each pass shows its own
  values. `fig.activations` holds the summaries; the still figure is unchanged.
  `examples/images.py` draws a test picture with numpy.
- A third style, `etching`: black ink on white paper, the operations told apart by engraved
  textures (diagonal, cross-hatch, dots, grid, zigzag…) in a strip at the start of each box and
  over a volume's face, instead of colour; wires told apart by line style; Helvetica, 0.75 pt
  lines. For journals printed in black and white and for every reader. Textures are a new
  capability any style can use (`texture = "diagonal"` in a category); a new check keeps them
  distinct.
- Palettes: `tg.draw(model, x, palette="colorblind")` recolours the operations of any style:
  `colorblind` (Okabe–Ito with Paul Tol's colour-blind-safe hues), `tab10` (matplotlib's
  default colours), `vivid` (saturated, for slides and posters) and `mono` (one hue, layers
  with weights darker). Fills keep text at 4.5 : 1, outlines show at 3 : 1, coloured wires
  follow their category; every palette passes the style checks with every variant
  (`load_style("origami", "dark", "colorblind")`).
- Papers: `tg.draw(model, x, width="3.5in")` (also `"8.9cm"`, `"88mm"`, `"252pt"` or inches)
  lays the figure out for that width with the style's text size (box names 9 pt), the shortest
  layout that fits; shrinks only if none does, never taller than a page (`height=`: the most
  it may take, 9 in by default), and warns under 7 pt. Saved exactly that wide, so
  `\includegraphics` places it at true size; SVGs state their size in inches.
- `legend_width`, a layout option: the most the legend may take (it stacks into fewer
  columns); a figure fitted to a width sets it.

### Changed

- The contrast check of a style judges translucent cards by the colour they show on the
  background.
- An encoder–decoder with nested skips is folded into the U whole: the layers before the
  outermost skip and after its merge join the U's top row (U-Net's first convolutions and its
  1×1 output layer were drawn outside it).
- A crop stays in the drawing as a box (U-Net's "copy and crop"); every other indexing
  operation is still left out at the layer level.
- More compact figures: less space between layers (32 px), neighbours (16 px), wrapped rows
  (40 px) and around frames, labels and margins; about a fifth less area on average.
- A module fed partway along by another module stands beside it, both starting level, the
  wire entering from the side: the Transformer's encoder and decoder run in parallel, as in
  the paper (before, the decoder came after the whole encoder).
- Top → bottom figures taller than `wrap_height` (1100 px by default, a new layout option)
  stand in columns side by side, like a newspaper's, frames included: ResNet-34 in four
  columns, 1829 × 1007 instead of 844 × 2958 px.
- A row holding only an input or output pill joins its neighbour, even if that row grows a
  little long: no lone `output` at the end of a wrapped figure.
- A row (or column) shorter than the longest is centred: the wire from the end of the row
  before comes back to the middle, and the white space is shared on both sides (AlexNet's
  classifier under its features).

- PNG and PDF figures render about a fifth faster (matplotlib no longer recomputes its axis
  limits for every shape); the output is unchanged.

### Fixed

- A frame that wraps into rows inside its parent kept to the page width less its frame
  padding, but not less the room its own wires need to turn into the next row.

## [0.1.0] — 2026-10-13

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
- Models on PyTorch's `meta` device (`model.to("meta")`) are traced without allocating
  memory for weights or activations (#2, @HarrisonKramer).
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
- Development on Windows: `uv sync` (CPU PyTorch) and the tests work there too (#1,
  @HarrisonKramer).
