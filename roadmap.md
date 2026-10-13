# Tensorgami — Roadmap

## Main idea
A python-based software that I can install locally and use it to draw the architectures and
information flow of different deep-learning/machine-learning models that I use locally.

## Main rules
1. We work in python for most of the code.
2. We create tests for all important part of the code.
3. Code should follow SOLID design rules.

Conventions derived from the rules (and from the sibling `Bayesball` project):
- Python ≥ 3.11 (tested on 3.11 and 3.12), managed with `uv`; `pytest` for tests, `ruff` for
  lint/format.
- `src/` layout, fully type-hinted, immutable dataclasses for value objects.
- Every module has one reason to change (S); new shapes / themes / importers / renderers are
  added by *registering* a new class or data file, not by editing existing code (O); all
  renderers and importers are interchangeable behind small `Protocol`s (L, I); high-level code
  depends on those protocols, never on a concrete backend (D).
- Runtime dependency: **matplotlib** only (figures, PNG, Jupyter display). PyTorch is used but
  not installed by Tensorgami (optional extra `tensorgami[torch]`), so it never replaces the
  user's own CUDA/CPU build of torch.

Status legend: ✅ done · 🔶 waiting on user · ⬜ not started

---

## Phase A — Design (user skeleton)

### 0) Analyse the roadmap.md and complete it with suggested steps. Use the users steps as skeleton. ✅
- This document. Phases B–D below are the suggested continuation.

### 1) Literature survey of plotting styles for DL models → PDF report (max 10 pages) ✅
Covers CNNs, VAE, RNN, LSTM, ConvLSTM, Transformers, GAN, U-Net and general DL.
- Output: [`docs/survey/tensorgami-literature-survey.pdf`](docs/survey/tensorgami-literature-survey.pdf)
- Source: `docs/survey/survey.typ` (Typst) + `docs/survey/make_figures.py` (matplotlib sketches)
- Rebuild: `uv run python docs/survey/build.py`
- Contents: taxonomy of 9 diagram styles, canonical depictions per architecture family, tool
  landscape (Netron, TensorBoard, Model Explorer, torchview, TorchLens, NN-SVG, PlotNeuralNet,
  visualkeras, Net2Vis, Treescope…), empirical guidelines (Net2Vis, Marshall et al. 2025),
  and the design requirements R1–R10 that Tensorgami derives from them.

### 2) Create a drawing style baseline ✅ (adopted as the `default` style)
- Style guide: [`docs/style/style-baseline.md`](docs/style/style-baseline.md) — 10 principles,
  17 colour-coded categories, shapes, 5 edge kinds, typography, layout rules, levels of detail
  L0–L4, views, and the open style questions (§10).
- Style as code: `src/tensorgami/vocabulary.py` (`OpCategory`, `EdgeKind`),
  `src/tensorgami/style/` (colour maths, style records, theme loader, validation rules) with
  the data in `src/tensorgami/style/themes/default.toml` (renamed from `baseline.toml`).
- Scene + SVG rendering layer used to prove the style: `src/tensorgami/scene.py`,
  `src/tensorgami/render/` (painter per shape, orthogonal edge router, groups, auto legend,
  SVG canvas).
- Reference specimens (vocabulary sheet, CNN block + volume view, VAE, RNN folded/unrolled,
  LSTM cell, ConvLSTM, Transformer encoder): `docs/style/specimens/*.svg`,
  rebuilt with `uv run python docs/style/build_specimens.py`.
- Tests: `uv run pytest` — 85 tests (colour maths & WCAG contrast, theme parsing/validation,
  scene invariants, routing, painters, SVG output, legend, specimens as golden files).

### 3) Interview the user if needed at any step ✅
Answers (2026-10-09):
1. **Frameworks:** PyTorch only for now; other backends move to the backlog (step 16).
2. **Purpose:** figures for scientific reports and presentations. Later, *dynamic* figures that
   show how information flows through the architecture (step 14).
3. **Workflow:** define and load a model in a Python script or notebook, then call
   TensorGami on it to plot the architecture.
4. **Output:** a figure object that displays inline in Jupyter (or a window from a script);
   saving is optional and explicit, primarily as **PNG**.
5. **Style:** ✅ the current look is adopted as the standard style **`default`**; the style is
   an input (`style="default"`), and more named styles follow: **`origami`** (planned, see
   step 13) and a third one to be decided (ADR-0008).
6. **Models to support first:** CNNs, LSTMs, cGANs, VAEs, ConvLSTMs, Transformers (and others).
7. **Distribution:** private install from GitHub first; public PyPI package later (step 15).
   → 2026-10-12: public from the first release instead, MIT licence, on PyPI (step 15).

Target usage that drives Phases B–D:

```python
import tensorgami as tg

model = MyConvLSTM(...)                        # any torch.nn.Module, already loaded
fig = tg.draw(model, torch.randn(1, 10, 1, 64, 64), level=2)
fig                                            # shows inline in Jupyter
fig.save("convlstm.png", dpi=300)              # only if you decide to keep it
```

---

## Phase B — From a PyTorch model to a figure

### 4) Requirements & architecture decision records ✅
- Requirements: [`docs/requirements.md`](docs/requirements.md) — usage scenario, FR1–FR14
  and NFR1–NFR9 with priorities, acceptance criteria, traceability to survey R1–R10.
- ADRs: [`docs/adr/`](docs/adr/README.md) — 0001 pipeline architecture, 0002 IR, 0003
  PyTorch capture by eager tracing (spike-verified on torch 2.2.2: CNN, LSTM, ConvLSTM, VAE,
  Transformer; `torch.fx` fails on the ConvLSTM loop), 0004 pure-Python layout, 0005
  matplotlib figures, 0006 dependencies (Python ≥ 3.11, torch ≥ 2.2 not installed by us),
  0007 public API `tg.draw(...) → tg.Figure`.
- Architecture overview with the pipeline drawn by Tensorgami itself:
  [`docs/architecture.md`](docs/architecture.md).
- Fitness tests: `tests/test_architecture.py` enforces the package import rules and where
  torch / matplotlib may be imported; the suite passes on Python 3.11 and 3.12 with
  byte-identical figures.

### 5) Intermediate representation (IR) ✅
- `src/tensorgami/ir/`: immutable `Graph` of `Node` (op, category, module path, call index,
  params, output shapes), `Edge` (ports, shape, kind), `Group` (one *invocation* of a module:
  `cell`, `cell#1`, … so recurrent steps stay separable) and `Repeat` (LAYERS or TIME).
- `GraphBuilder` for importers/tests (auto ids, ports and shapes); query helpers
  (predecessors, children, members, nodes_under, depth…).
- 7 validation rules (unique ids, group tree, edge endpoints/ports, shape consistency,
  acyclic data flow with state/tied loops allowed, inputs/outputs, repeats).
- Versioned JSON (`tensorgami.ir` v1), lossless round trip; `format_shape` → `64×56×56`.
- Tests: 53 new (reference graphs for residual CNN, ConvLSTM, VAE, Transformer in
  `tests/ir_fixtures.py`, reused by later steps); 2 000-node graph validates in ~7 ms.

### 6) PyTorch importer ✅
- `src/tensorgami/importers/torch/`: `trace(model, example_input, overrides=…, leaf_types=…)`
  → `ir.Graph`, by eager execution tracing (module hooks + `TorchFunctionMode`, ADR-0003).
  Small modules: `catalog` (categories, leaf modules, kept hyper-parameters), `recorder`,
  `convert` (records → IR), `isolation` (no side effects), `overrides`, `importer`.
- Handles: functional ops (`x + y`, `cat`, `split`, gates), Python loops (each call of a
  cell is its own group), tuple/dict inputs and outputs, in-place ops, buffers and outside
  constants, and plain indexing such as `h[-1]` that torch 2.2 does not report (recovered
  as a `getitem` view with the right shape). Shape-only ops (`randn_like`, `new_zeros`)
  become sources.
- Leaves the model untouched (weights, BatchNorm statistics, train/eval flags, hooks, the
  caller's RNG); explains failures ("missing batch dimension?"); warns about operations not
  in the catalogue instead of failing.
- `graph.summary()` prints a text table of the captured operations.
- Dev environment: torch 2.2.2 (Intel Mac) / torch ≥ 2.2 CPU wheels (Linux), `numpy<2`
  on Intel Mac; `tensorgami[torch]` extra for users.
- Tests: 36 new on 7 reference models in `tests/models/` (CNN with residual blocks, LSTM,
  cGAN generator + discriminator, VAE, ConvLSTM encoder, Transformer), including
  "model untouched" checks (verified to fail when the RNG protection is removed).

### 7) Graph transforms (levels of detail) ✅
- `src/tensorgami/transforms/`: small `Graph → Graph` transforms composed by `Pipeline`:
  `DetectRecurrence` (step-to-step edges become *state*, steps a TIME repeat),
  `DetectLayerRepeats` (identical chained siblings), `RemoveLayoutOps` (view/permute/getitem…),
  `RemoveConstants` (zero initial states), `FoldIntoPrevious` (activation, dropout, scalar
  arithmetic → `params["folded"]`), `CollapseCells` (custom recurrent/attention modules →
  one node per call), `FoldRepeats` (×N stacks and time steps kept once with `count = N`;
  folded time steps get state self-loops), `CollapseGroups`, `CollapseToSystem`,
  `PruneEmptyGroups`.
- Level presets: `simplify(graph, level)` with L0 system · L1 stages · **L2 layers (default)**
  · L3 operators · L4 (= L3 until the cell view, step 11); `fold_time=False` keeps
  recurrent steps unrolled.
- IR: `count` on nodes and groups (a folded copy stands for N).
- Results on the reference models (raw → L2): ConvLSTM 82 → 5 (two cells ×3 with h/c state
  loops), CNN 23 → 12 (residual block ×2), Transformer 28 → 13 (encoder layer ×2).
- Tests: 66 new (hand-built graphs without torch + real traces); 48-layer stack → L2 in ~30 ms.

### 8) Layout engine ✅
- `src/tensorgami/layout/`: `LayeredLayout` (pure Python, no Graphviz) behind the
  `LayoutEngine` protocol; `layout(graph, theme, options)` → `Scene`.
- Recursive compound layout: every module group is laid out first and placed as one block,
  so frames never overlap other boxes; ranks → border **ports** (residual/U-Net skips enter
  or leave mid-block through their own lanes) → lane slots for long edges → barycentre
  ordering → alignment of connected boxes → gaps sized for shape labels.
- Orthogonal routes, state self-loops with delay markers, ×N badges, frame titles
  (attribute name for `Sequential`), balanced row wrapping for wide models, top→bottom
  direction, automatic legend.
- `Labeler`: style-baseline text per operation ("Conv 3×3" / "64, s2, ReLU", "+", "‖",
  "Multi-head attention" / "8 heads", "ε ~ N(0, I)"); shapes on edges only where they change.
- Preview gallery drawn fully automatically from the reference models: `docs/gallery/`.
- Tests: 127 new — geometric invariants (no overlaps, boxes inside their frames, frames never
  cut foreign boxes, orthogonal edges, **no edge crosses a box it does not connect**, forward
  flow) on hand-built graphs and on real traces of all reference models, at levels L0–L3, in
  both directions; label rules; determinism. 338-node graph lays out in ~60 ms.
- Moved to step 11 (views): the time × depth grid for *unrolled* recurrence (the general
  layout draws it correctly but as a staircase) and the U-shape for encoder/decoder skips.

### 9) Matplotlib figure backend ✅
- `src/tensorgami/render/mpl.py`: `MatplotlibCanvas` implements the same `Canvas` protocol as
  the SVG canvas, so both draw the identical scene (1 px = 0.75 pt; `dpi` alone sets the
  PNG resolution). Installed fonts of the style's font stack only (no warnings), weights
  snapped like browsers do, `x_{t}` markup → mathtext, explicit paint order.
- `src/tensorgami/api/figure.py`: **`tg.Figure`** — inline in Jupyter (rendered at 2× for
  sharp screens, shown at natural size), `fig.save("model.png", dpi=300)` (also `.svg`
  — Tensorgami's own vector output — `.pdf`, `.jpg`, `.tif`), `fig.show()` for scripts,
  `fig.figure` / `fig.ax` for matplotlib access, `fig.draw_on(ax)` to place a diagram in a
  panel of your own `plt.subplots` layout (text and lines scale with the panel).
- Translucent fills in both canvases (prepared for the `origami` style).
- matplotlib is now the single runtime dependency; `import tensorgami` stays light (it loads
  matplotlib only when a figure is drawn).
- Fixed on the way: an arrow entering a frame exactly at its first box lost its arrowhead
  (zero-length last segment) in SVG and PNG; routes now drop repeated points and the layout
  invariants check every arrow ends with a real segment.
- `examples/alexnet.py` writes SVG + 300 dpi PNG; `examples/alexnet.ipynb` shows inline
  display, saving and multi-panel use.
- Tests: 17 new — PNG pixels sampled inside every box match the box colours (geometry
  identical to the SVG), output formats, notebook display, no leftover pyplot figures,
  subplot scaling, quiet fonts, light import.

### 10) Public API ✅ (2026-10-11)
- `tg.draw(model, example_input, *, level=2, view="auto", style="default", overrides=None,
  direction=None, **layout_options) → tg.Figure` (ADR-0007, ADR-0008), in
  `src/tensorgami/api/draw.py`; `import tensorgami` stays light (matplotlib loads with the
  figure, torch only when a model is drawn).
- Options after the input are keyword-only. `level` by number or name (`"stages"`);
  `style` by name, path or `Theme`; any `LayoutOptions` field as a keyword (`legend=False`).
- Views: a `View` protocol with a registry (`tensorgami.views`); `block` and `auto` now,
  `volume`, `unrolled`, `cell` in step 11 (asking for them says so).
- Clear errors, raised **before the model runs**: misspelt options ("did you mean
  'direction'?", `theme=` → `style=`), unknown styles, levels, directions and views with the
  valid choices; `LayoutOptions` checks every value's type. Model failures keep their hints
  (missing batch dimension…), and passing the class instead of a model says
  "create it first, e.g. AlexNet()". Warnings point at the user's own `tg.draw` line.
- Fixed on the way: a `label` from `overrides` now names the box in the figure (it reached
  the graph but the labeler ignored it for known layers).
- The gallery, the AlexNet example and its notebook use `tg.draw`; every figure stayed
  byte-identical.
- Tests: `tests/test_draw.py`, end to end from each reference model to a PNG in both styles,
  no side effects (weights, training flags, RNG, open figures, files), errors.

## Phase C — Your model families

### 11) Views ✅ (11a unrolled, 11b U-shape, 11c volume, 11d cell, 11e generative; 2026-10-12)
- Plan, in parts: **11a** unrolled time view ✅ · 11b U-shape layout for encoder–decoder skips
  and folding activations that come *before* a layer · 11c tensor-volume view for CNNs ·
  11d cell view (LSTM/GRU gates) · 11e generative figures (below).
- **11a Unrolled time view** ✅ (2026-10-11): `view="unrolled"`, and `view="auto"` picks it
  when a module is called step after step with state passed along (ADR-0007).
  - `UnrollRecurrence` (transform): traced loops (TIME repeats; one node per call) and fused
    `nn.LSTM`/`GRU`/`RNN` (expanded into `num_layers × steps` cells, the time axis from
    `batch_first`) become one grid group; an outside tensor feeding every step becomes the
    per-step inputs x₁…x_T (time axis dropped from the shape), a sequence read from every step
    the outputs h₁…h_T; longer than `max_steps` (4): the first steps, "⋯" and the last one,
    with the state bridged. Bidirectional fused modules stay one box (for now).
  - `UnrolledLayout`: the layered engine with each grid group laid out as a block: a column
    per step, a row per layer (the first at the bottom), x_t below on an input bus, state
    along the rows, data up the columns, one shape label per layer boundary; top → bottom
    figures turn the grid (time down, layers across). Blocks can now choose the side an edge
    enters by and carry notes ("⋯").
  - Gallery: ConvLSTM unrolled (auto) and folded (`view="block"`), LSTM unrolled.
- **11b U-shape** ✅ (2026-10-11): encoder–decoders whose skips nest (U-Net, pix2pix) fold
  into a U in the block view (`u_shape=False` keeps them on one line).
  - `FoldUShape` (transform): ≥ 3 skips nested inside each other (an encoder tensor reaching
    the merge of its decoder level) → the nodes from the outermost skip's source to its merge
    go into one `u-shape` group, each marked with its level and side (encoder, decoder,
    bottleneck); the modules inside lose their frames. Residual skips (one after another)
    are untouched.
  - U layout (block layout): a row per level, encoder layers on the left, decoder layers on
    the right, the skip a short arrow across the middle, the bottleneck in its own row; top
    → bottom figures turn it (levels become columns). The pix2pix generator (8 levels): 1086 ×
    964 px instead of 8195 × 1177.
  - Activations that open a module (pix2pix's in-place `LeakyReLU → Conv`) now fold into the
    layer before them, across the module boundary (activations only; a `Dropout` opening a
    classifier stays); the decoder's ReLU after each concat folds into the merge circle.
  - Merge notes (e.g. a folded ReLU) sit on a side of the circle no wire uses. Rows wrap only
    when that makes a figure ≥ 20 % narrower (no row for a lone pill).
  - Specialised layouts are now block layouts registered by group type in the layered
    engine (ADR-0004); pix2pix generator and discriminator joined the reference models.
- **11c Volume view** ✅ (2026-10-11): `view="volume"` draws every layer whose output is an
  image tensor (N, C, H, W) as a volume (face height ∝ √H, oblique depth ∝ √W, thickness ∝
  √C), the input image too, and flat vectors (Flatten, Linear) as thin bars; each named with
  its shape below, as in the AlexNet paper. Normalisation, activations and dropout fold into
  the layer before them (across modules too: a volume is a layer's output); merges keep their
  circles; always left → right, no shapes on the wires. `MarkVolumes` (transform) +
  `params["volume"]` read by the layered layout, which reserves the volume's footprint and
  centres its front face so wires stay level.
- **11d Cell view** ✅ (2026-10-12): `view="cell"` draws a model's recurrent cells as their
  gates, one framed cell per distinct cell type (`cell_figure`, a new graph):
  - LSTMs as the classic figure (a block layout, `lstm-cell`): the cell state along the top
    (× forget, + input), the four gates in a row (σ forget, σ input, tanh candidate, σ
    output), tanh(c) and the output × on the right, the hidden state along the bottom where
    h_{t-1} joins x_t; shapes on the state and input pills.
  - Fused `nn.LSTM`/`LSTMCell` from the module's settings; custom cells recognised as LSTMs
    from their data flow (the σ multiplied with the incoming state is the forget gate, the σ
    with the tanh candidate the input gate, the σ with tanh(c) the output gate), their gates
    coloured as the layer computing them (ConvLSTM: a convolution).
  - GRU (reset, update, candidate, 1−z) and RNN cells: their textbook gates, laid out by the
    layered engine (a GRU template would be tidier: its h_{t-1} feeds three places); any
    other cell (a module called in a loop with state): its traced operations.
  - Block layouts can keep their frame (`framed = True`); `params["caption"]` sets a node's
    second line.
- **11e Generative figures** ✅ (2026-10-12): what no `nn.Module` contains (FR11).
  - Several models in one figure: `tg.draw({"generator": G, "discriminator": D}, {name:
    example input}, links=[…])`; each model traced, framed and named (`compose`), ids and
    module paths prefixed with its name; levels count from the models (level 0: each model
    one box, the classic GAN overview).
  - `links=[(source, target[, label])]`: a tensor feeds another model's input, which
    disappears (the generator's output becomes the image its discriminator judges; one
    condition feeds both).
  - `annotations=[…]`: `tg.Loss(name, *tensors, trains=[…])` (a loss box; dashed training
    arrows from it, along a lane under the figure, into the frames of the models it trains,
    through a clear column or round the left), `tg.Input(name, to=…)` (e.g. the real images),
    `tg.Note(text, at=…)` (a second line). References: input/output names, keyword names,
    module paths, `model.name`; unknown ones list the names there are.
  - Training edges may close a loop (gradients flow backwards). Fixed on the way: nested
    frames are painted outermost first (an outer frame no longer covers an inner one).
  - Gallery: the cGAN (overview and layers), the VAE with its reconstruction and KL losses.
- Examples: `examples/` holds only scripts and notebooks; the notebooks (`alexnet`,
  `pix2pix`, `tour`) are run and keep their figures. AlexNet's figures moved to the gallery
  (`docs/gallery/alexnet*.svg`, every style, with the volume view).
- Views: block (default), tensor-volume view for CNNs, cell view (LSTM/GRU gates), unrolled
  time view (RNN, LSTM, ConvLSTM).
- Generative models: VAE (sampling node, μ/σ, loss terms) and cGAN (generator +
  discriminator in one figure, condition inputs, adversarial loop). Losses and the GAN loop
  are not part of `nn.Module`, so the API accepts annotations and several models per figure.
- Views are `View` strategies selecting transforms + layout + glyph mapping.

### 12) Acceptance gallery ✅ (2026-10-12)
- [`docs/gallery/`](docs/gallery/README.md): the reference models in `tests/models/` (CNN,
  LSTM, ConvLSTM, cGAN, VAE, Transformer, and the user's pix2pix), AlexNet, and composed
  figures (a cGAN with its adversarial loop, a VAE with its losses, pix2pix whole in 3-D), in
  every view that suits them (block, stages, unrolled, volume, cell, U-shape) and in every
  built-in style: 23 figures × 3 styles, drawn by `build_gallery.py` with `tg.draw`.
- `tests/test_gallery.py` accepts them: every figure keeps the layout invariants (no
  overlapping boxes, no wire through a box, orthogonal wires, boxes inside their frames) and
  matches its committed SVG byte for byte; no stray files; the gallery page shows each one.
  A change to the drawing fails until it is reviewed on the page and accepted by rebuilding.
- SVG is the snapshot format (deterministic and diffable; GitHub shows it); PNGs differ
  between matplotlib versions and fonts, so they are made on demand (`fig.save("x.png")`).
- On the way: the volume view folds U-Nets too (the U-Net paper's figure); a volume's face
  sits in the middle of its slot (wires meet it straight), and a volume hands down from its
  side (its name stays clear); each of several models is always framed; a U's skips start at
  an encoder layer and stay inside one model; a model's name refers to its linked output;
  training arrows rise in separate columns. The pix2pix notebook draws the whole system in
  3-D (`ConditionalDiscriminator` stacks A and B, as pix2pix's training code does).
- Then, for a compact 3-D pix2pix: volumes ~30 % smaller; a U's rows packed by each node's
  own reach (a volume's oblique top above, its name below) and closer together; `u_levels=N`
  (layout option) draws at most N levels of a U, with ⋮ for the middle ones (8-level
  generator: 1186 → 737 px tall); training arrows are labelled with their loss where a model
  is trained by several (pix2pix's generator: GAN and L1), labels spread apart.
- A loss loops back into the model right beside it over the top, counterclockwise (out of
  its side, up past the output, down into the model: GAN loss → discriminator); into models
  further away its arrows run underneath.

## Phase D — Reports, presentations and release

### 13) Styles and presentation output 🔶 (`default`, `origami`, `origami-extended` ✅; third to decide)
- Named styles, one TOML file each, chosen with `style=` (ADR-0008): `load_style("origami")`
  or `tg.draw(model, x, style="origami")`; a style may set layout preferences (`[layout]`:
  spacing, direction, wrapping) and box sizes (`[boxes]`).
- **`default`** ✅: the current light, print-friendly look.
- **Origami styles** ✅ (2026-10-11, [`docs/style/origami.md`](docs/style/origami.md)), after the
  radial figure in DeepMind's essay *Bending the curve of discovery*: hairline wires with dot
  terminals, folded-paper cards in translucent pastels with the same category meaning, solid
  hairline frames with capital titles, serif labels. They differ in how a card holds its text:
  - **`origami`**: left → right by default, with **upright cards** (one line of text reading
    bottom → top, the fold on top, all cards equally tall, upright tensor shapes between
    them): a layer takes ~ 60 px of width (AlexNet 1047 px in one row, the pix2pix U-Net
    generator 3464 px instead of 9306). Top → bottom: flat single-line cards.
  - **`origami-extended`**: top → bottom by default, with **flat single-line cards** of equal
    width, name left and details right, as in the Transformer paper's block flowcharts
    (survey Fig. 1d): figures fit a notebook's width and scroll down. Left → right: name
    above details, each card as wide as its text.
  - Chosen with the user from side-by-side comparisons at the same scale (the first origami
    look, with two-line cards as wide as their text, was replaced).
  - Renderer capabilities added, usable by any style: `folded` shape, fill `opacity`, dot
    terminals, unfilled frames, `caps` fonts; the `[boxes]` table (text `stacked` /
    `inline` / `rotated` per flow direction, padding, heights, equal lengths); rotated text
    in both canvases; upright edge labels beside upright cards. `default` is unchanged.
  - Fixed on the way (all styles): frame titles are painted above the wires on a small
    background patch; wrapped edges bend outside frames; rows wrap only when that makes a
    figure at least 20 % narrower.
- **Third style:** to be decided.
- Presentation variants: slide (16:9 sizing, larger fonts), dark, greyscale/print.
- Export for LaTeX reports (PDF/SVG at exact column widths).

### 14) Dynamic information flow ⬜
- Animate how information flows through the architecture: layers light up in execution
  order, tensor shapes evolve along the edges, recurrent steps unfold over time.
- Outputs: GIF/MP4 for presentations (`matplotlib.animation`) and interactive playback in
  Jupyter; optionally show real activations (feature maps) from the example input.

### 15) Packaging, docs & CI 🔶 (prepared 2026-10-12; the first release waits for the PyPI setup)
- Decided with the user: a public package on PyPI as `pip install tensorgami` (the name is
  free on PyPI and TestPyPI), **MIT licence**; PyTorch stays an optional extra
  (`tensorgami[torch]`), matplotlib the only dependency.
- Licences: `LICENSE` (MIT); `THIRD_PARTY_NOTICES.md` keeps the BSD notices of the code
  adapted from torchvision (AlexNet, `examples/alexnet.py`) and from
  pytorch-CycleGAN-and-pix2pix (`tests/models/pix2pix.py`, `examples/pix2pix.ipynb`).
- Metadata: one version source (`__version__`, read by hatch), classifiers, project URLs; the
  sdist holds the package and the top-level documents only; `CHANGELOG.md` (Keep a
  Changelog), `CITATION.cff`.
- CI (`.github/workflows/ci.yml`, every push and pull request): ruff, the tests on Python
  3.11–3.13 with CPU PyTorch, and the built wheel installed alone (no PyTorch) drawing in
  every style, with the newest dependencies and with the oldest accepted (matplotlib 3.7,
  whose full test run also passes).
- Release (`.github/workflows/release.yml`, a `vX.Y.Z` tag): the tag must match the version;
  build, `twine check`, the same smoke test; TestPyPI, then PyPI after an approval (Trusted
  Publishing: no tokens); a GitHub Release with the files. Steps in
  [`docs/RELEASING.md`](docs/RELEASING.md); `tests/test_release.py` keeps version, citation,
  changelog, licence, classifiers and the README's absolute links consistent.
- README rewritten for PyPI: install, quick start, what can be drawn, gallery figures. Its
  links are relative (GitHub shows them in a private repository too); the build makes them
  absolute for the PyPI page (`hatch-fancy-pypi-readme`).
- Before the first release (2026-10-13, with the user): a ResNet example
  (`examples/resnet.py`: ResNet-18/34/50 as in torchvision; `resnet.ipynb`, with how to read
  the figure) and ResNet-18 in the gallery; wide figures fold into rows inside frames too
  (every layout level wraps when longer than `wrap_width` less the frames around it; rows
  read like lines of text, chosen over a snake fold; a pill stays beside a frame that nearly
  fills the width): AlexNet at the operator level 2248 → 1278 px wide, ResNet-50 9239 → 1850.
- Left to the user: make the repository public, register the pending publishers on PyPI and
  TestPyPI, create the `testpypi` / `pypi` environments, set `PUBLISH_TO_PYPI`, push the tag.

## Phase E — Backlog

### 16) Other backends ⬜
- ONNX, Keras/TensorFlow, JAX importers behind the same `ModelImporter` protocol.

### 17) Declarative model spec ⬜
- YAML / Python builder for drawing architectures without code (e.g. from a paper).

### 18) Extras ⬜
- Interactive HTML viewer (collapse/expand, hover), TikZ export, command-line interface.
