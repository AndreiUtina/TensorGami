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

### 11) Views ⬜
- Block view (default), tensor-volume view for CNNs, cell view (LSTM/GRU gates), unrolled
  time view (RNN, LSTM, ConvLSTM).
- Generative models: VAE (sampling node, μ/σ, loss terms) and cGAN (generator +
  discriminator in one figure, condition inputs, adversarial loop). Losses and the GAN loop
  are not part of `nn.Module`, so the API accepts annotations and several models per figure.
- Views are `View` strategies selecting transforms + layout + glyph mapping.

### 12) Acceptance gallery ⬜
- Reference PyTorch implementations of CNN, LSTM, cGAN, VAE, ConvLSTM and Transformer in
  `tests/models/`, rendered to `docs/gallery/*.png` and snapshot-tested.
- Add the user's own models when available.

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

### 15) Packaging, docs & CI ⬜
- Private install from GitHub first:
  `uv add git+https://github.com/AndreiUtina/Tensorgami` (or `pip install git+…`).
- GitHub Actions (ruff + pytest), version tags, changelog, README with the gallery.
- Later: public PyPI release (name check, trusted publishing).

## Phase E — Backlog

### 16) Other backends ⬜
- ONNX, Keras/TensorFlow, JAX importers behind the same `ModelImporter` protocol.

### 17) Declarative model spec ⬜
- YAML / Python builder for drawing architectures without code (e.g. from a paper).

### 18) Extras ⬜
- Interactive HTML viewer (collapse/expand, hover), TikZ export, command-line interface.
