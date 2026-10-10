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

### 9) Matplotlib figure backend ⬜
- `MatplotlibCanvas` implementing the existing `Canvas` protocol (the SVG canvas stays for
  vector output).
- `tg.Figure`: inline display in Jupyter (`_repr_png_`), `.show()` from scripts,
  `.save("net.png", dpi=300)` (PNG first; SVG/PDF also possible), and `.figure`/`.ax` to
  reach the underlying matplotlib objects (e.g. to place a diagram in a multi-panel figure).
- Tests: image-level checks with tolerance plus scene-level assertions.

### 10) Public API ⬜
- `tg.draw(model, example_input, level=2, view="block", theme="baseline", **options)`.
- Sensible defaults, clear errors (e.g. "model could not be traced, falling back to hooks").
- Tests: end-to-end from a `torch.nn.Module` to a PNG.

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

### 13) Styles and presentation output 🔶 (`default` ✅, `origami` planned)
- Named styles, one TOML file each, chosen with `style=` (ADR-0008): `load_style("default")`
  now, `tg.draw(model, x, style="origami")` from step 10; a style may set layout
  preferences (`[layout]`: spacing, direction, wrapping).
- **`default`** ✅: the current light, print-friendly look.
- **`origami`** (planned), after the radial figure in DeepMind's essay *Bending the curve of
  discovery*:
  - hairline "wires" (≈ 0.75–1 px, soft slate) with small round dot terminals and fine
    arrowheads;
  - layer boxes as **folded paper**: two translucent pastel facets split by a diagonal
    crease, with hairline outlines, so overlapping elements show through;
  - pastel palette with the same category meaning (sage green, sky blue, lavender, peach,
    blush pink, mint, butter yellow);
  - solid hairline frames with no fill and small-caps grey titles; merges as hollow hairline
    circles;
  - serif labels (e.g. Georgia / Source Serif) with grey sans details, and airier spacing via
    `[layout]`;
  - renderer additions needed (added once, referenced by name from `origami.toml`):
    translucent fills, a folded-paper box painter, dot terminals.
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
