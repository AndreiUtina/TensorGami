# Tensorgami — Roadmap

## Main idea
A python-based software that I can install locally and use it to draw the architectures and
information flow of different deep-learning/machine-learning models that I use locally.

## Main rules
1. We work in python for most of the code.
2. We create tests for all important part of the code.
3. Code should follow SOLID design rules.

Conventions derived from the rules (and from the sibling `Bayesball` project):
- Python ≥ 3.12, managed with `uv`; `pytest` for tests, `ruff` for lint/format.
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

### 2) Create a drawing style baseline ✅ (draft, awaiting review)
- Style guide: [`docs/style/style-baseline.md`](docs/style/style-baseline.md) — 10 principles,
  17 colour-coded categories, shapes, 5 edge kinds, typography, layout rules, levels of detail
  L0–L4, views, and the open style questions (§10).
- Style as code: `src/tensorgami/vocabulary.py` (`OpCategory`, `EdgeKind`),
  `src/tensorgami/style/` (colour maths, style records, theme loader, validation rules) with
  the data in `src/tensorgami/style/themes/baseline.toml`.
- Scene + SVG rendering layer used to prove the style: `src/tensorgami/scene.py`,
  `src/tensorgami/render/` (painter per shape, orthogonal edge router, groups, auto legend,
  SVG canvas).
- Reference specimens (vocabulary sheet, CNN block + volume view, VAE, RNN folded/unrolled,
  LSTM cell, ConvLSTM, Transformer encoder): `docs/style/specimens/*.svg`,
  rebuilt with `uv run python docs/style/build_specimens.py`.
- Tests: `uv run pytest` — 85 tests (colour maths & WCAG contrast, theme parsing/validation,
  scene invariants, routing, painters, SVG output, legend, specimens as golden files).

### 3) Interview the user if needed at any step ✅ (style review still open)
Answers (2026-10-09):
1. **Frameworks:** PyTorch only for now; other backends move to the backlog (step 16).
2. **Purpose:** figures for scientific reports and presentations. Later, *dynamic* figures that
   show how information flows through the architecture (step 14).
3. **Workflow:** define and load a model in a Python script or notebook, then call
   TensorGami on it to plot the architecture.
4. **Output:** a figure object that displays inline in Jupyter (or a window from a script);
   saving is optional and explicit, primarily as **PNG**.
5. **Style:** 🔶 still open. The defaults in `docs/style/style-baseline.md` §10 apply until
   reviewed.
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

### 4) Requirements & architecture decision records ⬜
- Turn survey requirements R1–R10 + the interview answers into a short requirements list.
- ADRs in `docs/adr/`: IR format, PyTorch capture strategy (fx tracing vs. hooks), layout
  engine, matplotlib as the figure backend, dependency policy, public API.

### 5) Intermediate representation (IR) ⬜
- `tensorgami.ir`: `Graph`, `Node` (op category, op type, params, tensor shapes),
  `Edge` (tensor flow + shape + kind), `Group` (module hierarchy), `Repeat` (×N).
- Validation (acyclic except declared recurrent edges, dangling ports, shape consistency).
- JSON serialisation (versioned) for caching and test fixtures.
- Tests: construction, validation errors, round-trip serialisation.

### 6) PyTorch importer ⬜
- Capture the graph by running the user's example input: `torch.fx` symbolic tracing with
  module-hierarchy capture; forward-hook fallback for models with Python control flow (LSTM
  loops, custom ConvLSTM cells) that fx cannot trace.
- Record tensor shapes per edge; map `torch.nn` modules and functional ops to `OpCategory`.
- User-defined modules (e.g. a `ConvLSTMCell`) become groups; their category can be given
  explicitly (`categories={"ConvLSTMCell": "recurrent"}`) or guessed from their contents.
- Tests with small reference models (no GPU needed, `meta`/CPU tensors).

### 7) Graph transforms (levels of detail) ⬜
- Collapse module hierarchy to a target level (L0 system … L4 cell/operator).
- Detect repeated sub-graphs → `×N` groups; fold activation/norm into the previous layer
  label; hide trivial ops (reshape, dropout, identity).
- Each transform is a `GraphTransform` object; pipelines are composed (open/closed).
- Tests per transform with tiny graphs.

### 8) Layout engine ⬜
- Pure-Python layered (Sugiyama) layout: rank assignment, crossing minimisation, coordinate
  assignment, orthogonal edge routing — no Graphviz system dependency.
- Specialised layouts behind the same `LayoutEngine` protocol: linear chain (with row
  wrapping), U-shape (encoder/decoder with skips), unrolled-in-time grid, vertical stack.
- Output is a `Scene` (defined in step 2), so styling is reused unchanged.
- Tests: no overlaps, edge monotonicity, determinism.

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

### 13) Presentation output ⬜
- Slide theme (16:9 sizing, larger fonts), dark theme, print/greyscale theme.
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
