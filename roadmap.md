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
- Core package has **zero runtime dependencies**; framework importers (PyTorch, ONNX, Keras)
  are optional extras.

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

### 2) Create a drawing style baseline ⬜
Planned deliverables:
- Style guide: `docs/style/style-baseline.md`
- Style as code: `src/tensorgami/style/` (colours, op categories, glyphs, typography, edges,
  theme loader) with the data in a TOML theme file (`baseline.toml`).
- Minimal scene + SVG rendering layer used to prove the style: `src/tensorgami/scene.py`,
  `src/tensorgami/render/`.
- Reference specimens (vocabulary sheet, CNN flat + volume view, VAE, RNN folded/unrolled,
  LSTM cell, ConvLSTM, Transformer block): `docs/style/specimens/*.svg`.
- Tests: `uv run pytest` (colour maths & contrast, theme loading/validation, glyph painters,
  SVG canvas, scene rendering, specimen smoke test).

### 3) Interview the user if needed at any step 🔶
Steps 0–1 were executed with documented default assumptions. Before Phase B starts the
following answers are needed (they change scope, not the work already done):
1. **Frameworks**: which frameworks are your local models written in? (assumed: PyTorch first,
   then ONNX as a framework-neutral path, Keras later)
2. **Primary purpose**: paper/thesis figures, personal understanding/debugging, or teaching?
   (assumed: publication-quality figures *and* exploration of local models)
3. **Input mode**: automatic extraction from model objects, a hand-written spec (YAML/Python),
   or both? (assumed: both, sharing one intermediate representation)
4. **Outputs**: SVG only, or also PDF/PNG, TikZ for LaTeX, interactive HTML viewer?
   (assumed: SVG first, PDF/PNG second, HTML viewer optional)
5. **Style**: review `docs/style/style-baseline.md` — palette, left→right flow, flat vs. 3-D
   volume default for CNNs, level of detail defaults.
6. **Acceptance models**: which 3–5 of *your* models should be the acceptance tests?
7. **Distribution**: private local install (`uv tool install` from git) or a public PyPI package?

---

## Phase B — Core architecture (suggested)

### 4) Requirements & architecture decision records ⬜
- Turn survey requirements R1–R10 + interview answers into a short requirements list.
- Write ADRs in `docs/adr/` (IR format, layout engine choice, renderer strategy, dependency
  policy). Draw the package diagram with Tensorgami itself as soon as possible (dog-fooding).

### 5) Intermediate representation (IR) ⬜
- `tensorgami.ir`: `Graph`, `Node` (op category, op type, params, tensor shapes),
  `Edge` (tensor flow + shape + kind), `Group` (hierarchy / module tree), `Repeat` (×N).
- JSON serialisation (versioned schema) so graphs can be cached, diffed and hand-edited.
- Validation (acyclic except declared recurrent edges, dangling ports, shape consistency).
- Tests: construction, validation errors, round-trip serialisation.

### 6) Declarative model spec (no framework needed) ⬜
- YAML/TOML spec + fluent Python builder → IR (`tg.Sequential(...)`, `tg.block(...)`).
- Enables drawing paper architectures and models you cannot import.
- Tests: spec parsing, error messages with line numbers, builder ↔ YAML equivalence.

### 7) Graph transforms (levels of detail) ⬜
- Collapse module hierarchy to a target level (L0 system … L4 cell/operator).
- Detect repeated sub-graphs → `×N` groups; fold activation/norm into the previous layer
  label; hide trivial ops (reshape, dropout, identity); merge parallel branches.
- Each transform is a `GraphTransform` object; pipelines are composed (open/closed).
- Tests per transform with tiny graphs.

### 8) Layout engine ⬜
- Pure-Python layered (Sugiyama) layout: rank assignment, crossing minimisation, coordinate
  assignment, orthogonal edge routing — no Graphviz system dependency.
- Specialised layouts behind the same `LayoutEngine` protocol: linear chain, U-shape
  (encoder/decoder with skips), unrolled-in-time grid, stacked block (Transformer).
- Output is a `Scene` (already defined in step 2), so styling/rendering is reused unchanged.
- Tests: no overlaps, edge monotonicity, determinism (same input → same SVG).

## Phase C — Inputs & outputs (suggested)

### 9) Renderers ⬜
- SVG (exists), PDF/PNG (optional extra: cairosvg or matplotlib backend), TikZ export.
- Golden-file snapshot tests for every renderer.

### 10) Framework importers ⬜
- PyTorch: `torch.fx` / `torch.export` tracing with module-hierarchy capture, forward-hook
  fallback for untraceable models; shape inference with a sample input or `meta` device.
- ONNX: read graph + `onnx.shape_inference`.
- Keras: `model.layers` / functional graph.
- Each importer implements `ModelImporter` → IR; installed as optional extras
  (`tensorgami[torch]`, `tensorgami[onnx]`, `tensorgami[keras]`).

### 11) Alternative views ⬜
- Flat block view (default), tensor-volume (3-D cuboid) view for CNNs, cell view (LSTM/GRU
  gates), unrolled-time view (RNN/ConvLSTM), neuron view for tiny MLPs.
- Views are `View` strategies selecting transforms + layout + glyph mapping.

### 12) CLI & Python API ⬜
- `tensorgami draw model.py:build_model --input 1x3x224x224 --level 2 --view flat -o net.svg`
- `tensorgami draw spec.yaml`, `tensorgami themes`, `tensorgami legend`.
- Python: `tensorgami.draw(model, example_input, level=2).save("net.svg")`.

## Phase D — Quality & release (suggested)

### 13) Reference gallery & acceptance tests ⬜
- LeNet, VGG, ResNet block, U-Net, VAE, GAN, RNN/LSTM/GRU, ConvLSTM, Transformer,
  plus the user's own models (interview Q6). Snapshot-tested and published in `docs/gallery/`.

### 14) Interactive HTML viewer (optional) ⬜
- Single self-contained HTML: collapse/expand groups, hover for shapes/params, export.

### 15) Packaging, docs & CI ⬜
- `uv tool install`-able CLI, README with gallery, GitHub Actions (ruff + pytest + coverage).
- Theme authoring guide (custom TOML themes, dark theme, print/greyscale theme).
