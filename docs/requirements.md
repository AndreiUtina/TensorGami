# Tensorgami — Requirements

**Status:** baseline for Phase B (roadmap step 4) · 2026-10-09
**Sources:** survey requirements R1–R10 ([survey](survey/tensorgami-literature-survey.pdf)
§6), the interview answers ([roadmap](../roadmap.md) step 3), and the
[style baseline](style/style-baseline.md).

Priorities use MoSCoW: **Must** (needed for the first usable release), **Should** (in scope,
can follow shortly), **Could** (planned for later phases), and **Won't (now)** (backlog).

## 1. Usage scenario

A researcher defines and loads a PyTorch model in a script or Jupyter notebook, calls
Tensorgami with the model and an example input, looks at the figure inline, and saves it as
PNG only when it is worth keeping. The figure goes into a scientific report or a
presentation.

```python
import tensorgami as tg

fig = tg.draw(model, example_input)   # shows inline in Jupyter
fig.save("model.png", dpi=300)
```

## 2. Functional requirements

| ID | Requirement | Priority | Source | Step |
|---|---|---|---|---|
| FR1 | Draw any loaded `torch.nn.Module` from an example input (a tensor, a tuple of tensors, or keyword arguments). | Must | Q1, Q3 | 6, 10 |
| FR2 | Return a figure object that renders inline in Jupyter and opens a window from a plain script (`fig.show()`). | Must | Q4 | 9, 10 |
| FR3 | Save explicitly, primarily as PNG with a chosen resolution; SVG and PDF as well. Nothing is written to disk unless asked. | Must (PNG) · Should (SVG, PDF) | Q4 | 9 |
| FR4 | Levels of detail L0–L4 from one model; default L2 (layers). | Must (L1–L3) · Should (L0, L4) | R1 | 7 |
| FR5 | Tensor shapes on edges (`C×H×W`, batch omitted). | Must | R4 | 6, 8 |
| FR6 | Repeated blocks collapse into one frame with a `×N` badge; trivial ops (activation, reshape, dropout) fold into neighbours at L2. | Must | R5 | 7 |
| FR7 | Automatic legend for the categories and edge kinds used. | Must | R6 | 9 |
| FR8 | Views: block view; unrolled-time view for recurrent models; tensor-volume view for CNNs; gate-level cell view for LSTM/GRU. | Must (block, unrolled) · Should (volume, cell) | R2, R7 | 11 |
| FR9 | The six reference families draw correctly without manual edits: CNN, LSTM, cGAN, VAE, ConvLSTM, Transformer. | Must | Q6 | 12 |
| FR10 | User-defined modules (e.g. a custom `ConvLSTMCell`) appear as named groups; the user can override category, label and level per module. | Must | Q6, R9 | 6, 10 |
| FR11 | Elements that are not part of `nn.Module` can be added: loss terms, sampling notes, condition inputs, and several models in one figure (cGAN generator + discriminator). | Should | R8, Q6 | 11 |
| FR12 | Themes: baseline now; presentation (16:9, larger fonts), dark and greyscale later. | Must (baseline) · Could (others) | Q2 | 13 |
| FR13 | Dynamic figures: animate the information flow (execution order, evolving shapes, unfolding time steps) as GIF/MP4 and in Jupyter. | Could | Q2 | 14 |
| FR14 | Importers for ONNX, Keras/TensorFlow and JAX; a hand-written model spec; an HTML viewer; a CLI. | Won't (now) | Q1 | 16–18 |

## 3. Non-functional requirements

| ID | Requirement | Priority | Step |
|---|---|---|---|
| NFR1 | **Installation:** `pip`/`uv` install from GitHub; runtime dependency only on matplotlib; Tensorgami never installs or upgrades torch (optional extra `tensorgami[torch]`). | Must | 15 |
| NFR2 | **Compatibility:** Python ≥ 3.11; torch ≥ 2.2 (2.2.2 is the last release with Intel-Mac wheels, so it is the local floor). | Must | 4, 6 |
| NFR3 | **Non-invasive:** drawing leaves the model exactly as it was: weights and buffers (e.g. BatchNorm statistics), training/eval mode, hooks, and the RNG state of the caller. | Must | 6 |
| NFR4 | **Deterministic:** the same model and input produce an identical figure. | Must | 6–9 |
| NFR5 | **Performance:** typical research models (≤ 500 modules) draw in < 5 s on a laptop CPU; large models can be traced without real computation (`meta` device). | Should | 6, 8 |
| NFR6 | **Offline:** no network access at run time. | Must | all |
| NFR7 | **Robustness:** unknown operations never crash the drawing (they fall back to the `other` category) and every fallback is reported in plain words. | Must | 6, 10 |
| NFR8 | **Style compliance:** every figure uses a validated theme (contrast, distinct categories, one data-flow line style). | Must | 9 |
| NFR9 | **Code quality:** SOLID design with small protocols, type hints, ruff clean, tests for all important parts; architecture rules checked by tests. | Must | all |

## 4. Acceptance criteria for the first usable release (end of Phase C)

1. `tg.draw(model, x)` works in a notebook for the six reference models in
   `tests/models/` and renders them inline.
2. The PNG output of each reference model matches its approved snapshot in
   `docs/gallery/`.
3. After drawing, every reference model has the same `state_dict`, training flag and hook
   count as before (NFR3, tested).
4. Installing from GitHub into a fresh environment that already has torch adds only
   Tensorgami and matplotlib.

## 5. Traceability: survey requirements → this document

| Survey | Covered by |
|---|---|
| R1 multiple levels of detail | FR4 |
| R2 operator- and tensor-centric views | FR8 |
| R3 semantic, consistent encoding | NFR8, style baseline |
| R4 tensor shapes as annotations | FR5 |
| R5 aggregation of repeated structure | FR6 |
| R6 automatic legend | FR7 |
| R7 time-aware layouts | FR8 |
| R8 stochastic and training-time elements | FR11 |
| R9 automatic extraction + hand-written spec | FR1 (now), FR14 (spec later) |
| R10 deterministic output, no heavy system dependencies | NFR1, NFR4 |
