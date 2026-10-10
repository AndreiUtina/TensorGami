# ADR-0001: A layered pipeline from model to figure

Status: Accepted · 2026-10-09

## Context

Tensorgami turns a loaded PyTorch model into a figure (requirements FR1–FR3). The roadmap
already plans other inputs (ONNX, Keras, a hand-written spec), several views, themes, and
outputs (PNG, SVG, animation). The project rules require SOLID design and tests for every
important part. The drawing-style baseline (step 2) already defines a `Scene` (positioned
items) and a `Canvas` protocol with an SVG implementation.

## Decision

Tensorgami is a pipeline of independent stages. Each stage talks to the next through a plain
data structure, and each implementation is replaceable behind a small protocol:

```
nn.Module + example input
  └─ ModelImporter ──────────► ir.Graph            (what the model computes)
       └─ GraphTransform* ───► ir.Graph            (the chosen level of detail)
            └─ LayoutEngine ─► scene.Scene         (where everything goes)
                 └─ SceneRenderer + Theme ─► Canvas (how it looks)
                      └─ tg.Figure                 (display, save)
```

A `View` (block, volume, unrolled, cell) is a strategy that chooses the transforms, the
layout engine and the glyph mapping. `tg.draw` only wires the stages together.

Package layout (existing packages marked ✓):

| Package | Responsibility | May import |
|---|---|---|
| `vocabulary` ✓ | `OpCategory`, `EdgeKind` | — |
| `style` ✓ | colours, themes, validation | vocabulary |
| `scene` ✓ | positioned diagram items | vocabulary, style |
| `render` ✓ | `Canvas` protocol, SVG/matplotlib canvases, painters | vocabulary, style, scene |
| `ir` | graph model | vocabulary |
| `importers` | framework → IR (`importers.torch`) | vocabulary, ir |
| `transforms` | levels of detail, folding, repeats | vocabulary, ir |
| `layout` | IR → Scene | vocabulary, ir, scene, style, render.text |
| `views` | strategies combining the above | all of the above except `api` |
| `api` | `draw`, `Figure` | everything |

Dependencies point down the table and never up. In particular, nothing below `importers`
knows about torch, and nothing below `render` knows about matplotlib.

## Alternatives considered

- **One monolithic `draw` function** (the approach of most existing tools). It's fast to
  write, but every new view or framework means editing it, which violates the open/closed
  principle and makes testing expensive.
- **Render straight from the framework graph** (no IR). This ties layout and styling to
  PyTorch and blocks the planned ONNX, Keras and spec inputs.

## Consequences

- Every stage is testable in isolation with tiny hand-made inputs. The existing 85 tests
  already cover `style`, `scene` and `render` this way.
- The import rules in the table are enforced by `tests/test_architecture.py`, so a violation
  fails the build instead of eroding the design.
- Text measurement (`render.text`) is needed by layout. When step 8 starts, it may move to a
  neutral module if `layout → render` becomes more than that one helper.
