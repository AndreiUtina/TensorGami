# ADR-0007: `tg.draw(...)` returns a `tg.Figure`

Status: Accepted · 2026-10-09 · implemented in step 10 (2026-10-11); the `theme=` parameter is
called `style=` ([ADR-0008](0008-named-styles.md))

## Context

The user's workflow is: load the model in a script or notebook, call Tensorgami, look at the
result, and save it only if it is worth keeping (interview Q3, Q4). The API must be small
enough to remember, and open enough for views, themes, overrides and annotations to grow
without breaking calls.

## Decision

```python
import tensorgami as tg

fig = tg.draw(
    model,                    # torch.nn.Module (already loaded, any device)
    example_input,            # tensor | tuple of tensors | dict of keyword arguments
    level=2,                  # 0 system … 4 cell internals
    view="auto",              # "block" | "volume" | "unrolled" | "cell" | "auto"
    style="default",          # "default" | "origami" | path of a TOML style (ADR-0008)
    overrides=None,           # {module path or type name: {"category": ..., "label": ...}}
    direction=None,           # "LR" | "TB"; further layout options as keywords (legend=False)
)
fig                           # Jupyter: renders inline (_repr_png_)
fig.show()                    # scripts: opens a matplotlib window
fig.save("model.png", dpi=300)          # format from the suffix: .png .svg .pdf
fig.figure, fig.ax            # the underlying matplotlib objects
fig.graph, fig.scene          # the IR and the positioned scene (inspection, tests)
```

Rules:

- **Keyword-only options** after `example_input`, so new options never break positional
  calls.
- **No side effects:** `draw` writes no files and changes neither the model nor global
  state (ADR-0003). Saving is always an explicit `fig.save(...)`.
- **The example input is moved to the model's device** when they differ. The model is never
  moved.
- **Errors explain what to do.** Example: "model call failed with the example input: expected
  4-D input, got shape (3, 32, 32). Pass a batch, e.g. `x.unsqueeze(0)`." Fallbacks (an
  unknown op drawn as "other") become warnings that list the affected modules, not failures.
- `view="auto"` picks the view from the graph: unrolled if a module is called repeatedly with
  state passed between calls, the block view otherwise.
  Implemented in step 11a: a traced loop over time (TIME repeat) gives the unrolled view; a
  fused `nn.LSTM` is one call, so it stays a block unless `view="unrolled"` is asked for.

Extension points: `annotations=` for loss terms and extra inputs (FR11) and a mapping of
several models with `links=` for cGANs — both implemented in step 11e (`tg.Loss`, `tg.Input`,
`tg.Note`; `tg.draw({name: model}, {name: input}, links=[…])`) — and `tg.animate(...)` for the
dynamic figures (FR13).

*Fitted to a page (2026-10-15, step 13c):* `tg.draw(..., fit="slide")` returns a figure
made for a fixed canvas: the slide style (larger text and lines), the layout whose text is
largest once it fills the page, centred on the page's proportions. The `Figure` keeps the page
(`fig.fitted`, `fig.inches`) and is saved at its size; at its natural size otherwise.

## Alternatives considered

- **A class-based API** (`tg.Drawer(model).set_level(2).render()`). It is more verbose for
  the common one-line case, and fluent setters hide the moment the model runs.
- **Returning a matplotlib `Figure` directly.** It is familiar, but leaves no home for the
  graph/scene accessors or for saving at the scene's natural size, and couples the API to
  one backend.
- **Saving inside `draw` (`draw(..., path="x.png")`).** This contradicts "decide afterwards
  whether to keep it".

## Consequences

- `tg.Figure` wraps a scene plus a canvas backend, so the same object can later produce SVG,
  PDF or an animation.
- Notebook display relies on `_repr_png_`. The figure must therefore render quickly at a
  screen DPI, independent of the resolution used for saving.
