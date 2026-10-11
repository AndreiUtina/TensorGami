# ADR-0008: Named styles, selected as an input

Status: Accepted · 2026-10-10 · supersedes the `theme=` parameter name of ADR-0007

## Context

The user adopted the current look as the standard and wants to pass the style as an input
when running Tensorgami (`style="default"`), with more styles added over time: an
**origami** style next, and a third one to be decided. Until now the look was one
"baseline" theme loaded implicitly.

## Decision

- A **style** is one TOML file: colours, fonts, glyph, edge and frame styles, metrics, and
  optional layout preferences (`[layout]`: spacing, direction, wrapping; names as in
  `tensorgami.layout.LayoutOptions`). Built-in styles live in
  `src/tensorgami/style/themes/<name>.toml`, and users can pass a path to their own.
- The current look is the built-in style **`default`** (the file formerly called
  `baseline.toml`) and is used whenever no style is given.
- Selection: `load_style("default")` today; at step 10 `tg.draw(model, x, style="default")`.
  The parameter is called `style`, not `theme` as sketched in ADR-0007. `available_styles()`
  lists the built-ins.
- Option precedence: library defaults < the style's `[layout]` < options passed explicitly.
- New visual capabilities that a style needs, such as translucent fills, folded-paper boxes or
  dot terminals, are added to the renderer once and then referenced *by name* from style files.
  A new style is therefore mostly a new TOML file, and existing styles are untouched
  (open/closed).
- Every style passes the same validation rules (contrast, distinct categories, distinct edge
  kinds) and gets its own specimens and gallery snapshots.

## Planned styles

| Style | Status | Character |
|---|---|---|
| `default` | ✅ | light, print-friendly, semantic colour per operation category |
| `origami` | ✅ ([style page](../style/origami.md)) | hairline "wire" look with translucent folded-paper boxes, after the radial figure in DeepMind's essay *Bending the curve of discovery* |
| third | to be decided | — |

## Consequences

- `layout(graph, style)` without explicit options uses the style's layout preferences.
- Snapshot tests (step 12) are kept per style, so changing one style never silently changes
  another.
