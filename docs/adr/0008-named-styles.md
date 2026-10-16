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
| `origami` | ✅ ([style page](../style/origami.md)) | hairline "wire" look with translucent folded-paper cards, after the radial figure in DeepMind's essay *Bending the curve of discovery*; upright cards left → right, flat single-line cards top → bottom |
| `origami-extended` | ✅ ([style page](../style/origami.md)) | the origami look with level text: flat single-line cards top → bottom (default), stacked cards left → right |
| third | to be decided | — |

## Variants (2026-10-14, step 13a)

A *variant* recolours a style for where the figure goes, without a TOML file of its own:
`tg.draw(model, x, style="origami", variant="dark")`, or `load_style("origami", "dark")`.
A variant is a function `Theme → Theme`, so it works for every style, built-in or not, and
several apply in order (`variant=["dark", "greyscale"]`). The result is validated like any
style.

- `dark`: a dark board; cards keep their fills and dark text (made opaque), while wires,
  outlines, frames and the text on the background are lightened just enough to pass the
  contrast rules, so each keeps its hue.
- `greyscale`: every colour becomes the grey of the same luminance (every contrast ratio is
  kept); the category pastels, nearly equally light, are spread over a wider range of greys
  in the same order, and wire kinds that differed only by hue are kept apart.

Text drawn beside a node on the background (a volume's name, the note beside a merge circle)
uses `Theme.outside_font`: the node's font, or the background text colour where the node's
would be too faint. The contrast rule judges a translucent card by the colour it shows on the
background.

## Palettes (2026-10-16, step 13)

A *palette* recolours the operation categories of any style: `palette="colorblind"`, `"tab10"`,
`"vivid"` or `"mono"` (`load_style(style, variant, palette)`; the palette first, then the
variants). It gives a base colour per category; the fill is the lightest-needed tint on which
the style's text reaches 4.5 : 1 (opaque and as seen through a translucent card), the outline
the colour darkened to 3 : 1 on the background, and the coloured wires (state, sampling,
training) follow their category. Shapes, fonts and the neutral categories stay the style's.
The colours of `colorblind` and `vivid` were chosen by a search over candidate hues per
category so that every pair stays apart in every style (CIE76 ΔE ≥ 10 for fills, 15 for
outlines); `mono` uses one neutral outline for all, so its categories are told apart by
lightness and label.

## Consequences

- `layout(graph, style)` without explicit options uses the style's layout preferences.
- Snapshot tests (step 12) are kept per style, so changing one style never silently changes
  another.
