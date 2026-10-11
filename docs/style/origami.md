# Tensorgami drawing style — `origami`

**Status:** built-in style (2026-10-11, ADR-0008) · **Style file:**
[`src/tensorgami/style/themes/origami.toml`](../../src/tensorgami/style/themes/origami.toml)

A lighter, more illustrative alternative to the [`default`](style-baseline.md) style, after
the radial figure in DeepMind's essay *Bending the curve of discovery*: thin "wires", paper,
pastels and white space. It keeps every rule of the default style (colour = operation
category, one line style for data flow, shapes where they change, ×N for repeats) and passes
the same validation (text contrast ≥ 4.5:1, outlines ≥ 3:1, distinct category colours).

```python
import tensorgami as tg

fig = tg.draw(model, example_input, style="origami")
```

![Origami vocabulary](specimens/origami/vocabulary.svg)

## What makes it origami

| Element | `default` | `origami` |
|---|---|---|
| Layers | rounded boxes, solid tint | **folded paper cards**: a slanted crease and a slightly darker folded strip on the right; sharp corners; translucent pastel |
| Lines | 1.25 px, filled arrowheads | **hairline wires** (0.9 px, soft slate) starting with a small **dot terminal**, fine arrowheads |
| Frames | dashed, light fill | **solid hairline, no fill**, small capital titles |
| Labels | sans-serif, semibold | **serif** (Source Serif / Georgia), regular; details in grey sans |
| Palette | saturated tints | pastels: sky, lavender, sage, peach, butter, blush, mint, sand, coral |
| Spacing | standard | airier (wider gaps and frame padding, from the style's `[layout]`) |

The text of a folded card sits on its flat face: the layout widens folded cards by the
strip's width, so the crease never crosses a label.

## Examples

AlexNet ([`examples/alexnet.py`](../../examples/alexnet.py)):

![AlexNet, origami](../../examples/alexnet-origami.svg)

ConvLSTM with recurrent state loops, and the VAE with its sampling node:

![ConvLSTM, origami](../gallery/origami/convlstm.svg)

![VAE, origami](../gallery/origami/vae.svg)

More in the [gallery](../gallery/README.md#origami-style) and in
[`specimens/origami/`](specimens/origami/).

## How it is built

Origami is one TOML file. The renderer gained five capabilities, which any style can now use
by name: the `folded` shape, translucent fills (`opacity`), dot terminals
(`start = "dot"` / `arrow = "dot"`), frames without a fill (`fill_opacity = 0`) and
`caps = true` fonts. The `default` style does not use them; its only visible change is shared
with origami: frame titles and ×N badges are now drawn above the wires, on a small patch of
the frame's background, so a wire never runs through a title.
