"""The Tensorgami logo: "Folded Layers" mark + "TensorGami" wordmark.

An accordion-folded sheet where every crease is a layer of neurons (3 → 4 → 4 → 2) and the
connections run across the paper panels: origami folds as network depth. Colours come from
the drawing-style baseline. Rebuild with::

    uv run python docs/brand/make_logo.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

OUT_DIR = Path(__file__).parent

Pt = tuple[float, float]

INK = "#0f172a"
BLUE_100, BLUE_300, BLUE_700, BLUE_900 = "#dbeafe", "#93c5fd", "#1d4ed8", "#1e3a8a"
ORANGE_400 = "#fb923c"
WORDMARK_FONT = "'Avenir Next', 'Avenir', 'Helvetica Neue', Helvetica, Arial, sans-serif"

# Mark geometry on a 256 × 256 grid
FOLD_X = [34, 96, 160, 222]  # x of each crease (= layer)
HIGH, LOW = (46, 186), (78, 218)  # (top, bottom) of creases that point back / forward
FOLDS = [HIGH, LOW, HIGH, LOW]
NEURONS = [3, 4, 4, 2]  # neurons per layer
PANEL_TONES = [BLUE_100, BLUE_300, BLUE_100]  # lit, shaded, lit


def _n(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _layer(i: int) -> list[Pt]:
    top, bottom = FOLDS[i]
    step = (bottom - top) / (NEURONS[i] + 1)
    return [(FOLD_X[i], top + step * (k + 1)) for k in range(NEURONS[i])]


def mark() -> str:
    """The icon, drawn on a 256 × 256 grid."""
    parts = []
    for i, tone in enumerate(PANEL_TONES):
        (t0, b0), (t1, b1) = FOLDS[i], FOLDS[i + 1]
        x0, x1 = FOLD_X[i], FOLD_X[i + 1]
        pts = " ".join(f"{_n(x)},{_n(y)}" for x, y in [(x0, t0), (x1, t1), (x1, b1), (x0, b0)])
        parts.append(
            f'<polygon points="{pts}" fill="{tone}" stroke="{BLUE_700}" stroke-width="3.5" '
            f'stroke-linejoin="round"/>'
        )
    layers = [_layer(i) for i in range(len(FOLD_X))]
    for a_layer, b_layer in itertools.pairwise(layers):
        for (x1, y1), (x2, y2) in itertools.product(a_layer, b_layer):
            parts.append(
                f'<line x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}" '
                f'stroke="{BLUE_900}" stroke-width="2.4" stroke-opacity="0.55" '
                f'stroke-linecap="round"/>'
            )
    for i, layer in enumerate(layers):
        fill = ORANGE_400 if i == len(layers) - 1 else "#ffffff"
        parts += [
            f'<circle cx="{_n(x)}" cy="{_n(y)}" r="8.5" fill="{fill}" stroke="{INK}" '
            f'stroke-width="3.5"/>'
            for x, y in layer
        ]
    return "\n".join(parts)


def wordmark(x: float, y: float, size: float) -> str:
    """'TensorGami': one regular weight; the capital G and the blue mark the two halves."""
    return (
        f'<text x="{x}" y="{y}" font-family="{WORDMARK_FONT}" font-size="{size}" '
        f'font-weight="500" fill="{INK}">Tensor<tspan fill="{BLUE_700}">Gami</tspan></text>'
    )


def svg(width: float, height: float, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_n(width)}" height="{_n(height)}" '
        f'viewBox="0 0 {_n(width)} {_n(height)}" role="img">\n<title>{title}</title>\n'
        f"{body}\n</svg>\n"
    )


def main(out_dir: Path = OUT_DIR) -> list[Path]:
    icon = mark()
    placed_icon = f'<g transform="translate(8 8) scale({112 / 256:.4f})">\n{icon}\n</g>'
    lockup = placed_icon + "\n" + wordmark(132, 80, 46)
    files = {
        "logo.svg": svg(470, 128, lockup, "TensorGami"),
        "logo-mark.svg": svg(256, 256, icon, "TensorGami"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in files.items():
        path = out_dir / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in main():
        print(p)
