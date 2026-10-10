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


def wordmark(x: float, y: float, size: float, tensor_color: str) -> str:
    """'TensorGami' in one regular weight: "Tensor" in ``tensor_color``, "Gami" in blue."""
    return (
        f'<text x="{x}" y="{y}" font-family="{WORDMARK_FONT}" font-size="{size}" '
        f'font-weight="500" fill="{tensor_color}">Tensor<tspan fill="{BLUE_700}">Gami</tspan>'
        f"</text>"
    )


def svg(width: float, height: float, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_n(width)}" height="{_n(height)}" '
        f'viewBox="0 0 {_n(width)} {_n(height)}" role="img">\n<title>{title}</title>\n'
        f"{body}\n</svg>\n"
    )


BANNER_W, BANNER_H, BANNER_SCALE = 1600, 400, 2.0
BANNER_BG = "#0d1117"  # GitHub's dark-theme page background


def lockup(tensor_color: str) -> str:
    """Icon + wordmark on a 470 × 128 canvas."""
    placed_icon = f'<g transform="translate(8 8) scale({112 / 256:.4f})">\n{mark()}\n</g>'
    return placed_icon + "\n" + wordmark(132, 80, 46, tensor_color)


def banner() -> str:
    """README banner: the white-"Tensor" logo centred on its own dark card.

    The card makes one image legible on light pages, dark pages and in the GitHub mobile
    app, which supports neither theme-switching <picture> elements nor sized <img> tags.
    """
    dx = (BANNER_W - 470 * BANNER_SCALE) / 2
    dy = (BANNER_H - 128 * BANNER_SCALE) / 2
    return (
        f'<rect width="{BANNER_W}" height="{BANNER_H}" rx="48" fill="{BANNER_BG}"/>\n'
        f'<g transform="translate({_n(dx)} {_n(dy)}) scale({BANNER_SCALE})">\n'
        f"{lockup('#ffffff')}\n</g>"
    )


def to_png(svg_path: Path, png_path: Path) -> Path:
    """Rasterise with typst (dev dependency); 1 SVG px → 1 PNG px, transparent corners."""
    import tempfile

    import typst

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "banner.typ"
        (Path(tmp) / "banner.svg").write_text(svg_path.read_text(encoding="utf-8"))
        source.write_text(
            f"#set page(width: {BANNER_W}pt, height: {BANNER_H}pt, margin: 0pt, fill: none)\n"
            f'#image("banner.svg", width: {BANNER_W}pt)\n',
            encoding="utf-8",
        )
        png_path.write_bytes(typst.compile(str(source), format="png", ppi=72))
    return png_path


def main(out_dir: Path = OUT_DIR) -> list[Path]:
    files = {
        # main logo, for dark backgrounds: white "Tensor"
        "logo.svg": svg(470, 128, lockup("#ffffff"), "TensorGami"),
        # for light backgrounds (papers, slides), where white would vanish: dark "Tensor"
        "logo-light.svg": svg(470, 128, lockup(INK), "TensorGami"),
        "logo-mark.svg": svg(256, 256, mark(), "TensorGami"),
        "logo-banner.svg": svg(BANNER_W, BANNER_H, banner(), "TensorGami"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in files.items():
        path = out_dir / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    paths.append(to_png(out_dir / "logo-banner.svg", out_dir / "logo-banner.png"))
    return paths


if __name__ == "__main__":
    for p in main():
        print(p)
