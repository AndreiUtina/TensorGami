"""Build the literature-survey PDF: regenerate figures, then compile the Typst source.

    uv run python docs/survey/build.py [--png PREVIEW_DIR]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import typst

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import make_figures  # noqa: E402

SOURCE = HERE / "survey.typ"
PDF = HERE / "tensorgami-literature-survey.pdf"
MAX_PAGES = 10


def count_pages(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page(?![A-Za-z])", pdf))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=Path, help="also render page previews into this folder")
    args = parser.parse_args()

    make_figures.main()
    pdf, warnings = typst.compile_with_warnings(str(SOURCE), root=str(HERE))
    for w in warnings:
        print(f"typst warning: {w}", file=sys.stderr)
    PDF.write_bytes(pdf)
    pages = count_pages(pdf)
    print(f"wrote {PDF.relative_to(HERE.parent.parent)} ({pages} pages)")

    if args.png:
        args.png.mkdir(parents=True, exist_ok=True)
        typst.compile(str(SOURCE), output=str(args.png / "page-{n}.png"), format="png", ppi=110,
                      root=str(HERE))
    if pages > MAX_PAGES:
        print(f"error: report exceeds {MAX_PAGES} pages", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
