"""Generated documentation figures must match the code that draws them."""

from __future__ import annotations

import importlib.util
from pathlib import Path

DOCS = Path(__file__).parents[1] / "docs"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_architecture_diagram_is_up_to_date(tmp_path, theme):
    builder = _load(DOCS / "build_architecture.py")
    fresh = builder.build(theme, tmp_path / "architecture.svg").read_text(encoding="utf-8")
    committed = (DOCS / "architecture.svg").read_text(encoding="utf-8")
    assert fresh == committed, "run: uv run python docs/build_architecture.py"
