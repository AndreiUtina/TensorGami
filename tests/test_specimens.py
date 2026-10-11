"""The style specimens double as end-to-end and golden-file tests of the rendering stack."""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[1] / "docs" / "style"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_specimens", DOCS / "build_specimens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


@pytest.mark.parametrize("name", list(builder.SPECIMENS))
def test_specimen_scene_fits_its_canvas(theme, name):
    scene = builder.SPECIMENS[name](theme)
    for node in scene.nodes:
        b = node.box
        assert 0 <= b.x and b.right <= scene.width, f"{name}: node '{node.id}' outside x range"
        assert 0 <= b.y and b.bottom <= scene.height, f"{name}: node '{node.id}' outside y range"
    for group in scene.groups:
        assert group.box.right <= scene.width and group.box.bottom <= scene.height


def test_build_writes_well_formed_svgs(tmp_path, theme):
    paths = builder.build(tmp_path, theme)
    assert [p.stem for p in paths] == list(builder.SPECIMENS)
    for path in paths:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        assert root.tag.endswith("svg")


@pytest.mark.parametrize("style", ["default", "origami"])
def test_committed_specimens_are_up_to_date(tmp_path, style):
    """Fails when a style changed but its specimens were not rebuilt."""
    from tensorgami.style import load_theme

    for path in builder.build(tmp_path, load_theme(style)):
        committed = builder.specimen_dir(style) / path.name
        assert committed.exists(), f"missing {committed}; run docs/style/build_specimens.py"
        assert committed.read_text(encoding="utf-8") == path.read_text(encoding="utf-8"), (
            f"{committed} is stale; run: uv run python docs/style/build_specimens.py"
        )
