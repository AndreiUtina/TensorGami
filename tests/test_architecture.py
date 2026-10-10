"""Architecture fitness tests: the import rules of ADR-0001 and ADR-0006.

Every module under ``src/tensorgami`` belongs to a component (its first package level). A
component may only import the components listed in ``ALLOWED``. A new package must be added
here deliberately, which keeps the dependency direction visible in review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src" / "tensorgami"

_PIPELINE = {"vocabulary", "style", "scene", "render", "ir", "importers", "transforms", "layout"}

ALLOWED: dict[str, set[str]] = {
    "vocabulary": set(),
    "style": {"vocabulary"},
    "scene": {"vocabulary", "style"},
    "render": {"vocabulary", "style", "scene"},
    "ir": {"vocabulary"},
    "importers": {"vocabulary", "ir"},
    "transforms": {"vocabulary", "ir"},
    "layout": {"vocabulary", "ir", "scene", "style", "render"},
    "views": _PIPELINE,
    "api": _PIPELINE | {"views"},
    "__init__": _PIPELINE | {"views", "api"},  # the package facade
}

# third-party packages restricted to one place (ADR-0006)
RESTRICTED = {
    "torch": {"importers", "api"},
    "matplotlib": {"render", "api"},
}


def _component(path: Path) -> str:
    rel = path.relative_to(SRC)
    return rel.parts[0].removesuffix(".py") if len(rel.parts) > 1 else rel.stem


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


MODULES = sorted(SRC.rglob("*.py"))


def test_every_component_is_registered():
    unknown = {_component(p) for p in MODULES} - set(ALLOWED)
    assert not unknown, f"add {unknown} to ALLOWED in tests/test_architecture.py (ADR-0001)"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_imports_respect_the_layering(path: Path):
    component = _component(path)
    for name in _imports(path):
        parts = name.split(".")
        if parts[0] == "tensorgami" and len(parts) > 1:
            target = parts[1]
            assert target == component or target in ALLOWED[component], (
                f"{path.relative_to(SRC)} imports tensorgami.{target}; "
                f"'{component}' may only import {sorted(ALLOWED[component])}"
            )
        elif parts[0] in RESTRICTED:
            assert component in RESTRICTED[parts[0]], (
                f"{path.relative_to(SRC)} imports {parts[0]}, which is only allowed in "
                f"{sorted(RESTRICTED[parts[0]])} (ADR-0006)"
            )


def test_the_checker_detects_violations(tmp_path):
    """Guard against a silently broken checker."""
    bad = tmp_path / "bad.py"
    bad.write_text("import torch\nfrom tensorgami.render import svg\n", encoding="utf-8")
    assert _imports(bad) == ["torch", "tensorgami.render"]
