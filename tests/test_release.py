"""Release consistency (roadmap step 15): what a release reads must agree before a tag is
pushed. The version in the package, the citation and the changelog; the license in the
metadata and in LICENSE; the README's absolute links (PyPI shows the README with them, so a
moved file breaks the project page); the supported Pythons in the classifiers and in CI.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import tensorgami

ROOT = Path(__file__).parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
REPO = "AndreiUtina/TensorGami"


def _cff(key: str) -> str | None:
    match = re.search(rf'^{key}:\s*"?([^"\n]+)"?\s*$', (ROOT / "CITATION.cff").read_text(), re.M)
    return match.group(1) if match else None


def test_version_is_plain_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", tensorgami.__version__)


def test_citation_has_the_package_version():
    assert _cff("version") == tensorgami.__version__


def test_changelog_has_a_section_for_the_version():
    changelog = (ROOT / "CHANGELOG.md").read_text()
    heading = re.search(rf"^## \[{re.escape(tensorgami.__version__)}\](.*)$", changelog, re.M)
    assert heading, f"CHANGELOG.md has no '## [{tensorgami.__version__}]' section"
    released = _cff("date-released")
    if released:  # set at release time, in both places
        assert released in heading.group(1)


def test_license_is_mit_everywhere():
    assert PYPROJECT["project"]["license"] == "MIT"
    text = (ROOT / "LICENSE").read_text()
    assert text.startswith("MIT License")
    for author in PYPROJECT["project"]["authors"]:
        assert author["name"] in text
    assert _cff("license") == "MIT"


def test_sdist_includes_exist():
    for path in PYPROJECT["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]:
        assert (ROOT / path.lstrip("/")).exists(), path


def test_readme_links_into_the_repository_exist():
    readme = (ROOT / "README.md").read_text()
    raw = re.findall(rf"https://raw\.githubusercontent\.com/{REPO}/main/([^)\s\"]+)", readme)
    blob = re.findall(rf"https://github\.com/{REPO}/(?:blob|tree)/main/([^)\s\"#]+)", readme)
    assert raw and blob
    missing = [p for p in [*raw, *blob] if not (ROOT / p).exists()]
    assert not missing, missing


def test_readme_has_no_relative_links():
    """PyPI cannot resolve them: every link and image must be absolute."""
    readme = (ROOT / "README.md").read_text()
    relative = [
        target
        for target in re.findall(r"\]\(([^)]+)\)", readme)
        if not target.startswith(("https://", "http://", "#"))
    ]
    assert not relative, relative


def test_classifiers_match_the_python_versions_tested_in_ci():
    classifiers = {
        c.rsplit(" :: ", 1)[1]
        for c in PYPROJECT["project"]["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    }
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    matrix = re.search(r"python: \[(.*)\]", ci)
    assert matrix
    tested = set(re.findall(r'"(3\.\d+)"', matrix.group(1)))
    assert classifiers == tested
    lowest = min(tested, key=lambda v: int(v.split(".")[1]))
    assert PYPROJECT["project"]["requires-python"] == f">={lowest}"
