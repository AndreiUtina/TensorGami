"""Release consistency (roadmap step 15): what a release reads must agree before a tag is
pushed. The version in the package, the citation and the changelog; the license in the
metadata and in LICENSE; the README's links (relative on GitHub, made absolute for PyPI); the
supported Pythons in the classifiers and in CI.
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


def _links(markdown: str) -> list[str]:
    return re.findall(r"\]\(([^)\s]+)\)", markdown)


def _pypi_readme() -> str:
    """README.md as PyPI shows it: the substitutions of hatch-fancy-pypi-readme applied."""
    hook = PYPROJECT["tool"]["hatch"]["metadata"]["hooks"]["fancy-pypi-readme"]
    text = (ROOT / "README.md").read_text()
    for sub in hook["substitutions"]:
        text = re.sub(sub["pattern"], sub["replacement"], text)
    return text


def test_readme_links_to_files_that_exist():
    """Relative, so GitHub shows the images in a private repository and on every branch."""
    relative = [t for t in _links((ROOT / "README.md").read_text()) if "://" not in t]
    assert relative
    missing = [t for t in relative if not (ROOT / t.partition("#")[0]).exists()]
    assert not missing, missing


def test_pypi_readme_links_are_absolute_and_exist():
    """PyPI cannot resolve relative links: the build makes them absolute."""
    links = _links(_pypi_readme())
    relative = [t for t in links if not t.startswith(("https://", "http://", "#"))]
    assert not relative, relative
    raw = f"https://raw.githubusercontent.com/{REPO}/main/"
    blob = f"https://github.com/{REPO}/blob/main/"
    files = [t.removeprefix(raw).removeprefix(blob) for t in links if t.startswith((raw, blob))]
    assert files
    missing = [f for f in files if not (ROOT / f).exists()]
    assert not missing, missing
    images = re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", _pypi_readme())
    assert all(not i.startswith(blob) for i in images)  # images load from the raw files


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
