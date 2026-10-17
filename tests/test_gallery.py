"""The acceptance gallery (roadmap step 12): every figure, in every view and style, is drawn
without manual edits, keeps the layout invariants and matches its committed SVG."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

from layout_checks import check_scene  # noqa: E402

from tensorgami.style import available_styles  # noqa: E402
from tensorgami.vocabulary import EdgeKind  # noqa: E402

GALLERY = Path(__file__).parents[1] / "docs" / "gallery"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_gallery", GALLERY / "build_gallery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


@pytest.mark.parametrize("style", available_styles())
def test_gallery_figures_are_sound_and_up_to_date(style):
    stale, missing = [], []
    folder = builder.gallery_dir(style)
    for name, fig in builder.figures(style):
        check_scene(fig.scene, fig.graph, fig.style)  # no overlaps, no wire through a box, …
        committed = folder / f"{name}.svg"
        if not committed.exists():
            missing.append(committed.name)
        elif committed.read_text(encoding="utf-8") != fig.to_svg():
            stale.append(committed.name)
    hint = "review the change, then: uv run python docs/gallery/build_gallery.py"
    assert not missing, f"{style}: missing {missing}; {hint}"
    assert not stale, f"{style}: changed {stale}; {hint}"


def test_every_committed_figure_is_still_drawn():
    names = {name for name, _ in builder.FIGURES.items()} | set(builder.COMPOSED)
    for style in available_styles():
        files = {p.stem for p in builder.gallery_dir(style).glob("*.svg")}
        assert files == names, f"{style}: stray {sorted(files - names)}"


def test_the_gallery_page_shows_every_figure():
    page = (GALLERY / "README.md").read_text(encoding="utf-8")
    shown = {name for name in set(builder.FIGURES) | set(builder.COMPOSED) if f"{name}.svg" in page}
    assert shown == set(builder.FIGURES) | set(builder.COMPOSED), "add the new figures to README.md"


def test_the_gallery_page_shows_every_animation():
    page = (GALLERY / "README.md").read_text(encoding="utf-8")
    for name in builder.ANIMATED:
        assert (GALLERY / "animations" / f"{name}.gif").is_file(), "rebuild: --animations"
        assert f"animations/{name}.svg" in page, "add the new animation to README.md"


@pytest.mark.parametrize("name", builder.ANIMATED)
def test_animated_svgs_match_the_committed_ones(name):
    """Step 14c: the animated SVGs, like the still ones, change only on purpose."""
    committed = (GALLERY / "animations" / f"{name}.svg").read_text(encoding="utf-8")
    assert builder.figure(name).animate().to_svg() == committed, (
        f"{name}.svg changed: review it, then rebuild with build_gallery.py --animations"
    )


def test_every_gallery_figure_animates_and_ends_as_its_still():
    """Steps 14a, 14b: every view, composite and fitted figure lights up step by step, the
    last step is the still figure, and every wire into a layer carries data at some point."""
    for name, fig in builder.figures("default"):
        anim = fig.animate()
        steps = anim.steps
        assert len(steps) >= 2 and len(anim) > len(steps), name
        assert anim.to_svg(len(steps) - 1) == fig.to_svg(), name
        carried = {e for f in anim for e in f.pulses}
        for i, e in enumerate(fig.scene.edges):
            if isinstance(e.target, str) and not e.target.startswith("legend:"):
                assert i in carried or e.kind is EdgeKind.TRAINING, (name, e)
