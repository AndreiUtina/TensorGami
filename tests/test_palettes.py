"""Colour palettes (roadmap step 13): the operations recoloured, for any style."""

from __future__ import annotations

import pytest

from tensorgami.style import (
    TextRole,
    ThemeValidator,
    apply_palette,
    available_palettes,
    available_styles,
    load_style,
)
from tensorgami.style.palettes import PALETTES
from tensorgami.vocabulary import EdgeKind, OpCategory

VARIANTS = [None, "dark", "greyscale", ["dark", "greyscale"]]


def test_the_palettes():
    assert available_palettes() == ["colorblind", "tab10", "vivid", "mono"]


@pytest.mark.parametrize("style", available_styles())
@pytest.mark.parametrize("palette", available_palettes())
@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "+".join(v) if v else str(v))
def test_every_palette_of_every_style_passes_the_style_checks(style, palette, variant):
    theme = load_style(style, variant, palette)  # validated on load
    assert ThemeValidator().validate(theme) == []
    assert theme.name.startswith(f"{style}, {palette}")


@pytest.mark.parametrize("style", available_styles())
@pytest.mark.parametrize("palette", available_palettes())
def test_a_palette_recolours_the_operations_only(style, palette):
    base, coloured = load_style(style), load_style(style, palette=palette)
    for category, card in coloured.nodes.items():
        before = base.nodes[category]
        assert (card.shape, card.dash, card.opacity, card.text) == (
            before.shape,
            before.dash,
            before.opacity,
            before.text,
        )
        if category not in PALETTES[palette].colors:  # inputs, tensors, merges, reshapes …
            assert card == before
        else:  # opaque or seen through a translucent card, the text stays readable
            detail = coloured.font(TextRole.DETAIL).color
            for text in (card.text, detail):
                assert text.contrast_ratio(card.fill) >= 4.5
            assert card.stroke.contrast_ratio(coloured.background) >= 3.0
    assert coloured.fonts == base.fonts and coloured.boxes == base.boxes


@pytest.mark.parametrize("palette", ["colorblind", "tab10", "vivid"])
def test_coloured_wires_follow_their_category(palette):
    theme = load_style(palette=palette)
    assert theme.edge_style(EdgeKind.DATA) == load_style().edge_style(EdgeKind.DATA)
    for kind, category in [
        (EdgeKind.STATE, OpCategory.RECURRENT),
        (EdgeKind.STOCHASTIC, OpCategory.STOCHASTIC),
        (EdgeKind.TRAINING, OpCategory.LOSS),
    ]:
        assert theme.edge_style(kind).color == theme.node_style(category).stroke


def test_mono_marks_the_layers_with_weights():
    theme = load_style(palette="mono")
    weights = [OpCategory.CONV, OpCategory.LINEAR, OpCategory.ATTENTION]
    between = [OpCategory.NORM, OpCategory.ACTIVATION, OpCategory.RESAMPLE]
    lum = {c: theme.node_style(c).fill.relative_luminance for c in weights + between}
    assert max(lum[c] for c in weights) < min(lum[c] for c in between)  # darker
    assert len({theme.node_style(c).stroke for c in weights + between}) == 1  # one outline


def test_unknown_palettes_are_named():
    with pytest.raises(ValueError, match="unknown palette 'pastel'; available: colorblind"):
        apply_palette(load_style(), "pastel")
