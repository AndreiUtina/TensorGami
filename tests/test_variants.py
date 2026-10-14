"""Style variants (roadmap step 13): ``dark`` and ``greyscale``, for any style."""

from __future__ import annotations

import dataclasses

import pytest
from ir_fixtures import residual_cnn
from layout_checks import check_scene

from tensorgami.layout import layout
from tensorgami.render import render_svg
from tensorgami.style import (
    TextRole,
    ThemeError,
    ThemeValidator,
    apply_variants,
    available_styles,
    available_variants,
    load_style,
)
from tensorgami.style.variants import DARK_BACKGROUND
from tensorgami.transforms import simplify
from tensorgami.vocabulary import OpCategory

COMBINATIONS = [["dark"], ["greyscale"], ["dark", "greyscale"]]


def test_the_variants():
    assert available_variants() == ["dark", "greyscale"]


@pytest.mark.parametrize("style", available_styles())
@pytest.mark.parametrize("variant", COMBINATIONS, ids="+".join)
def test_every_variant_of_every_style_passes_the_style_checks(style, variant):
    theme = load_style(style, variant)  # validated on load
    assert ThemeValidator().validate(theme) == []
    assert theme.name == ", ".join([style, *variant])


@pytest.mark.parametrize("style", available_styles())
def test_dark_keeps_the_cards_and_darkens_the_board(style):
    light, dark = load_style(style), load_style(style, "dark")
    assert dark.background == DARK_BACKGROUND
    for category, card in dark.nodes.items():
        assert card.fill == light.nodes[category].fill  # the pastels stay
        assert card.text == light.nodes[category].text  # and so does their dark text
        assert card.opacity == 1.0
        assert card.stroke.contrast_ratio(dark.background) >= 3.0
    for style_ in dark.edges.values():
        assert style_.color.contrast_ratio(dark.background) >= 4.5
    for role in (TextRole.ANNOTATION, TextRole.LEGEND, TextRole.GROUP_TITLE, TextRole.LABEL):
        assert dark.font(role).color.contrast_ratio(dark.background) >= 4.5


@pytest.mark.parametrize("style", available_styles())
def test_greyscale_has_no_colour_and_keeps_wires_apart(style):
    grey = load_style(style, "greyscale")
    colours = [grey.background, grey.group.fill, grey.group.stroke]
    colours += [c for s in grey.nodes.values() for c in (s.fill, s.stroke, s.text)]
    colours += [s.color for s in grey.edges.values()] + [f.color for f in grey.fonts.values()]
    assert all(c.r == c.g == c.b for c in colours)
    signatures = {(s.color, s.dash, s.arrow) for s in grey.edges.values()}
    assert len(signatures) == len(grey.edges)


def test_greyscale_spreads_the_category_fills_in_their_order():
    light, grey = load_style("default"), load_style("default", "greyscale")
    coloured = [c for c, s in light.nodes.items() if s.fill.chroma >= 10]
    by_light = sorted(coloured, key=lambda c: light.nodes[c].fill.relative_luminance)
    levels = [grey.nodes[c].fill.r for c in by_light]
    assert levels == sorted(levels) and len(set(levels)) == len(levels)
    assert levels[-1] - levels[0] >= 80  # much further apart than the pastels' own greys


def test_unknown_variants_are_named():
    with pytest.raises(ValueError, match="unknown variant 'drak'; available: dark, greyscale"):
        apply_variants(load_style(), "drak")


def test_what_a_variant_makes_is_checked_like_every_style(monkeypatch):
    from tensorgami.style import variants

    def washed(theme):  # the board as dark as the convolutions' outline: they would vanish
        return dataclasses.replace(theme, background=theme.nodes[OpCategory.CONV].stroke)

    monkeypatch.setitem(variants.VARIANTS, "washed", washed)
    with pytest.raises(ThemeError, match="failed validation"):
        load_style("default", "washed")


def test_text_beside_a_node_stays_readable_on_a_dark_board():
    light, dark = load_style("default"), load_style("default", "dark")
    for role in (TextRole.LABEL, TextRole.DETAIL):
        assert light.outside_font(OpCategory.CONV, role) == light.node_font(OpCategory.CONV, role)
        font = dark.outside_font(OpCategory.CONV, role)
        assert font.color.contrast_ratio(dark.background) >= 4.5
        assert font.size == dark.node_font(OpCategory.CONV, role).size


def test_variants_only_recolour(theme):
    """Same layout, other colours: a variant never moves anything."""
    graph = simplify(residual_cnn(), 2)
    plain = layout(graph, theme)
    dark_theme = load_style("default", "dark")
    dark = layout(graph, dark_theme)
    assert [n.box for n in plain.nodes] == [n.box for n in dark.nodes]
    check_scene(dark, graph, dark_theme)
    assert DARK_BACKGROUND.hex in render_svg(dark, dark_theme)
