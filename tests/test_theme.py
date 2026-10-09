from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tensorgami.style import (
    Color,
    Shape,
    TextRole,
    ThemeError,
    ThemeValidator,
    available_themes,
    load_theme,
    parse_theme,
    read_theme,
)
from tensorgami.style.validation import (
    DistinctCategoriesRule,
    DistinctEdgesRule,
    GraphicContrastRule,
    TextContrastRule,
)
from tensorgami.vocabulary import EdgeKind, OpCategory

BASELINE_TOML = (Path(__file__).parents[1] / "src/tensorgami/style/themes/baseline.toml").read_text(
    encoding="utf-8"
)


# --------------------------------------------------------------------------- loading


def test_baseline_is_shipped_and_complete(theme):
    assert "baseline" in available_themes()
    assert set(theme.nodes) == set(OpCategory)
    assert set(theme.edges) == set(EdgeKind)
    assert set(theme.fonts) == set(TextRole)


def test_baseline_passes_every_validation_rule(theme):
    assert ThemeValidator().validate(theme) == []


def test_theme_can_be_loaded_from_a_path(tmp_path):
    path = tmp_path / "mine.toml"
    path.write_text(BASELINE_TOML.replace('name = "baseline"', 'name = "mine"'), encoding="utf-8")
    assert load_theme(path).name == "mine"
    assert load_theme(str(path)).name == "mine"


def test_unknown_theme_name_lists_builtins():
    with pytest.raises(ThemeError, match="baseline"):
        read_theme("does-not-exist")


def test_missing_theme_file():
    with pytest.raises(ThemeError, match="not found"):
        read_theme(Path("/nonexistent/theme.toml"))


def test_theme_mappings_are_read_only(theme):
    with pytest.raises(TypeError):
        theme.nodes[OpCategory.CONV] = None  # type: ignore[index]


def test_category_text_defaults_to_label_colour(theme):
    assert theme.node_style(OpCategory.CONV).text == theme.font(TextRole.LABEL).color


def test_node_font_uses_category_text_colour_except_for_details():
    conv = 'name = "Convolution"\nshape = "rect"'
    toml = BASELINE_TOML.replace(conv, conv + '\ntext = "#000000"')
    theme = parse_theme(toml)
    assert theme.node_font(OpCategory.CONV).color == Color(0, 0, 0)
    assert theme.node_font(OpCategory.CONV, TextRole.DETAIL) == theme.font(TextRole.DETAIL)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('fill = "#bfdbfe"', 'fill = "blue"', r"categories\.conv\.fill"),
        ('shape = "pill"\nfill = "#e2e8f0"', 'shape = "hexagon"\nfill = "#e2e8f0"', "shape"),
        ("[categories.conv]", "[categories.convolution]", "unknown key"),
        ('family = "mono"', 'family = "serif"', "unknown family"),
        ("[edges.tied]", "[edges.tied_weights]", "unknown key"),
        ("width = 1.25\narrow", 'width = "thick"\narrow', "expected a number"),
        ("dash = [4, 3]\ncorner_radius", "dash = [-1]\ncorner_radius", "dash"),
        ("cuboid_angle = 35", "cuboid_angle = 35\nwobble = 2", "unknown keys"),
        ("[theme]", "[theme_x]", r"missing table \[theme\]"),
        ("= 600", "== 600", "invalid TOML"),
    ],
)
def test_malformed_themes_report_the_offending_key(old, new, message):
    assert old in BASELINE_TOML
    with pytest.raises(ThemeError, match=message):
        parse_theme(BASELINE_TOML.replace(old, new, 1))


def test_incomplete_theme_is_rejected():
    start = BASELINE_TOML.index("[categories.loss]")
    end = BASELINE_TOML.index("# ---", start)
    with pytest.raises(ThemeError, match="category 'loss'"):
        parse_theme(BASELINE_TOML[:start] + BASELINE_TOML[end:])


# --------------------------------------------------------------------------- validation


def _with_node(theme, category, **changes):
    nodes = dict(theme.nodes)
    nodes[category] = dataclasses.replace(nodes[category], **changes)
    return dataclasses.replace(theme, nodes=nodes)


def test_text_contrast_rule_flags_low_contrast_fill(theme):
    bad = _with_node(theme, OpCategory.CONV, fill=Color.from_hex("#1e293b"))
    issues = list(TextContrastRule().check(bad))
    assert any("label on 'conv'" in i.message for i in issues)


def test_graphic_contrast_rule_flags_faint_strokes(theme):
    bad = _with_node(theme, OpCategory.NORM, stroke=Color.from_hex("#fde68a"))
    assert any("'norm'" in i.message for i in GraphicContrastRule().check(bad))


def test_distinct_categories_rule_flags_lookalike_hues(theme):
    conv = theme.node_style(OpCategory.CONV)
    bad = _with_node(theme, OpCategory.LINEAR, fill=conv.fill, stroke=conv.stroke)
    messages = [i.message for i in DistinctCategoriesRule().check(bad)]
    assert any("'conv' and 'linear'" in m for m in messages)


def test_distinct_edges_rule_flags_identical_edge_styles(theme):
    edges = dict(theme.edges)
    edges[EdgeKind.STATE] = edges[EdgeKind.DATA]
    bad = dataclasses.replace(theme, edges=edges)
    assert list(DistinctEdgesRule().check(bad))


def test_load_theme_refuses_invalid_theme(tmp_path):
    path = tmp_path / "ugly.toml"
    path.write_text(BASELINE_TOML.replace('fill = "#bfdbfe"', 'fill = "#0f172a"'), encoding="utf-8")
    with pytest.raises(ThemeError, match="failed validation"):
        load_theme(path)
    assert load_theme(path, validate=False).node_style(OpCategory.CONV).fill.hex == "#0f172a"


def test_validator_accepts_custom_rule_set(theme):
    class AlwaysFails:
        name = "always"

        def check(self, _theme):
            yield from [type("I", (), {"rule": "always", "message": "nope"})()]

    assert ThemeValidator([]).validate(theme) == []
    assert len(ThemeValidator([AlwaysFails()]).validate(theme)) == 1


def test_baseline_uses_documented_shape_conventions(theme):
    assert theme.node_style(OpCategory.IO).shape is Shape.PILL
    assert theme.node_style(OpCategory.MERGE).shape is Shape.CIRCLE
    assert theme.node_style(OpCategory.CONV).shape is Shape.RECT
    assert theme.node_style(OpCategory.STOCHASTIC).dash  # dashed outline
    assert theme.edge_style(EdgeKind.TRAINING).dash
    assert not theme.edge_style(EdgeKind.DATA).dash  # one solid style for data flow
