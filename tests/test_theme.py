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

DEFAULT_TOML = (Path(__file__).parents[1] / "src/tensorgami/style/themes/default.toml").read_text(
    encoding="utf-8"
)


# --------------------------------------------------------------------------- loading


def test_default_is_shipped_and_complete(theme):
    assert "default" in available_themes()
    assert set(theme.nodes) == set(OpCategory)
    assert set(theme.edges) == set(EdgeKind)
    assert set(theme.fonts) == set(TextRole)


def test_default_passes_every_validation_rule(theme):
    assert ThemeValidator().validate(theme) == []


def test_theme_can_be_loaded_from_a_path(tmp_path):
    path = tmp_path / "mine.toml"
    path.write_text(DEFAULT_TOML.replace('name = "default"', 'name = "mine"'), encoding="utf-8")
    assert load_theme(path).name == "mine"
    assert load_theme(str(path)).name == "mine"


def test_unknown_theme_name_lists_builtins():
    with pytest.raises(ThemeError, match="default"):
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
    toml = DEFAULT_TOML.replace(conv, conv + '\ntext = "#000000"')
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
    assert old in DEFAULT_TOML
    with pytest.raises(ThemeError, match=message):
        parse_theme(DEFAULT_TOML.replace(old, new, 1))


def test_incomplete_theme_is_rejected():
    start = DEFAULT_TOML.index("[categories.loss]")
    end = DEFAULT_TOML.index("# ---", start)
    with pytest.raises(ThemeError, match="category 'loss'"):
        parse_theme(DEFAULT_TOML[:start] + DEFAULT_TOML[end:])


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
    path.write_text(DEFAULT_TOML.replace('fill = "#bfdbfe"', 'fill = "#0f172a"'), encoding="utf-8")
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


def test_default_uses_documented_shape_conventions(theme):
    assert theme.node_style(OpCategory.IO).shape is Shape.PILL
    assert theme.node_style(OpCategory.MERGE).shape is Shape.CIRCLE
    assert theme.node_style(OpCategory.CONV).shape is Shape.RECT
    assert theme.node_style(OpCategory.STOCHASTIC).dash  # dashed outline
    assert theme.edge_style(EdgeKind.TRAINING).dash
    assert not theme.edge_style(EdgeKind.DATA).dash  # one solid style for data flow


# --------------------------------------------------------------------------- named styles


def test_default_style_is_the_standard():
    from tensorgami.style import DEFAULT_STYLE, available_styles, load_style

    assert DEFAULT_STYLE == "default" and "default" in available_styles()
    assert load_style().name == "default" == load_style("default").name


def test_unknown_style_lists_the_available_ones():
    from tensorgami.style import load_style

    with pytest.raises(ThemeError, match="built-in styles: default"):
        load_style("origami-not-yet")


def test_a_style_may_carry_layout_preferences():
    theme = parse_theme(DEFAULT_TOML + '\n[layout]\nrank_gap = 96\ndirection = "TB"\n')
    assert dict(theme.layout) == {"rank_gap": 96, "direction": "TB"}
    assert dict(parse_theme(DEFAULT_TOML).layout) == {}
    with pytest.raises(ThemeError, match="layout.rank_gap"):
        parse_theme(DEFAULT_TOML + "\n[layout]\nrank_gap = [1, 2]\n")


# --------------------------------------------------------------------------- origami


@pytest.mark.parametrize("name", ["origami", "origami-extended"])
def test_origami_styles_are_built_in_and_pass_every_rule(name):
    from tensorgami.style import available_styles, load_style

    assert available_styles() == ["default", "etching", "origami", "origami-extended"]
    origami = load_style(name)
    assert ThemeValidator().validate(origami) == []
    assert origami.node_style(OpCategory.CONV).shape is Shape.FOLDED
    assert origami.node_style(OpCategory.CONV).opacity < 1
    assert origami.font(TextRole.GROUP_TITLE).caps
    assert origami.group.fill_opacity == 0
    assert origami.edge_style(EdgeKind.DATA).start.value == "dot"
    assert origami.boxes.text(vertical=True).value == "inline"  # flat cards in a column


def test_the_two_origami_styles_differ_in_rows():
    from tensorgami.style import load_style

    origami, extended = load_style("origami"), load_style("origami-extended")
    assert (origami.layout["direction"], origami.boxes.text_lr.value) == ("LR", "rotated")
    assert (extended.layout["direction"], extended.boxes.text_lr.value) == ("TB", "stacked")


def test_new_style_keys_are_parsed_and_checked():
    toml = DEFAULT_TOML.replace(
        'fill = "#bfdbfe"\nstroke = "#1d4ed8"',
        'fill = "#bfdbfe"\nstroke = "#1d4ed8"\nopacity = 0.5',
    )
    assert parse_theme(toml).node_style(OpCategory.CONV).opacity == 0.5
    with pytest.raises(ThemeError, match="categories.conv.opacity"):
        parse_theme(toml.replace("opacity = 0.5", "opacity = 1.5"))
    with pytest.raises(ThemeError, match="edges.data.start"):
        parse_theme(
            DEFAULT_TOML.replace('arrow = "filled"\n', 'arrow = "filled"\nstart = "star"\n', 1)
        )
    default = parse_theme(DEFAULT_TOML)
    assert default.node_style(OpCategory.CONV).opacity == 1.0  # defaults keep the old look
    assert default.edge_style(EdgeKind.DATA).start.value == "none"
    assert not default.font(TextRole.GROUP_TITLE).caps and default.group.fill_opacity == 1.0


def test_box_geometry_is_parsed_and_checked():
    from tensorgami.style import BoxGeometry, TextLayout

    assert parse_theme(DEFAULT_TOML).boxes == BoxGeometry()  # defaults keep the old look
    toml = DEFAULT_TOML + '\n[boxes]\ntext = "inline"\nheight = 26\nuniform_width = true\n'
    boxes = parse_theme(toml).boxes
    assert (boxes.text_lr, boxes.text_tb) == (TextLayout.INLINE, TextLayout.INLINE)
    assert (boxes.height, boxes.uniform_width) == (26, True)
    per_direction = DEFAULT_TOML + '\n[boxes]\ntext = { TB = "inline", LR = "rotated" }\n'
    boxes = parse_theme(per_direction).boxes
    assert boxes.text(vertical=True) is TextLayout.INLINE
    assert boxes.text(vertical=False) is TextLayout.ROTATED
    for bad, message in [
        ('text = "sideways"', "boxes.text.TB: expected one of: stacked, inline, rotated"),
        ('text = { XY = "inline" }', r"boxes.text: use one arrangement or \{ TB = …, LR = … \}"),
        ("height = true", "boxes.height: expected a number"),
        ('uniform_width = "yes"', "boxes.uniform_width: expected true or false"),
        ("colour = 1", "boxes: unknown keys"),
    ]:
        with pytest.raises(ThemeError, match=message):
            parse_theme(DEFAULT_TOML + f"\n[boxes]\n{bad}\n")
