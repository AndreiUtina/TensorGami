from __future__ import annotations

import pytest

from tensorgami.style.color import BLACK, WHITE, Color


def test_hex_round_trip_and_short_form():
    assert Color.from_hex("#1D4ED8").hex == "#1d4ed8"
    assert Color.from_hex("#fff") == WHITE


@pytest.mark.parametrize("bad", ["1d4ed8", "#12345", "#gggggg", "", None, "#1234567"])
def test_invalid_hex_is_rejected(bad):
    with pytest.raises(ValueError):
        Color.from_hex(bad)


def test_channels_are_range_checked():
    with pytest.raises(ValueError):
        Color(256, 0, 0)


def test_contrast_ratio_matches_wcag_reference_values():
    assert WHITE.contrast_ratio(BLACK) == pytest.approx(21.0)
    assert WHITE.contrast_ratio(WHITE) == pytest.approx(1.0)
    # WCAG reference: #767676 on white is just above 4.5:1
    assert Color.from_hex("#767676").contrast_ratio(WHITE) == pytest.approx(4.54, abs=0.01)


def test_contrast_ratio_is_symmetric():
    a, b = Color.from_hex("#1d4ed8"), Color.from_hex("#fef08a")
    assert a.contrast_ratio(b) == pytest.approx(b.contrast_ratio(a))


def test_lab_of_white_and_black():
    assert WHITE.to_lab() == pytest.approx((100.0, 0.0, 0.0), abs=0.05)
    assert BLACK.to_lab() == pytest.approx((0.0, 0.0, 0.0), abs=0.05)


def test_delta_e_and_chroma():
    red, red2 = Color.from_hex("#b91c1c"), Color.from_hex("#ba1d1d")
    assert red.delta_e(red) == 0
    assert red.delta_e(red2) < 2.3  # below the just-noticeable difference
    assert Color.from_hex("#808080").chroma < 1
    assert red.chroma > 50


def test_mix_lighten_darken():
    grey = Color(100, 100, 100)
    assert grey.lighten(1.0) == WHITE
    assert grey.darken(1.0) == BLACK
    assert grey.mix(WHITE, 0.0) == grey
    with pytest.raises(ValueError):
        grey.mix(WHITE, 1.5)
