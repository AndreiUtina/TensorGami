from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # tests never open windows

import pytest

from tensorgami.style import Theme, load_theme


@pytest.fixture(scope="session")
def theme() -> Theme:
    return load_theme("default")
