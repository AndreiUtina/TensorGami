from __future__ import annotations

import pytest

from tensorgami.style import Theme, load_theme


@pytest.fixture(scope="session")
def theme() -> Theme:
    return load_theme("default")
