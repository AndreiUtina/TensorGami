"""Text helpers: the label mini-markup and text-width estimation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from tensorgami.style.glyphs import Font

SCRIPT_SCALE = 0.75  # size of sub/superscripts relative to the base font

_MARKUP = re.compile(r"([_^])\{([^{}]*)\}")


class Script(StrEnum):
    NORMAL = "normal"
    SUB = "sub"
    SUPER = "super"


@dataclass(frozen=True, slots=True)
class Run:
    text: str
    script: Script = Script.NORMAL


def parse_markup(text: str) -> list[Run]:
    """Split ``"x_{t-1}"`` into runs. Only braced forms are markup, so names such as
    ``batch_norm`` stay literal."""
    runs: list[Run] = []
    pos = 0
    for match in _MARKUP.finditer(text):
        if match.start() > pos:
            runs.append(Run(text[pos : match.start()]))
        script = Script.SUB if match.group(1) == "_" else Script.SUPER
        if match.group(2):
            runs.append(Run(match.group(2), script))
        pos = match.end()
    if pos < len(text):
        runs.append(Run(text[pos:]))
    return runs


def plain_text(text: str) -> str:
    return "".join(run.text for run in parse_markup(text))


class TextMeasurer(Protocol):
    def width(self, text: str, font: Font) -> float: ...


_NARROW = set("iljtfrI.,:;'|!()[]{} ")
_WIDE = set("mwMW@%")
_MONO_HINTS = ("mono", "courier", "consolas", "menlo")


class HeuristicTextMeasurer:
    """Font-independent width estimate (good to ~10 % for common sans fonts).

    Used for halos behind labels and legend layout, where an estimate is enough. A
    font-metrics measurer can replace it behind the same protocol.
    """

    def width(self, text: str, font: Font) -> float:
        mono = any(h in font.family[0].lower() for h in _MONO_HINTS) or (
            font.family[0] == "monospace"
        )
        bold = 1.06 if font.weight >= 600 else 1.0
        total = 0.0
        for run in parse_markup(text):
            scale = 1.0 if run.script is Script.NORMAL else SCRIPT_SCALE
            # fsum, not sum: sum() of floats changed in Python 3.12; fsum is identical on
            # every version, which keeps figures byte-identical (NFR4)
            em = math.fsum(0.6 if mono else self._char_em(c) for c in run.text)
            total += em * scale
        return total * font.size * bold

    @staticmethod
    def _char_em(char: str) -> float:
        if char in _NARROW:
            return 0.3
        if char in _WIDE:
            return 0.85
        if char.isupper():
            return 0.68
        if char.isdigit():
            return 0.56
        return 0.55
