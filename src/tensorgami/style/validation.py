"""Quality rules a theme must satisfy (legibility, accessibility, distinctness).

Each rule is a small object; new rules are added by passing them to :class:`ThemeValidator`
without touching the existing ones.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from tensorgami.style.color import Color
from tensorgami.style.glyphs import TextRole
from tensorgami.style.theme import Theme, ThemeError


@dataclass(frozen=True, slots=True)
class Issue:
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message}"


class ThemeRule(Protocol):
    name: str

    def check(self, theme: Theme) -> Iterable[Issue]: ...


@dataclass(frozen=True)
class TextContrastRule:
    """All text must reach ``min_ratio`` against whatever it is drawn on (WCAG AA = 4.5)."""

    min_ratio: float = 4.5
    name: str = "text-contrast"

    def check(self, theme: Theme) -> Iterable[Issue]:
        detail = theme.font(TextRole.DETAIL).color
        for category, style in theme.nodes.items():
            yield from self._pair(style.text, style.fill, f"label on '{category}'")
            yield from self._pair(detail, style.fill, f"detail text on '{category}'")
        on_background = (TextRole.ANNOTATION, TextRole.NOTE, TextRole.LEGEND, TextRole.GROUP_TITLE)
        for role in on_background:
            yield from self._pair(theme.font(role).color, theme.background, f"{role} text")
        yield from self._pair(
            theme.font(TextRole.GROUP_TITLE).color, theme.group.fill, "group title on group fill"
        )
        yield from self._pair(theme.font(TextRole.BADGE).color, theme.group.badge_fill, "badge")

    def _pair(self, fg: Color, bg: Color, what: str) -> Iterable[Issue]:
        ratio = fg.contrast_ratio(bg)
        if ratio < self.min_ratio:
            yield Issue(
                self.name, f"{what}: contrast {ratio:.2f} < {self.min_ratio} ({fg} on {bg})"
            )


@dataclass(frozen=True)
class GraphicContrastRule:
    """Outlines and edges must reach ``min_ratio`` against the background (WCAG 1.4.11)."""

    min_ratio: float = 3.0
    name: str = "graphic-contrast"

    def check(self, theme: Theme) -> Iterable[Issue]:
        items = [(f"stroke of '{c}'", s.stroke) for c, s in theme.nodes.items()]
        items += [(f"edge '{k}'", s.color) for k, s in theme.edges.items()]
        items.append(("group stroke", theme.group.stroke))
        for what, color in items:
            ratio = color.contrast_ratio(theme.background)
            if ratio < self.min_ratio:
                yield Issue(self.name, f"{what}: contrast {ratio:.2f} < {self.min_ratio}")


@dataclass(frozen=True)
class DistinctCategoriesRule:
    """Chromatic categories must be told apart by fill and by stroke (CIE76 ΔE)."""

    min_fill_delta: float = 10.0
    min_stroke_delta: float = 15.0
    chroma_threshold: float = 20.0  # strokes below this chroma count as neutral greys
    name: str = "distinct-categories"

    def check(self, theme: Theme) -> Iterable[Issue]:
        chromatic = [
            (c, s) for c, s in theme.nodes.items() if s.stroke.chroma >= self.chroma_threshold
        ]
        for (a, sa), (b, sb) in itertools.combinations(chromatic, 2):
            fill_d, stroke_d = sa.fill.delta_e(sb.fill), sa.stroke.delta_e(sb.stroke)
            if fill_d < self.min_fill_delta:
                yield Issue(self.name, f"fills of '{a}' and '{b}' too similar (ΔE {fill_d:.1f})")
            if stroke_d < self.min_stroke_delta:
                yield Issue(
                    self.name, f"strokes of '{a}' and '{b}' too similar (ΔE {stroke_d:.1f})"
                )


@dataclass(frozen=True)
class DistinctEdgesRule:
    """Each edge kind needs a unique (colour, dash, arrow) signature (Marshall guideline B)."""

    name: str = "distinct-edges"

    def check(self, theme: Theme) -> Iterable[Issue]:
        seen: dict[tuple, str] = {}
        for kind, style in theme.edges.items():
            signature = (style.color, style.dash, style.arrow)
            if signature in seen:
                other = seen[signature]
                yield Issue(self.name, f"edge kinds '{other}' and '{kind}' look identical")
            seen.setdefault(signature, kind)


def default_rules() -> list[ThemeRule]:
    return [
        TextContrastRule(),
        GraphicContrastRule(),
        DistinctCategoriesRule(),
        DistinctEdgesRule(),
    ]


class ThemeValidator:
    def __init__(self, rules: Sequence[ThemeRule] | None = None) -> None:
        self._rules = list(default_rules() if rules is None else rules)

    def validate(self, theme: Theme) -> list[Issue]:
        return [issue for rule in self._rules for issue in rule.check(theme)]

    def ensure_valid(self, theme: Theme) -> Theme:
        issues = self.validate(theme)
        if issues:
            lines = "\n  ".join(str(i) for i in issues)
            raise ThemeError(f"theme '{theme.name}' failed validation:\n  {lines}")
        return theme
