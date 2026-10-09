"""Load a :class:`Theme` from a TOML file (parsing only; validation lives elsewhere)."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from tensorgami.style.color import Color
from tensorgami.style.glyphs import (
    ArrowHead,
    EdgeStyle,
    Font,
    GroupStyle,
    Metrics,
    NodeStyle,
    Shape,
    TextRole,
)
from tensorgami.style.theme import Theme, ThemeError
from tensorgami.vocabulary import EdgeKind, OpCategory

_BUILTIN_PACKAGE = "tensorgami.style.themes"


def available_themes() -> list[str]:
    """Names of the themes shipped with Tensorgami."""
    folder = resources.files(_BUILTIN_PACKAGE)
    names = (p.name for p in folder.iterdir())
    return sorted(n.removesuffix(".toml") for n in names if n.endswith(".toml"))


def read_theme(source: str | Path) -> Theme:
    """Parse a built-in theme by name, or a ``.toml`` file by path."""
    if isinstance(source, Path) or str(source).endswith(".toml"):
        path = Path(source)
        if not path.is_file():
            raise ThemeError(f"theme file not found: {path}")
        text = path.read_text(encoding="utf-8")
    else:
        resource = resources.files(_BUILTIN_PACKAGE) / f"{source}.toml"
        if not resource.is_file():
            known = ", ".join(available_themes())
            raise ThemeError(f"unknown theme '{source}' (built-in themes: {known})")
        text = resource.read_text(encoding="utf-8")
    return parse_theme(text)


def parse_theme(text: str) -> Theme:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ThemeError(f"invalid TOML: {exc}") from exc
    return _ThemeParser(data).parse()


class _ThemeParser:
    """Turns the TOML document into style records, reporting errors by key path."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = data

    def parse(self) -> Theme:
        head = self._table(self._data, "theme")
        fonts = self._fonts(self._table(self._data, "fonts"))
        label_color = fonts[TextRole.LABEL].color
        nodes = self._keyed(
            self._table(self._data, "categories"),
            "categories",
            OpCategory,
            lambda t, p: self._node(t, p, label_color),
        )
        edges = self._keyed(self._table(self._data, "edges"), "edges", EdgeKind, self._edge)
        return Theme(
            name=self._str(head, "name", "theme"),
            description=str(head.get("description", "")),
            background=self._color(head, "background", "theme"),
            fonts=fonts,
            nodes=nodes,
            edges=edges,
            group=self._group(self._table(self._data, "group"), "group"),
            metrics=self._metrics(self._data.get("metrics", {}), "metrics"),
        )

    # ------------------------------------------------------------------ sections

    def _fonts(self, table: Mapping[str, Any]) -> dict[TextRole, Font]:
        families = {
            name: self._str_list(table.get("families", {}), name, "fonts.families")
            for name in table.get("families", {})
        }
        roles = {k: v for k, v in table.items() if k != "families"}

        def font(t: Mapping[str, Any], path: str) -> Font:
            family = self._str(t, "family", path)
            if family not in families:
                raise ThemeError(f"{path}.family: unknown family '{family}'")
            return Font(
                family=tuple(families[family]),
                size=self._number(t, "size", path),
                color=self._color(t, "color", path),
                weight=int(t.get("weight", 400)),
                italic=bool(t.get("italic", False)),
            )

        return self._keyed(roles, "fonts", TextRole, font)

    def _node(self, t: Mapping[str, Any], path: str, default_text: Color) -> NodeStyle:
        return NodeStyle(
            name=self._str(t, "name", path),
            shape=self._enum(t, "shape", path, Shape),
            fill=self._color(t, "fill", path),
            stroke=self._color(t, "stroke", path),
            text=self._color(t, "text", path) if "text" in t else default_text,
            dash=self._dash(t, path),
        )

    def _edge(self, t: Mapping[str, Any], path: str) -> EdgeStyle:
        return EdgeStyle(
            name=self._str(t, "name", path),
            color=self._color(t, "color", path),
            width=self._number(t, "width", path),
            arrow=self._enum(t, "arrow", path, ArrowHead) if "arrow" in t else ArrowHead.FILLED,
            dash=self._dash(t, path),
        )

    def _group(self, t: Mapping[str, Any], path: str) -> GroupStyle:
        return GroupStyle(
            fill=self._color(t, "fill", path),
            stroke=self._color(t, "stroke", path),
            width=self._number(t, "width", path),
            corner_radius=self._number(t, "corner_radius", path),
            padding=self._number(t, "padding", path),
            dash=self._dash(t, path),
            badge_fill=self._color(t, "badge_fill", path)
            if "badge_fill" in t
            else Color(255, 255, 255),
        )

    def _metrics(self, t: Mapping[str, Any], path: str) -> Metrics:
        known = Metrics.__dataclass_fields__
        unknown = sorted(set(t) - set(known))
        if unknown:
            raise ThemeError(f"{path}: unknown keys {unknown}")
        return Metrics(**{k: self._number(t, k, path) for k in t})

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _keyed(table, path, enum_type, build):
        out = {}
        for key, value in table.items():
            try:
                member = enum_type(key)
            except ValueError:
                valid = ", ".join(m.value for m in enum_type)
                raise ThemeError(f"{path}.{key}: unknown key (expected one of: {valid})") from None
            if not isinstance(value, Mapping):
                raise ThemeError(f"{path}.{key}: expected a table")
            out[member] = build(value, f"{path}.{key}")
        return out

    @staticmethod
    def _table(t: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        value = t.get(key)
        if not isinstance(value, Mapping):
            raise ThemeError(f"missing table [{key}]")
        return value

    @staticmethod
    def _str(t: Mapping[str, Any], key: str, path: str) -> str:
        value = t.get(key)
        if not isinstance(value, str) or not value:
            raise ThemeError(f"{path}.{key}: expected a non-empty string")
        return value

    @staticmethod
    def _str_list(t: Mapping[str, Any], key: str, path: str) -> list[str]:
        value = t.get(key)
        if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
            raise ThemeError(f"{path}.{key}: expected a non-empty list of strings")
        return value

    @staticmethod
    def _number(t: Mapping[str, Any], key: str, path: str) -> float:
        value = t.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ThemeError(f"{path}.{key}: expected a number")
        return float(value)

    @staticmethod
    def _color(t: Mapping[str, Any], key: str, path: str) -> Color:
        try:
            return Color.from_hex(t.get(key))
        except ValueError as exc:
            raise ThemeError(f"{path}.{key}: {exc}") from None

    @staticmethod
    def _enum(t: Mapping[str, Any], key: str, path: str, enum_type):
        try:
            return enum_type(t.get(key))
        except ValueError:
            valid = ", ".join(m.value for m in enum_type)
            raise ThemeError(f"{path}.{key}: expected one of: {valid}") from None

    @staticmethod
    def _dash(t: Mapping[str, Any], path: str) -> tuple[float, ...]:
        value = t.get("dash", [])
        if not isinstance(value, list) or not all(
            isinstance(v, int | float) and not isinstance(v, bool) and v > 0 for v in value
        ):
            raise ThemeError(f"{path}.dash: expected a list of positive numbers")
        return tuple(float(v) for v in value)
