"""User overrides per module path or class name (ADR-0007 ``overrides=``).

    overrides = {
        "ConvLSTMCell": {"category": "recurrent", "label": "ConvLSTM cell"},  # by class name
        "encoder.layers.0": {"label": "first layer"},                        # by module path
        "Swish": {"leaf": True},                    # draw a custom module as one operation
    }

A path entry wins over a class-name entry; class names match along the MRO.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from tensorgami.importers.base import ImporterError
from tensorgami.vocabulary import OpCategory

_KEYS = {"category", "label", "leaf"}


@dataclass(frozen=True)
class Override:
    category: OpCategory | None = None
    label: str | None = None
    leaf: bool | None = None

    def merged(self, other: Override) -> Override:
        """``other`` wins where it sets a value."""
        return Override(
            other.category if other.category is not None else self.category,
            other.label if other.label is not None else self.label,
            other.leaf if other.leaf is not None else self.leaf,
        )


class Overrides:
    def __init__(self, raw: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self._entries = {key: self._parse(key, value) for key, value in (raw or {}).items()}

    def lookup(self, path: str, mro_names: Iterable[str]) -> Override:
        result = Override()
        for name in reversed(tuple(mro_names)):  # most specific class applied last
            if name in self._entries:
                result = result.merged(self._entries[name])
        if path in self._entries:
            result = result.merged(self._entries[path])
        return result

    @staticmethod
    def _parse(key: str, value: Mapping[str, Any]) -> Override:
        if not isinstance(value, Mapping):
            raise ImporterError(f"overrides['{key}'] must be a dict, e.g. {{'label': '...'}}")
        unknown = set(value) - _KEYS
        if unknown:
            raise ImporterError(
                f"overrides['{key}'] has unknown keys {sorted(unknown)}; allowed: {sorted(_KEYS)}"
            )
        category = value.get("category")
        if category is not None:
            try:
                category = OpCategory(category)
            except ValueError:
                valid = ", ".join(c.value for c in OpCategory)
                raise ImporterError(
                    f"overrides['{key}']: unknown category {category!r} (one of: {valid})"
                ) from None
        label = value.get("label")
        leaf = value.get("leaf")
        if label is not None and not isinstance(label, str):
            raise ImporterError(f"overrides['{key}']['label'] must be a string")
        if leaf is not None and not isinstance(leaf, bool):
            raise ImporterError(f"overrides['{key}']['leaf'] must be True or False")
        return Override(category, label, leaf)
