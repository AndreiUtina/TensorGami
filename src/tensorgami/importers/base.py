"""Framework-independent importer contract."""

from __future__ import annotations

from typing import Any, Protocol

from tensorgami.ir import Graph


class ImporterError(RuntimeError):
    """The model could not be captured (e.g. it failed on the example input)."""


class UnclassifiedOpWarning(UserWarning):
    """Some operations are not in the catalogue and were drawn with the 'other' category."""


class ModelImporter(Protocol):
    """Turns a framework model plus an example input into an :class:`~tensorgami.ir.Graph`."""

    def import_model(self, model: Any, example_input: Any) -> Graph: ...
