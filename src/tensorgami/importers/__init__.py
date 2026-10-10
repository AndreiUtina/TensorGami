"""Model importers: framework → IR (ADR-0001).

Each framework lives in its own subpackage and is imported on demand, so ``tensorgami`` itself
works without any deep-learning framework installed::

    from tensorgami.importers.torch import trace
"""

from __future__ import annotations

from tensorgami.importers.base import ImporterError, ModelImporter, UnclassifiedOpWarning

__all__ = ["ImporterError", "ModelImporter", "UnclassifiedOpWarning"]
