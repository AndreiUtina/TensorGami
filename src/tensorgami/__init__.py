"""Tensorgami: draw the architecture and information flow of ML models."""

from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy top-level API, so ``import tensorgami`` stays light (matplotlib loads on use)."""
    if name == "Figure":
        from tensorgami.api import Figure

        return Figure
    raise AttributeError(f"module 'tensorgami' has no attribute {name!r}")


__all__ = ["Figure", "__version__"]
