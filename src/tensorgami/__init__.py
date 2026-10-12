"""Tensorgami: draw the architecture and information flow of ML models."""

from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy top-level API, so ``import tensorgami`` stays light: matplotlib loads when a
    figure is made, torch when a model is drawn."""
    if name in ("Figure", "Input", "Loss", "Note", "draw"):
        from tensorgami import api

        return getattr(api, name)
    raise AttributeError(f"module 'tensorgami' has no attribute {name!r}")


def __dir__() -> list[str]:  # tab completion lists the lazy names too
    return sorted({*globals(), *__all__})


__all__ = ["Figure", "Input", "Loss", "Note", "__version__", "draw"]
