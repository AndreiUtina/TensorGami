"""Views: how a captured model is drawn (ADR-0007 ``view=``; more views in roadmap step 11).

A view chooses the transforms and the layout engine. New views are added to ``VIEWS``; the
API looks them up by name with :func:`get_view`.
"""

from __future__ import annotations

from tensorgami.views.base import View
from tensorgami.views.block import AutoView, BlockView, CellView, UnrolledView, VolumeView

_BLOCK, _UNROLLED = BlockView(), UnrolledView()

VIEWS: dict[str, View] = {
    "auto": AutoView(_BLOCK, _UNROLLED),
    "block": _BLOCK,
    "unrolled": _UNROLLED,
    "volume": VolumeView(),
    "cell": CellView(),
}

#: views that are designed but not built yet
PLANNED: dict[str, str] = {}


def available_views() -> list[str]:
    return list(VIEWS)


def get_view(name: str) -> View:
    """The view called ``name``, or an error that lists the views there are."""
    if name in VIEWS:
        return VIEWS[name]
    known = ", ".join(repr(v) for v in VIEWS)
    if name in PLANNED:
        raise NotImplementedError(
            f"view={name!r} ({PLANNED[name]}) is planned for roadmap step 11; "
            f"available now: {known}"
        )
    planned = f" (planned: {', '.join(PLANNED)})" if PLANNED else ""
    raise ValueError(f"unknown view {name!r}; available: {known}{planned}")


__all__ = [
    "PLANNED",
    "VIEWS",
    "AutoView",
    "BlockView",
    "CellView",
    "UnrolledView",
    "VolumeView",
    "View",
    "available_views",
    "get_view",
]
