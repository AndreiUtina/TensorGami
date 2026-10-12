"""``tg.draw``: from a loaded model to a figure in one call (ADR-0007, ADR-0008)."""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tensorgami.api.figure import Figure
from tensorgami.ir import Graph
from tensorgami.layout import LayoutOptions
from tensorgami.style import DEFAULT_STYLE, Theme, load_style
from tensorgami.transforms import Level
from tensorgami.views import get_view

if TYPE_CHECKING:
    from torch import nn

LAYOUT_OPTIONS = tuple(f.name for f in fields(LayoutOptions))
_PARAMETERS = ("level", "view", "style", "overrides", "direction", *LAYOUT_OPTIONS)
_RENAMED = {"theme": "style"}  # the name sketched in ADR-0007, before ADR-0008


def draw(
    model: nn.Module,
    example_input: Any,
    *,
    level: int | str = Level.LAYERS,
    view: str = "auto",
    style: str | Path | Theme = DEFAULT_STYLE,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    direction: str | None = None,
    **layout_options: Any,
) -> Figure:
    """Draw ``model`` as it runs on ``example_input``.

    The model runs once, without gradients and in eval mode; its weights, training flags and
    the caller's random state are left as they were. Nothing is written to disk: the figure
    shows inline in Jupyter and is saved only on request, ``fig.save("model.png", dpi=300)``.

    Args:
        model: a loaded ``torch.nn.Module``, on any device (the input is moved to it).
        example_input: what the model is called with: a tensor, a tuple of positional
            arguments, or a dict of keyword arguments. Include the batch dimension.
        level: the detail, by number or name: 0 ``"system"`` (one box), 1 ``"stages"`` (the
            model's direct sub-modules), 2 ``"layers"`` (default), 3 ``"operators"``
            (everything traced), 4 ``"cells"``.
        view: ``"block"`` (boxes and wires), ``"unrolled"`` (recurrence as a time × layer
            grid, with per-step inputs x_t), ``"volume"`` (CNN layers as tensor volumes),
            ``"cell"`` (the recurrent cells drawn as their gates), or ``"auto"`` (default:
            unrolled when a module is called step after step with state passed along, block
            otherwise).
        style: a built-in style (``"default"``, ``"origami"``), the path of a ``.toml``
            style, or a loaded :class:`~tensorgami.style.Theme`.
        overrides: category, label or leaf-ness per module path or class name, e.g.
            ``{"ConvLSTMCell": {"category": "recurrent", "label": "ConvLSTM cell"}}``.
        direction: ``"LR"`` (left → right) or ``"TB"`` (top → bottom); by default the
            style's choice, else ``"LR"``.
        **layout_options: further :class:`~tensorgami.layout.LayoutOptions`, e.g.
            ``legend=False``, ``wrap_width=None`` (one row) or ``rank_gap=64``.

    Example::

        import tensorgami as tg

        fig = tg.draw(model, torch.randn(1, 3, 224, 224), style="origami")
        fig  # shows inline in Jupyter
        fig.save("model.png", dpi=300)
    """
    # every argument is checked before the model runs, so a typo costs no forward pass
    _check_names(layout_options)
    theme = style if isinstance(style, Theme) else load_style(style)
    if direction is not None:
        layout_options["direction"] = direction
    options = LayoutOptions.for_style(theme, **layout_options)
    level = Level.parse(level)
    chosen = get_view(view)

    shown, scene = chosen.draw(
        _capture(model, example_input, overrides), theme, level=level, options=options
    )
    return Figure(scene, theme, shown)


def _capture(model: nn.Module, example_input: Any, overrides: Mapping | None) -> Graph:
    from tensorgami.importers.torch import trace  # torch is imported only to draw a model

    return trace(model, example_input, overrides=overrides)


def _check_names(layout_options: Mapping[str, Any]) -> None:
    for name in layout_options:
        if name in LAYOUT_OPTIONS:
            continue
        close = (
            [_RENAMED[name]]
            if name in _RENAMED
            else difflib.get_close_matches(name, _PARAMETERS, n=1)
        )
        hint = (
            f"; did you mean {close[0]!r}?"
            if close
            else f"; layout options are: {', '.join(LAYOUT_OPTIONS)}"
        )
        raise TypeError(f"draw() got an unexpected keyword argument {name!r}{hint}")
