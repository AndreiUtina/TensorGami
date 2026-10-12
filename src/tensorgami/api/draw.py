"""``tg.draw``: from a loaded model to a figure in one call (ADR-0007, ADR-0008)."""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tensorgami.api.annotations import Annotation, Input, Loss, Note
from tensorgami.api.figure import Figure
from tensorgami.ir import Graph
from tensorgami.layout import LayoutOptions
from tensorgami.style import DEFAULT_STYLE, Theme, load_style
from tensorgami.transforms import Level
from tensorgami.transforms.annotate import add_input, add_loss, add_note, compose, link
from tensorgami.views import get_view

if TYPE_CHECKING:
    from torch import nn

LAYOUT_OPTIONS = tuple(f.name for f in fields(LayoutOptions))
_PARAMETERS = (
    "level",
    "view",
    "style",
    "overrides",
    "links",
    "annotations",
    "direction",
    *LAYOUT_OPTIONS,
)
_RENAMED = {"theme": "style"}  # the name sketched in ADR-0007, before ADR-0008


def draw(
    model: nn.Module | Mapping[str, nn.Module],
    example_input: Any,
    *,
    level: int | str = Level.LAYERS,
    view: str = "auto",
    style: str | Path | Theme = DEFAULT_STYLE,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    links: Sequence[tuple[str, ...]] = (),
    annotations: Sequence[Annotation] = (),
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
        links: with several models (``model`` a dict of name → module, ``example_input`` a
            dict of name → input), which tensor feeds which model input, as ``(source,
            target)`` or ``(source, target, label)``, e.g. ``("generator",
            "discriminator.image")``: the generator's output is the image its discriminator
            judges.
        annotations: what no module contains: ``tg.Loss`` terms (with ``trains=`` for the
            arrows back to the models they train), extra ``tg.Input`` and ``tg.Note``.
            Tensors are named by input/output names (``"input"``, ``"output 0"``, keyword
            names) or module paths (``"mu"``); with several models, after the model's name
            (``"discriminator.image"``; the name alone is the model's output).
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
    _check_extras(model, example_input, links, annotations)
    theme = style if isinstance(style, Theme) else load_style(style)
    if direction is not None:
        layout_options["direction"] = direction
    options = LayoutOptions.for_style(theme, **layout_options)
    level = Level.parse(level)
    chosen = get_view(view)

    graph = _annotate(_capture(model, example_input, overrides), links, annotations)
    shown, scene = chosen.draw(graph, theme, level=level, options=options)
    return Figure(scene, theme, shown)


def _capture(model, example_input: Any, overrides: Mapping | None) -> Graph:
    from tensorgami.importers.torch import trace  # torch is imported only to draw a model

    if not isinstance(model, Mapping):
        return trace(model, example_input, overrides=overrides)
    graphs = {
        name: trace(module, example_input[name], overrides=_own(name, model, overrides))
        for name, module in model.items()
    }
    return compose(graphs)


def _own(name: str, models: Mapping, overrides: Mapping | None) -> dict | None:
    """The overrides for one of several models: its own (``"generator.fc"`` → ``"fc"``) and
    the shared ones (class names and paths without a model's name)."""
    if not overrides:
        return None
    own = {}
    for key, value in overrides.items():
        head, _, rest = key.partition(".")
        if head == name and rest:
            own[rest] = value
        elif head not in models:
            own[key] = value
    return own


def _annotate(graph: Graph, links: Sequence[tuple[str, ...]], annotations) -> Graph:
    for a in annotations:  # extra inputs first: a link may then replace the input they join
        if isinstance(a, Input):
            graph = add_input(graph, a.name, a.to)
    for source, target, *label in links:
        graph = link(graph, source, target, *label)
    for a in annotations:
        if isinstance(a, Loss):
            graph = add_loss(graph, a.name, a.inputs, a.trains)
        elif isinstance(a, Note):
            graph = add_note(graph, a.text, a.at)
    return graph


def _check_extras(model, example_input, links, annotations) -> None:
    if isinstance(model, Mapping):
        if not model or not all(isinstance(k, str) and "." not in k for k in model):
            raise ValueError("several models: a dict of name → model, names without dots")
        missing = (
            sorted(set(model) - set(example_input)) if isinstance(example_input, Mapping) else []
        )
        if not isinstance(example_input, Mapping) or missing:
            raise ValueError(
                "several models need a dict of example inputs with the same names "
                f"({', '.join(model)}); missing: {', '.join(missing) or 'all'}"
            )
    for entry in links:
        if not (isinstance(entry, tuple) and len(entry) in (2, 3)):
            raise TypeError(
                f"links: each entry is (source, target) or (source, target, label); got {entry!r}"
            )
    for a in annotations:
        if not isinstance(a, Loss | Input | Note):
            raise TypeError(f"annotations: tg.Loss, tg.Input or tg.Note; got {a!r}")


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
