"""Level-of-detail presets (style baseline §7): one pipeline per level."""

from __future__ import annotations

from enum import IntEnum

from tensorgami.ir import Graph, RepeatKind
from tensorgami.transforms.base import Pipeline
from tensorgami.transforms.cleanup import (
    FoldIntoPrevious,
    PruneEmptyGroups,
    RemoveConstants,
    RemoveLayoutOps,
)
from tensorgami.transforms.collapse import CollapseCells, CollapseGroups, CollapseToSystem
from tensorgami.transforms.recurrence import DetectRecurrence
from tensorgami.transforms.repeats import DetectLayerRepeats, FoldRepeats


class Level(IntEnum):
    SYSTEM = 0  # input → model → output
    STAGES = 1  # the model's direct sub-modules
    LAYERS = 2  # every layer; activations folded, repeats ×N, cells as one box (default)
    OPERATORS = 3  # everything the trace recorded
    CELLS = 4  # operators, with recurrent cells expanded to gates by the cell view (step 11)

    @classmethod
    def parse(cls, value: int | str) -> Level:
        """A level from its number (``2``) or name (``"layers"``), with a helpful error."""
        try:
            if isinstance(value, bool):  # True is an int, but never a level
                raise ValueError
            if isinstance(value, str):
                return cls(int(value)) if value.isdigit() else cls[value.upper()]
            return cls(value)
        except (KeyError, ValueError, TypeError):
            choices = ", ".join(f"{level.value} {level.name.lower()}" for level in cls)
            raise ValueError(f"level must be one of {choices}; got {value!r}") from None


def pipeline_for(
    level: int | str = Level.LAYERS, *, fold_layers: bool = True, fold_time: bool = True
) -> Pipeline:
    """The transforms for a level. ``fold_layers``: draw ×N stacks once; ``fold_time``: draw
    a recurrent cell once with a state loop (False keeps time steps unrolled)."""
    level = Level.parse(level)
    if level is Level.SYSTEM:
        return Pipeline([CollapseToSystem()])
    detect = [DetectRecurrence(), DetectLayerRepeats()]
    if level >= Level.OPERATORS:
        return Pipeline(detect)
    tidy = [
        RemoveLayoutOps(),
        RemoveConstants(),
        FoldIntoPrevious(),
        CollapseCells(),
        PruneEmptyGroups(),
    ]
    kinds = [k for k, on in ((RepeatKind.LAYERS, fold_layers), (RepeatKind.TIME, fold_time)) if on]
    fold = [FoldRepeats(kinds)] if kinds else []
    if level is Level.LAYERS:
        return Pipeline(detect + tidy + fold)
    return Pipeline(detect + tidy + fold + [CollapseGroups(depth=1), PruneEmptyGroups()])


def simplify(
    graph: Graph,
    level: int | str = Level.LAYERS,
    *,
    fold_layers: bool = True,
    fold_time: bool = True,
) -> Graph:
    """Reduce a full trace to the requested level of detail."""
    return pipeline_for(level, fold_layers=fold_layers, fold_time=fold_time).apply(graph)
