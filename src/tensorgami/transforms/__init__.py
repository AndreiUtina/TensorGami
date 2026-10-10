"""Graph transforms: from a full trace to the chosen level of detail (roadmap step 7).

Each transform is a small ``Graph → Graph`` object; :func:`simplify` applies the preset
pipeline for a level, and :class:`Pipeline` composes custom ones.
"""

from __future__ import annotations

from tensorgami.transforms.base import GraphTransform, Pipeline
from tensorgami.transforms.cleanup import (
    FoldIntoPrevious,
    PruneEmptyGroups,
    RemoveConstants,
    RemoveLayoutOps,
)
from tensorgami.transforms.collapse import CollapseCells, CollapseGroups, CollapseToSystem
from tensorgami.transforms.levels import Level, pipeline_for, simplify
from tensorgami.transforms.recurrence import DetectRecurrence
from tensorgami.transforms.repeats import DetectLayerRepeats, FoldRepeats

__all__ = [
    "CollapseCells",
    "CollapseGroups",
    "CollapseToSystem",
    "DetectLayerRepeats",
    "DetectRecurrence",
    "FoldIntoPrevious",
    "FoldRepeats",
    "GraphTransform",
    "Level",
    "Pipeline",
    "PruneEmptyGroups",
    "RemoveConstants",
    "RemoveLayoutOps",
    "pipeline_for",
    "simplify",
]
