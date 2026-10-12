"""Framework-neutral intermediate representation of a model (ADR-0002)."""

from __future__ import annotations

from tensorgami.ir.builder import GraphBuilder, invocation_id
from tensorgami.ir.model import (
    INPUT_OP,
    LAYER,
    OUTPUT_OP,
    ROLE,
    ROOT,
    STEP,
    U_LEVEL,
    U_SIDE,
    UNROLLED,
    USHAPE,
    VOLUME,
    Edge,
    Graph,
    Group,
    Node,
    ParamValue,
    Repeat,
    RepeatKind,
    Shape,
)
from tensorgami.ir.serialize import SCHEMA_VERSION, from_dict, from_json, to_dict, to_json
from tensorgami.ir.shapes import format_shape
from tensorgami.ir.summary import summarize
from tensorgami.ir.validation import GraphError, GraphValidator, Issue

__all__ = [
    "INPUT_OP",
    "LAYER",
    "OUTPUT_OP",
    "ROLE",
    "ROOT",
    "STEP",
    "UNROLLED",
    "USHAPE",
    "VOLUME",
    "U_LEVEL",
    "U_SIDE",
    "SCHEMA_VERSION",
    "Edge",
    "Graph",
    "GraphBuilder",
    "GraphError",
    "GraphValidator",
    "Group",
    "Issue",
    "Node",
    "ParamValue",
    "Repeat",
    "RepeatKind",
    "Shape",
    "format_shape",
    "from_dict",
    "from_json",
    "invocation_id",
    "summarize",
    "to_dict",
    "to_json",
]
