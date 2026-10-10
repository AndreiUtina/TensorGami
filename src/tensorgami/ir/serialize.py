"""Versioned JSON serialisation of :class:`Graph` (fixtures, caching, hand-editing)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from tensorgami.ir.model import Edge, Graph, Group, Node, ParamValue, Repeat, RepeatKind, Shape
from tensorgami.ir.validation import GraphError, GraphValidator
from tensorgami.vocabulary import EdgeKind, OpCategory

SCHEMA = "tensorgami.ir"
SCHEMA_VERSION = 1


def to_dict(graph: Graph) -> dict[str, Any]:
    """Plain JSON-compatible data; field order is fixed so output is deterministic."""
    return {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "name": graph.name,
        "metadata": dict(graph.metadata),
        "groups": [
            {
                "id": g.id,
                "type": g.type_name,
                "parent": g.parent,
                "module": g.module,
                "call_index": g.call_index,
                "label": g.label,
                "category": g.category.value if g.category else None,
            }
            for g in graph.groups
        ],
        "nodes": [
            {
                "id": n.id,
                "op": n.op,
                "category": n.category.value,
                "label": n.label,
                "group": n.group,
                "call_index": n.call_index,
                "params": {k: _param_out(v) for k, v in n.params.items()},
                "outputs": [list(s) for s in n.output_shapes],
                "module": n.module,
            }
            for n in graph.nodes
        ],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "kind": e.kind.value,
                "source_port": e.source_port,
                "target_port": e.target_port,
                "shape": list(e.shape) if e.shape is not None else None,
            }
            for e in graph.edges
        ],
        "repeats": [
            {"id": r.id, "kind": r.kind.value, "groups": list(r.groups)} for r in graph.repeats
        ],
    }


def from_dict(data: Mapping[str, Any], *, validate: bool = True) -> Graph:
    if data.get("schema") != SCHEMA:
        raise GraphError(f"not a Tensorgami graph (schema={data.get('schema')!r})")
    if data.get("version") != SCHEMA_VERSION:
        raise GraphError(
            f"unsupported graph version {data.get('version')!r} (this Tensorgami reads "
            f"version {SCHEMA_VERSION})"
        )
    try:
        graph = Graph(
            groups=tuple(
                Group(
                    g["id"],
                    g["type"],
                    parent=g["parent"],
                    module=g.get("module", ""),
                    call_index=g.get("call_index", 0),
                    label=g.get("label", ""),
                    category=OpCategory(g["category"]) if g.get("category") else None,
                )
                for g in data["groups"]
            ),
            nodes=tuple(
                Node(
                    n["id"],
                    n["op"],
                    OpCategory(n.get("category", OpCategory.OTHER)),
                    n.get("label", ""),
                    n.get("group", ""),
                    n.get("call_index", 0),
                    {k: _param_in(v) for k, v in n.get("params", {}).items()},
                    tuple(_shape_in(s) for s in n.get("outputs", [])),
                    n.get("module", ""),
                )
                for n in data["nodes"]
            ),
            edges=tuple(
                Edge(
                    e["source"],
                    e["target"],
                    EdgeKind(e.get("kind", EdgeKind.DATA)),
                    e.get("source_port", 0),
                    e.get("target_port", 0),
                    _shape_in(e["shape"]) if e.get("shape") is not None else None,
                )
                for e in data["edges"]
            ),
            repeats=tuple(
                Repeat(r["id"], RepeatKind(r["kind"]), tuple(r["groups"]))
                for r in data.get("repeats", [])
            ),
            name=data.get("name", ""),
            metadata=data.get("metadata", {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GraphError(f"malformed graph data: {type(exc).__name__}: {exc}") from exc
    return GraphValidator().ensure_valid(graph) if validate else graph


def to_json(graph: Graph, *, indent: int | None = 2) -> str:
    return json.dumps(to_dict(graph), indent=indent, ensure_ascii=False)


def from_json(text: str, *, validate: bool = True) -> Graph:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GraphError(f"invalid JSON: {exc}") from exc
    return from_dict(data, validate=validate)


def _param_out(value: ParamValue) -> Any:
    return [_param_out(v) for v in value] if isinstance(value, tuple) else value


def _param_in(value: Any) -> ParamValue:
    return tuple(_param_in(v) for v in value) if isinstance(value, list) else value


def _shape_in(value: Any) -> Shape:
    if not isinstance(value, list) or not all(v is None or isinstance(v, int) for v in value):
        raise GraphError(f"invalid shape {value!r}: expected a list of ints or nulls")
    return tuple(value)
