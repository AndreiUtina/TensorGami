"""Structural rules for :class:`Graph` (ADR-0002). Each rule is a small object; the
:class:`GraphValidator` runs a list of them, so new rules never change existing ones."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from tensorgami.ir.model import ROOT, Graph
from tensorgami.vocabulary import EdgeKind


class GraphError(ValueError):
    """Raised when a graph violates the IR rules."""


@dataclass(frozen=True, slots=True)
class Issue:
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message}"


class GraphRule(Protocol):
    name: str

    def check(self, graph: Graph) -> Iterable[Issue]: ...


@dataclass(frozen=True)
class UniqueIdsRule:
    name: str = "unique-ids"

    def check(self, graph: Graph) -> Iterable[Issue]:
        for what, ids in (
            ("node", [n.id for n in graph.nodes]),
            ("group", [g.id for g in graph.groups]),
            ("repeat", [r.id for r in graph.repeats]),
        ):
            for ident, count in Counter(ids).items():
                if count > 1:
                    yield Issue(self.name, f"{what} id '{ident}' is used {count} times")


@dataclass(frozen=True)
class GroupTreeRule:
    """Exactly one root (id ``""``), every parent exists, no cycles, every node placed."""

    name: str = "group-tree"

    def check(self, graph: Graph) -> Iterable[Issue]:
        roots = [g for g in graph.groups if g.parent is None]
        if len(roots) != 1 or roots[0].id != ROOT:
            yield Issue(self.name, f"expected one root group with id '', found {len(roots)}")
        for g in graph.groups:
            if g.parent is not None and not graph.has_group(g.parent):
                yield Issue(self.name, f"group '{g.id}' has unknown parent '{g.parent}'")
        for g in graph.groups:
            seen, current = {g.id}, g
            while current.parent is not None and graph.has_group(current.parent):
                current = graph.group(current.parent)
                if current.id in seen:
                    yield Issue(self.name, f"group '{g.id}' is part of a parent cycle")
                    break
                seen.add(current.id)
        for n in graph.nodes:
            if not graph.has_group(n.group):
                yield Issue(self.name, f"node '{n.id}' is in unknown group '{n.group}'")


@dataclass(frozen=True)
class EdgeEndpointsRule:
    """Edges join existing nodes, through ports that exist when shapes are known."""

    name: str = "edge-endpoints"

    def check(self, graph: Graph) -> Iterable[Issue]:
        for e in graph.edges:
            for end in (e.source, e.target):
                if not graph.has_node(end):
                    yield Issue(self.name, f"edge {e.source} → {e.target}: unknown node '{end}'")
            if graph.has_node(e.source):
                outputs = graph.node(e.source).output_shapes
                if outputs and not 0 <= e.source_port < len(outputs):
                    yield Issue(
                        self.name,
                        f"edge {e.source} → {e.target}: source port {e.source_port} does not "
                        f"exist ({len(outputs)} outputs)",
                    )
            if e.target_port < 0:
                yield Issue(self.name, f"edge {e.source} → {e.target}: negative target port")


@dataclass(frozen=True)
class ShapeConsistencyRule:
    """An edge's shape, when given, equals the shape its source produces on that port."""

    name: str = "shape-consistency"

    def check(self, graph: Graph) -> Iterable[Issue]:
        for e in graph.edges:
            if e.shape is None or not graph.has_node(e.source):
                continue
            outputs = graph.node(e.source).output_shapes
            if 0 <= e.source_port < len(outputs) and outputs[e.source_port] != e.shape:
                yield Issue(
                    self.name,
                    f"edge {e.source} → {e.target}: shape {e.shape} differs from source output "
                    f"{outputs[e.source_port]}",
                )


@dataclass(frozen=True)
class AcyclicDataFlowRule:
    """Data flow is acyclic. Only ``state`` edges (folded recurrence), undirected ``tied``
    edges and ``training`` signals (gradients flow backwards) may close a loop."""

    name: str = "acyclic"
    cycle_kinds: frozenset[EdgeKind] = frozenset({EdgeKind.STATE, EdgeKind.TIED, EdgeKind.TRAINING})

    def check(self, graph: Graph) -> Iterable[Issue]:
        successors: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
        for e in graph.edges:
            if e.kind not in self.cycle_kinds and e.source in successors:
                successors[e.source].append(e.target)
        white, grey, black = 0, 1, 2
        colour = dict.fromkeys(successors, white)
        for start in successors:  # iterative DFS: deep graphs must not hit the recursion limit
            if colour[start] != white:
                continue
            stack = [(start, iter(successors[start]))]
            colour[start] = grey
            while stack:
                node, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    colour[node] = black
                    stack.pop()
                elif colour.get(nxt) == grey:
                    yield Issue(self.name, f"data-flow cycle through '{node}' → '{nxt}'")
                    return
                elif colour.get(nxt) == white:
                    colour[nxt] = grey
                    stack.append((nxt, iter(successors[nxt])))


@dataclass(frozen=True)
class InputOutputRule:
    """Graph inputs receive no data; graph outputs send none."""

    name: str = "inputs-outputs"

    def check(self, graph: Graph) -> Iterable[Issue]:
        for n in graph.inputs:
            if any(e.kind is EdgeKind.DATA for e in graph.in_edges(n.id)):
                yield Issue(self.name, f"input '{n.id}' has incoming data edges")
        for n in graph.outputs:
            if any(e.kind is EdgeKind.DATA for e in graph.out_edges(n.id)):
                yield Issue(self.name, f"output '{n.id}' has outgoing data edges")


@dataclass(frozen=True)
class RepeatRule:
    """A repeat has ≥ 2 existing groups with a common parent, and a group repeats once."""

    name: str = "repeats"

    def check(self, graph: Graph) -> Iterable[Issue]:
        used: Counter[str] = Counter()
        for r in graph.repeats:
            if r.count < 2:
                yield Issue(self.name, f"repeat '{r.id}' has {r.count} group(s); needs ≥ 2")
            missing = [g for g in r.groups if not graph.has_group(g)]
            if missing:
                yield Issue(self.name, f"repeat '{r.id}' references unknown groups {missing}")
                continue
            parents = {graph.group(g).parent for g in r.groups}
            if len(parents) > 1:
                yield Issue(self.name, f"repeat '{r.id}' spans different parents {sorted(parents)}")
            used.update(r.groups)
        for g, count in used.items():
            if count > 1:
                yield Issue(self.name, f"group '{g}' belongs to {count} repeats")


@dataclass(frozen=True)
class CountRule:
    """A node or group stands for at least one invocation."""

    name: str = "counts"

    def check(self, graph: Graph) -> Iterable[Issue]:
        for what, items in (("node", graph.nodes), ("group", graph.groups)):
            for item in items:
                if item.count < 1:
                    yield Issue(self.name, f"{what} '{item.id}' has count {item.count}; needs ≥ 1")


def default_rules() -> list[GraphRule]:
    return [
        UniqueIdsRule(),
        GroupTreeRule(),
        EdgeEndpointsRule(),
        ShapeConsistencyRule(),
        AcyclicDataFlowRule(),
        InputOutputRule(),
        RepeatRule(),
        CountRule(),
    ]


class GraphValidator:
    def __init__(self, rules: Sequence[GraphRule] | None = None) -> None:
        self._rules = list(default_rules() if rules is None else rules)

    def validate(self, graph: Graph) -> list[Issue]:
        return [issue for rule in self._rules for issue in rule.check(graph)]

    def ensure_valid(self, graph: Graph) -> Graph:
        issues = self.validate(graph)
        if issues:
            lines = "\n  ".join(str(i) for i in issues)
            raise GraphError(f"graph '{graph.name}' is invalid:\n  {lines}")
        return graph
