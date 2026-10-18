"""Which box of a figure shows which activation (roadmap step 14d).

:func:`resolve` finds, for each box of a drawn graph, the summary of what it outputs, among
those captured while the model ran (an :class:`ActivationStore` per model):

* a box that stands for one operation keeps its traced id; it shows the output of the last
  operation folded into it (a convolution with its ReLU shows the maps after the ReLU);
* a collapsed module (``<encoder>``) shows that module call's output;
* inside a block folded ×N, pass *k* shows the *k*-th copy's output (``layers.1``, or the
  second call of a recurrent cell), matched by position;
* an unrolled recurrence's per-step input x_t shows the slice that feeds that step's cell;
* with several models, ids carry the model's name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from tensorgami.activations import Activation
from tensorgami.ir.model import ROLE, Graph
from tensorgami.vocabulary import EdgeKind


@dataclass
class ActivationStore:
    """The summaries captured while one model ran: by traced node id (its first output) and
    by module call (group id: ``"encoder"``, ``"cell#2"``; ``""``: the whole model)."""

    graph: Graph
    nodes: dict[str, Activation] = field(default_factory=dict)
    groups: dict[str, Activation] = field(default_factory=dict)


def resolve(
    stores: Mapping[str, ActivationStore], shown: Graph
) -> dict[str, tuple[Activation, ...]]:
    """For each box of the drawn graph ``shown``, what it outputs, once per pass (one, or N
    inside a block folded ×N). ``stores``: by model name, ``""`` for a single model."""
    resolver = _Resolver(stores, shown)
    out: dict[str, tuple[Activation, ...]] = {}
    for node in shown.nodes:
        found = resolver.activations(node.id)
        if found:
            out[node.id] = found
    return out


class _Resolver:
    def __init__(self, stores: Mapping[str, ActivationStore], shown: Graph) -> None:
        self.stores, self.shown = stores, shown
        self.drawn: dict[str, set[str]] = {}  # model name → the trace ids drawn as boxes
        for node in shown.nodes:
            local = self._local(node.id)
            if local is not None:
                self.drawn.setdefault(local[0], set()).add(local[1])

    def _local(self, node_id: str) -> tuple[str, str] | None:
        """(model name, the id in that model's trace) of a box."""
        if "" in self.stores:
            return "", node_id
        if node_id.startswith("<") and node_id.endswith(">"):
            name, _, rest = node_id[1:-1].partition(".")
            return name, f"<{rest}>"
        name, slash, rest = node_id.partition("/")
        return (name, rest) if slash else None

    def activations(self, node_id: str) -> tuple[Activation, ...]:
        role = self.shown.node(node_id).params.get(ROLE)
        if role in ("in", "out"):
            return self._step(node_id, role)
        local = self._local(node_id)
        if local is None or local[0] not in self.stores:
            return ()
        name, lid = local
        store = self.stores[name]
        trace = store.graph
        copies = self._copies(node_id, name, trace)
        if lid.startswith("<") and lid.endswith(">"):  # a collapsed module: its call's output
            gid = lid[1:-1]
            found = [store.groups.get(gid)]
            if copies:
                found = [store.groups.get(_moved(gid, copies[0], c)) for c in copies]
        elif lid in {n.id for n in trace.nodes}:
            source = _follow(trace, lid, self.drawn.get(name, set()))
            found = [store.nodes.get(source)]
            if copies:
                found = [store.nodes.get(_same_place(trace, source, copies[0], c)) for c in copies]
        else:
            return ()
        if any(a is None for a in found):
            found = [found[0]] * len(found) if found[0] is not None else []
        return tuple(a for a in found if a is not None)

    def _step(self, node_id: str, role: str) -> tuple[Activation, ...]:
        """A step's input x_t (or output h_t) in an unrolled recurrence: what feeds the cell
        call of that step from outside, with x_t's shape (what that call produces)."""
        if role == "out":
            (edge,) = self.shown.in_edges(node_id)[:1] or (None,)
            return self.activations(edge.source) if edge is not None else ()
        pill = self.shown.node(node_id)
        for edge in self.shown.out_edges(node_id):
            local = self._local(edge.target)
            if local is None or local[0] not in self.stores or not local[1].startswith("<"):
                continue
            store, gid = self.stores[local[0]], local[1][1:-1]
            inside = {n.id for n in store.graph.nodes_under(gid)}
            for n in store.graph.nodes_under(gid):
                for e in store.graph.in_edges(n.id):
                    source = store.graph.node(e.source)
                    if e.source not in inside and source.output_shapes[:1] == pill.output_shapes:
                        found = store.nodes.get(e.source)
                        if found is not None:
                            return (found,)
        return ()

    def _copies(self, node_id: str, name: str, trace: Graph) -> list[str] | None:
        """The copies of the block folded ×N around a box, in the model's trace."""
        node = self.shown.node(node_id)
        chain = [self.shown.group(node.group), *self.shown.ancestors(node.group)]
        folded = [g for g in chain if g.count > 1]
        if not folded:
            return None
        outer = folded[-1]
        gid = outer.id.removeprefix(f"{name}.") if name else outer.id
        ids = {g.id for g in trace.groups}
        if f"{gid}#1" in ids:  # one module called again and again: a recurrence
            copies = [gid] + [f"{gid}#{k}" for k in range(1, outer.count)]
        else:  # stacked modules: layers.0, layers.1, …
            head, dot, last = gid.rpartition(".")
            if not last.isdigit():
                return None
            copies = [f"{head}{dot}{int(last) + k}" for k in range(outer.count)]
        return copies if all(c in ids for c in copies) else None


def _follow(trace: Graph, node_id: str, shown: set[str]) -> str:
    """The operation whose output a box shows: the box's own, or the last of the operations
    of the same shape folded into it (the ReLU after a convolution, an eval-mode dropout)."""
    current = trace.node(node_id)
    while True:
        outs = [e for e in trace.out_edges(current.id) if e.kind is EdgeKind.DATA]
        if len(outs) != 1:
            return current.id
        following = trace.node(outs[0].target)
        ins = [e for e in trace.in_edges(following.id) if e.kind is EdgeKind.DATA]
        if (
            following.id in shown
            or following.is_output
            or following.group != current.group
            or following.output_shapes[:1] != current.output_shapes[:1]
            or len(ins) != 1
        ):
            return current.id
        current = following


def _moved(gid: str, first: str, copy: str) -> str:
    """A module call's id in another copy of the block ``first``."""
    if gid == first:
        return copy
    if copy.startswith(f"{first}#"):  # the k-th call: the module inside it, called k-th too
        return f"{gid}{copy[len(first) :]}"
    return copy + gid[len(first) :] if gid.startswith(f"{first}.") else gid


def _same_place(trace: Graph, node_id: str, first: str, copy: str) -> str | None:
    """The node at the same place in another copy of a block (copies run the same
    operations in the same order)."""
    if copy == first:
        return node_id
    a, b = trace.nodes_under(first), trace.nodes_under(copy)
    if len(a) != len(b) or any(x.op != y.op for x, y in zip(a, b, strict=True)):
        return None
    index = [n.id for n in a].index(node_id) if node_id in {n.id for n in a} else None
    return b[index].id if index is not None else None
