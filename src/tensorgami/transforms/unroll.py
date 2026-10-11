"""Unrolled recurrence (roadmap step 11): time steps side by side, layers stacked.

:class:`UnrollRecurrence` prepares a graph for the unrolled view. Each recurrence ends up in
one group of type ``UNROLLED`` (``tensorgami.ir``) whose nodes carry their place in the grid:
``params[STEP]``
(0-based time step), ``params[LAYER]`` (0 = the layer that reads the input) and
``params[ROLE]`` (``"cell"``, or ``"in"`` / ``"out"`` for the per-step tensors x_t / h_t).
The layout arranges such a group as a time × layer grid.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict

from tensorgami.ir import LAYER, ROLE, STEP, Edge, Graph, Group, Node, RepeatKind
from tensorgami.ir import UNROLLED as GRID
from tensorgami.transforms.base import rebuild
from tensorgami.transforms.collapse import _collapse_groups
from tensorgami.vocabulary import EdgeKind, OpCategory

FUSED_STATES = {"LSTM": 2, "GRU": 1, "RNN": 1}  # fused module → state tensors per cell (h, c)


class UnrollRecurrence:
    """Turn each recurrence into a grid group:

    1. fused modules (``nn.LSTM`` / ``GRU`` / ``RNN``: one call over the whole sequence)
       become ``num_layers × steps`` cells;
    2. cells called step by step (TIME repeats) become one node per call, and repeats that
       feed each other at the same step become the layers of one grid;
    3. an outside tensor feeding every step of the first layer is split into per-step
       tensors x₁ … x_T, and a sequence read from every step of the top layer into h₁ … h_T;
    4. a sequence longer than ``max_steps`` keeps its first ``max_steps - 1`` steps and the
       last one (the layout marks the gap with "⋯").

    Bidirectional fused modules are left as one box.
    """

    name = "unroll-recurrence"

    def __init__(self, max_steps: int = 4) -> None:
        if max_steps < 2:
            raise ValueError("max_steps must be ≥ 2 (the first and the last step)")
        self.max_steps = max_steps

    def apply(self, graph: Graph) -> Graph:
        graph, fused = _expand_fused(graph)
        graph, looped = _looped_grids(graph)
        for n, cells in enumerate([*fused, *looped]):
            graph = _Gather(graph, f"{GRID}:{n}", cells, self.max_steps).apply()
        return graph


# --------------------------------------------------------------------------- fused modules


def _steps_and_batch(params, shape) -> tuple[int, int | None] | None:
    if not shape or len(shape) not in (2, 3):
        return None
    if len(shape) == 2:  # unbatched: (T, features)
        return shape[0], None
    steps, batch = (shape[1], shape[0]) if params.get("batch_first") else (shape[0], shape[1])
    return (steps, batch) if steps else None


def _expand_fused(graph: Graph) -> tuple[Graph, list[dict[str, tuple[int, int]]]]:
    nodes, edges, grids, replaced = [], [], [], {}
    for node in graph.nodes:
        states = FUSED_STATES.get(node.op)
        inputs = [e for e in graph.in_edges(node.id) if e.target_port == 0]
        if not states or node.count > 1 or node.params.get("bidirectional") or not inputs:
            nodes.append(node)
            continue
        x = inputs[0]
        source_shapes = graph.node(x.source).output_shapes
        shape = source_shapes[x.source_port] if x.source_port < len(source_shapes) else None
        sizes = _steps_and_batch(node.params, shape)
        hidden = node.params.get("hidden_size")
        if sizes is None or not isinstance(hidden, int):
            nodes.append(node)
            continue
        steps, batch = sizes
        layers = int(node.params.get("num_layers") or 1)
        out = ((batch, hidden) if batch is not None else (hidden,),) * states
        cell = {}
        for t in range(steps):
            for k in range(layers):
                cell[k, t] = f"{node.id}.{k}.{t}"
                nodes.append(
                    Node(
                        cell[k, t],
                        f"{node.op}Cell",
                        OpCategory.RECURRENT,
                        group=node.group,
                        call_index=t,
                        params={"hidden_size": hidden, STEP: t, LAYER: k, ROLE: "cell"},
                        output_shapes=out,
                        module=node.module,
                    )
                )
        top = layers - 1
        replaced[node.id] = (cell, steps, top, states)
        grids.append({cell[k, t]: (k, t) for k in range(layers) for t in range(steps)})
        for t in range(steps):
            for k in range(layers):
                if k < top:
                    edges.append(Edge(cell[k, t], cell[k + 1, t]))
                if t + 1 < steps:
                    edges += [
                        Edge(cell[k, t], cell[k, t + 1], EdgeKind.STATE, source_port=s)
                        for s in range(states)
                    ]
    if not replaced:
        return graph, []
    for e in graph.edges:
        if e.target in replaced:
            cell, steps, top, _ = replaced[e.target]
            if e.target_port == 0:  # the sequence: step t of the first layer reads x_t
                edges += [Edge(e.source, cell[0, t], e.kind, e.source_port) for t in range(steps)]
            else:  # initial states: the first step of every layer
                edges += [Edge(e.source, cell[k, 0], e.kind, e.source_port) for k in range(top + 1)]
        elif e.source in replaced:
            cell, steps, top, states = replaced[e.source]
            if e.source_port == 0:  # the output sequence: h_t of the top layer, every step
                edges += [Edge(cell[top, t], e.target, e.kind) for t in range(steps)]
            else:  # final states (h_n, c_n): the last step of the top layer
                port = min(e.source_port - 1, states - 1)
                edges.append(Edge(cell[top, steps - 1], e.target, e.kind, port))
        else:
            edges.append(e)
    return rebuild(graph, nodes=nodes, edges=edges), grids


# --------------------------------------------------------------------------- traced loops


def _looped_grids(graph: Graph) -> tuple[Graph, list[dict[str, tuple[int, int]]]]:
    """Grids from TIME repeats: one node per call (multi-node calls are collapsed first);
    repeats that feed each other at the same step are stacked as layers."""
    time = [r for r in graph.repeats if r.kind is RepeatKind.TIME]
    if not time:
        return graph, []
    bulky = [
        graph.group(g)
        for r in time
        for g in r.groups
        if len(graph.nodes_under(g)) > 1 or graph.children(g)
    ]
    if bulky:
        graph = _collapse_groups(graph, bulky)
        time = [r for r in graph.repeats if r.kind is RepeatKind.TIME]
    place: dict[str, tuple[str, int]] = {}  # node → (repeat, step)
    for r in time:
        for t, g in enumerate(r.groups):
            for n in graph.members(g):
                place[n.id] = (r.id, t)
    above: dict[str, set[str]] = defaultdict(set)  # repeat → repeats it feeds at the same step
    for e in graph.edges:
        a, b = place.get(e.source), place.get(e.target)
        if a and b and a[0] != b[0] and a[1] == b[1] and e.kind is EdgeKind.DATA:
            above[a[0]].add(b[0])
    component = {r.id: r.id for r in time}  # union-find over linked repeats

    def find(x: str) -> str:
        while component[x] != x:
            component[x] = component[component[x]]
            x = component[x]
        return x

    for a, bs in above.items():
        for b in bs:
            component[find(a)] = find(b)
    members: dict[str, list] = defaultdict(list)
    for r in time:
        members[find(r.id)].append(r)
    grids = []
    for repeats in members.values():
        if len({r.count for r in repeats}) != 1:
            continue  # layers with different numbers of steps: not a grid
        layer: dict[str, int] = {}
        for r in repeats:  # longest path from the layers nobody feeds
            _layer_of(r.id, above, layer, {x.id for x in repeats})
        grids.append({n: (layer[rid], t) for n, (rid, t) in place.items() if rid in layer})
    return graph, grids


def _layer_of(rid: str, above: dict, layer: dict, scope: set, seen: frozenset = frozenset()) -> int:
    if rid in layer:
        return layer[rid]
    below = [a for a in scope if rid in above.get(a, ()) and a not in seen]
    layer[rid] = max(
        (_layer_of(a, above, layer, scope, seen | {rid}) + 1 for a in below), default=0
    )
    return layer[rid]


# --------------------------------------------------------------------------- one grid


class _Gather:
    """Per-step tensors, elided middle steps, and one GRID group for one recurrence."""

    def __init__(self, graph: Graph, gid: str, cells: dict, max_steps: int) -> None:
        self.g, self.gid, self.cells = graph, gid, dict(cells)
        self.steps = max(t for _, t in cells.values()) + 1
        self.top = max(k for k, _ in cells.values())
        self.max_steps = max_steps

    def apply(self) -> Graph:
        g = self.g
        nodes = {n.id: n for n in g.nodes}
        edges = list(g.edges)
        new_nodes: list[Node] = []
        edges, pills_in = self._split(edges, nodes, role="in")
        edges, pills_out = self._split(edges, nodes, role="out")
        new_nodes += pills_in + pills_out
        keep = self._kept_steps()
        if keep is not None:
            edges = self._elide(edges, keep, {p.id: p.params[STEP] for p in new_nodes})
            new_nodes = [p for p in new_nodes if p.params[STEP] in keep]
        dropped = {n for n, (_, t) in self.cells.items() if keep is not None and t not in keep}
        dropped |= {p.id for p in pills_in + pills_out if p not in new_nodes}
        edges = [e for e in edges if e.source not in dropped and e.target not in dropped]

        parent = self._common_parent()
        grid = Group(self.gid, GRID, parent=parent)
        out_nodes: list[Node] = []
        first = min(i for i, n in enumerate(g.nodes) if n.id in self.cells)
        for i, n in enumerate(g.nodes):
            if i == first:
                out_nodes += [p for p in new_nodes if p.params[ROLE] == "in"]
            if n.id in dropped:
                continue
            if n.id in self.cells:
                k, t = self.cells[n.id]
                params = {**n.params, STEP: t, LAYER: k, ROLE: "cell"}
                n = dataclasses.replace(n, group=self.gid, params=params)
            out_nodes.append(n)
        out_nodes += [p for p in new_nodes if p.params[ROLE] == "out"]
        used = {n.group for n in out_nodes}
        groups = [*g.groups, grid]
        groups = [x for x in groups if not self._emptied(x, used, groups)]
        return rebuild(g, nodes=out_nodes, edges=edges, groups=groups)

    # ------------------------------------------------------------------ per-step tensors

    def _split(self, edges: list[Edge], nodes: dict, role: str) -> tuple[list[Edge], list[Node]]:
        """An outside tensor feeding layer 0 at several steps (``in``), or an outside node
        reading the top layer at several steps (``out``), gets one pill per step."""
        layer = 0 if role == "in" else self.top
        by_end: dict[tuple, dict[int, list[Edge]]] = defaultdict(lambda: defaultdict(list))
        for e in edges:
            inside, outside = (e.target, e.source) if role == "in" else (e.source, e.target)
            if inside in self.cells and outside not in self.cells and e.kind is EdgeKind.DATA:
                k, t = self.cells[inside]
                if k == layer:
                    key = (outside, e.source_port) if role == "in" else (outside,)
                    by_end[key][t].append(e)
        pills: list[Node] = []
        replaced: set[int] = set()
        new_edges: list[Edge] = []
        letter = "x" if role == "in" else "h"
        for n, per_step in enumerate(by_end.values()):
            if len(per_step) < 2:
                continue
            suffix = "" if n == 0 else str(n + 1)
            for t, step_edges in sorted(per_step.items()):
                pill = Node(
                    f"{self.gid}.{role}{suffix}.{t}",
                    "step",
                    OpCategory.TENSOR,
                    label=f"{letter}{suffix}_{{{t + 1}}}",
                    group=self.gid,
                    params={STEP: t, ROLE: role},
                    output_shapes=self._step_shape(nodes, step_edges[0], role),
                )
                pills.append(pill)
                for e in step_edges:  # source → pill → target
                    replaced.add(id(e))
                    new_edges += [
                        Edge(e.source, pill.id, e.kind, e.source_port),
                        Edge(pill.id, e.target, e.kind),
                    ]
        kept = [e for e in edges if id(e) not in replaced]
        return kept + new_edges, pills

    def _step_shape(self, nodes: dict, edge: Edge, role: str) -> tuple:
        source = nodes[edge.source]
        shapes = source.output_shapes
        shape = shapes[edge.source_port] if edge.source_port < len(shapes) else None
        if role == "out" or not shape:
            return (shape,) if shape else ()
        for axis in range(1, len(shape)):  # drop the time axis (not the batch)
            if shape[axis] == self.steps:
                return (shape[:axis] + shape[axis + 1 :],)
        return ()

    # ------------------------------------------------------------------ long sequences

    def _kept_steps(self) -> set[int] | None:
        if self.steps <= self.max_steps:
            return None
        return set(range(self.max_steps - 1)) | {self.steps - 1}

    def _elide(self, edges: list[Edge], keep: set[int], pill_steps: dict) -> list[Edge]:
        """Bridge the state from the last early step to the final one."""
        early, last = self.max_steps - 2, self.steps - 1
        at = {(k, t): n for n, (k, t) in self.cells.items()}
        bridges = []
        for e in edges:
            if e.kind is EdgeKind.STATE and e.source in self.cells and e.target in self.cells:
                (k, t), (k2, t2) = self.cells[e.source], self.cells[e.target]
                if k == k2 and t == early and t2 == early + 1 and (k, last) in at:
                    bridges.append(dataclasses.replace(e, target=at[k, last]))
        return edges + bridges

    # ------------------------------------------------------------------ group tree

    def _common_parent(self) -> str:
        chains = []
        for n in self.cells:
            gid = self.g.node(n).group
            chain = [gid, *(a.id for a in self.g.ancestors(gid))]
            chains.append(chain)
        # the call groups of a loop are replaced by the grid: start from their parents
        own = {self.g.node(n).group for n in self.cells}
        common = None
        for chain in chains:
            ids = [x for x in chain if x not in own]
            common = ids if common is None else [x for x in common if x in ids]
        return common[0] if common else ""

    def _emptied(self, group: Group, used: set[str], groups: list[Group]) -> bool:
        """A call group whose only node moved into the grid disappears."""
        if group.id == self.gid or group.parent is None:
            return False
        moved = {self.g.node(n).group for n in self.cells}
        if group.id not in moved:
            return False
        has_children = any(x.parent == group.id for x in groups)
        return group.id not in used and not has_children
