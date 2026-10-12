"""What no single ``nn.Module`` contains (roadmap step 11e, FR11): several models in one figure,
the links between them, extra inputs, loss terms and notes.

Tensors are named by *references*: an input or output name (``"input"``, ``"output 0"``, a
keyword argument such as ``"labels"``), or a module path (``"mu"``, ``"decoder"``). In a figure
of several models a reference starts with the model's name (``"discriminator.image"``), and a
model's name alone is its output (or, for ``trains``, the model itself).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping

from tensorgami.ir import (
    CAPTION,
    INPUT_OP,
    MODELS,
    ROOT,
    TRAINS,
    Edge,
    Graph,
    GraphValidator,
    Group,
    Node,
    Repeat,
)
from tensorgami.transforms.base import rebuild
from tensorgami.vocabulary import EdgeKind, OpCategory


class UnknownReference(ValueError):
    """A reference names no tensor or module of the figure."""


# --------------------------------------------------------------------------- several models


def compose(graphs: Mapping[str, Graph]) -> Graph:
    """Several traced models in one graph: each model's modules in a group named after it
    (its frame), its inputs and outputs outside; ids and module paths are prefixed with the
    model's name (``generator/n3``, ``generator.fc``)."""
    nodes: list[Node] = []
    edges: list[Edge] = []
    groups: list[Group] = []
    repeats: list[Repeat] = []
    for name, g in graphs.items():

        def group_id(gid: str, name: str = name) -> str:
            return name if gid == ROOT else f"{name}.{gid}"

        groups.append(Group(name, g.root.type_name, parent=ROOT, module=name, label=name))
        for grp in g.groups:
            if grp.parent is not None:
                groups.append(
                    dataclasses.replace(
                        grp,
                        id=group_id(grp.id),
                        parent=group_id(grp.parent),
                        module=f"{name}.{grp.module}" if grp.module else "",
                    )
                )
        for n in g.nodes:
            io = n.is_input or n.is_output
            nodes.append(
                dataclasses.replace(
                    n,
                    id=f"{name}/{n.id}",
                    group=ROOT if io else group_id(n.group),
                    module=f"{name}.{n.module}" if n.module else "",
                )
            )
        edges += [
            dataclasses.replace(e, source=f"{name}/{e.source}", target=f"{name}/{e.target}")
            for e in g.edges
        ]
        repeats += [
            dataclasses.replace(r, id=f"{name}.{r.id}", groups=[group_id(x) for x in r.groups])
            for r in g.repeats
        ]
    root = Group(ROOT, " + ".join(g.root.type_name for g in graphs.values()), parent=None)
    metadata = {k: v for g in graphs.values() for k, v in g.metadata.items()}
    metadata[MODELS] = ",".join(graphs)
    composed = Graph(
        tuple(nodes),
        tuple(edges),
        (root, *groups),
        tuple(repeats),
        " + ".join(graphs),
        metadata,
    )
    return GraphValidator().ensure_valid(composed)


# --------------------------------------------------------------------------- references


def _models(graph: Graph) -> list[str]:
    return [m for m in graph.metadata.get(MODELS, "").split(",") if m]


def _scope(graph: Graph, ref: str) -> tuple[str, str, list[Node]]:
    """(model, the rest of the reference, the model's nodes) — or ("", ref, every node)."""
    head, _, tail = ref.partition(".")
    if head in _models(graph):
        return head, tail, [n for n in graph.nodes if n.id.startswith(f"{head}/")]
    return "", ref, list(graph.nodes)


def _names(graph: Graph, nodes: list[Node]) -> str:
    """Names worth suggesting: inputs and outputs, top-level modules, models."""
    io = [n.label for n in nodes if n.is_input or n.is_output]
    dots = 1 if _models(graph) else 0  # inside a model, after its name
    top = {g.module for g in graph.groups if g.module and g.module.count(".") == dots}
    top |= {n.module for n in nodes if n.module and n.module.count(".") == dots}
    return ", ".join(dict.fromkeys([*io, *sorted(top), *_models(graph)]))


def resolve(graph: Graph, ref: str) -> str:
    """The node producing the tensor ``ref`` names (an output: the node behind it)."""
    model, rest, nodes = _scope(graph, ref)
    if model and not rest:  # a model's name: its output
        outputs = [n for n in nodes if n.is_output]
        if outputs:
            return outputs[0].id
    module = f"{model}.{rest}" if model else rest
    for n in nodes:
        if (n.is_input or n.is_output) and n.label == rest:
            return n.id
    leaves = [n for n in nodes if n.module == module]
    if leaves:
        return leaves[-1].id
    for g in reversed(graph.groups):
        if g.parent is not None and (g.id == module or g.module == module):
            under = graph.nodes_under(g.id)
            if under:
                return under[-1].id
    raise UnknownReference(f"nothing called {ref!r}; try one of: {_names(graph, nodes)}")


def _producer(graph: Graph, node_id: str) -> tuple[str, int]:
    """For an output: the node behind it (the output pill stays an output)."""
    node = graph.node(node_id)
    if node.is_output:
        e = graph.in_edges(node_id)[0]
        return e.source, e.source_port
    return node_id, 0


def _trained(graph: Graph, ref: str) -> str | None:
    """For ``trains``: the group (a model, a module) the reference names, if any."""
    model, rest, _ = _scope(graph, ref)
    if model and not rest:
        return model
    module = f"{model}.{rest}" if model else rest
    for g in graph.groups:
        if g.parent is not None and (g.id == module or g.module == module):
            return g.id
    return None


# --------------------------------------------------------------------------- edits


def link(graph: Graph, source: str, target: str, label: str | None = None) -> Graph:
    """The tensor ``source`` names feeds the model input ``target``, which disappears: a
    generator's output becomes the image its discriminator judges. A linked output becomes
    an intermediate tensor, named ``label`` (else after the input it replaces)."""
    src, dst = resolve(graph, source), resolve(graph, target)
    if not graph.node(dst).is_input:
        raise UnknownReference(f"a link ends at a model input; {target!r} is not one")
    nodes = []
    for n in graph.nodes:
        if n.id == dst:
            continue
        if n.id == src and n.is_output:  # now inside the figure: a tensor
            n = dataclasses.replace(
                n, op="tensor", category=OpCategory.TENSOR, label=label or graph.node(dst).label
            )
        nodes.append(n)
    edges = [
        dataclasses.replace(e, source=src, source_port=0) if e.source == dst else e
        for e in graph.edges
    ]
    return rebuild(graph, nodes=nodes, edges=edges)


def add_input(graph: Graph, name: str, to: str) -> Graph:
    """Another input for the tensor ``to`` names (e.g. the real images a discriminator also
    judges): it feeds what that tensor feeds."""
    target = resolve(graph, to)
    t = graph.node(target)
    shapes = t.output_shapes[:1]
    new = Node(f"in:{name}", INPUT_OP, OpCategory.IO, label=name, output_shapes=shapes)
    if t.is_input:
        extra = [Edge(new.id, e.target, e.kind) for e in graph.out_edges(target)]
    else:
        extra = [Edge(new.id, target)]
    return rebuild(graph, nodes=[new, *graph.nodes], edges=[*graph.edges, *extra])


def add_loss(graph: Graph, name: str, inputs: Iterable[str], trains: Iterable[str] = ()) -> Graph:
    """A loss term fed by the tensors it compares; dashed training arrows go back to the
    modules its gradient updates (``trains``)."""
    edges, groups = list(graph.edges), []
    for ref in trains:
        group = _trained(graph, ref)
        if group is not None:  # an arrow into its frame, drawn by the layout
            groups.append(group)
        else:  # a single layer: an edge into it
            edges.append(Edge(f"loss:{name}", resolve(graph, ref), EdgeKind.TRAINING))
    params = {TRAINS: tuple(groups)} if groups else {}
    loss = Node(f"loss:{name}", "loss", OpCategory.LOSS, label=name, params=params)
    for ref in inputs:
        source, port = _producer(graph, resolve(graph, ref))
        edges.append(Edge(source, loss.id, EdgeKind.DATA, port))
    return rebuild(graph, nodes=[*graph.nodes, loss], edges=edges)


def add_note(graph: Graph, text: str, at: str) -> Graph:
    """``text`` as the second line of the node the reference names (e.g. ε ~ N(0, I))."""
    target = resolve(graph, at)
    nodes = [
        dataclasses.replace(n, params={**n.params, CAPTION: text}) if n.id == target else n
        for n in graph.nodes
    ]
    return rebuild(graph, nodes=nodes)
