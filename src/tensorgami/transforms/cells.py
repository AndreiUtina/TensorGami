"""The cell view (roadmap step 11d): a model's recurrent cells drawn as their gates.

:func:`cell_figure` builds a new graph with one framed cell per distinct recurrent cell:

* fused PyTorch modules (``nn.LSTM`` / ``LSTMCell`` / ``GRU`` / ``GRUCell`` / ``RNN`` /
  ``RNNCell``) run no traceable internals, so their textbook gates are drawn from the
  module's settings;
* custom cells (e.g. ConvLSTM) are read from the trace: an LSTM is recognised by its data flow
  (the σ multiplied with the incoming state is the forget gate, the σ multiplied with the tanh
  candidate the input gate, the σ multiplied with tanh(c) the output gate), and its gates
  carry the category of the layer that computes them (a convolution, for ConvLSTM); any other
  cell shows its traced operations.

LSTMs become a group of type ``LSTM_CELL`` (drawn as the classic figure); the others are
ordinary groups for the layered layout.
"""

from __future__ import annotations

from tensorgami.ir import (
    CAPTION,
    CELL_ROLE,
    LSTM_CELL,
    Graph,
    GraphBuilder,
    RepeatKind,
    Shape,
)
from tensorgami.transforms.recurrence import DetectRecurrence
from tensorgami.vocabulary import EdgeKind, OpCategory

FUSED = {
    "LSTM": "lstm",
    "LSTMCell": "lstm",
    "GRU": "gru",
    "GRUCell": "gru",
    "RNN": "rnn",
    "RNNCell": "rnn",
}
GATES = (
    ("f", "σ", "forget"),
    ("i", "σ", "input"),
    ("g", "tanh", "candidate"),
    ("o", "σ", "output"),
)


def cell_figure(graph: Graph) -> Graph | None:
    """One framed cell per distinct recurrent cell of ``graph``; ``None`` if it has none."""
    b = GraphBuilder(f"{graph.name} cells", root_type=graph.root.type_name)
    seen: set = set()
    for node in graph.nodes:  # fused modules
        kind = FUSED.get(node.op)
        if not kind or not node.module:
            continue
        key = (node.op, node.params.get("input_size"), node.params.get("hidden_size"))
        if key in seen:
            continue
        seen.add(key)
        gid = f"cell:{len(seen)}"
        sizes = _fused_sizes(graph, node)
        if kind == "lstm":
            title = f"{node.op} cell · {sizes['x'][-1]} → {sizes['h'][-1]}"
            _lstm(b, gid, title, OpCategory.LINEAR, sizes)
        else:
            _CANONICAL[kind](b, gid, f"{node.op} cell · {sizes['x'][-1]} → {sizes['h'][-1]}", sizes)
    graph = DetectRecurrence().apply(graph)  # a module called step after step with state
    looped = {g for r in graph.repeats if r.kind is RepeatKind.TIME for g in r.groups}

    def recurrent(group) -> bool:
        return group.category is OpCategory.RECURRENT or group.id in looped

    for group in graph.groups:  # custom cells: the first call of each cell type
        if group.parent is None or not recurrent(group) or group.type_name in seen:
            continue
        if any(recurrent(a) for a in graph.ancestors(group.id) if a.parent is not None):
            continue  # inside a cell already drawn
        seen.add(group.type_name)
        gid = f"cell:{len(seen)}"
        parts = _match_lstm(graph, group.id)
        if parts is not None:
            gate_layer = graph.node(parts.pop("layer"))
            title = f"{group.type_name} · gates: {gate_layer.op}"
            _lstm(b, gid, title, gate_layer.category, _traced_sizes(graph, parts))
        else:
            _copy(b, gid, graph, group.id)
    return b.build() if seen else None


# --------------------------------------------------------------------------- sizes


def _shape_of(graph: Graph, node_id: str, port: int = 0) -> Shape | None:
    shapes = graph.node(node_id).output_shapes
    return shapes[port] if port < len(shapes) else None


def _fused_sizes(graph: Graph, node) -> dict[str, Shape]:
    hidden = int(node.params.get("hidden_size") or 0)
    features = int(node.params.get("input_size") or 0)
    batch = None
    for e in graph.in_edges(node.id):
        shape = _shape_of(graph, e.source, e.source_port)
        if e.target_port == 0 and shape and len(shape) >= 2:
            batch = shape[0] if (node.params.get("batch_first") or len(shape) == 2) else shape[1]
    lead = (batch,) if batch is not None else ()
    return {"x": (*lead, features), "h": (*lead, hidden), "c": (*lead, hidden)}


def _traced_sizes(graph: Graph, parts: dict[str, str]) -> dict[str, Shape]:
    return {
        "x": _shape_of(graph, parts["x"]) if parts.get("x") else (),
        "h": _shape_of(graph, parts["h"]),
        "c": _shape_of(graph, parts["c"]),
    }


def _detail(shape) -> str:
    """A shape without its batch axis (``(1, 8, 16, 16)`` → ``"8×16×16"``)."""
    if not shape:
        return ""
    dims = shape[1:] if len(shape) > 1 else shape
    return "×".join("?" if d is None else str(d) for d in dims)


# --------------------------------------------------------------------------- LSTM


def _lstm(b: GraphBuilder, gid: str, title: str, gate_category: OpCategory, sizes) -> None:
    b.group(gid, LSTM_CELL, module="", label=title)

    def part(role: str, op: str, category: OpCategory, label: str = "", caption=None) -> str:
        params = {CELL_ROLE: role} | ({CAPTION: caption} if caption is not None else {})
        return b.node(
            op, category=category, label=label, group=gid, params=params, node_id=f"{gid}.{role}"
        )

    tensor = OpCategory.TENSOR
    c_prev = part("c_prev", "state", tensor, "c_{t-1}", _detail(sizes["c"]))
    h_prev = part("h_prev", "state", tensor, "h_{t-1}", _detail(sizes["h"]))
    x = part("x", "input", tensor, "x_{t}", _detail(sizes["x"]))
    concat = part("concat", "cat", OpCategory.MERGE)
    gate = {r: part(r, "gate", gate_category, label, caption) for r, label, caption in GATES}
    mul_f = part("mul_f", "mul", OpCategory.MERGE)
    mul_ig = part("mul_ig", "mul", OpCategory.MERGE)
    add_c = part("add_c", "add", OpCategory.MERGE)
    tanh_c = part("tanh_c", "tanh", OpCategory.ACTIVATION, "tanh")
    mul_o = part("mul_o", "mul", OpCategory.MERGE)
    c = part("c", "state", tensor, "c_{t}", _detail(sizes["c"]))
    h = part("h", "state", tensor, "h_{t}", _detail(sizes["h"]))
    state = EdgeKind.STATE
    for a, z, kind in [
        (c_prev, mul_f, state),
        (mul_f, add_c, state),
        (add_c, c, state),
        (add_c, tanh_c, EdgeKind.DATA),
        (h_prev, concat, state),
        (x, concat, EdgeKind.DATA),
        *((concat, gate[r], EdgeKind.DATA) for r in "figo"),
        (gate["f"], mul_f, EdgeKind.DATA),
        (gate["i"], mul_ig, EdgeKind.DATA),
        (gate["g"], mul_ig, EdgeKind.DATA),
        (mul_ig, add_c, EdgeKind.DATA),
        (gate["o"], mul_o, EdgeKind.DATA),
        (tanh_c, mul_o, EdgeKind.DATA),
        (mul_o, h, state),
    ]:
        b.edge(a, z, kind=kind)


def _match_lstm(graph: Graph, gid: str) -> dict[str, str] | None:
    """The LSTM's parts in one traced call (node ids), or ``None`` if it is not an LSTM."""
    inside = {n.id for n in graph.nodes_under(gid)}

    def op(n: str) -> str:
        return graph.node(n).op.rstrip("_")

    def sources(n: str) -> list[str]:
        return [e.source for e in graph.in_edges(n)]

    def consumers(n: str, name: str) -> list[str]:
        return [e.target for e in graph.out_edges(n) if e.target in inside and op(e.target) == name]

    for add in (n.id for n in graph.nodes_under(gid) if op(n.id) == "add"):
        muls = sources(add)
        if len(muls) != 2 or not all(m in inside and op(m) == "mul" for m in muls):
            continue
        for mul_f, mul_ig in (muls, muls[::-1]):
            f = [s for s in sources(mul_f) if s in inside and op(s) == "sigmoid"]
            c_prev = [s for s in sources(mul_f) if s not in inside]
            i = [s for s in sources(mul_ig) if op(s) == "sigmoid"]
            g = [s for s in sources(mul_ig) if op(s) == "tanh"]
            if not (len(f) == len(c_prev) == len(i) == len(g) == 1):
                continue
            for tanh_c in consumers(add, "tanh"):
                for mul_o in consumers(tanh_c, "mul"):
                    o = [s for s in sources(mul_o) if s != tanh_c and op(s) == "sigmoid"]
                    if o:
                        return {"c": add, "h": mul_o, **_gate_inputs(graph, f[0], inside)}
    return None


def _gate_inputs(graph: Graph, gate: str, inside: set[str]) -> dict[str, str]:
    """The layer computing the gates (through a split, if any) and the step's input x."""
    layer = graph.in_edges(gate)[0].source
    if graph.node(layer).op == "split":
        layer = graph.in_edges(layer)[0].source
    x = ""
    for e in graph.in_edges(layer):
        feed = graph.node(e.source)
        if feed.op == "cat":  # [x, h]: x is the input from outside the cell that is not h
            outside = [s.source for s in graph.in_edges(feed.id) if s.source not in inside]
            x = outside[0] if outside else ""
    return {"layer": layer, "x": x}


# --------------------------------------------------------------------------- GRU, RNN


def _gru(b: GraphBuilder, gid: str, title: str, sizes) -> None:
    b.group(gid, "cell", module="", label=title)

    def part(op, category, label="", caption=None):
        params = {CAPTION: caption} if caption is not None else {}
        return b.node(op, category=category, label=label, group=gid, params=params)

    tensor, merge, linear = OpCategory.TENSOR, OpCategory.MERGE, OpCategory.LINEAR
    h_prev = part("state", tensor, "h_{t-1}", _detail(sizes["h"]))
    x = part("input", tensor, "x_{t}", _detail(sizes["x"]))
    concat = part("cat", merge)
    r = part("gate", linear, "σ", "reset")
    z = part("gate", linear, "σ", "update")
    mul_r = part("mul", merge)
    n = part("gate", linear, "tanh", "candidate")
    one_minus = part("rsub", merge, "1−")
    mul_n = part("mul", merge)
    mul_z = part("mul", merge)
    add_h = part("add", merge)
    h = part("state", tensor, "h_{t}", _detail(sizes["h"]))
    for a, c in [
        (h_prev, concat),
        (x, concat),
        (concat, r),
        (concat, z),
        (r, mul_r),
        (h_prev, mul_r),
        (mul_r, n),
        (x, n),
        (z, one_minus),
        (one_minus, mul_n),
        (n, mul_n),
        (z, mul_z),
        (h_prev, mul_z),
        (mul_n, add_h),
        (mul_z, add_h),
        (add_h, h),
    ]:
        b.edge(a, c)


def _rnn(b: GraphBuilder, gid: str, title: str, sizes) -> None:
    b.group(gid, "cell", module="", label=title)
    h_prev = b.node(
        "state",
        category=OpCategory.TENSOR,
        label="h_{t-1}",
        group=gid,
        params={CAPTION: _detail(sizes["h"])},
    )
    x = b.node(
        "input",
        category=OpCategory.TENSOR,
        label="x_{t}",
        group=gid,
        params={CAPTION: _detail(sizes["x"])},
    )
    concat = b.node("cat", category=OpCategory.MERGE, group=gid)
    tanh = b.node(
        "gate",
        category=OpCategory.LINEAR,
        label="tanh",
        group=gid,
        params={CAPTION: "W [h, x] + b"},
    )
    h = b.node(
        "state",
        category=OpCategory.TENSOR,
        label="h_{t}",
        group=gid,
        params={CAPTION: _detail(sizes["h"])},
    )
    for a, c in [(h_prev, concat), (x, concat), (concat, tanh), (tanh, h)]:
        b.edge(a, c)


_CANONICAL = {"gru": _gru, "rnn": _rnn}


# --------------------------------------------------------------------------- any other cell


def _copy(b: GraphBuilder, gid: str, graph: Graph, cell: str) -> None:
    """A cell that is not an LSTM: its traced operations, with its inputs and outputs."""
    group = graph.group(cell)
    b.group(gid, "cell", module="", label=group.type_name)
    inside = [n for n in graph.nodes_under(cell)]
    ids = {n.id for n in inside}
    new: dict = {}
    for e in graph.edges:
        if e.target in ids and e.source not in ids and (e.source, e.source_port) not in new:
            new[e.source, e.source_port] = b.node(
                "input",
                category=OpCategory.TENSOR,
                label=f"in {sum(isinstance(k, tuple) for k in new) + 1}",
                group=gid,
                params={CAPTION: _detail(_shape_of(graph, e.source, e.source_port))},
            )
    for n in inside:
        new[n.id] = b.node(
            n.op,
            category=n.category,
            label=n.label,
            group=gid,
            params=dict(n.params),
            outputs=n.output_shapes,
            module=n.module,
        )
    for e in graph.edges:
        if e.target in ids:
            source = new[e.source] if e.source in ids else new[e.source, e.source_port]
            b.edge(source, new[e.target], source_port=e.source_port if e.source in ids else 0)
    outs = 0
    for e in graph.edges:
        if e.source in ids and e.target not in ids and ("out", e.source, e.source_port) not in new:
            outs += 1
            pill = b.node(
                "output_state", category=OpCategory.TENSOR, label=f"out {outs}", group=gid
            )
            new["out", e.source, e.source_port] = pill
            b.edge(new[e.source], pill, source_port=e.source_port)
