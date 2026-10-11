"""Display text for IR nodes and edges, following the style baseline (§5).

``Conv2d(kernel 3, out 64, stride 2) + ReLU`` → label ``"Conv 3×3"``, detail ``"64, s2, ReLU"``.
The :class:`Labeler` is a small rule table; views can replace it (open/closed).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tensorgami.ir import Edge, Graph, Node, ParamValue, format_shape
from tensorgami.vocabulary import EdgeKind, OpCategory

TIMES = "×"
GENERIC_CONTAINERS = frozenset({"Sequential", "ModuleList", "ModuleDict"})


@dataclass(frozen=True)
class NodeText:
    label: str
    detail: str = ""


PRETTY = {
    "relu": "ReLU",
    "relu6": "ReLU6",
    "gelu": "GELU",
    "silu": "SiLU",
    "elu": "ELU",
    "selu": "SELU",
    "leaky_relu": "LeakyReLU",
    "sigmoid": "Sigmoid",
    "tanh": "tanh",
    "softmax": "Softmax",
    "log_softmax": "LogSoftmax",
    "softplus": "Softplus",
    "mish": "Mish",
    "exp": "exp",
    "log": "log",
    "dropout": "Dropout",
    "flatten": "Flatten",
    "layer_norm": "LayerNorm",
    "batch_norm": "BatchNorm",
    "group_norm": "GroupNorm",
    "interpolate": "Upsample",
    "embedding": "Embedding",
    "linear": "Linear",
    "conv2d": "Conv",
    "conv1d": "Conv",
    "matmul": "MatMul",
    "bmm": "MatMul",
    "einsum": "einsum",
    "split": "split",
    "chunk": "chunk",
    "scaled_dot_product_attention": "Attention",
    "normalize": "Normalize",
}
MERGE_SYMBOLS = {
    "add": "+",
    "sub": "−",
    "mul": TIMES,
    "div": "÷",
    "cat": "‖",
    "concat": "‖",
    "concatenate": "‖",
    "stack": "‖",
    "hstack": "‖",
    "vstack": "‖",
    "maximum": "max",
    "minimum": "min",
    "where": "?",
    "matmul": "@",
}
MODULE_NAMES = {
    "BatchNorm1d": "BatchNorm",
    "BatchNorm2d": "BatchNorm",
    "BatchNorm3d": "BatchNorm",
    "InstanceNorm1d": "InstanceNorm",
    "InstanceNorm2d": "InstanceNorm",
    "InstanceNorm3d": "InstanceNorm",
    "MultiheadAttention": "Multi-head attention",
    "AdaptiveAvgPool1d": "AdaptiveAvgPool",
    "AdaptiveAvgPool2d": "AdaptiveAvgPool",
    "AdaptiveAvgPool3d": "AdaptiveAvgPool",
    "AdaptiveMaxPool2d": "AdaptiveMaxPool",
}
NOISE = {
    "randn": "ε ~ N(0, I)",
    "randn_like": "ε ~ N(0, I)",
    "normal": "ε ~ N(μ, σ)",
    "rand": "U(0, 1)",
    "rand_like": "U(0, 1)",
    "bernoulli": "Bernoulli",
}


def _num(value: ParamValue) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _square(value: ParamValue) -> str:
    """``(3, 3)`` → ``3×3``; ``3`` → ``3``; ``(1, 3)`` → ``1×3``."""
    if isinstance(value, tuple):
        return TIMES.join(_num(v) for v in value)
    return _num(value)


def _per_axis(value: ParamValue, op: str) -> ParamValue:
    """``MaxPool2d(kernel_size=2)`` → ``(2, 2)``: spell out one value per spatial axis."""
    dims = op[-2:]
    if isinstance(value, int) and not isinstance(value, bool) and dims in ("1d", "2d", "3d"):
        return (value,) * int(dims[0])
    return value


def _uniform(value: ParamValue) -> ParamValue:
    """``(2, 2)`` → ``2``: a tuple of equal values collapses to one."""
    if isinstance(value, tuple) and value and all(v == value[0] for v in value):
        return value[0]
    return value


def pretty(op: str) -> str:
    return PRETTY.get(op.rstrip("_"), op)


class Labeler:
    """Turns IR nodes into (label, detail) and decides which edges show their shape."""

    def __init__(self) -> None:
        self._rules: list[tuple[Callable[[Node], bool], Callable[[Node], NodeText]]] = [
            (lambda n: n.category is OpCategory.IO, self._io),
            (lambda n: "collapsed" in n.params, self._collapsed),
            (lambda n: n.category is OpCategory.MERGE, self._merge),
            (lambda n: n.op.startswith("ConvTranspose") or n.op.startswith("Conv"), self._conv),
            (lambda n: n.op in ("Linear", "LazyLinear"), self._linear),
            (
                lambda n: n.op in ("LSTM", "GRU", "RNN", "LSTMCell", "GRUCell", "RNNCell"),
                self._recurrent,
            ),
            (lambda n: n.op == "MultiheadAttention", self._attention),
            (lambda n: "Pool" in n.op and "Adaptive" not in n.op, self._pool),
            (lambda n: n.op.startswith("Adaptive"), self._adaptive),
            (lambda n: n.op in ("Embedding", "EmbeddingBag"), self._embedding),
            (lambda n: n.op.startswith("Dropout") or n.op == "AlphaDropout", self._dropout),
            (lambda n: n.category is OpCategory.STOCHASTIC, self._noise),
            (lambda n: n.category is OpCategory.OTHER and n.params.get("args"), self._arithmetic),
        ]

    # ------------------------------------------------------------------ nodes

    def describe(self, node: Node) -> NodeText:
        for applies, describe in self._rules:
            if applies(node):
                text = describe(node)
                break
        else:
            text = NodeText(
                node.label or MODULE_NAMES.get(node.op, pretty(node.op)),
                self._join(self._folded(node)),
            )
        if node.label:  # set by the user (overrides) or a transform: it names the box
            text = NodeText(node.label, text.detail)
        if node.count > 1:
            text = NodeText(text.label, self._join([text.detail, f"{TIMES}{node.count} steps"]))
        return text

    @staticmethod
    def _join(parts: list[str]) -> str:
        return ", ".join(p for p in parts if p)

    @staticmethod
    def _folded(node: Node) -> list[str]:
        return [pretty(name) for name in node.params.get("folded", ())]

    def _io(self, n: Node) -> NodeText:
        return NodeText(n.label or n.op)

    def _collapsed(self, n: Node) -> NodeText:
        if not n.label and n.op in GENERIC_CONTAINERS and n.module:
            layers = n.params["collapsed"]
            return NodeText(n.module.rsplit(".", 1)[-1], f"{layers} layers" if layers > 1 else "")
        return NodeText(n.label or n.op, self._join(self._folded(n)))

    def _merge(self, n: Node) -> NodeText:
        return NodeText(MERGE_SYMBOLS.get(n.op.rstrip("_"), n.op), self._join(self._folded(n)))

    def _conv(self, n: Node) -> NodeText:
        p = n.params
        base = "ConvT" if n.op.startswith("ConvTranspose") else "Conv"
        label = f"{base} {_square(p['kernel_size'])}" if "kernel_size" in p else base
        parts = [_num(p["out_channels"])] if "out_channels" in p else []
        stride = _uniform(p.get("stride", 1))
        if stride not in (1, None):
            parts.append(f"s{_square(stride)}")
        if p.get("groups", 1) not in (1, None):
            parts.append(f"g{p['groups']}")
        return NodeText(label, self._join(parts + self._folded(n)))

    def _linear(self, n: Node) -> NodeText:
        p = n.params
        dims = f"{p['in_features']}→{p['out_features']}" if "out_features" in p else ""
        return NodeText("Linear", self._join([dims, *self._folded(n)]))

    def _recurrent(self, n: Node) -> NodeText:
        p = n.params
        parts = [f"{p['hidden_size']} hidden"] if "hidden_size" in p else []
        if p.get("num_layers", 1) not in (1, None):
            parts.append(f"{p['num_layers']} layers")
        if p.get("bidirectional"):
            parts.append("bidirectional")
        return NodeText(n.op, self._join(parts + self._folded(n)))

    def _attention(self, n: Node) -> NodeText:
        heads = n.params.get("num_heads")
        return NodeText(
            "Multi-head attention",
            self._join([f"{heads} heads" if heads else "", *self._folded(n)]),
        )

    def _pool(self, n: Node) -> NodeText:
        p = n.params
        kind = "MaxPool" if n.op.startswith("Max") else "AvgPool"
        kernel = _uniform(p.get("kernel_size"))
        size = _per_axis(p.get("kernel_size"), n.op)
        label = f"{kind} {_square(size)}" if kernel is not None else kind
        stride = _uniform(p.get("stride"))
        parts = [f"s{_square(stride)}"] if stride not in (None, kernel) else []
        return NodeText(label, self._join(parts + self._folded(n)))

    def _adaptive(self, n: Node) -> NodeText:
        size = _per_axis(n.params.get("output_size"), n.op)
        detail = f"→ {_square(size)}" if size is not None else ""
        return NodeText(MODULE_NAMES.get(n.op, n.op), self._join([detail, *self._folded(n)]))

    def _embedding(self, n: Node) -> NodeText:
        p = n.params
        dims = f"{p['num_embeddings']}→{p['embedding_dim']}" if "embedding_dim" in p else ""
        return NodeText("Embedding", self._join([dims, *self._folded(n)]))

    def _dropout(self, n: Node) -> NodeText:
        p = n.params.get("p")
        return NodeText("Dropout", f"p = {_num(p)}" if p is not None else "")

    def _noise(self, n: Node) -> NodeText:
        return NodeText(n.label or NOISE.get(n.op, n.op))

    def _arithmetic(self, n: Node) -> NodeText:
        symbol = MERGE_SYMBOLS.get(n.op.rstrip("_"), n.op)
        constants = ", ".join(_num(a) for a in n.params["args"])
        return NodeText(f"{symbol} {constants}", self._join(self._folded(n)))

    # ------------------------------------------------------------------ edges

    def edge_label(self, graph: Graph, edge: Edge) -> str:
        """The tensor shape, but only where it changes: on edges leaving a graph input, and
        on edges whose shape differs from what their source received."""
        if edge.kind is not EdgeKind.DATA or edge.shape is None:
            return ""
        source = graph.node(edge.source)
        if not source.is_input:
            inputs = [e.shape for e in graph.in_edges(source.id) if e.kind is EdgeKind.DATA]
            if not inputs or edge.shape in inputs:
                return ""
        return format_shape(edge.shape)
