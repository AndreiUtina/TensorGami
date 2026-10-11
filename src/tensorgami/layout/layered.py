"""Recursive layered (Sugiyama-style) layout: IR graph → positioned :class:`Scene`.

Each module group is laid out on its own and becomes one rectangular block in its parent, so
frames never overlap other boxes. Inside a block:

1. **ranks** along the flow (longest path over forward edges; pure sources are pulled next
   to their first consumer);
2. **ports**: an edge entering the block at a node that is not in its first rank (e.g. a
   residual skip into the block's ``+``) enters through a port on the border; likewise for an
   edge leaving from inside (a U-Net skip). Ports get their own lanes;
3. **lanes**: edges spanning several ranks get dummy slots, so they route *around* boxes;
4. **order** within each rank by barycentre sweeps (fewer crossings), ties by execution order;
5. **cross coordinates** aligning each box with the boxes that feed it (straight chains);
6. **flow coordinates**, with gaps wide enough for the tensor-shape labels.

Everything is computed in (flow, cross) coordinates; ``direction`` maps them to (x, y).
Deterministic: ties are always broken by execution order.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from tensorgami.ir import ROOT, Graph, Group
from tensorgami.layout.base import Direction, LayoutOptions
from tensorgami.layout.labels import GENERIC_CONTAINERS, Labeler, NodeText
from tensorgami.render.legend import add_legend, build_legend
from tensorgami.render.text import HeuristicTextMeasurer, TextMeasurer
from tensorgami.scene import Box, Edge, Node, Point, Scene, Side
from tensorgami.scene import Group as SceneGroup
from tensorgami.style import Shape, TextRole, Theme
from tensorgami.vocabulary import EdgeKind

GRID = 8.0
DUMMY_CROSS = 6.0
PORT_GAP = 28.0

FC = tuple[float, float]  # (flow, cross)


def _ceil(value: float, step: float = GRID) -> float:
    return math.ceil(value / step) * step


def _shift(points: list[FC], df: float, dc: float) -> list[FC]:
    return [(f + df, c + dc) for f, c in points]


@dataclass
class _Route:
    edge: int  # index into graph.edges
    points: list[FC]
    exit: str  # "flow_end" | "flow_start" | "cross_after" | "cross_before"
    entry: str


@dataclass
class _Block:
    flow: float
    cross: float
    nodes: dict[str, FC] = field(default_factory=dict)  # node → box top-left
    frames: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    routes: list[_Route] = field(default_factory=list)  # edges complete inside the block
    in_paths: dict[int, list[FC]] = field(default_factory=dict)  # border → target
    out_paths: dict[int, list[FC]] = field(default_factory=dict)  # source → border

    def shifted(self, df: float, dc: float) -> _Block:
        return _Block(
            self.flow,
            self.cross,
            {k: (f + df, c + dc) for k, (f, c) in self.nodes.items()},
            {k: (f + df, c + dc, fs, cs) for k, (f, c, fs, cs) in self.frames.items()},
            [_Route(r.edge, _shift(r.points, df, dc), r.exit, r.entry) for r in self.routes],
            {e: _shift(p, df, dc) for e, p in self.in_paths.items()},
            {e: _shift(p, df, dc) for e, p in self.out_paths.items()},
        )


@dataclass
class _Item:
    key: str
    order: float
    flow_size: float
    cross_size: float
    node: str | None = None
    block: _Block | None = None
    inner: FC = (0.0, 0.0)  # offset of the content (node box or block) inside the item
    dummy: bool = False  # lane slot or port: no visible content
    port: bool = False
    rank: int = 0
    flow: float = 0.0
    cross: float = 0.0


class LayeredLayout:
    """The general layout engine (ADR-0004)."""

    def __init__(
        self,
        options: LayoutOptions | None = None,
        labeler: Labeler | None = None,
        measurer: TextMeasurer | None = None,
    ) -> None:
        self.options = options  # None: the style's [layout] preferences over the defaults
        self.labeler = labeler or Labeler()
        self.measurer = measurer or HeuristicTextMeasurer()

    def layout(self, graph: Graph, theme: Theme) -> Scene:
        options = self.options or LayoutOptions.for_style(theme)
        return _Run(graph, theme, options, self.labeler, self.measurer).scene()


class _Run:
    """State of one layout computation."""

    def __init__(
        self,
        graph: Graph,
        theme: Theme,
        opts: LayoutOptions,
        labeler: Labeler,
        measurer: TextMeasurer,
    ) -> None:
        self.g, self.theme, self.opts = graph, theme, opts
        self.labeler, self.measurer = labeler, measurer
        self.lr = opts.direction is Direction.LEFT_RIGHT
        self.exec = {n.id: i for i, n in enumerate(graph.nodes)}
        self.texts: dict[str, NodeText] = {n.id: labeler.describe(n) for n in graph.nodes}
        self.sizes = {n.id: self._node_size(n.id) for n in graph.nodes}
        self.loops: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(graph.edges):
            if e.source == e.target:
                self.loops[e.source].append(i)
        self.labels = self._edge_labels() if opts.shapes_on_edges else {}
        self.chain = {
            n.id: [n.group, *(a.id for a in graph.ancestors(n.group))] for n in graph.nodes
        }

    # ------------------------------------------------------------------ geometry helpers

    def fc(self, w: float, h: float) -> FC:
        return (w, h) if self.lr else (h, w)

    def xy(self, f: float, c: float) -> tuple[float, float]:
        return (f, c) if self.lr else (c, f)

    def _node_size(self, node_id: str) -> tuple[float, float]:
        node = self.g.node(node_id)
        text = self.texts[node_id]
        shape = self.theme.node_style(node.category).shape
        if shape is Shape.CIRCLE:  # a note (e.g. folded ReLU) sits centred above the circle
            return 24.0, (48.0 if text.detail else 24.0)
        label_w = self.measurer.width(text.label, self.theme.node_font(node.category))
        detail_w = (
            self.measurer.width(text.detail, self.theme.font(TextRole.DETAIL))
            if text.detail
            else 0.0
        )
        width = _ceil(max(label_w, detail_w) + 32)
        if shape is Shape.FOLDED:  # the folded strip is extra; text stays on the flat face
            width = _ceil(width + self.theme.metrics.fold_width)
        if shape is Shape.PILL:
            return max(width, 64.0), (42.0 if text.detail else 32.0)
        return max(width, 96.0), (46.0 if text.detail else 36.0)

    def _loop_pads(self, node_id: str) -> FC:
        """Space for state self-loops: above the node (LR) or to its right (TB)."""
        n = len(self.loops.get(node_id, ()))
        pad = self.opts.loop_clearance + 10 * (n - 1) + 6 if n else 0.0
        return (pad, 0.0) if self.lr else (0.0, pad)  # (cross before, cross after)

    def _edge_labels(self) -> dict[int, str]:
        labels, seen = {}, set()
        for i, e in enumerate(self.g.edges):
            key = (e.source, e.source_port)
            if key in seen or e.source == e.target:
                continue
            text = self.labeler.edge_label(self.g, e)
            if text:
                labels[i] = text
                seen.add(key)
        return labels

    def label_width(self, text: str) -> float:
        return self.measurer.width(text, self.theme.font(TextRole.ANNOTATION))

    def node_fc(self, node_id: str) -> FC:
        return self.fc(*self.sizes[node_id])

    # ------------------------------------------------------------------ frames

    def framed(self, group: Group) -> bool:
        """Frame a group unless it is a lone collapsed node (count 1) or a pass-through
        container holding exactly one sub-group."""
        nodes = self.g.members(group.id)
        children = self.g.children(group.id)
        if group.count > 1:
            return True
        if not children and len(nodes) <= 1:
            return False
        return not (not nodes and len(children) == 1)

    def title(self, group: Group) -> str:
        if len(self.g.members(group.id)) <= 1 and not self.g.children(group.id):
            return ""  # a single collapsed node already carries the name
        if group.label:
            return group.label
        if group.type_name in GENERIC_CONTAINERS and group.module:
            return group.module.rsplit(".", 1)[-1]
        return group.type_name

    @staticmethod
    def badge(group: Group) -> str:
        return f"×{group.count}" if group.count > 1 else ""

    def _frame_pads(self, group: Group) -> tuple[float, float, float, float, float]:
        """(flow before, flow after, cross before, cross after, minimum width) of a frame."""
        pad = self.opts.group_padding
        title, badge = self.title(group), self.badge(group)
        top = pad + (self.opts.title_height if title else 0) + (6 if badge and not title else 0)
        title_w = self.measurer.width(title, self.theme.font(TextRole.GROUP_TITLE))
        badge_w = self.measurer.width(badge, self.theme.font(TextRole.BADGE)) + 14 if badge else 0
        min_w = title_w + badge_w + 2 * pad + 24
        if self.lr:
            return pad, pad, top, pad, min_w
        return top, pad, pad, pad, min_w

    # ------------------------------------------------------------------ recursion

    def lift(self, node_id: str, group_id: str) -> str | None:
        """The item of ``group_id``'s level containing ``node_id`` (None if outside)."""
        if self.g.node(node_id).group == group_id:
            return node_id
        for gid in self.chain[node_id]:
            if gid != ROOT and self.g.group(gid).parent == group_id:
                return f"g:{gid}"
        return None

    def layout_group(self, group_id: str) -> _Block:
        items: dict[str, _Item] = {}
        for child in self.g.children(group_id):
            block = self.layout_group(child.id)
            if self.framed(child):
                fb, fa, cb, ca, min_w = self._frame_pads(child)
            else:
                fb = fa = cb = ca = min_w = 0.0
            flow_size, cross_size = fb + block.flow + fa, cb + block.cross + ca
            if self.lr:
                flow_size = max(flow_size, min_w)
            else:
                cross_size = max(cross_size, min_w)
            order = min((self.exec[n.id] for n in self.g.nodes_under(child.id)), default=1e9)
            items[f"g:{child.id}"] = _Item(
                f"g:{child.id}", order, flow_size, cross_size, block=block, inner=(fb, cb)
            )
        for node in self.g.members(group_id):
            fs, cs = self.node_fc(node.id)
            before, after = self._loop_pads(node.id)
            items[node.id] = _Item(
                node.id,
                self.exec[node.id],
                fs,
                before + cs + after,
                node=node.id,
                inner=(0.0, before),
            )
        return _Level(self, group_id, items).solve()

    # ------------------------------------------------------------------ scene

    def scene(self) -> Scene:
        block = self.layout_group(ROOT)
        m = self.opts.margin
        width, height = self.xy(block.flow, block.cross)
        scene_nodes: list[Node] = []
        boxes: dict[str, Box] = {}
        for node in self.g.nodes:
            x, y = self.xy(*block.nodes[node.id])
            w, h = self.sizes[node.id]
            box = Box(m + x, m + y, w, h)
            boxes[node.id] = box
            text = self.texts[node.id]
            scene_nodes.append(Node(node.id, box, node.category, text.label, text.detail))

        frames = []
        for gid, (f, c, fs, cs) in block.frames.items():
            x, y = self.xy(f, c)
            w, h = self.xy(fs, cs)
            group = self.g.group(gid)
            frames.append(
                SceneGroup(gid, Box(m + x, m + y, w, h), self.title(group), self.badge(group))
            )

        edges = [self._route_edge(r, boxes, m) for r in block.routes]
        for node_id, indices in self.loops.items():
            edges += self._loop_edges(indices, boxes[node_id])

        scene = Scene(width + 2 * m, height + 2 * m, title=self.g.name)
        scene.add(*frames, *scene_nodes, *edges)
        if self.opts.legend:
            self._add_legend(scene, m + height + 28)
        return scene

    def _sides(self, name: str) -> Side:
        if self.lr:
            table = {
                "flow_end": Side.RIGHT,
                "flow_start": Side.LEFT,
                "cross_after": Side.BOTTOM,
                "cross_before": Side.TOP,
            }
        else:
            table = {
                "flow_end": Side.BOTTOM,
                "flow_start": Side.TOP,
                "cross_after": Side.RIGHT,
                "cross_before": Side.LEFT,
            }
        return table[name]

    def _route_edge(self, route: _Route, boxes: dict[str, Box], m: float) -> Edge:
        e = self.g.edges[route.edge]
        via = _simplify([Point(m + x, m + y) for x, y in (self.xy(f, c) for f, c in route.points)])
        exit_side, entry_side = self._sides(route.exit), self._sides(route.entry)
        label = self.labels.get(route.edge, "")
        label_at = None
        if label:
            start = boxes[e.source].anchor(exit_side)
            end = boxes[e.target].anchor(entry_side)
            label_at = self._label_position([start, *via, end], label)
        return Edge(
            e.source, e.target, EdgeKind(e.kind), exit_side, entry_side, tuple(via), label, label_at
        )

    def _label_position(self, path: list[Point], label: str) -> Point:
        font = self.theme.font(TextRole.ANNOTATION)
        width = self.label_width(label)
        segments = list(zip(path, path[1:], strict=False))
        if self.lr:
            flat = [(a, b) for a, b in segments if abs(a.y - b.y) < 0.5] or segments
            fitting = [s for s in flat if abs(s[1].x - s[0].x) >= width + 8]
            a, b = (fitting or sorted(flat, key=lambda s: -abs(s[1].x - s[0].x)))[0]
            return Point((a.x + b.x) / 2, a.y - font.size / 2 - 4)
        upright = [(a, b) for a, b in segments if abs(a.x - b.x) < 0.5] or segments
        fitting = [s for s in upright if abs(s[1].y - s[0].y) >= font.size + 8]
        a, b = (fitting or sorted(upright, key=lambda s: -abs(s[1].y - s[0].y)))[0]
        return Point(a.x + 6 + width / 2, (a.y + b.y) / 2)

    def _loop_edges(self, indices: list[int], box: Box) -> list[Edge]:
        edges = []
        for k, index in enumerate(indices):
            lift = self.opts.loop_clearance + 10 * k
            span = 10 + 7 * k
            if self.lr:
                start, end = Point(box.cx + span, box.y), Point(box.cx - span, box.y)
                via = (Point(start.x, box.y - lift), Point(end.x, box.y - lift))
            else:
                start, end = Point(box.right, box.cy - span), Point(box.right, box.cy + span)
                via = (Point(box.right + lift, start.y), Point(box.right + lift, end.y))
            kind = EdgeKind(self.g.edges[index].kind)
            edges.append(Edge(start, end, kind, via=via, delay=kind is EdgeKind.STATE))
        return edges

    def _add_legend(self, scene: Scene, top: float) -> None:
        m = self.opts.margin
        categories, kinds = scene.categories(), scene.edge_kinds()
        for columns in (4, 3, 2, 1):
            legend = build_legend(
                self.theme, categories, kinds, Point(m, top), columns, self.measurer
            )
            if legend.box.w <= max(scene.width - 2 * m, 560) or columns == 1:
                break
        scene.width = max(scene.width, legend.box.right + m)
        scene.height = legend.box.bottom + m
        add_legend(scene, self.theme, Point(m, top), columns)


def _simplify(points: list[Point]) -> list[Point]:
    """Drop repeated and collinear middle points."""
    out: list[Point] = []
    for p in points:
        if out and abs(out[-1].x - p.x) < 0.01 and abs(out[-1].y - p.y) < 0.01:
            continue
        if len(out) >= 2:
            a, b = out[-2], out[-1]
            if (abs(a.x - b.x) < 0.01 and abs(b.x - p.x) < 0.01) or (
                abs(a.y - b.y) < 0.01 and abs(b.y - p.y) < 0.01
            ):
                out[-1] = p
                continue
        out.append(p)
    return out


def _best_cuts(
    levels: list[int], cuts: list[int], sizes: dict, gaps: dict, width: float
) -> list[int]:
    """Row breaks (after the given ranks) for wrapping: the fewest rows whose widest row is
    no wider than ``width`` (or than the widest unbreakable block + 15 %, when that block alone
    is wider), balanced so that the widest row is as narrow as possible.

    The legal cuts split the ranks into unbreakable units; greedy packing finds the fewest
    rows, then a small dynamic programme balances them (O(units² × rows))."""
    start, end, flow = {}, {}, 0.0
    for i, r in enumerate(levels):
        start[r] = flow
        flow += sizes[r]
        end[r] = flow
        flow += gaps[r] if i < len(levels) - 1 else 0
    units: list[tuple[int, int]] = []  # (first rank, last rank)
    first = levels[0]
    for i, r in enumerate(levels):
        if r in cuts or i == len(levels) - 1:
            units.append((first, r))
            if i < len(levels) - 1:
                first = levels[i + 1]

    def span(i: int, j: int) -> float:  # width of a row holding units i..j
        return end[units[j][1]] - start[units[i][0]]

    widest_unit = max(span(i, i) for i in range(len(units)))
    # a block wider than ``width`` sets the pace; 15 % slack beats a row holding a lone pill
    limit = width if widest_unit <= width else widest_unit * 1.15
    rows, i = 0, 0
    while i < len(units):  # greedy: as many units per row as fit → fewest rows
        j = i
        while j + 1 < len(units) and span(i, j + 1) <= limit + 0.5:
            j += 1
        rows, i = rows + 1, j + 1

    n = len(units)
    inf = float("inf")
    best = [[inf] * (rows + 1) for _ in range(n + 1)]  # best[j][k]: units[:j] in k rows
    back = [[0] * (rows + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for k in range(1, rows + 1):
        for j in range(1, n + 1):
            for i in range(k - 1, j):
                value = max(best[i][k - 1], span(i, j - 1))
                if value < best[j][k] - 1e-9:
                    best[j][k], back[j][k] = value, i
    breaks, j = [], n
    for k in range(rows, 0, -1):
        i = back[j][k]
        if i > 0:
            breaks.append(units[i - 1][1])
        j = i
    return sorted(breaks)


class _Level:
    """Layout of the items directly inside one group."""

    def __init__(self, run: _Run, group_id: str, items: dict[str, _Item]) -> None:
        self.run, self.gid, self.items = run, group_id, items
        self.forward: list[tuple[int, str, str]] = []  # (edge, source item, target item)
        self.back: list[tuple[int, str, str]] = []
        self.entering: list[tuple[int, str]] = []  # (edge, target item): source is outside
        self.leaving: list[tuple[int, str]] = []  # (edge, source item): target is outside
        for i, e in enumerate(run.g.edges):
            if e.source == e.target:
                continue
            a, b = run.lift(e.source, group_id), run.lift(e.target, group_id)
            if a is None and b is not None:
                self.entering.append((i, b))
            elif b is None and a is not None:
                self.leaving.append((i, a))
            elif a is not None and a != b:
                (self.forward if items[a].order < items[b].order else self.back).append((i, a, b))

    # ------------------------------------------------------------------ pipeline

    def solve(self) -> _Block:
        if not self.items:
            return _Block(0.0, 0.0)
        self._rank()
        self._add_ports()
        self._add_lanes()
        self._order()
        self._place_cross()
        rows = self._rows()
        self._place_flow(rows)
        return self._compose(rows)

    def _rank(self) -> None:
        preds, succs = defaultdict(set), defaultdict(set)
        for _, a, b in self.forward:
            preds[b].add(a)
            succs[a].add(b)
        ordered = sorted(self.items.values(), key=lambda it: it.order)
        for it in ordered:
            it.rank = max((self.items[p].rank + 1 for p in preds[it.key]), default=0)
        for it in ordered:  # pull pure sources next to their first consumer
            if not preds[it.key] and succs[it.key]:
                it.rank = max(0, min(self.items[s].rank for s in succs[it.key]) - 1)

    def _add_ports(self) -> None:
        """A border port for each outside tensor that reaches *past* the first rank (or for
        each inside tensor leaving from *before* the last rank)."""
        self.port_of: dict[int, str] = {}  # edge → port item key
        last = max(it.rank for it in self.items.values())
        entering: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
        for index, b in self.entering:
            e = self.run.g.edges[index]
            entering[(e.source, e.source_port)].append((index, b))
        needs_in = {k: v for k, v in entering.items() if any(self.items[b].rank > 0 for _, b in v)}
        leaving: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
        for index, a in self.leaving:
            e = self.run.g.edges[index]
            leaving[(e.source, e.source_port)].append((index, a))
        needs_out = {
            k: v for k, v in leaving.items() if any(self.items[a].rank < last for _, a in v)
        }
        if needs_in:
            for it in self.items.values():
                it.rank += 1
            last += 1
        for n, edges in enumerate(needs_in.values()):
            key = f"pi:{n}"
            self.items[key] = _Item(key, -1 + n * 1e-3, 0.0, DUMMY_CROSS, dummy=True, port=True)
            for index, _ in edges:
                self.port_of[index] = key
        for n, edges in enumerate(needs_out.values()):
            key = f"po:{n}"
            self.items[key] = _Item(
                key, 1e9 + n, 0.0, DUMMY_CROSS, dummy=True, port=True, rank=last + 1
            )
            for index, _ in edges:
                self.port_of[index] = key

    def _pairs(self) -> list[tuple[int, str, str]]:
        """Every link routed at this level: internal edges and port connections."""
        pairs = list(self.forward)
        pairs += [(i, self.port_of[i], b) for i, b in self.entering if i in self.port_of]
        pairs += [(i, a, self.port_of[i]) for i, a in self.leaving if i in self.port_of]
        return pairs

    def _add_lanes(self) -> None:
        self.lanes: dict[tuple[str, str], list[str]] = {}
        for _, a, b in self._pairs():
            if (a, b) in self.lanes:
                continue
            ra, rb = self.items[a].rank, self.items[b].rank
            keys = []
            for r in range(ra + 1, rb):
                key = f"d:{a}>{b}:{r}"
                self.items[key] = _Item(
                    key, self.items[a].order + 0.5, 0.0, DUMMY_CROSS, dummy=True, rank=r
                )
                keys.append(key)
            self.lanes[(a, b)] = keys

    def _order(self) -> None:
        ranks: dict[int, list[str]] = defaultdict(list)
        for it in sorted(self.items.values(), key=lambda it: (it.order, it.key)):
            ranks[it.rank].append(it.key)
        up, down = defaultdict(list), defaultdict(list)
        for (a, b), lanes in self.lanes.items():
            path = [a, *lanes, b]
            for x, y in zip(path, path[1:], strict=False):
                up[y].append(x)
                down[x].append(y)
        pos = {k: i for keys in ranks.values() for i, k in enumerate(keys)}
        levels = sorted(ranks)
        for sweep in range(4):
            sequence = levels[1:] if sweep % 2 == 0 else list(reversed(levels[:-1]))
            neighbours = up if sweep % 2 == 0 else down
            for r in sequence:
                keys = ranks[r]

                def centre(k: str, nb=neighbours) -> float:
                    linked = nb[k]
                    return sum(pos[x] for x in linked) / len(linked) if linked else pos[k]

                keys.sort(key=lambda k: (centre(k), pos[k]))
                for i, k in enumerate(keys):
                    pos[k] = i
        self.ranks = ranks

    # ------------------------------------------------------------------ cross axis

    def _offset(self, item_key: str, edge: int, side: str) -> float:
        """Cross-axis position, from the top of the item, where ``edge`` leaves (``out``) or
        enters (``in``) it."""
        it = self.items[item_key]
        if it.dummy:
            return it.cross_size / 2
        e = self.run.g.edges[edge]
        if it.node is not None:
            node_cross = self.run.node_fc(it.node)[1]
            return it.inner[1] + node_cross / 2
        paths = it.block.out_paths if side == "out" else it.block.in_paths
        point = paths[edge][-1] if side == "out" else paths[edge][0]
        return it.inner[1] + point[1] if edge in paths else it.cross_size / 2 + 0 * e.source_port

    def _constraints(self) -> tuple[dict, dict]:
        """For each item: (neighbour, own offset, neighbour offset) along every link."""
        ups, downs = defaultdict(list), defaultdict(list)
        for index, a, b in self._pairs():
            path = [a, *self.lanes[(a, b)], b]
            for x, y in zip(path, path[1:], strict=False):
                off_x = self._offset(x, index, "out")
                off_y = self._offset(y, index, "in")
                ups[y].append((x, off_y, off_x))
                downs[x].append((y, off_x, off_y))
        return ups, downs

    def _place_cross(self) -> None:
        ups, downs = self._constraints()
        gap = self.run.opts.node_gap

        def place(keys: list[str], use_successors: bool) -> None:
            prev_end = None
            for k in keys:
                it = self.items[k]
                links = ups[k] or (downs[k] if use_successors else [])
                wanted = None
                if links:
                    wanted = sum(self.items[n].cross + n_off - own for n, own, n_off in links)
                    wanted /= len(links)
                spacing = gap / 2 if it.dummy else gap
                if prev_end is None:
                    top = wanted if wanted is not None else 0.0
                else:
                    floor = prev_end + spacing
                    top = floor if wanted is None else max(wanted, floor)
                it.cross = top
                prev_end = top + it.cross_size

        for second in (False, True):
            for r in sorted(self.ranks):
                place(self.ranks[r], use_successors=second)
        low = min(it.cross for it in self.items.values())
        for it in self.items.values():
            it.cross -= low

    # ------------------------------------------------------------------ flow axis

    def _gap(self, r: int) -> float:
        run = self.run
        keys = self.ranks.get(r, [])
        nxt = self.ranks.get(r + 1, [])
        if (
            keys
            and all(self.items[k].port for k in keys)
            or (nxt and all(self.items[k].port for k in nxt))
        ):
            return PORT_GAP
        gap = run.opts.rank_gap
        for index, a, b in self._pairs():
            label = run.labels.get(index)
            if not label or self.items[a].rank != r:
                continue
            straight = (
                self.items[b].rank == r + 1
                and abs(
                    (self.items[a].cross + self._offset(a, index, "out"))
                    - (self.items[b].cross + self._offset(b, index, "in"))
                )
                < 0.5
            )
            if run.lr:
                need = run.label_width(label) + 24
                gap = max(gap, need if straight else 2 * need - 8)
            else:
                gap = max(gap, run.theme.font(TextRole.ANNOTATION).size + 28)
        return gap

    def _rows(self) -> list[list[int]]:
        """Split the top level into balanced rows when it is wider than ``wrap_width``.

        A row may only end where exactly one connection continues into the next row, and no
        row may be left nearly empty (e.g. a lone output)."""
        levels = sorted(self.ranks)
        run = self.run
        width = run.opts.wrap_width
        if self.gid != ROOT or not run.lr or width is None or len(levels) < 2:
            return [levels]
        sizes = {r: max(self.items[k].flow_size for k in self.ranks[r]) for r in levels}
        ends, flow = {}, 0.0
        for i, r in enumerate(levels):
            flow += sizes[r]
            ends[r] = flow
            flow += self._gap(r) if i < len(levels) - 1 else 0
        total = flow
        if total <= width:
            return [levels]
        crossing = dict.fromkeys(levels, 0)
        for a, b in self.lanes:
            for r in range(self.items[a].rank, self.items[b].rank):
                crossing[r] += 1
        cuts = [r for r in levels[:-1] if crossing[r] == 1]
        chosen = _best_cuts(levels, cuts, sizes, {r: self._gap(r) for r in levels}, width)
        rows, current = [], []
        for r in levels:
            current.append(r)
            if r in chosen:
                rows.append(current)
                current = []
        rows.append(current)
        return [row for row in rows if row]

    def _place_flow(self, rows: list[list[int]]) -> None:
        self.rank_span: dict[int, tuple[float, float]] = {}
        self.gaps: dict[int, float] = {}
        self.row_of: dict[int, int] = {}
        self.row_bounds: list[tuple[float, float]] = []  # cross extent of each row
        cross_offset = 0.0
        for row_index, row in enumerate(rows):
            flow = 0.0
            for i, r in enumerate(row):
                size = max(self.items[k].flow_size for k in self.ranks[r])
                self.rank_span[r] = (flow, flow + size)
                self.row_of[r] = row_index
                for k in self.ranks[r]:
                    it = self.items[k]
                    it.flow = flow + (size - it.flow_size) / 2
                gap = self._gap(r)
                self.gaps[r] = gap
                flow += size + (gap if i < len(row) - 1 else 0)
            keys = [k for r in row for k in self.ranks[r]]
            low = min(self.items[k].cross for k in keys)
            for k in keys:
                self.items[k].cross += cross_offset - low
            high = max(self.items[k].cross + self.items[k].cross_size for k in keys)
            self.row_bounds.append((cross_offset, high))
            cross_offset = high + self.run.opts.row_gap

    # ------------------------------------------------------------------ result

    def _compose(self, rows: list[list[int]]) -> _Block:
        run = self.run
        block = _Block(0.0, 0.0)
        child_in: dict[int, list[FC]] = {}
        child_out: dict[int, list[FC]] = {}
        for it in self.items.values():
            if it.dummy:
                continue
            if it.node is not None:
                block.nodes[it.node] = (it.flow + it.inner[0], it.cross + it.inner[1])
                continue
            inner = it.block.shifted(it.flow + it.inner[0], it.cross + it.inner[1])
            block.nodes.update(inner.nodes)
            block.frames.update(inner.frames)
            block.routes += inner.routes
            child_in.update(inner.in_paths)
            child_out.update(inner.out_paths)
            group_id = it.key[2:]
            if run.framed(run.g.group(group_id)):
                block.frames[group_id] = (it.flow, it.cross, it.flow_size, it.cross_size)
        block.flow = max(it.flow + it.flow_size for it in self.items.values())
        block.cross = max(it.cross + it.cross_size for it in self.items.values())

        for index, a, b in self.forward:
            start = self._exit_cross(a, index)
            end = self._entry_cross(b, index)
            if self.row_of.get(self.items[a].rank) != self.row_of.get(self.items[b].rank):
                middle = self._wrap(a, b, index, start, end, block, child_out, child_in)
            else:
                middle = self._lanes(a, b, start, end)
            points = child_out.get(index, []) + middle + child_in.get(index, [])
            block.routes.append(_Route(index, points, "flow_end", "flow_start"))

        for index, b in self.entering:
            if index in self.port_of:
                port = self.items[self.port_of[index]]
                cross = port.cross + port.cross_size / 2
                head = [(port.flow, cross)] + self._lanes(
                    port.key, b, cross, self._entry_cross(b, index)
                )
            else:
                head = [(0.0, self._entry_cross(b, index))]
            block.in_paths[index] = head + child_in.get(index, [])
        for index, a in self.leaving:
            if index in self.port_of:
                port = self.items[self.port_of[index]]
                cross = port.cross + port.cross_size / 2
                tail = self._lanes(a, port.key, self._exit_cross(a, index), cross)
                tail.append((port.flow, cross))
            else:
                tail = [(block.flow, self._exit_cross(a, index))]
            block.out_paths[index] = child_out.get(index, []) + tail

        lane = block.cross + 12
        for k, (index, _, _) in enumerate(self.back):
            e = run.g.edges[index]
            src, tgt = block.nodes[e.source], block.nodes[e.target]
            sf = src[0] + run.node_fc(e.source)[0] / 2
            tf = tgt[0] + run.node_fc(e.target)[0] / 2
            y = lane + 10 * k
            block.routes.append(_Route(index, [(sf, y), (tf, y)], "cross_after", "cross_after"))
        if self.back:
            block.cross = lane + 10 * len(self.back)
        return block

    def _exit_cross(self, item_key: str, edge: int) -> float:
        return self.items[item_key].cross + self._offset(item_key, edge, "out")

    def _entry_cross(self, item_key: str, edge: int) -> float:
        return self.items[item_key].cross + self._offset(item_key, edge, "in")

    def _lanes(self, a: str, b: str, start: float, end: float) -> list[FC]:
        """Orthogonal points from item ``a`` to item ``b`` through the lane slots between
        them; bends sit in the middle of the gaps between ranks."""
        points: list[FC] = []
        current = start
        lanes = self.lanes.get((a, b), [])
        for step, r in enumerate(range(self.items[a].rank, self.items[b].rank)):
            elbow = self.rank_span[r][1] + self.gaps[r] / 2
            if step < len(lanes):
                slot = self.items[lanes[step]]
                nxt = slot.cross + slot.cross_size / 2
            else:
                nxt = end
            if abs(nxt - current) >= 0.5:
                points += [(elbow, current), (elbow, nxt)]
            current = nxt
        return points

    def _wrap(
        self,
        a: str,
        b: str,
        edge: int,
        start: float,
        end: float,
        block: _Block,
        child_out: dict,
        child_in: dict,
    ) -> list[FC]:
        """Into the next row: out to the right, down between the rows, back to the left."""
        row_a = self.row_of[self.items[a].rank]
        middle = (self.row_bounds[row_a][1] + self.row_bounds[row_a + 1][0]) / 2
        e = self.run.g.edges[edge]
        source, target = self.items[a], self.items[b]
        # turn just outside the source/target *item*: for a framed group that is outside its
        # frame, so the bend never runs through the padding where the frame title sits
        if edge in child_out:
            leave = source.flow + source.flow_size
        else:
            leave = block.nodes[e.source][0] + self.run.node_fc(e.source)[0]
        if edge in child_in:
            arrive = target.flow
        else:
            arrive = block.nodes[e.target][0]
        right, left = leave + 14, arrive - 14
        return [(right, start), (right, middle), (left, middle), (left, end)]
