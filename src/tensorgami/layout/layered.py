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

import dataclasses
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from tensorgami.ir import MODELS, ROOT, TRAINS, VOLUME, Graph, Group
from tensorgami.ir import Node as GraphNode
from tensorgami.layout.base import Direction, LayoutOptions
from tensorgami.layout.labels import GENERIC_CONTAINERS, Labeler, NodeText
from tensorgami.layout.volume import volume_geometry, volume_label
from tensorgami.render.legend import add_legend, build_legend
from tensorgami.render.text import HeuristicTextMeasurer, TextMeasurer
from tensorgami.scene import Box, Edge, Node, Note, Point, Scene, Side, TextAnchor
from tensorgami.scene import Group as SceneGroup
from tensorgami.style import Shape, TextLayout, TextRole, Theme
from tensorgami.vocabulary import EdgeKind, OpCategory

GRID = 8.0
DUMMY_CROSS = 6.0
PORT_GAP = 20.0
WRAP_GUTTER = 16.0  # beside rows wrapped inside a frame: where wires turn into the next row

#: group type → layout of such a group as one block (e.g. the unrolled view's time × layer
#: grid); registered by the modules that define them, framed only if the class says so
BLOCK_LAYOUTS: dict[str, type] = {}

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
    # sides an edge leaves its source / enters its target by, when not the flow sides
    # (e.g. a grid's per-step input enters its cell from below): edge → (exit, entry)
    sides: dict[int, tuple[str | None, str | None]] = field(default_factory=dict)
    notes: list[tuple[FC, str]] = field(default_factory=list)  # free text, e.g. "⋯"

    def shifted(self, df: float, dc: float) -> _Block:
        return _Block(
            self.flow,
            self.cross,
            {k: (f + df, c + dc) for k, (f, c) in self.nodes.items()},
            {k: (f + df, c + dc, fs, cs) for k, (f, c, fs, cs) in self.frames.items()},
            [_Route(r.edge, _shift(r.points, df, dc), r.exit, r.entry) for r in self.routes],
            {e: _shift(p, df, dc) for e, p in self.in_paths.items()},
            {e: _shift(p, df, dc) for e, p in self.out_paths.items()},
            dict(self.sides),
            [((f + df, c + dc), text) for (f, c), text in self.notes],
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
        self.boxes = theme.boxes
        self.text = theme.boxes.text(vertical=not self.lr)  # the arrangement in this flow
        self.exec = {n.id: i for i, n in enumerate(graph.nodes)}
        self.texts: dict[str, NodeText] = {n.id: labeler.describe(n) for n in graph.nodes}
        for n in graph.nodes:  # a tensor volume is named, and shows its shape below
            if VOLUME in n.params:
                self.texts[n.id] = NodeText(self.texts[n.id].label, volume_label(n.params[VOLUME]))
        self.sizes = self._uniform({n.id: self._node_size(n.id) for n in graph.nodes})
        self.loops: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(graph.edges):
            if e.source == e.target:
                self.loops[e.source].append(i)
        self.labels = self._edge_labels() if opts.shapes_on_edges else {}
        self.chain = {
            n.id: [n.group, *(a.id for a in graph.ancestors(n.group))] for n in graph.nodes
        }
        self.side_inputs = self._side_inputs()
        for index in self.side_inputs:  # its wire runs between two frames: no room for a label
            self.labels.pop(index, None)

    def _side_inputs(self) -> set[int]:
        """Edges into a module *partway along it* from a branch independent of the module's
        main input, e.g. the Transformer's encoder output into each decoder layer's second
        attention. Such a module may stand beside that branch (both stacks in parallel, as in
        the paper's figure) and the edge enters it from the side, level with its target.

        Not a side input: an edge into where the module begins, one from a pure source (an
        input or a buffer, placed next to its consumer anyway), one from a loose tensor rather
        than another module (a generated image the discriminator judges), and one from
        upstream of the module's main input (a residual skip leaves from the block's own
        input)."""
        g = self.g
        preds: dict[str, set[str]] = defaultdict(set)
        data = []
        for i, e in enumerate(g.edges):
            if e.kind is EdgeKind.DATA and e.source != e.target:
                preds[e.target].add(e.source)
                data.append(i)
        upstream: dict[str, set[str]] = {}

        def ancestors(node: str) -> set[str]:
            if node not in upstream:
                seen, stack = set(), [node]
                while stack:
                    for p in preds[stack.pop()]:
                        if p not in seen:
                            seen.add(p)
                            stack.append(p)
                upstream[node] = seen
            return upstream[node]

        side = set()
        for i in data:
            s, t = g.edges[i].source, g.edges[i].target
            if not preds[s]:
                continue
            outside = set(self.chain[s])
            holder = next(
                (gid for gid in reversed(self.chain[t]) if gid != ROOT and gid not in outside),
                None,
            )
            if holder is None:
                continue
            parent = g.group(holder).parent
            if not any(gid != ROOT and g.group(gid).parent == parent for gid in self.chain[s]):
                continue  # two stacks side by side: the source too comes out of a module
            path = self.chain[t][: self.chain[t].index(holder) + 1]
            if any(g.group(gid).type_name in BLOCK_LAYOUTS for gid in path):
                continue  # a block layout of its own (U, grid, cell) places its entries
            inside = {n.id for n in g.nodes_under(holder)}
            if not preds[t] & inside:
                continue  # it enters where the module begins
            mains = {  # what enters where the module begins
                p for n in inside if not preds[n] & inside for p in preds[n] if p not in inside
            }
            if mains and not any(s == m or s in ancestors(m) for m in mains):
                side.add(i)
        return side

    # ------------------------------------------------------------------ geometry helpers

    def fc(self, w: float, h: float) -> FC:
        return (w, h) if self.lr else (h, w)

    def xy(self, f: float, c: float) -> tuple[float, float]:
        return (f, c) if self.lr else (c, f)

    def _shape(self, node_id: str) -> Shape:
        return self.theme.node_style(self.g.node(node_id).category).shape

    def text_of(self, node_id: str) -> TextLayout:
        """How the node's text is arranged: the flow's arrangement, except that data pills
        never turn upright (they stay level, on one line)."""
        if self.text is TextLayout.ROTATED and self._shape(node_id) is Shape.PILL:
            return TextLayout.INLINE
        return self.text

    def _node_size(self, node_id: str) -> tuple[float, float]:
        node = self.g.node(node_id)
        text = self.texts[node_id]
        shape = self._shape(node_id)
        if VOLUME in node.params:
            return self._volume_size(node, text)
        if shape is Shape.CIRCLE:  # a note (e.g. a folded ReLU) sits above the circle, or
            if not text.detail:  # beside it in a top → bottom flow, where wires come from above
                return 24.0, 24.0
            if self.lr:
                return 24.0, 48.0
            note = self.measurer.width(text.detail, self.theme.font(TextRole.DETAIL))
            return _ceil(24 + 2 * (note + 6)), 24.0
        boxes, arrangement = self.boxes, self.text_of(node_id)
        label_w = self.measurer.width(text.label, self.theme.node_font(node.category))
        detail_w = (
            self.measurer.width(text.detail, self.theme.font(TextRole.DETAIL))
            if text.detail
            else 0.0
        )
        if arrangement.single_line:  # label, gap, detail
            text_w = label_w + (boxes.gap + detail_w if text.detail else 0.0)
            thickness = boxes.height
        else:
            text_w = max(label_w, detail_w)
            thickness = boxes.detail_height if text.detail else boxes.height
        length = _ceil(text_w + boxes.padding)
        if shape is Shape.FOLDED:  # the folded strip is extra; text stays on the flat face
            length = _ceil(length + self.theme.metrics.fold_width)
        if shape is Shape.RECT and self.theme.node_style(node.category).texture is not None:
            length = _ceil(length + self.theme.metrics.texture_strip)  # the engraved strip
        if shape is Shape.PILL:
            return max(length, 64.0), thickness - 4
        length = max(length, boxes.min_width)
        if arrangement is TextLayout.ROTATED:  # upright: thin along x, long along y
            return thickness, length
        return length, thickness

    def _volume_geometry(self, node_id: str) -> tuple[float, float, float, float]:
        """(face width, face height, oblique depth, its horizontal part) of a volume."""
        fw, fh, depth = volume_geometry(self.g.node(node_id).params[VOLUME])
        return fw, fh, depth, depth * math.cos(math.radians(self.theme.metrics.cuboid_angle))

    def volume_extent(self, node_id: str) -> tuple[float, float]:
        """How far a volume reaches above and below its centre line: the oblique top above,
        the name and shape below (its footprint reserves the larger of the two both ways)."""
        fw, fh, depth, _ = self._volume_geometry(node_id)
        node = self.g.node(node_id)
        dy = depth * math.sin(math.radians(self.theme.metrics.cuboid_angle))
        below = 6 + self.theme.node_font(node.category).size * 1.15
        if self.texts[node_id].detail:
            below += self.theme.metrics.line_gap + self.theme.font(TextRole.DETAIL).size * 1.15
        return fh / 2 + dy, fh / 2 + below

    def _volume_size(self, node: GraphNode, text: NodeText) -> tuple[float, float]:
        """A volume's footprint: its face in the middle, with room for the oblique depth on
        both sides and above, and for the name and shape below (as much above), so wires
        meet the face level from the sides and straight from above or below."""
        fw, fh, depth, dx = self._volume_geometry(node.id)
        dy = depth * math.sin(math.radians(self.theme.metrics.cuboid_angle))
        label_font = self.theme.node_font(node.category)
        detail_font = self.theme.font(TextRole.DETAIL)
        below = 6 + label_font.size * 1.15
        widths = [self.measurer.width(text.label, label_font)]
        if text.detail:
            below += self.theme.metrics.line_gap + detail_font.size * 1.15
            widths.append(self.measurer.width(text.detail, detail_font))
        return _ceil(max(fw + 2 * dx, *widths) + 8), _ceil(fh + 2 * max(dy, below))

    def _uniform(self, sizes: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        """With ``uniform_width``, single-line boxes all take the longest one's length (pills
        and circles keep theirs): a column (or row of upright boxes) reads like a table."""
        if not (self.boxes.uniform_width and self.text.single_line):
            return sizes
        boxes = [
            i
            for i in sizes
            if self._shape(i) not in (Shape.CIRCLE, Shape.PILL)
            and VOLUME not in self.g.node(i).params
        ]
        if not boxes:
            return sizes
        axis = 1 if self.text is TextLayout.ROTATED else 0  # the length runs along y upright
        longest = max(sizes[i][axis] for i in boxes)
        return {
            i: (size[:axis] + (longest,) + size[axis + 1 :]) if i in boxes else size
            for i, size in sizes.items()
        }

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
        """Frame a group unless it is a lone collapsed node (count 1), a pass-through
        container holding exactly one sub-group, or laid out by a block layout."""
        if group.type_name in BLOCK_LAYOUTS:  # framed only if the block layout asks for it
            return getattr(BLOCK_LAYOUTS[group.type_name], "framed", False)
        if group.parent == ROOT and group.id in self.g.metadata.get(MODELS, "").split(","):
            return True  # one of several models: always framed and named
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

    def room(self, group_id: str) -> float | None:
        """How long a group's content may run along the flow before it wraps into rows (left
        → right) or columns (top → bottom): ``wrap_width`` or ``wrap_height``, less the padding
        of its own frame and of every frame around it."""
        width = self.opts.wrap_width if self.lr else self.opts.wrap_height
        if width is None:
            return None
        for group in [self.g.group(group_id), *self.g.ancestors(group_id)]:
            if group.id != ROOT and self.framed(group):
                before, after, *_ = self._frame_pads(group)
                width -= before + after
        return width

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
        special = BLOCK_LAYOUTS.get(self.g.group(group_id).type_name) if group_id != ROOT else None
        if special is not None:
            return special(self, group_id).solve()
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
            text = self.texts[node.id]
            if VOLUME in node.params:  # the front face, in the middle of the footprint
                fw, fh, depth, dx = self._volume_geometry(node.id)
                face = Box(box.cx - fw / 2, box.cy - fh / 2, fw, fh)
                boxes[node.id] = face
                scene_nodes.append(
                    Node(
                        node.id,
                        face,
                        node.category,
                        text.label,
                        text.detail,
                        shape=Shape.CUBOID,
                        depth=depth,
                    )
                )
                continue
            boxes[node.id] = box
            scene_nodes.append(
                Node(
                    node.id, box, node.category, text.label, text.detail, text=self.text_of(node.id)
                )
            )

        frames = []
        outer_first = sorted(block.frames.items(), key=lambda kv: self.g.depth(kv[0]))
        for gid, (f, c, fs, cs) in outer_first:  # painted in order: inner frames on top
            x, y = self.xy(f, c)
            w, h = self.xy(fs, cs)
            group = self.g.group(gid)
            frames.append(
                SceneGroup(gid, Box(m + x, m + y, w, h), self.title(group), self.badge(group))
            )

        edges = [self._route_edge(r, boxes, m) for r in block.routes]
        for node_id, indices in self.loops.items():
            edges += self._loop_edges(indices, boxes[node_id])
        scene_nodes = self._place_notes(scene_nodes, edges)

        notes = []
        for (f, c), text in block.notes:
            x, y = self.xy(f, c)
            notes.append(Note(Point(m + x, m + y), text, TextAnchor.MIDDLE, TextRole.LABEL))
        training = self._training_arrows(boxes, frames, bottom=m + height)
        edges += training
        height += 16 + 10 * self.under_lanes if self.under_lanes else 0
        scene = Scene(width + 2 * m, height + 2 * m, title=self.g.name)
        scene.add(*frames, *scene_nodes, *edges, *notes)
        if self.opts.legend:
            self._add_legend(scene, m + height + 20)
        return scene

    def _training_arrows(self, boxes: dict[str, Box], frames: list, bottom: float) -> list[Edge]:
        """A loss's dashed arrows to the models or modules it trains (``params[TRAINS]``).

        Into the model right beside the loss (the nearest thing on its left), *over the top*:
        out of the loss's side, up past what stands above it, along above the frame and down
        into it — a short counterclockwise loop (GAN loss → discriminator). Into any other,
        *underneath*: down to a lane under the figure, along it and up into the frame through
        a clear column (or, if every column is blocked, up a gutter on the left and in from
        the side)."""
        frame_of = {f.id: f.box for f in frames}
        losses_per: dict[str, int] = defaultdict(int)
        for node in self.g.nodes:
            for gid in node.params.get(TRAINS, ()):
                losses_per[gid] += 1
        font = self.theme.font(TextRole.ANNOTATION)
        arrows: list[Edge] = []
        lanes = overs = 0
        for node in self.g.nodes:
            for gid in node.params.get(TRAINS, ()):
                target = frame_of.get(gid) or self._bounds(gid, boxes)
                if target is None:
                    continue
                source = boxes[node.id]
                others = [f.box for f in frames if f.id != gid and not self._inside(gid, f.id)]
                obstacles = [b for b in [*boxes.values(), *others] if b is not source]
                width = self.measurer.width(node.label, font)
                named = losses_per[gid] > 1  # which loss is which, where several meet
                label = node.label if named else ""

                lane = bottom + 16 + 10 * lanes
                taken = [a.target.x for a in arrows if isinstance(a.target, Point)]
                x = self._clear_column(target, lane, obstacles, taken)
                if x is not None:
                    via = (Point(source.cx, lane), Point(x, lane))
                    end = Point(x, target.bottom)
                    label_at = Point(x + 6 + width / 2, target.bottom + font.size + 4)
                else:
                    every = [*boxes.values(), *frame_of.values()]
                    gutter = max(min(b.x for b in every) - 12 - 6 * lanes, 4.0)
                    via = (Point(source.cx, lane), Point(gutter, lane), Point(gutter, target.cy))
                    end = Point(target.x, target.cy)
                    label_at = Point(target.x - 6 - width / 2, target.cy - font.size)
                over = self._over_the_top(node.id, source, target, obstacles, overs, label, font)
                if over is not None:  # the model right beside the loss: a short loop back
                    arrows.append(over)
                    overs += 1
                else:
                    arrows.append(
                        Edge(
                            node.id,
                            end,
                            EdgeKind.TRAINING,
                            Side.BOTTOM,
                            via=via,
                            label=label,
                            label_at=label_at if named else None,
                        )
                    )
                    lanes += 1
        self.under_lanes = lanes
        return self._spread_labels(arrows, font.size)

    def _over_the_top(
        self, loss: str, source: Box, target: Box, obstacles: list[Box], k: int, label, font
    ) -> Edge | None:
        """Out of the loss's right side, up past everything above it, left above the target
        and down into its top — or ``None`` unless the target is the nearest thing on the
        loss's left (no other model or box in between) and the way is clear."""
        if source.x < target.right:
            return None
        if any(b.x >= target.right - 1 and b.right <= source.x + 1 for b in obstacles):
            return None  # it would jump over another model: the route underneath reads better
        out = source.right + 14 + 8 * k
        enter = target.right - 24 - 10 * k
        spanned = [b for b in obstacles if b.right > enter - 12 and b.x < out + 12]
        top = min([target.y, *(b.y for b in spanned if b.bottom > target.y - 1)]) - 14 - 8 * k
        if top < 8:
            return None
        legs = [
            (Point(out, source.cy), Point(out, top)),
            (Point(out, top), Point(enter, top)),
            (Point(enter, top), Point(enter, target.y)),
        ]
        for a, b in legs:
            if any(_crosses(a, b, box) for box in obstacles):
                return None
        via = (Point(out, source.cy), Point(out, top), Point(enter, top))
        at = (
            Point(enter + 6 + self.measurer.width(label, font) / 2, top - font.size)
            if label
            else None
        )
        return Edge(
            loss,
            Point(enter, target.y),
            EdgeKind.TRAINING,
            Side.RIGHT,
            via=via,
            label=label,
            label_at=at,
        )

    def _spread_labels(self, arrows: list[Edge], size: float) -> list[Edge]:
        """Labels of arrows rising into the same box side by side: the leftmost labelled on
        its left, the others on their right, each a line lower, so no two overlap."""
        by_end: dict[float, list[int]] = defaultdict(list)
        for i, a in enumerate(arrows):
            if a.label and isinstance(a.target, Point):
                by_end[round(a.target.y, 1)].append(i)
        spread = list(arrows)
        for indices in by_end.values():
            indices.sort(key=lambda i: arrows[i].target.x)
            font = self.theme.font(TextRole.ANNOTATION)
            for k, i in enumerate(indices):
                a = arrows[i]
                width = self.measurer.width(a.label, font)
                dx = -(6 + width / 2) if k == 0 and len(indices) > 1 else 6 + width / 2
                at = Point(a.target.x + dx, a.target.y + size + 4 + (size + 4) * max(k - 1, 0))
                spread[i] = dataclasses.replace(a, label_at=at)
        return spread

    def _bounds(self, gid: str, boxes: dict[str, Box]) -> Box | None:
        """The box around a group's nodes (a group without a frame)."""
        under = [boxes[n.id] for n in self.g.nodes_under(gid) if n.id in boxes]
        if not under:
            return None
        left, top = min(b.x for b in under), min(b.y for b in under)
        return Box(
            left, top, max(b.right for b in under) - left, max(b.bottom for b in under) - top
        )

    def _inside(self, gid: str, other: str) -> bool:
        """``other`` contains ``gid`` (an enclosing frame is no obstacle)."""
        return other in {a.id for a in self.g.ancestors(gid)}

    @staticmethod
    def _clear_column(
        target: Box, lane: float, obstacles: list[Box], taken: Sequence[float] = ()
    ) -> float | None:
        """An x under ``target``, nearest its middle, from which a line down to the lane
        touches no obstacle and no other training arrow."""

        def clear(x: float) -> bool:
            if any(abs(x - other) < 10 for other in taken):
                return False
            return not any(
                b.x - 12 < x < b.right + 12 and b.bottom > target.bottom + 0.5 and b.y < lane
                for b in obstacles
            )

        steps = int((target.w / 2 - 12) // 8)
        for k in range(steps + 1):
            for x in (target.cx - 8 * k, target.cx + 8 * k):
                if clear(x):
                    return x
        return None

    def _place_notes(self, nodes: list[Node], edges: list[Edge]) -> list[Node]:
        """A merge circle's note goes on a side no wire uses: above or below a tall box,
        right or left of a wide one (the layout sized the box for one of the two)."""
        used: dict[str, set[Side]] = defaultdict(set)
        for e in edges:
            if isinstance(e.source, str) and e.exit:
                used[e.source].add(e.exit)
            if isinstance(e.target, str) and e.entry:
                used[e.target].add(e.entry)
        placed = []
        for n in nodes:
            circle = self.theme.node_style(n.category).shape is Shape.CIRCLE
            if circle and n.detail and n.box.w != n.box.h:
                options = (Side.RIGHT, Side.LEFT) if n.box.w > n.box.h else (Side.TOP, Side.BOTTOM)
                free = [s for s in options if s not in used[n.id]]
                n = dataclasses.replace(n, note_side=(free or options)[0])
            placed.append(n)
        return placed

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
        label_at, upright = None, False
        if label:
            start = boxes[e.source].anchor(exit_side)
            end = boxes[e.target].anchor(entry_side)
            label_at, upright = self._label_position([start, *via, end], label)
        return Edge(
            e.source,
            e.target,
            EdgeKind(e.kind),
            exit_side,
            entry_side,
            tuple(via),
            label,
            label_at,
            label_upright=upright,
        )

    @property
    def upright_labels(self) -> bool:
        """Beside upright boxes (rotated text, left → right) the tensor shapes stand upright
        too, so a gap between two boxes needs one line of text, not a whole label."""
        return self.lr and self.text is TextLayout.ROTATED

    def _label_position(self, path: list[Point], label: str) -> tuple[Point, bool]:
        """Where the label's centre goes, and whether it stands upright."""
        font = self.theme.font(TextRole.ANNOTATION)
        width = self.label_width(label)
        segments = list(zip(path, path[1:], strict=False))
        if self.lr and all(abs(a.x - b.x) < 0.5 for a, b in segments):  # a vertical wire
            a, b = max(segments, key=lambda s: abs(s[1].y - s[0].y))
            return Point(a.x + 6 + width / 2, (a.y + b.y) / 2), False  # level, beside it
        if self.lr:
            along = font.size if self.upright_labels else width  # the label's extent along x
            flat = [(a, b) for a, b in segments if abs(a.y - b.y) < 0.5] or segments
            fitting = [s for s in flat if abs(s[1].x - s[0].x) >= along + 8]
            a, b = (fitting or sorted(flat, key=lambda s: -abs(s[1].x - s[0].x)))[0]
            if self.upright_labels:  # standing on the line, reading upwards
                return Point((a.x + b.x) / 2, a.y - 4 - width / 2), True
            return Point((a.x + b.x) / 2, a.y - font.size / 2 - 4), False
        if all(abs(a.y - b.y) < 0.5 for a, b in segments):  # a level wire across the flow
            a, b = max(segments, key=lambda s: abs(s[1].x - s[0].x))
            return Point((a.x + b.x) / 2, a.y - font.size / 2 - 4), False  # above it
        upright = [(a, b) for a, b in segments if abs(a.x - b.x) < 0.5] or segments
        fitting = [s for s in upright if abs(s[1].y - s[0].y) >= font.size + 8]
        a, b = (fitting or sorted(upright, key=lambda s: -abs(s[1].y - s[0].y)))[0]
        return Point(a.x + 6 + width / 2, (a.y + b.y) / 2), False

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
        room = self.opts.legend_width
        if room is None:
            room = max(scene.width - 2 * m, 560)
        for columns in (4, 3, 2, 1):
            legend = build_legend(
                self.theme, categories, kinds, Point(m, top), columns, self.measurer
            )
            if legend.box.w <= room or columns == 1:
                break
        scene.width = max(scene.width, legend.box.right + m)
        scene.height = legend.box.bottom + m
        add_legend(scene, self.theme, Point(m, top), columns, text=self.text)


def _crosses(a: Point, b: Point, box: Box, pad: float = 2.0) -> bool:
    """An axis-aligned segment passes through ``box`` (grown by ``pad``)."""
    if abs(a.x - b.x) < 0.5:
        low, high = sorted((a.y, b.y))
        return box.x - pad < a.x < box.right + pad and low < box.bottom + pad and high > box.y - pad
    low, high = sorted((a.x, b.x))
    return box.y - pad < a.y < box.bottom + pad and low < box.right + pad and high > box.x - pad


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
    no wider than ``width`` (or than the widest unbreakable block + 15 %, when that block is
    nearly as wide or wider), balanced so that the widest row is as narrow as possible.

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
    # a block (nearly) as wide as ``width`` sets the pace — e.g. a frame that wrapped itself
    # to fit: 15 % slack beats a row holding a lone pill
    limit = max(width, widest_unit * 1.15)
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
    if best[n][rows] > 0.8 * span(0, n - 1):
        return []  # a second row that saves < 20 % of the width would only strand small items
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

    def _side(self, index: int) -> bool:
        return index in self.run.side_inputs

    def _rank(self) -> None:
        preds: dict[str, dict[str, int]] = defaultdict(dict)  # item → predecessor → min. step
        succs = defaultdict(set)
        for index, a, b in self.forward:
            step = 0 if self._side(index) else 1  # fed from the side: beside its source
            preds[b][a] = max(preds[b].get(a, 0), step)
            succs[a].add(b)
        ordered = sorted(self.items.values(), key=lambda it: it.order)
        for it in ordered:
            it.rank = max(
                (self.items[p].rank + step for p, step in preds[it.key].items()), default=0
            )
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
            if self._side(index):
                continue
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
        # the source of a side input stands before the module it feeds (on the side its
        # wire enters), in every sweep, so the ranks around follow that order
        beside = [
            (a, b)
            for index, a, b in self.forward
            if self._side(index) and self.items[a].rank == self.items[b].rank
        ]

        def keep_beside(keys: list[str]) -> None:
            for a, b in beside:
                if a in keys and keys.index(a) > keys.index(b):
                    keys.remove(a)
                    keys.insert(keys.index(b), a)

        for keys in ranks.values():
            keep_beside(keys)
        self.beside = {k for pair in beside for k in pair}  # start level, as parallel stacks
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
                keep_beside(keys)
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
            if self._side(index):
                continue
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
                font_size = run.theme.font(TextRole.ANNOTATION).size
                need = (font_size if run.upright_labels else run.label_width(label)) + 16
                gap = max(gap, need if straight else 2 * need)  # bent: on one half of the gap
            else:
                gap = max(gap, run.theme.font(TextRole.ANNOTATION).size + 18)
        return gap

    def _rows(self) -> list[list[int]]:
        """Split the level into balanced rows when it is longer than its room (``wrap_width``
        less the frames around it): a frame too wide for the page wraps inside, and its rows
        read like lines of text. Top → bottom, the same with ``wrap_height``: a tall network
        stands in columns side by side, like a newspaper's.

        A row may only end where exactly one connection continues into the next row (so a
        residual block, with its skip alongside, stays whole), and no row may be left nearly
        empty (e.g. a lone output)."""
        levels = sorted(self.ranks)
        run = self.run
        width = run.room(self.gid)
        nested = self.gid != ROOT
        if width is None or len(levels) < 2 or (nested and self.back):
            return [levels]
        if nested:  # wrapped inside a frame, the rows get a gutter on both sides
            width -= 2 * WRAP_GUTTER
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
        return self._no_lone_pills([row for row in rows if row])

    def _no_lone_pills(self, rows: list[list[int]]) -> list[list[int]]:
        """A row holding nothing but an input or output pill joins its neighbour, even if
        that row grows a little longer than the room: a lone ``output`` on a line of its own,
        with a wire across the page to reach it, reads worse than a slightly longer row."""

        def lone(row: list[int]) -> bool:
            shown = [self.items[k] for r in row for k in self.ranks[r] if not self.items[k].dummy]
            return bool(shown) and all(
                it.node is not None and self.run.g.node(it.node).category is OpCategory.IO
                for it in shown
            )

        merged: list[list[int]] = []
        for row in rows:
            if merged and (lone(row) or lone(merged[-1])):
                merged[-1] = merged[-1] + row
            else:
                merged.append(row)
        return merged

    def _place_flow(self, rows: list[list[int]]) -> None:
        self.rank_span: dict[int, tuple[float, float]] = {}
        self.gaps: dict[int, float] = {}
        self.row_of: dict[int, int] = {}
        self.row_bounds: list[tuple[float, float]] = []  # cross extent of each row
        # wrapped inside a frame: room on both sides for the wires turning into the next row
        # (at the top level the figure's margin gives it)
        self.gutter = WRAP_GUTTER if len(rows) > 1 and self.gid != ROOT else 0.0
        cross_offset = 0.0
        for row_index, row in enumerate(rows):
            flow = self.gutter
            for i, r in enumerate(row):
                size = max(self.items[k].flow_size for k in self.ranks[r])
                self.rank_span[r] = (flow, flow + size)
                self.row_of[r] = row_index
                for k in self.ranks[r]:
                    it = self.items[k]
                    centred = k not in self.beside
                    it.flow = flow + ((size - it.flow_size) / 2 if centred else 0.0)
                gap = self._gap(r)
                self.gaps[r] = gap
                flow += size + (gap if i < len(row) - 1 else 0)
            keys = [k for r in row for k in self.ranks[r]]
            low = min(self.items[k].cross for k in keys)
            for k in keys:
                self.items[k].cross += cross_offset - low
            high = max(self.items[k].cross + self.items[k].cross_size for k in keys)
            self.row_bounds.append((cross_offset, high))
            cross_offset = high + self._row_gap(rows, row_index)
        if len(rows) > 1:
            self._centre_rows(rows)

    def _row_gap(self, rows: list[list[int]], index: int) -> float:
        """Space before the next row. Between columns (top → bottom) the wire into the next
        column runs up the middle of the gap with its shape beside it: room for that label."""
        gap = self.run.opts.row_gap
        if self.run.lr or index + 1 >= len(rows):
            return gap
        here, there = set(rows[index]), set(rows[index + 1])
        for edge, a, b in self.forward:
            label = self.run.labels.get(edge)
            if label and self.items[a].rank in here and self.items[b].rank in there:
                gap = max(gap, 2 * (self.run.label_width(label) + 10))
        return gap

    def _centre_rows(self, rows: list[list[int]]) -> None:
        """A row shorter than the longest is centred under it: the wire from the row before
        comes back to the middle, and the white space is shared on both sides."""
        spans = []
        for row in rows:
            keys = [k for r in row for k in self.ranks[r]]
            spans.append(max(self.items[k].flow + self.items[k].flow_size for k in keys))
        longest = max(spans)
        for row, end in zip(rows, spans, strict=True):
            shift = (longest - end) / 2
            if shift < 0.5:
                continue
            for r in row:
                start, stop = self.rank_span[r]
                self.rank_span[r] = (start + shift, stop + shift)
                for k in self.ranks[r]:
                    self.items[k].flow += shift

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
            block.sides.update(inner.sides)
            block.notes += inner.notes
            child_in.update(inner.in_paths)
            child_out.update(inner.out_paths)
            group_id = it.key[2:]
            if run.framed(run.g.group(group_id)):
                block.frames[group_id] = (it.flow, it.cross, it.flow_size, it.cross_size)
        block.flow = max(it.flow + it.flow_size for it in self.items.values()) + self.gutter
        block.cross = max(it.cross + it.cross_size for it in self.items.values())

        for index, a, b in self.forward:
            start = self._exit_cross(a, index)
            if self._side(index):  # beside its source: along the gap before it, in from the side
                points = (
                    child_out.get(index, [])
                    + self._side_route(a, b, index, start, child_in)
                    + child_in.get(index, [])
                )
                exit_side = block.sides.get(index, (None, None))[0]
                block.routes.append(_Route(index, points, exit_side or "flow_end", "cross_before"))
                continue
            end = self._entry_cross(b, index)
            if self.row_of.get(self.items[a].rank) != self.row_of.get(self.items[b].rank):
                middle = self._wrap(a, b, index, start, end, block, child_out, child_in)
            else:
                middle = self._lanes(a, b, start, end)
            points = child_out.get(index, []) + middle + child_in.get(index, [])
            exit_side, entry_side = block.sides.get(index, (None, None))
            block.routes.append(
                _Route(index, points, exit_side or "flow_end", entry_side or "flow_start")
            )

        for index, b in self.entering:
            if self._side(index):  # in through the side border, level with its target
                level = self._side_level(b, index, child_in)
                block.in_paths[index] = [(level, 0.0)] + child_in.get(index, [])
                if index not in child_in:
                    block.sides[index] = (block.sides.get(index, (None, None))[0], "cross_before")
                continue
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

    def _side_level(self, item_key: str, edge: int, child_in: dict) -> float:
        """Where along the flow a side input reaches ``item_key``: level with its target
        (a node's middle, or the side entry of the module holding it)."""
        it = self.items[item_key]
        if edge in child_in:
            return child_in[edge][0][0]
        return it.flow + it.inner[0] + self.run.node_fc(it.node)[0] / 2

    def _side_route(self, a: str, b: str, edge: int, start: float, child_in: dict) -> list[FC]:
        """From ``a``'s end to ``b``'s side: just past ``a``, across into the gap before
        ``b``, along it (forwards or back) to the level of the target, then in."""
        source, target = self.items[a], self.items[b]
        channel = target.cross - self.run.opts.node_gap / 2
        if source.rank == target.rank:
            end = self.rank_span[source.rank][1]
            last = source.rank == max(self.rank_span)
            past = end + (8.0 if last else self.gaps[source.rank] / 2)
            points = [(past, start), (past, channel)]
        else:
            points = self._lanes(a, b, start, channel)
        points.append((self._side_level(b, edge, child_in), channel))
        return points

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
