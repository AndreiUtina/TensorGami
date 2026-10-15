"""The unrolled view's layout: every ``unrolled`` group becomes a time × layer grid.

Time steps run along the flow (left → right), the first layer sits at the bottom with the
per-step inputs x_t below it, and per-step outputs h_t sit above the top layer: the classic
unrolled RNN figure. State passes from step to step along each row; data climbs each column.
The rest of the model is laid out around the grid by the layered engine, which treats the grid
as one block (top → bottom figures get the grid turned with them).
"""

from __future__ import annotations

from collections import defaultdict

from tensorgami.ir import LAYER, ROLE, STEP, UNROLLED
from tensorgami.layout.layered import BLOCK_LAYOUTS, FC, _Block, _Route, _Run

BUS = 18.0  # distance of the input (output) bus below (above) the grid
ELIDED = 36.0  # extra width of a gap that hides time steps ("⋯")


class _Grid:
    """One grid, computed left → right with the first layer at the bottom; in a top → bottom
    figure the cross axis is mirrored, so the first layer sits on the left."""

    def __init__(self, run: _Run, group_id: str) -> None:
        self.run, self.gid = run, group_id
        nodes = run.g.members(group_id)
        self.ids = {n.id for n in nodes}
        self._label_layers()
        self.top = top = max(
            (n.params[LAYER] for n in nodes if n.params.get(ROLE) == "cell"), default=0
        )
        level = {"in": -1, "out": top + 1}  # row of the per-step tensors
        self.place = {  # node → (row level: -1 inputs, 0… layers, top + 1 outputs; column)
            n.id: (level.get(n.params.get(ROLE), n.params.get(LAYER, 0)), n.params[STEP])
            for n in nodes
        }
        self.steps = sorted({t for _, t in self.place.values()})
        self.column = {t: j for j, t in enumerate(self.steps)}
        self.levels = sorted({lvl for lvl, _ in self.place.values()}, reverse=True)  # top first

    def _label_layers(self) -> None:
        """Inside the grid, a shape is shown only between layers, once, at the first step."""
        run = self.run
        for i, e in enumerate(run.g.edges):
            if e.source in self.ids and e.target in self.ids:
                run.labels.pop(i, None)
                a, b = run.g.node(e.source).params, run.g.node(e.target).params
                cells = a.get(ROLE) == b.get(ROLE) == "cell"
                if cells and a[STEP] == b[STEP] == 0 and a[LAYER] != b[LAYER]:
                    text = run.labeler.edge_label(run.g, e) if run.opts.shapes_on_edges else ""
                    if text:
                        run.labels[i] = text

    # ------------------------------------------------------------------ geometry

    def solve(self) -> _Block:
        run = self.run
        size = {n: run.node_fc(n) for n in self.ids}
        width = max(fs for fs, _ in size.values())
        row_h = defaultdict(float)
        for n, (lvl, _) in self.place.items():
            row_h[lvl] = max(row_h[lvl], size[n][1])
        col_gap = max(run.opts.rank_gap, 32.0)
        self.gaps = [
            col_gap + (ELIDED if b - a > 1 else 0.0)
            for a, b in zip(self.steps, self.steps[1:], strict=False)
        ]
        row_gap = max(run.opts.node_gap + 10, 28.0)
        if not run.lr:  # layers side by side: a shape between them needs its width
            between = [run.labels[i] for i in self._layer_labels()]
            row_gap = max([row_gap, *(run.label_width(text) + 16 for text in between)])

        self.entering, self.leaving, self.internal = [], [], []
        for i, e in enumerate(run.g.edges):
            if e.source == e.target:
                continue
            a, b = e.source in self.ids, e.target in self.ids
            if a and b:
                self.internal.append(i)
            elif b:
                self.entering.append(i)
            elif a:
                self.leaving.append(i)
        last = len(self.steps) - 1
        below = any(not self._direct_in(i) for i in self.entering)
        above = any(not self._direct_out(i, last) for i in self.leaving)

        self.x, x = [], 0.0
        for j in range(len(self.steps)):
            self.x.append(x)
            x += width + (self.gaps[j] if j < last else 0.0)
        flow = self.x[-1] + width
        self.y, y = {}, BUS if above else 0.0
        for lvl in self.levels:
            self.y[lvl] = y
            y += row_h[lvl] + row_gap
        rows_end = y - row_gap
        cross = rows_end + (BUS if below else 0.0)
        self.width, self.row_h = width, row_h
        self.bus_above, self.bus_below = BUS / 2, rows_end + BUS / 2

        block = _Block(flow, cross)
        for n, (lvl, t) in self.place.items():
            fs, cs = size[n]
            block.nodes[n] = (self._cx(t) - fs / 2, self._cy(lvl) - cs / 2)
        for i in self.internal:
            block.routes.append(self._internal(i))
        for i in self.entering:
            self._enter(i, block)
        for i in self.leaving:
            self._leave(i, block, flow, last)
        for j, gap in enumerate(self.gaps):
            if gap > col_gap:  # hidden steps
                mid = self.x[j] + width + gap / 2
                for lvl in self.levels:
                    shift = 9 if 0 <= lvl <= self.top else 0  # clear of the state wire
                    block.notes.append(((mid, self._cy(lvl) - shift), "⋯"))
        return block if run.lr else _mirror(block, size)

    def _layer_labels(self) -> list[int]:
        """Edges between layers that carry a shape label."""
        return [
            i
            for i, e in enumerate(self.run.g.edges)
            if i in self.run.labels and e.source in self.ids and e.target in self.ids
        ]

    def _cx(self, step: int) -> float:
        return self.x[self.column[step]] + self.width / 2

    def _cy(self, level: int) -> float:
        return self.y[level] + self.row_h[level] / 2

    # ------------------------------------------------------------------ edges

    def _internal(self, index: int) -> _Route:
        e = self.run.g.edges[index]
        (la, ta), (lb, tb) = self.place[e.source], self.place[e.target]
        if la == lb:  # along a row: state, forwards or backwards in time
            sides = ("flow_end", "flow_start") if ta < tb else ("flow_start", "flow_end")
            return _Route(index, [], *sides)
        if ta == tb:  # up (or down) a column: data between layers
            sides = ("cross_before", "cross_after") if la < lb else ("cross_after", "cross_before")
            return _Route(index, [], *sides)
        j = self.column[ta]  # anything else: through the gap after the source's column
        x = self.x[j] + self.width + (self.gaps[j] / 2 if j < len(self.gaps) else 12)
        return _Route(index, [(x, self._cy(la)), (x, self._cy(lb))], "flow_end", "flow_start")

    def _direct_in(self, index: int) -> bool:
        """Enters straight from the left: a cell of the first column."""
        target = self.run.g.edges[index].target
        lvl, t = self.place[target]
        return lvl >= 0 and self.column[t] == 0 and self.run.g.node(target).params.get(ROLE) != "in"

    def _direct_out(self, index: int, last: int) -> bool:
        """Leaves straight to the right: a cell of the last column."""
        source = self.run.g.edges[index].source
        _, t = self.place[source]
        return self.column[t] == last and self.run.g.node(source).params.get(ROLE) != "out"

    def _enter(self, index: int, block: _Block) -> None:
        target = self.run.g.edges[index].target
        lvl, t = self.place[target]
        if self._direct_in(index):
            block.in_paths[index] = [(0.0, self._cy(lvl))]
        elif self.run.g.node(target).params.get(ROLE) == "in":  # along the bus, up into x_t
            block.in_paths[index] = [(0.0, self.bus_below), (self._cx(t), self.bus_below)]
            block.sides[index] = (None, "cross_after")
        else:  # along the bus, up the gap before its column, in from the left
            j = self.column[t]
            x = self.x[j] - self.gaps[j - 1] / 2
            block.in_paths[index] = [
                (0.0, self.bus_below),
                (x, self.bus_below),
                (x, self._cy(lvl)),
            ]

    def _leave(self, index: int, block: _Block, flow: float, last: int) -> None:
        source = self.run.g.edges[index].source
        lvl, t = self.place[source]
        if self._direct_out(index, last):
            block.out_paths[index] = [(flow, self._cy(lvl))]
        elif self.run.g.node(source).params.get(ROLE) == "out":  # up from h_t, along the bus
            block.out_paths[index] = [(self._cx(t), self.bus_above), (flow, self.bus_above)]
            block.sides[index] = ("cross_before", None)
        else:  # out to the right, up the gap after its column, along the bus
            j = self.column[t]
            x = self.x[j] + self.width + self.gaps[j] / 2
            block.out_paths[index] = [
                (x, self._cy(lvl)),
                (x, self.bus_above),
                (flow, self.bus_above),
            ]


def _mirror(block: _Block, size: dict[str, FC]) -> _Block:
    """Flip the cross axis (top → bottom figures: the first layer on the left)."""
    total = block.cross

    def flip(points: list[FC]) -> list[FC]:
        return [(f, total - c) for f, c in points]

    swap = {"cross_before": "cross_after", "cross_after": "cross_before"}
    return _Block(
        block.flow,
        block.cross,
        {n: (f, total - c - size[n][1]) for n, (f, c) in block.nodes.items()},  # box corners
        {},
        [
            _Route(r.edge, flip(r.points), swap.get(r.exit, r.exit), swap.get(r.entry, r.entry))
            for r in block.routes
        ],
        {i: flip(p) for i, p in block.in_paths.items()},
        {i: flip(p) for i, p in block.out_paths.items()},
        {i: (swap.get(a, a), swap.get(b, b)) for i, (a, b) in block.sides.items()},
        [((f, total - c), text) for (f, c), text in block.notes],
    )


BLOCK_LAYOUTS[UNROLLED] = _Grid
