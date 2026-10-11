"""The U layout for encoder–decoder models with nested skips (U-Net, pix2pix; step 11b).

A ``u-shape`` group becomes one block: a row per level, its encoder layers on the left
(right-aligned against the middle) and its decoder layers on the right (left-aligned from
the middle), so each skip is a short arrow across the middle; the bottleneck sits in a row of
its own below, centred. Encoder rows hand on downwards, decoder rows upwards, through the
gaps between the rows. The block is entered at the first encoder layer and left from the last
decoder layer, both on the top row, so the rest of the model continues on the same line. In a
top → bottom figure the rows become columns.
"""

from __future__ import annotations

from collections import defaultdict

from tensorgami.ir import U_LEVEL, U_SIDE, USHAPE
from tensorgami.layout.layered import BLOCK_LAYOUTS, _Block, _Route, _Run
from tensorgami.style import TextRole

BUS = 16.0  # distance of the bus for unusual entries / exits above the top row


class _UShape:
    def __init__(self, run: _Run, group_id: str) -> None:
        self.run = run
        order = {n.id: i for i, n in enumerate(run.g.nodes)}
        nodes = sorted(run.g.members(group_id), key=lambda n: order[n.id])
        self.ids = {n.id for n in nodes}
        self.rows: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.where: dict[str, tuple[int, str]] = {}
        for n in nodes:
            level, side = n.params[U_LEVEL], n.params[U_SIDE]
            self.rows[level][side].append(n.id)
            self.where[n.id] = (level, side)
        self.levels = sorted(self.rows)

    # ------------------------------------------------------------------ geometry

    def _gap(self, a: str, b: str) -> float:
        """Space between two neighbours along the flow: room for the shape label between."""
        run = self.run
        gap = max(run.opts.rank_gap, 28.0)
        for i, e in enumerate(run.g.edges):
            if e.source == a and e.target == b and i in run.labels:
                font = run.theme.font(TextRole.ANNOTATION).size
                along = font if run.upright_labels else run.label_width(run.labels[i])
                gap = max(gap, along + 24)
        return gap

    def _segment(self, ids: list[str]) -> tuple[list[float], float]:
        """Offsets of the nodes of one segment from its start, and its length."""
        offsets, at = [], 0.0
        for k, n in enumerate(ids):
            offsets.append(at)
            at += self.run.node_fc(n)[0]
            if k + 1 < len(ids):
                at += self._gap(n, ids[k + 1])
        return offsets, at

    def solve(self) -> _Block:
        run = self.run
        size = {n: run.node_fc(n) for n in self.ids}
        segments = {
            (level, side): self._segment(ids)
            for level, sides in self.rows.items()
            for side, ids in sides.items()
        }
        left = max((segments[k][1] for k in segments if k[1] == "enc"), default=0.0)
        middle = max(run.opts.rank_gap, 40.0)
        for e_index, e in enumerate(run.g.edges):  # a skip's shape needs the middle's width
            if e.source in self.ids and e.target in self.ids and e_index in run.labels:
                if self.where[e.source][1] == "enc" and self.where[e.target][1] == "dec":
                    middle = max(middle, self._gap(e.source, e.target))
        centre = left + middle / 2

        flow: dict[str, float] = {}
        for (level, side), (offsets, length) in segments.items():
            start = {"enc": left - length, "dec": left + middle, "bottom": centre - length / 2}
            for n, offset in zip(self.rows[level][side], offsets, strict=True):
                flow[n] = start[side] + offset
        shift = -min(flow.values())  # a long bottleneck may stick out on the left
        for n in flow:
            flow[n] += shift

        entering, leaving, internal = [], [], []
        for i, e in enumerate(run.g.edges):
            a, b = e.source in self.ids, e.target in self.ids
            if a and b and e.source != e.target:
                internal.append(i)
            elif b and not a:
                entering.append(i)
            elif a and not b:
                leaving.append(i)
        first = self.rows[self.levels[0]]["enc"][:1]
        last = self.rows[self.levels[0]]["dec"][-1:]
        bus = any(run.g.edges[i].target not in first for i in entering) or any(
            run.g.edges[i].source not in last for i in leaving
        )

        row_gap = max(run.opts.node_gap + 26, 44.0)
        self.top, self.height, cross = {}, {}, BUS if bus else 0.0
        for level in self.levels:
            self.top[level] = cross
            self.height[level] = max(size[n][1] for ids in self.rows[level].values() for n in ids)
            cross += self.height[level] + row_gap
        total_cross = cross - row_gap
        total_flow = max(flow[n] + size[n][0] for n in self.ids)

        block = _Block(total_flow, total_cross)
        for n in self.ids:
            level, _ = self.where[n]
            block.nodes[n] = (flow[n], self._cy(level) - size[n][1] / 2)
        self.flow, self.size = flow, size
        for i in internal:
            block.routes.append(self._internal(i))
        for i in entering:
            target = run.g.edges[i].target
            if target in first:
                block.in_paths[i] = [(0.0, self._cy(self.levels[0]))]
            else:  # along the bus above the top row, then down into it
                x = flow[target] + size[target][0] / 2
                block.in_paths[i] = [(0.0, BUS / 2), (x, BUS / 2)]
                block.sides[i] = (None, "cross_before")
        for i in leaving:
            source = run.g.edges[i].source
            if source in last:
                block.out_paths[i] = [(total_flow, self._cy(self.levels[0]))]
            else:
                x = flow[source] + size[source][0] / 2
                block.out_paths[i] = [(x, BUS / 2), (total_flow, BUS / 2)]
                block.sides[i] = ("cross_before", None)
        return block

    def _cy(self, level: int) -> float:
        return self.top[level] + self.height[level] / 2

    def _cx(self, node: str) -> float:
        return self.flow[node] + self.size[node][0] / 2

    # ------------------------------------------------------------------ edges

    def _internal(self, index: int) -> _Route:
        e = self.run.g.edges[index]
        (la, sa), (lb, sb) = self.where[e.source], self.where[e.target]
        row_a, row_b = self.rows[la][sa], self.rows[lb][sb]
        if la == lb:
            a, b = row_a.index(e.source) if sa == sb else -1, row_b.index(e.target)
            neighbours = sa == sb and b == a + 1
            skip = sa == "enc" and sb == "dec" and e.source == row_a[-1] and b == 0
            if neighbours or skip:  # along the row (a skip crosses the middle)
                return _Route(index, [], "flow_end", "flow_start")
            above = self.top[la] - 8  # anything else in a row: over the top
            points = [(self._cx(e.source), above), (self._cx(e.target), above)]
            return _Route(index, points, "cross_before", "cross_before")
        if lb == la + 1 or (lb > la and sb == "bottom"):  # handing on to the next row down
            y = (self.top[lb] + self.top[la] + self.height[la]) / 2 - 3
            points = [(self._cx(e.source), y), (self._cx(e.target), y)]
            return _Route(index, points, "cross_after", "cross_before")
        if lb == la - 1 or (la > lb and sa == "bottom"):  # handing back up a row
            y = (self.top[la] + self.top[lb] + self.height[lb]) / 2 + 3
            points = [(self._cx(e.source), y), (self._cx(e.target), y)]
            return _Route(index, points, "cross_before", "cross_after")
        x = self.flow[e.source] + self.size[e.source][0] + 10  # rare: down the side
        return _Route(index, [(x, self._cy(la)), (x, self._cy(lb))], "flow_end", "flow_start")


BLOCK_LAYOUTS[USHAPE] = _UShape
