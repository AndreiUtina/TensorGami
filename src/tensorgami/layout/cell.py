"""The classic LSTM figure (the cell view, step 11d).

The cell state c runs along the top (c_{t-1} → × forget → + input → c_t), the four gates sit
in a row below it (σ forget, σ input, tanh candidate, σ output), tanh(c) and the output × on
the right, and the hidden state runs along the bottom: h_{t-1} joins x_t (‖) and feeds every
gate, h_t leaves on the right. Every node is placed by its role (``params[CELL_ROLE]``).
"""

from __future__ import annotations

from tensorgami.ir import CELL_ROLE, LSTM_CELL
from tensorgami.layout.layered import BLOCK_LAYOUTS, _Block, _Route, _Run

GAP = 36.0  # between columns
GATE_GAP = 24.0  # between neighbouring gates
ROW_GAP = 24.0  # between rows

STRAIGHT = ("flow_end", "flow_start")
UP = ("cross_before", "cross_after")
DOWN = ("cross_after", "cross_before")


class _LSTMCell:
    framed = True  # a cell keeps its frame and title

    def __init__(self, run: _Run, group_id: str) -> None:
        self.run = run
        self.id = {n.params[CELL_ROLE]: n.id for n in run.g.members(group_id)}

    def _w(self, role: str) -> float:
        return self.run.node_fc(self.id[role])[0]

    def _h(self, role: str) -> float:
        return self.run.node_fc(self.id[role])[1]

    def solve(self) -> _Block:
        w, h = self._w, self._h
        left = max(w("c_prev"), w("h_prev"))
        x = {"c_prev": left / 2, "h_prev": left / 2}
        x["concat"] = left + GAP + w("concat") / 2
        x["x"] = x["concat"]
        at = x["concat"] + w("concat") / 2 + GAP
        for role in "figo":
            x[role] = at + w(role) / 2
            at += w(role) + GATE_GAP
        x["mul_f"] = x["f"]
        x["mul_ig"] = x["add_c"] = (x["i"] + x["g"]) / 2
        x["tanh_c"] = x["mul_o"] = at - GATE_GAP + GAP + w("tanh_c") / 2
        right = max(w("c"), w("h"))
        x["c"] = x["h"] = x["tanh_c"] + w("tanh_c") / 2 + GAP + right / 2

        line = max(h("c_prev"), h("c"), h("mul_f"), h("add_c"))
        middle = max(h("mul_ig"), h("tanh_c"))
        gates = max(h(r) for r in "figo")
        bottom = max(h("h_prev"), h("concat"), h("h"))
        y = {r: line / 2 for r in ("c_prev", "mul_f", "add_c", "c")}
        y["mul_ig"] = y["tanh_c"] = line + ROW_GAP + middle / 2
        gate_row = line + middle + 2 * ROW_GAP + gates / 2
        y.update({r: gate_row for r in ("f", "i", "g", "o", "mul_o")})
        y_h = line + middle + gates + 3 * ROW_GAP + 4 + bottom / 2
        y.update({r: y_h for r in ("h_prev", "concat", "h")})
        y["x"] = y_h + bottom / 2 + ROW_GAP + h("x") / 2
        self.x, self.y = x, y

        block = _Block(x["c"] + right / 2, y["x"] + h("x") / 2)
        for role, node in self.id.items():
            block.nodes[node] = (x[role] - w(role) / 2, y[role] - h(role) / 2)
        role_of = {node: role for role, node in self.id.items()}
        for i, e in enumerate(self.run.g.edges):
            if e.source in role_of and e.target in role_of:
                block.routes.append(self._route(i, role_of[e.source], role_of[e.target]))
        return block

    def _route(self, index: int, a: str, b: str) -> _Route:
        x, y = self.x, self.y
        if (a, b) == ("add_c", "tanh_c"):  # a branch of the cell state, down into tanh
            return _Route(index, [(x["tanh_c"], y["add_c"])], "flow_end", "cross_before")
        if (a, b) in (("f", "mul_f"), ("mul_ig", "add_c"), ("x", "concat")):
            return _Route(index, [], *UP)
        if (a, b) == ("i", "mul_ig"):
            return _Route(index, [(x["i"], y["mul_ig"])], "cross_before", "flow_start")
        if (a, b) == ("g", "mul_ig"):
            return _Route(index, [(x["g"], y["mul_ig"])], "cross_before", "flow_end")
        if (a, b) == ("tanh_c", "mul_o"):
            return _Route(index, [], *DOWN)
        if (a, b) == ("mul_o", "h"):
            return _Route(index, [(x["mul_o"], y["h"])], "cross_after", "flow_start")
        if a == "concat" and b in "figo":  # along the hidden-state line, up into the gate
            return _Route(index, [(x[b], y["concat"])], "flow_end", "cross_after")
        return _Route(index, [], *STRAIGHT)


BLOCK_LAYOUTS[LSTM_CELL] = _LSTMCell
