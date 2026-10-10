"""Plain-text table of a graph: a quick check before any figure is drawn."""

from __future__ import annotations

from tensorgami.ir.model import Graph
from tensorgami.ir.shapes import format_shape


def summarize(graph: Graph, *, max_rows: int | None = None) -> str:
    """One row per node in execution order: op, category, group, module, inputs, output."""
    header = ("#", "op", "category", "group", "module", "inputs", "output")
    rows = []
    for index, node in enumerate(graph.nodes):
        sources = ", ".join(e.source for e in graph.in_edges(node.id))
        shape = node.output_shapes[0] if node.output_shapes else None
        extra = f" (+{len(node.output_shapes) - 1})" if len(node.output_shapes) > 1 else ""
        rows.append(
            (
                str(index),
                node.display_label,
                node.category.value,
                node.group or "·",
                node.module or "·",
                sources or "·",
                format_shape(shape, drop_batch=False) + extra,
            )
        )
    hidden = 0
    if max_rows is not None and len(rows) > max_rows:
        hidden, rows = len(rows) - max_rows, rows[:max_rows]
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)).rstrip()

    title = (
        f"{graph.name or 'graph'}: {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
        f"{len(graph.groups)} groups"
    )
    out = [title, line(header), line(tuple("─" * w for w in widths)), *map(line, rows)]
    if hidden:
        out.append(f"… {hidden} more rows")
    return "\n".join(out)
