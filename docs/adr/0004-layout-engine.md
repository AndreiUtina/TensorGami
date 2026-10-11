# ADR-0004: Pure-Python layered layout plus specialised layouts

Status: Accepted · 2026-10-09

## Context

The layout turns an IR graph (after transforms) into a `Scene` with positioned nodes and
routed edges. The style baseline fixes the conventions: main flow left → right with
orthogonal edges, deep stacks top → bottom, time on the x-axis for unrolled recurrence,
U-shapes for encoder/decoder models, and frames with `×N` badges. After collapsing to L2
(FR4, FR6), typical graphs have tens to a few hundred nodes. Installation must not require
system packages (NFR1).

## Decision

Implement layout in pure Python behind a `LayoutEngine` protocol (`Graph → Scene`):

1. **Layered (Sugiyama) layout** as the general engine:
   - rank assignment along the flow direction;
   - crossing minimisation (barycentre heuristic with a few sweeps);
   - coordinate assignment on the 8 px grid;
   - orthogonal edge routing, with residual edges in a side lane;
   - groups laid out recursively as frames.
2. **Specialised layouts** for shapes the general engine draws poorly:
   - linear chain with row wrapping;
   - vertical stack;
   - unrolled time × depth grid;
   - U-shape for skip-connected encoder/decoders.
3. A **view** picks the engine. The layered layout is the fallback.

*Implemented (step 11):* the specialised layouts are **block layouts** inside the layered
engine, registered by group type (`BLOCK_LAYOUTS`): a transform gathers the nodes into a group
of that type (`unrolled`: the time × layer grid, 11a; `u-shape`: the U, 11b) and the layered
engine places the group as one block, so the rest of the model is laid out around it.

All engines are deterministic: ties are broken by execution order, never by hash or set
order.

## Alternatives considered

- **Graphviz `dot`** (used by torchview, torchviz, Keras `plot_model`). Its layouts are good,
  but it needs a system binary, its output is hard to restyle to our conventions, and its
  results shift between versions.
- **ELK (Eclipse Layout Kernel)**. It is excellent for layered, orthogonal layouts, but runs
  on Java or JavaScript, so it would need a runtime bridge.
- **networkx / grandalf**. networkx has no layered layout with orthogonal routing.
  grandalf (pure-Python Sugiyama) is small but unmaintained, and does not support groups or
  ports.

## Consequences

- No system dependencies, and layouts tuned exactly to the style baseline.
- Layout quality depends on our implementation. Tests check invariants (no overlaps, edges
  monotone in flow direction, groups contain their members, determinism) instead of exact
  coordinates.
- Hand-positioned scenes (the specimens) keep working, because layout only produces scenes.
