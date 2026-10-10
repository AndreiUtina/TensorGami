# ADR-0002: One framework-neutral graph IR

Status: Accepted · 2026-10-09

## Context

Between "what the model computes" and "what the figure shows" sit several transformations:
collapsing hierarchy (FR4), folding trivial ops and repeats (FR6), and choosing views
(FR8). They need a representation that is independent of PyTorch (ADR-0001) and rich enough
for every target family: CNN, LSTM, cGAN, VAE, ConvLSTM and Transformer (FR9). The capture
spike (ADR-0003) showed what a trace actually contains: operations in execution order,
their module path, the producing operation of every input tensor, and output shapes.

## Decision

`tensorgami.ir` holds an immutable directed graph:

- **`Node`**: id, `op` (e.g. `Conv2d`, `add`), `category: OpCategory`, `label`, `params`
  (selected hyper-parameters such as kernel size or channels), `module_path`, `call_index`
  (which invocation of a module this was; it separates the time steps of a recurrent cell),
  and the shapes of its outputs.
- **`Edge`**: source node and output index → target node and input index, the tensor shape,
  and `kind: EdgeKind` (data, state, stochastic, training, tied).
- **`Group`**: a node of the module tree (path, type name, label, category override) with
  child groups and member nodes. This is the hierarchy that levels of detail collapse.
- **`Repeat`**: a group recognised as N repetitions of the same structure, either sequential
  layers or the time steps of a recurrent cell.
- **Graph inputs and outputs** are explicit nodes (category `io`).

Rules:

- Edges come only from real data flow. *Shape-only* dependencies (`zeros_like`,
  `new_zeros`, `randn_like`, `empty_like`, `ones_like`, `full_like`) create a source node,
  not an edge; random sources get category `stochastic`. (The spike showed that without this
  rule the VAE's noise looks as if it were computed from μ.)
- Parameters and buffers are attributes of a node, not separate inputs.
- Validation: unique ids, no dangling edges, acyclic data flow, and every node in exactly
  one group.
- JSON serialisation with a schema version, for test fixtures and caching.

## Alternatives considered

- **networkx graphs.** They are flexible, but untyped (attributes are free-form dicts), add a
  runtime dependency, and invariants would be checked by convention only.
- **Use `torch.fx.Graph` as the IR.** It is PyTorch-specific and cannot represent traces fx
  fails on (see ADR-0003).
- **Draw the module tree only (no op-level graph).** This loses functional operations such
  as `x + y` (residuals), `torch.cat` and `split` (ConvLSTM gates), which carry the essential
  structure of most target models.

## Consequences

- Transforms are pure functions `Graph → Graph`, which are easy to test and compose.
- Recurrence is first-class: an unrolled trace (one cell called T times) can be shown
  unrolled or folded into a cell with a state self-loop and a delay marker.
- Importers for other frameworks (step 16) only need to emit this IR.
