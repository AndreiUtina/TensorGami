# Architecture decision records

Each ADR records one decision: its context, the decision, the alternatives considered and
the consequences. ADRs are not edited after acceptance. A changed decision gets a new ADR
that *supersedes* the old one.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-pipeline-architecture.md) | A layered pipeline: importer → IR → transforms → layout → scene → canvas → figure | Accepted |
| [0002](0002-intermediate-representation.md) | One framework-neutral graph IR with hierarchy, shapes and repeats | Accepted |
| [0003](0003-pytorch-capture.md) | Capture PyTorch models by eager execution tracing (hooks + `TorchFunctionMode`) | Accepted (spike-verified) |
| [0004](0004-layout-engine.md) | Pure-Python layered layout plus specialised layouts; no Graphviz | Accepted |
| [0005](0005-figure-backend.md) | matplotlib for figures and PNG; the SVG canvas stays for vector output | Accepted |
| [0006](0006-dependencies-and-compatibility.md) | Runtime deps: matplotlib only; Python ≥ 3.11; torch ≥ 2.2 not installed by us | Accepted |
| [0007](0007-public-api.md) | `tg.draw(model, example_input, …)` returns a `tg.Figure` | Accepted |

The overall picture is in [architecture.md](../architecture.md).

## Template

```markdown
# ADR-NNNN: <decision as a short sentence>

Status: Proposed | Accepted | Superseded by ADR-NNNN · <date>

## Context
## Decision
## Alternatives considered
## Consequences
```
