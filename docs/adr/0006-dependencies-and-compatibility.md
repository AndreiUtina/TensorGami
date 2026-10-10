# ADR-0006: Dependencies, supported versions and distribution

Status: Accepted · 2026-10-09

## Context

Tensorgami is installed into environments that already contain the user's PyTorch, often a
specific CUDA build or a version pinned by a project. The development machine is an Intel
Mac: PyTorch publishes no macOS x86-64 wheels after **2.2.2**, so that is the newest torch
that can be tested locally. GPU or cloud notebooks typically run newer torch and Python
3.11–3.13. The user wants a private install from GitHub first and a public PyPI package
later.

## Decision

- **Runtime dependencies:** `matplotlib` only.
- **PyTorch:** never installed or upgraded by Tensorgami. It is declared as an optional extra
  (`tensorgami[torch]`, `torch>=2.2`) and imported lazily, only inside
  `tensorgami.importers.torch`. A missing torch gives a clear message.
- **Python:** `>=3.11` (the code needs `enum.StrEnum` and `tomllib`). The test matrix covers
  3.11 and 3.12 locally with torch 2.2.2, and newer torch/Python on Linux in CI.
- **Distribution:** install from GitHub now, e.g.
  `uv add "tensorgami @ git+https://github.com/AndreiUtina/Tensorgami"` or
  `pip install "git+https://github.com/AndreiUtina/Tensorgami"`. Releases are git tags
  (`v0.x.y`), and a PyPI release follows once the API is stable.
- **Development-only tools** (typst, the survey figures, pytest, ruff) stay in dependency
  groups and never reach users.

## Alternatives considered

- **Depend on `torch` directly**. A plain `pip install tensorgami` could then replace a CUDA
  build with a CPU wheel or upgrade a pinned version, which is the most common way
  visualisation tools break ML environments.
- **Python ≥ 3.12 only** (the earlier project default). It excludes notebooks still on 3.11
  for no technical gain.
- **Zero runtime dependencies** (the earlier rule). That conflicts with the
  notebook/PNG requirement (ADR-0005).

## Consequences

- `tests/test_architecture.py` enforces that `torch` is imported only inside
  `tensorgami.importers` and `matplotlib` only inside `tensorgami.render` (and the API
  facade), so the core stays importable without either.
- The importer must be written against the torch 2.2 API (e.g.
  `torch.overrides.TorchFunctionMode`, `register_forward_hook(..., with_kwargs=True)`).
  Newer APIs need a guarded fallback.
- torch 2.2 wheels need `numpy<2`. The dev environment pins that; users with newer torch
  are unaffected.
