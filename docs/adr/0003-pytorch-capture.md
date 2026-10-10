# ADR-0003: Capture PyTorch models by eager execution tracing

Status: Accepted (verified by a spike on torch 2.2.2) · 2026-10-09

## Context

The user loads a model in a notebook and calls `tg.draw(model, example_input)` (FR1). The
importer must recover the operations, their data flow, the module hierarchy and the tensor
shapes. The target families include models with Python control flow: a ConvLSTM loops over
time steps inside `forward`, and custom cells use `torch.cat` / `torch.split` / point-wise
gates. Drawing must not change the model (NFR3).

## Decision

Run the model once on the example input and record what happens:

1. **Forward pre/post hooks** on every module maintain the module stack, which gives the
   hierarchy and the call index of repeated invocations.
2. A **`torch.overrides.TorchFunctionMode`** records every tensor-producing torch call
   (including operators such as `+` and `[...]`) made outside a *leaf* module.
3. **Leaf modules** (`Conv2d`, `Linear`, `LSTM`, `MultiheadAttention`, norms, activations,
   …) are recorded as one operation; their internals are not traced. The leaf set is a
   registry, so users can extend it.
4. Data flow is recovered from **tensor identity**: each output tensor is mapped to its
   producing operation, and recorded tensors are kept alive during the trace so ids are never
   reused.
5. The trace runs under `torch.no_grad()`, with `model.eval()` temporarily (so BatchNorm
   statistics do not update), and inside `torch.random.fork_rng()` (so sampling inside the
   model, e.g. `randn_like`, does not advance the caller's random state). All hooks are
   removed and the training flag is restored in a `finally` block.

### Spike results (torch 2.2.2, CPU)

| Model | Nodes | Time | Notes |
|---|---|---|---|
| CNN with residual | 8 | 7 ms | functional `relu`, `add`, `flatten` captured |
| 2-layer LSTM classifier | 3 | 9 ms | `nn.LSTM` as one leaf; `out[:, -1]` captured |
| ConvLSTM (custom cell, 4 steps) | 55 | 7 ms | loop unrolled; `cat`, `split`, gates, state edges between steps |
| VAE | 11 | 1 ms | reparameterisation (`exp`, `randn_like`, `mul`, `add`) captured |
| Transformer encoder (2 layers) | 22 | 9 ms | sub-layers, residual `add`, norms captured |

For every model: `state_dict` unchanged, zero hooks left, training flag restored.
`torch.fx.symbolic_trace` on the same ConvLSTM fails with
`TypeError: 'Proxy' object cannot be interpreted as an integer`.

## Alternatives considered

- **`torch.fx.symbolic_trace`.** It gives a clean graph, but fails on data-dependent
  control flow, which is exactly the ConvLSTM / custom-RNN case (shown above).
- **`torch.export` / Dynamo.** It handles control flow, but produces ATen-level graphs (far
  below the layer level), is strict about unsupported Python, and its API changed
  substantially across 2.x releases. That is risky on the torch 2.2 floor (ADR-0006).
- **Module hooks only** (the approach of torchsummary-style tools). This misses functional
  operations (`x + y`, `torch.cat`) and therefore residuals, merges and gates.
- **TorchScript tracing (`torch.jit.trace`).** It is in maintenance mode, loses module
  structure in inlined graphs, and warns on Python control flow.

## Consequences

- Loops are captured as they actually ran (unrolled). Folding them back into a recurrent
  cell is a transform (ADR-0002), and the unrolled view comes for free.
- Only the executed path is drawn. Branches not taken for the example input are not drawn
  (documented behaviour).
- The model runs once with real data, so cost equals one inference. A `meta`-device mode for
  very large models (NFR5) is a later optimisation.
- `nn.TransformerEncoderLayer` has a fused "fast path" in eval mode. It was not taken during
  the spike (an active `TorchFunctionMode` disables it), but the importer must detect the
  fused op (`_transformer_encoder_layer_fwd`) and report it if a future torch takes it.
- The trace keeps references to intermediate tensors until it finishes, so peak memory is
  that of one forward pass with all activations kept.
