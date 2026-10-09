"""Semantic vocabulary shared by every layer of Tensorgami (IR, scene, style, renderers).

These enums say *what* something is, never how it looks; the mapping to visuals lives in a
:class:`~tensorgami.style.theme.Theme`.
"""

from __future__ import annotations

from enum import StrEnum


class OpCategory(StrEnum):
    """Category of a diagram node; one fill colour per category (survey R3)."""

    # data and structure (neutral colours)
    IO = "io"  # model input / output data
    TENSOR = "tensor"  # named intermediate tensor or state (mu, sigma, h_t, context, ...)
    MODULE = "module"  # composite block shown collapsed (encoder, stage, ResNet-18, ...)
    MERGE = "merge"  # add, concatenate, multiply, ... (drawn as a symbol circle)
    RESHAPE = "reshape"  # flatten, reshape, permute, split, ...
    REGULARIZATION = "regularization"  # dropout, drop-path, ...
    OTHER = "other"  # anything not covered
    # operations (one hue each)
    CONV = "conv"
    LINEAR = "linear"
    RECURRENT = "recurrent"
    ATTENTION = "attention"
    NORM = "norm"
    ACTIVATION = "activation"
    RESAMPLE = "resample"  # pooling, up-sampling, interpolation
    EMBEDDING = "embedding"  # token / positional embeddings
    STOCHASTIC = "stochastic"  # sampling nodes and noise sources
    LOSS = "loss"


class EdgeKind(StrEnum):
    """Meaning of an edge; one line style per kind (survey R3, Marshall guideline B)."""

    DATA = "data"  # forward information flow, including skip connections
    STATE = "state"  # recurrent state passed between time steps
    STOCHASTIC = "stochastic"  # noise injection / sampling
    TRAINING = "training"  # loss, gradient and other training-only signals
    TIED = "tied"  # weight sharing between nodes (undirected)
