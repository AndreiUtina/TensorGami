"""What Tensorgami knows about PyTorch operations: names, categories, leaves, parameters.

Pure data plus small lookup functions. Classes are matched by *name* along the MRO, so
subclasses (e.g. ``NonDynamicallyQuantizableLinear``) inherit their parent's category and the
catalogue never needs the class objects themselves. Name lists are whitespace-separated
strings to keep the tables compact.
"""

from __future__ import annotations

from collections.abc import Iterable

from tensorgami.vocabulary import OpCategory as C


def _names(text: str) -> tuple[str, ...]:
    return tuple(text.split())


# --------------------------------------------------------------------------- modules

_MODULES: dict[C, tuple[str, ...]] = {
    C.CONV: _names("""Conv1d Conv2d Conv3d ConvTranspose1d ConvTranspose2d ConvTranspose3d
                      LazyConv1d LazyConv2d LazyConv3d"""),
    C.LINEAR: _names("Linear LazyLinear Bilinear"),
    C.RECURRENT: _names("RNN LSTM GRU RNNCell LSTMCell GRUCell"),
    C.ATTENTION: _names("MultiheadAttention"),
    C.NORM: _names("""BatchNorm1d BatchNorm2d BatchNorm3d SyncBatchNorm LayerNorm GroupNorm
                      InstanceNorm1d InstanceNorm2d InstanceNorm3d LocalResponseNorm RMSNorm"""),
    C.ACTIVATION: _names("""ReLU ReLU6 LeakyReLU PReLU RReLU ELU SELU CELU GELU SiLU Mish
                            Sigmoid Tanh Softmax LogSoftmax Softplus Softsign Hardtanh
                            Hardswish Hardsigmoid GLU Tanhshrink Threshold"""),
    C.RESAMPLE: _names("""MaxPool1d MaxPool2d MaxPool3d AvgPool1d AvgPool2d AvgPool3d
                          AdaptiveAvgPool1d AdaptiveAvgPool2d AdaptiveAvgPool3d
                          AdaptiveMaxPool1d AdaptiveMaxPool2d AdaptiveMaxPool3d Upsample
                          UpsamplingNearest2d UpsamplingBilinear2d PixelShuffle
                          PixelUnshuffle"""),
    C.EMBEDDING: _names("Embedding EmbeddingBag"),
    C.REGULARIZATION: _names(
        "Dropout Dropout1d Dropout2d Dropout3d AlphaDropout FeatureAlphaDropout"
    ),
    C.RESHAPE: _names("""Flatten Unflatten ZeroPad1d ZeroPad2d ZeroPad3d ConstantPad1d
                         ConstantPad2d ConstantPad3d ReflectionPad1d ReflectionPad2d
                         ReflectionPad3d ReplicationPad1d ReplicationPad2d ReplicationPad3d"""),
    C.OTHER: _names("Identity"),
}
MODULE_CATEGORY: dict[str, C] = {name: cat for cat, names in _MODULES.items() for name in names}

#: Modules drawn as one operation; their internals are not traced (ADR-0003).
LEAF_MODULES: frozenset[str] = frozenset(MODULE_CATEGORY)

#: Hyper-parameters worth keeping per module type (first match along the MRO).
MODULE_PARAMS: dict[str, tuple[str, ...]] = {
    "_ConvNd": _names("in_channels out_channels kernel_size stride padding dilation groups"),
    "Linear": _names("in_features out_features"),
    "Bilinear": _names("in1_features in2_features out_features"),
    "RNNBase": _names("input_size hidden_size num_layers bidirectional batch_first"),
    "RNNCellBase": _names("input_size hidden_size"),
    "MultiheadAttention": _names("embed_dim num_heads batch_first"),
    "_NormBase": _names("num_features"),
    "LayerNorm": _names("normalized_shape"),
    "GroupNorm": _names("num_groups num_channels"),
    "_MaxPoolNd": _names("kernel_size stride"),
    "_AvgPoolNd": _names("kernel_size stride"),
    "_AdaptiveAvgPoolNd": _names("output_size"),
    "_AdaptiveMaxPoolNd": _names("output_size"),
    "Upsample": _names("scale_factor size mode"),
    "Embedding": _names("num_embeddings embedding_dim"),
    "EmbeddingBag": _names("num_embeddings embedding_dim"),
    "_DropoutNd": _names("p"),
    "LeakyReLU": _names("negative_slope"),
    "Softmax": _names("dim"),
    "LogSoftmax": _names("dim"),
    "Flatten": _names("start_dim end_dim"),
}

# --------------------------------------------------------------------------- functions

_FUNCTIONS: dict[C, tuple[str, ...]] = {
    C.CONV: _names("""conv1d conv2d conv3d conv_transpose1d conv_transpose2d conv_transpose3d
                      convolution"""),
    C.LINEAR: _names("linear bilinear matmul mm bmm addmm baddbmm einsum mv addmv"),
    C.ACTIVATION: _names("""relu relu6 leaky_relu elu selu celu gelu silu mish sigmoid tanh
                            softmax log_softmax softplus softsign hardtanh hardswish hardsigmoid
                            glu logsigmoid prelu rrelu threshold tanhshrink exp log"""),
    C.NORM: _names("""batch_norm layer_norm group_norm instance_norm local_response_norm
                      normalize rms_norm"""),
    C.RESAMPLE: _names("""max_pool1d max_pool2d max_pool3d avg_pool1d avg_pool2d avg_pool3d
                          adaptive_avg_pool1d adaptive_avg_pool2d adaptive_avg_pool3d
                          adaptive_max_pool1d adaptive_max_pool2d adaptive_max_pool3d
                          interpolate upsample upsample_nearest upsample_bilinear pixel_shuffle
                          pixel_unshuffle grid_sample"""),
    C.ATTENTION: _names("scaled_dot_product_attention multi_head_attention_forward"),
    C.EMBEDDING: _names("embedding embedding_bag one_hot"),
    C.REGULARIZATION: _names("""dropout dropout1d dropout2d dropout3d alpha_dropout
                                feature_alpha_dropout"""),
    C.RESHAPE: _names("""view reshape flatten unflatten permute transpose t squeeze unsqueeze
                         contiguous getitem split chunk unbind expand expand_as repeat tile
                         narrow select movedim swapaxes swapdims view_as reshape_as flip roll
                         pad to float double half bfloat16 int long bool type clone detach T mT
                         H mH index_select gather masked_select data real imag"""),
    C.MERGE: _names("cat concat concatenate stack hstack vstack dstack"),
    C.STOCHASTIC: _names("""randn rand randint randn_like rand_like randint_like normal
                            bernoulli multinomial randperm poisson"""),
    C.TENSOR: _names("""zeros ones full empty zeros_like ones_like full_like empty_like arange
                        linspace logspace eye tensor as_tensor new_zeros new_ones new_full
                        new_empty new_tensor tril triu"""),
    C.LOSS: _names("""mse_loss l1_loss cross_entropy nll_loss binary_cross_entropy
                      binary_cross_entropy_with_logits kl_div smooth_l1_loss huber_loss
                      cosine_embedding_loss margin_ranking_loss triplet_margin_loss"""),
}
FUNCTION_CATEGORY: dict[str, C] = {n: cat for cat, names in _FUNCTIONS.items() for n in names}

#: Element-wise binary operations: a *merge* when they combine two tensors, plain arithmetic
#: (category 'other') when one operand is a Python number, e.g. ``0.5 * logvar``.
ARITHMETIC: frozenset[str] = frozenset(
    _names("""add sub mul div true_divide floor_divide maximum minimum where remainder fmod
              atan2 hypot logaddexp""")
)

#: Known operations without a dedicated category: drawn as 'other', without a warning.
KNOWN_OTHER: frozenset[str] = frozenset(
    _names("""neg abs sqrt rsqrt pow square reciprocal mean sum prod max min amax amin argmax
              argmin std var norm clamp clip sign round floor ceil trunc cumsum cumprod
              logsumexp sort topk eq ne lt le gt ge logical_not logical_and logical_or
              logical_xor masked_fill fill copy zero index_put scatter scatter_add nan_to_num
              isnan isinf sin cos erf all any count_nonzero diff outer cross dot trace diag
              diagonal setitem""")
)

#: Operations that read their tensor argument only for its *shape*: they become sources
#: without data edges (ADR-0002), e.g. ``randn_like(mu)`` is noise, not a function of μ.
SHAPE_ONLY: frozenset[str] = frozenset(
    _names("""zeros_like ones_like full_like empty_like randn_like rand_like randint_like
              new_zeros new_ones new_full new_empty new_tensor""")
)

#: Fused operations that hide sub-module structure (ADR-0003 consequence).
FUSED: frozenset[str] = frozenset(
    _names("_transformer_encoder_layer_fwd _native_multi_head_attention")
)

_REFLECTED = frozenset(
    _names("add sub mul truediv floordiv mod pow matmul and or xor lshift rshift")
)
_RENAMES = {
    "truediv": "div",
    "floordiv": "floor_divide",
    "mod": "remainder",
    "and": "logical_and",
    "or": "logical_or",
    "xor": "logical_xor",
}

_GROUP_KEYWORDS: tuple[tuple[tuple[str, ...], C], ...] = (
    (("lstm", "gru", "rnn", "recurrent"), C.RECURRENT),
    (("attention", "attn"), C.ATTENTION),
    (("embed",), C.EMBEDDING),
    (("conv",), C.CONV),
    (("norm",), C.NORM),
)

# --------------------------------------------------------------------------- lookups


def normalize_op_name(name: str) -> str:
    """``__add__``/``__radd__`` → ``add``; ``__iadd__`` → ``add_``; ``__getitem__`` →
    ``getitem``; ``__truediv__`` → ``div``."""
    if name.startswith("__") and name.endswith("__"):
        core = name.strip("_")
        if core.startswith("r") and core[1:] in _REFLECTED:
            core = core[1:]
        elif core.startswith("i") and core[1:] in _REFLECTED:
            return _RENAMES.get(core[1:], core[1:]) + "_"
        return _RENAMES.get(core, core)
    return name


def base_op(name: str) -> str:
    """In-place variants share the category of their base op: ``add_`` → ``add``."""
    return name[:-1] if name.endswith("_") and not name.startswith("_") else name


def module_category(mro_names: Iterable[str]) -> C | None:
    for name in mro_names:
        if name in MODULE_CATEGORY:
            return MODULE_CATEGORY[name]
    return None


def module_param_names(mro_names: Iterable[str]) -> tuple[str, ...]:
    for name in mro_names:
        if name in MODULE_PARAMS:
            return MODULE_PARAMS[name]
    return ()


def function_category(name: str, tensor_inputs: int) -> tuple[C, bool]:
    """Category of a functional op and whether it is known to the catalogue."""
    op = base_op(name)
    if op in ARITHMETIC:
        return (C.MERGE if tensor_inputs >= 2 else C.OTHER), True
    if op in FUNCTION_CATEGORY:
        return FUNCTION_CATEGORY[op], True
    return C.OTHER, op in KNOWN_OTHER


def guess_group_category(type_name: str) -> C | None:
    """Category hint for a user-defined module from its class name (``ConvLSTMCell`` →
    recurrent). Only used when the module is drawn collapsed."""
    lower = type_name.lower()
    for keywords, category in _GROUP_KEYWORDS:
        if any(k in lower for k in keywords):
            return category
    return None
