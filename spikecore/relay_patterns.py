"""Relay pattern matching for SpikeCore-supported operations.

Defines composite patterns that map TVM Relay subgraphs to SpikeCore
primitives. The pattern table is consumed by relay.transform.MergeComposite.

Pattern mappings:
  qnn.dense + bias_add + clip(0,127)  →  ACC + FIRE (dense_fire)
  nn.dense + bias_add + relu           →  ACC + FIRE (dense_relu, float path)
  nn.dense + bias_add                   →  ACC       (dense_bias)
"""

from __future__ import annotations

try:
    import tvm
    from tvm import relay
    from tvm.relay.dataflow_pattern import (
        is_op,
        wildcard,
        is_constant,
        is_tuple_get_item,
    )

    HAS_TVM = True
except ImportError:
    HAS_TVM = False


def _make_dense_relu_pattern():
    """Match: nn.dense(x, w) → bias_add → relu  (float pre-quantization path)."""
    x = wildcard()
    w = is_constant()
    b = is_constant()
    dense = is_op("nn.dense")(x, w)
    biased = is_op("nn.bias_add")(dense, b)
    activated = is_op("nn.relu")(biased)
    return activated


def _make_dense_bias_pattern():
    """Match: nn.dense(x, w) → bias_add  (output layer, no activation)."""
    x = wildcard()
    w = is_constant()
    b = is_constant()
    dense = is_op("nn.dense")(x, w)
    biased = is_op("nn.bias_add")(dense, b)
    return biased


def _make_qnn_dense_clip_pattern():
    """Match: qnn.dense → bias_add → clip(0, 127)  (quantized path)."""
    x = wildcard()
    w = is_constant()
    izp = is_constant()  # input zero point
    wzp = is_constant()  # weight zero point
    iscale = is_constant()
    wscale = is_constant()
    dense = is_op("qnn.dense")(x, w, izp, wzp, iscale, wscale)
    b = is_constant()
    biased = is_op("add")(dense, b)
    clipped = is_op("clip")(biased)
    return clipped


def spikecore_pattern_table() -> list[tuple[str, "tvm.relay.dataflow_pattern.DFPattern"]]:
    """Return the SpikeCore composite pattern table for MergeComposite.

    Returns:
        List of (pattern_name, pattern) tuples.
    """
    if not HAS_TVM:
        raise RuntimeError("TVM is required for pattern table generation")

    return [
        ("spikecore.dense_relu", _make_dense_relu_pattern()),
        ("spikecore.dense_bias", _make_dense_bias_pattern()),
        ("spikecore.qnn_dense_clip", _make_qnn_dense_clip_pattern()),
    ]
