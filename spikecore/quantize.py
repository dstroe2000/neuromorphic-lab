"""Quantization configuration for SpikeCore int8 targeting.

Wraps TVM's quantization passes to produce int8 weights and activations
compatible with SpikeCore's integer-only datapath (int8 weights, int16
accumulator, threshold-based activation).
"""

from __future__ import annotations

import numpy as np

try:
    import tvm
    from tvm import relay
    from tvm.relay.quantize import quantize as tvm_quantize

    HAS_TVM = True
except ImportError:
    HAS_TVM = False


def quantize_for_spikecore(
    mod: "tvm.IRModule",
    params: dict[str, "tvm.nd.NDArray"],
    calibration_dataset: np.ndarray | None = None,
) -> tuple["tvm.IRModule", dict[str, "tvm.nd.NDArray"]]:
    """Apply int8 quantization tuned for SpikeCore hardware.

    Args:
        mod: Relay module (float32).
        params: Model parameters.
        calibration_dataset: Optional calibration data for scale estimation.

    Returns:
        Quantized (mod, params) tuple.
    """
    if not HAS_TVM:
        raise RuntimeError("TVM is required for quantization")

    with tvm.transform.PassContext(opt_level=3):
        qconfig = relay.quantize.qconfig(
            nbit_input=8,
            nbit_weight=8,
            nbit_activation=16,  # SpikeCore int16 accumulator
            dtype_input="int8",
            dtype_weight="int8",
            dtype_activation="int16",
            calibrate_mode="global_scale",
            global_scale=8.0,
        )
        with qconfig:
            qmod = relay.quantize.quantize(mod, params)
    return qmod, params


def manual_quantize_weights(
    weights_fp32: np.ndarray,
    bits: int = 8,
) -> tuple[np.ndarray, float, int]:
    """Manually quantize float32 weights to int8 (for non-TVM path).

    Uses symmetric quantization: q = round(w / scale), scale = max(|w|) / 127.

    Args:
        weights_fp32: Float32 weight tensor.
        bits: Quantization bit width (default 8).

    Returns:
        (weights_int8, scale, zero_point) tuple.
    """
    qmax = (1 << (bits - 1)) - 1  # 127 for 8-bit
    abs_max = np.max(np.abs(weights_fp32))
    if abs_max < 1e-10:
        return np.zeros_like(weights_fp32, dtype=np.int8), 1.0, 0

    scale = abs_max / qmax
    quantized = np.clip(np.round(weights_fp32 / scale), -qmax - 1, qmax)
    return quantized.astype(np.int8), float(scale), 0


def manual_quantize_activations(
    activations_fp32: np.ndarray,
    bits: int = 8,
) -> tuple[np.ndarray, float, int]:
    """Quantize activations (typically after ReLU, so unsigned).

    Uses asymmetric quantization for ReLU outputs (min=0).

    Args:
        activations_fp32: Float32 activation tensor (assumed non-negative).
        bits: Quantization bit width.

    Returns:
        (activations_int8, scale, zero_point) tuple.
    """
    qmax = (1 << bits) - 1  # 255 for 8-bit unsigned
    act_max = np.max(activations_fp32)
    if act_max < 1e-10:
        return np.zeros_like(activations_fp32, dtype=np.int8), 1.0, 0

    scale = act_max / qmax
    quantized = np.clip(np.round(activations_fp32 / scale), 0, qmax)
    # Store as int8 (values 0-127 fit; 128-255 wraps but SpikeCore handles unsigned)
    return quantized.astype(np.int8), float(scale), 0
