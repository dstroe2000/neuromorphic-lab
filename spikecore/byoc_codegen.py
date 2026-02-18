"""TVM BYOC (Bring Your Own Codegen) target registration for SpikeCore.

This module registers "spikecore" as a BYOC target in TVM's compilation
pipeline. The flow:

  1. MergeComposite — fuses Relay ops into composite patterns
  2. AnnotateTarget  — marks composites for offloading to "spikecore"
  3. PartitionGraph  — splits the Relay graph into host / spikecore subgraphs
  4. Codegen callback — traverses partitioned subgraphs, emits SpikeCore assembly

For environments without TVM, a standalone `compile_from_torch` function
performs the equivalent pipeline using manual quantization and direct
SpikeCore code generation.
"""

from __future__ import annotations

import math

import numpy as np

from .assembly import Instruction, Opcode, disassemble
from .quantize import manual_quantize_weights


def _compute_fire_shift(fan_in: int) -> int:
    """Compute right-shift for FIRE to rescale accumulator to int8 range.

    The ACC accumulator holds sum(int8 * int8) over fan_in inputs.
    Max value ≈ fan_in * 127 * 127. We need to shift right so the
    output fits in [-128, 127] for the downstream bus.

    shift = ceil(log2(fan_in)) + 7
    """
    if fan_in <= 1:
        return 7  # just the 127*127 → 127 scaling
    return math.ceil(math.log2(fan_in)) + 7

try:
    import tvm
    from tvm import relay
    from tvm.relay.build_module import bind_params_by_name
    from .relay_patterns import spikecore_pattern_table

    HAS_TVM = True
except ImportError:
    HAS_TVM = False


# ---------------------------------------------------------------------------
# TVM BYOC path
# ---------------------------------------------------------------------------

def _spikecore_codegen_callback(func: "relay.Function") -> str:
    """BYOC codegen callback: Relay subgraph → SpikeCore assembly text.

    Called by TVM for each partitioned subgraph annotated as "spikecore".
    Traverses the function body and emits ACC/FIRE/LEAK instructions.
    """
    instructions: list[str] = []
    core_counter = [0]  # mutable counter for closure

    def _visit(expr):
        if isinstance(expr, relay.Call):
            if isinstance(expr.op, relay.Function):
                # Composite function — check pattern name
                pattern = expr.op.attrs.get("Composite", "")
                _emit_composite(pattern, expr, instructions, core_counter)
            else:
                for arg in expr.args:
                    _visit(arg)
        elif isinstance(expr, relay.Tuple):
            for f in expr.fields:
                _visit(f)
        elif isinstance(expr, relay.TupleGetItem):
            _visit(expr.tuple_value)

    def _emit_composite(pattern, call, insts, counter):
        core_id = counter[0]
        if "dense_relu" in pattern or "dense_clip" in pattern or "qnn_dense_clip" in pattern:
            # Dense + activation → ACC + FIRE
            insts.append(f"ACC  core_{core_id}  weight_bank_0  spike_in_[0:64]")
            insts.append(f"FIRE core_{core_id}  threshold_64")
            insts.append(f"LEAK core_{core_id}  decay_240")
        elif "dense_bias" in pattern:
            # Dense without activation → ACC only (output layer)
            insts.append(f"ACC  core_{core_id}  weight_bank_0  spike_in_[0:64]")
        counter[0] += 1

    _visit(func.body)
    instructions.append("HALT")
    return "\n".join(instructions)


def register_spikecore_target() -> None:
    """Register 'spikecore' as a BYOC target in TVM.

    After calling this, relay.transform.AnnotateTarget(["spikecore"])
    will recognize SpikeCore-compatible subgraphs.
    """
    if not HAS_TVM:
        raise RuntimeError("TVM is required for BYOC target registration")

    @tvm.register_func("relay.ext.spikecore")
    def _spikecore_compiler(func):
        """External compiler entry point for BYOC."""
        asm_text = _spikecore_codegen_callback(func)
        # Return assembly as a runtime module (string representation)
        # In a real target, this would emit binary; here we return text
        return asm_text

    @tvm.register_func("relay.ext.spikecore.cost_estimator")
    def _cost_estimator(func):
        return 1  # Always prefer offloading to SpikeCore


def partition_for_spikecore(
    mod: "tvm.IRModule",
    params: dict[str, "tvm.nd.NDArray"],
) -> "tvm.IRModule":
    """Run the full BYOC partitioning pipeline for SpikeCore.

    Steps: MergeComposite → AnnotateTarget → PartitionGraph

    Args:
        mod: Relay module.
        params: Bound parameters.

    Returns:
        Partitioned Relay module with SpikeCore subgraphs marked.
    """
    if not HAS_TVM:
        raise RuntimeError("TVM is required for graph partitioning")

    mod["main"] = bind_params_by_name(mod["main"], params)

    patterns = spikecore_pattern_table()

    seq = tvm.transform.Sequential([
        relay.transform.MergeComposite(patterns),
        relay.transform.AnnotateTarget(["spikecore"]),
        relay.transform.PartitionGraph(),
    ])

    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    return mod


# ---------------------------------------------------------------------------
# Standalone path (no TVM required)
# ---------------------------------------------------------------------------

def compile_from_torch(
    state_dict: dict[str, np.ndarray],
    layer_configs: list[dict],
) -> tuple[list[Instruction], list[tuple[np.ndarray, float]]]:
    """Compile PyTorch model weights directly to SpikeCore assembly.

    This is the non-TVM path for environments where TVM isn't available.
    It manually quantizes weights and generates SpikeCore instructions.

    Args:
        state_dict: Model weights as numpy arrays.
            Expected keys: "layers.0.weight", "layers.0.bias", etc.
        layer_configs: Per-layer configuration dicts with keys:
            - name: layer name prefix
            - in_features: input size
            - out_features: output size
            - activation: "relu" or None

    Returns:
        (program, quantized_weights) where quantized_weights is a list
        of (int8_weights, scale) per layer.
    """
    program: list[Instruction] = []
    quantized_weights: list[tuple[np.ndarray, float]] = []
    core_offset = 0

    for layer in layer_configs:
        name = layer["name"]
        in_feat = layer["in_features"]
        out_feat = layer["out_features"]
        activation = layer.get("activation", None)

        # Quantize weights
        w_key = f"{name}.weight"
        w_fp32 = state_dict[w_key]
        w_int8, scale, _ = manual_quantize_weights(w_fp32)
        quantized_weights.append((w_int8, scale))

        # Compute FIRE shift based on fan-in
        shift = _compute_fire_shift(in_feat)

        # Generate instructions for each output neuron
        for i in range(out_feat):
            core_id = core_offset + i
            # ACC: accumulate weighted inputs
            program.append(Instruction(
                Opcode.ACC, core_id, (0, 0, in_feat)
            ))
            # FIRE: threshold activation (for hidden layers with ReLU)
            if activation == "relu":
                program.append(Instruction(
                    Opcode.FIRE, core_id, (0, shift)
                ))
                program.append(Instruction(
                    Opcode.LEAK, core_id, (240,)
                ))

        core_offset += out_feat

    program.append(Instruction(Opcode.HALT))
    return program, quantized_weights


def compile_nn_to_spikecore(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    activations: list[str | None],
) -> tuple[list[Instruction], list[tuple[np.ndarray, float]], list[tuple[np.ndarray, float]]]:
    """Higher-level compilation: takes weight/bias arrays directly.

    Args:
        weights: List of float32 weight matrices, one per layer.
        biases: List of float32 bias vectors, one per layer.
        activations: List of activation types ("relu" or None) per layer.

    Returns:
        (program, quantized_weights, quantized_biases)
    """
    program: list[Instruction] = []
    q_weights: list[tuple[np.ndarray, float]] = []
    q_biases: list[tuple[np.ndarray, float]] = []
    core_offset = 0

    for layer_idx, (w, b, act) in enumerate(zip(weights, biases, activations)):
        out_feat, in_feat = w.shape

        # Quantize
        w_int8, w_scale, _ = manual_quantize_weights(w)
        b_int8, b_scale, _ = manual_quantize_weights(b)
        q_weights.append((w_int8, w_scale))
        q_biases.append((b_int8, b_scale))

        shift = _compute_fire_shift(in_feat)

        for i in range(out_feat):
            core_id = core_offset + i
            program.append(Instruction(Opcode.ACC, core_id, (0, 0, in_feat)))
            if act == "relu":
                program.append(Instruction(Opcode.FIRE, core_id, (0, shift)))
                program.append(Instruction(Opcode.LEAK, core_id, (240,)))

        core_offset += out_feat

    program.append(Instruction(Opcode.HALT))
    return program, q_weights, q_biases
