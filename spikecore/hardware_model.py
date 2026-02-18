"""SpikeCore hardware simulator.

Models a fictional neuromorphic chip with:
  - 128 neuron cores, each with 256 bytes of local weight memory
  - Integer-only arithmetic: int8 weights, int16 accumulator
  - Event-driven execution (spikes propagate between timesteps)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from .assembly import Instruction, Opcode

NUM_CORES = 128
WEIGHT_MEM_BYTES = 256  # per core


@dataclass
class NeuronCore:
    """A single neuron core with local state."""

    core_id: int
    accumulator: int = 0
    membrane_potential: int = 0
    weights: np.ndarray = field(default_factory=lambda: np.zeros(WEIGHT_MEM_BYTES, dtype=np.int8))
    has_spiked: bool = False

    def reset(self) -> None:
        self.accumulator = 0
        self.membrane_potential = 0
        self.has_spiked = False


class SpikeCoreCPU:
    """Event-driven simulator for the SpikeCore architecture."""

    def __init__(self, num_cores: int = NUM_CORES):
        self.num_cores = num_cores
        self.cores = [NeuronCore(core_id=i) for i in range(num_cores)]
        self.program: list[Instruction] = []
        self.spike_log: list[tuple[int, int]] = []  # (timestep, core_id)
        self._next_core_offset = 0

    def load_program(self, program: list[Instruction]) -> None:
        """Load an assembled program."""
        self.program = list(program)

    def load_weights(self, layer_id: int, weights_int8: np.ndarray,
                     core_offset: int | None = None) -> None:
        """Distribute int8 weight matrix across cores.

        Args:
            layer_id: Layer index (used in error messages).
            weights_int8: Shape (out_features, in_features), dtype int8.
            core_offset: Starting core index. If None, uses cumulative tracking.
        """
        out_features, in_features = weights_int8.shape
        if core_offset is None:
            core_offset = self._next_core_offset
        for i in range(out_features):
            core_idx = core_offset + i
            if core_idx >= self.num_cores:
                raise ValueError(
                    f"Layer {layer_id}: need core {core_idx} but only {self.num_cores} available"
                )
            n = min(in_features, WEIGHT_MEM_BYTES)
            self.cores[core_idx].weights[:n] = weights_int8[i, :n]
        self._next_core_offset = core_offset + out_features

    def _exec_acc(self, inst: Instruction, bus: np.ndarray) -> None:
        """ACC: weighted accumulate from activation bus (int32 arithmetic)."""
        core = self.cores[inst.core_id]
        bank, start, end = inst.operands
        # Clamp range to actual bus length
        actual_end = min(end, len(bus))
        if actual_end <= start:
            return
        n = actual_end - start
        spike_slice = bus[start:actual_end].astype(np.int32)
        weight_slice = core.weights[:n].astype(np.int32)
        acc_value = int(np.sum(spike_slice * weight_slice))
        core.accumulator += acc_value

    def _exec_fire(self, inst: Instruction, bus: np.ndarray) -> None:
        """FIRE: if accumulator >= threshold, emit spike and write to bus.

        Operands: (threshold, shift) where shift is right-shift bits applied
        to the activation before writing to the bus. This rescales the wide
        accumulator output to int8 range for the next layer.
        """
        core = self.cores[inst.core_id]
        threshold = inst.operands[0]
        shift = inst.operands[1] if len(inst.operands) > 1 else 0
        total = core.accumulator + core.membrane_potential
        if total >= threshold:
            core.has_spiked = True
            # Right-shift and clamp to int8 for downstream bus
            scaled = total >> shift if shift > 0 else total
            if inst.core_id < len(bus):
                bus[inst.core_id] = int(np.clip(scaled, -128, 127))
            core.membrane_potential = 0
            core.accumulator = 0
        else:
            core.membrane_potential = total
            core.accumulator = 0
            core.has_spiked = False

    def _exec_leak(self, inst: Instruction) -> None:
        """LEAK: decay membrane potential (fixed-point multiply by decay/256)."""
        core = self.cores[inst.core_id]
        (decay,) = inst.operands
        leaked = (core.membrane_potential * decay) >> 8
        core.membrane_potential = leaked

    def run(self, input_spikes: np.ndarray, timesteps: int = 1) -> np.ndarray:
        """Execute the loaded program for the given number of timesteps.

        Uses an activation bus for inter-layer communication with layer-aware
        snapshotting: within a layer, all ACC instructions read from the same
        snapshot (simulating parallel execution). When ACC input range changes
        (new layer), the snapshot refreshes to include prior FIRE outputs.

        Args:
            input_spikes: int8 array representing input activations/spikes.
            timesteps: Number of simulation timesteps.

        Returns:
            Array of spike counts per core over all timesteps.
        """
        self.spike_log.clear()
        spike_counts = np.zeros(self.num_cores, dtype=np.int32)

        # Reset all cores
        for core in self.cores:
            core.reset()

        # Activation bus: sized to hold both input and core outputs
        bus_size = max(self.num_cores, len(input_spikes))
        bus = np.zeros(bus_size, dtype=np.int16)
        bus[:len(input_spikes)] = input_spikes.astype(np.int16)

        for t in range(timesteps):
            # Re-inject original input at the start of each timestep
            if t > 0:
                bus[:] = 0
                bus[:len(input_spikes)] = input_spikes.astype(np.int16)

            # Layer-aware execution: snapshot bus at layer boundaries
            # so same-layer cores read consistent input (parallel semantics)
            last_acc_range = None
            snapshot = bus.copy()

            for inst in self.program:
                if inst.opcode == Opcode.ACC:
                    acc_range = (inst.operands[1], inst.operands[2])
                    if acc_range != last_acc_range:
                        # New layer — refresh snapshot from live bus
                        snapshot = bus.copy()
                        last_acc_range = acc_range
                    self._exec_acc(inst, snapshot)
                elif inst.opcode == Opcode.FIRE:
                    self._exec_fire(inst, bus)
                elif inst.opcode == Opcode.LEAK:
                    self._exec_leak(inst)
                elif inst.opcode == Opcode.HALT:
                    break

            # Collect spikes from this timestep
            for core in self.cores:
                if core.has_spiked:
                    self.spike_log.append((t, core.core_id))
                    spike_counts[core.core_id] += 1
                    core.has_spiked = False

        return spike_counts

    def get_spike_log(self) -> list[tuple[int, int]]:
        """Return spike history as list of (timestep, core_id) tuples."""
        return list(self.spike_log)

    def get_output_activations(self, output_core_ids: list[int]) -> np.ndarray:
        """Read final accumulator values from designated output cores.

        For single-timestep rate-coded inference, the accumulator values
        (before FIRE) represent the layer's output activations.
        """
        return np.array(
            [int(self.cores[c].accumulator) + int(self.cores[c].membrane_potential)
             for c in output_core_ids],
            dtype=np.int32,
        )
