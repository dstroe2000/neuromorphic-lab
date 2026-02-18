"""Unit tests for SpikeCore hardware simulator and assembly."""

import numpy as np
import pytest

from spikecore.assembly import Instruction, Opcode, assemble, disassemble
from spikecore.hardware_model import SpikeCoreCPU, NeuronCore, NUM_CORES


# ---- Assembly tests ----

class TestAssembly:
    def test_assemble_acc(self):
        prog = assemble("ACC core_3 weight_bank_0 spike_in_[0:8]")
        assert len(prog) == 1
        assert prog[0].opcode == Opcode.ACC
        assert prog[0].core_id == 3
        assert prog[0].operands == (0, 0, 8)

    def test_assemble_fire(self):
        prog = assemble("FIRE core_5 threshold_64")
        assert len(prog) == 1
        assert prog[0].opcode == Opcode.FIRE
        assert prog[0].core_id == 5
        assert prog[0].operands == (64,)

    def test_assemble_leak(self):
        prog = assemble("LEAK core_0 decay_240")
        assert len(prog) == 1
        assert prog[0].opcode == Opcode.LEAK
        assert prog[0].core_id == 0
        assert prog[0].operands == (240,)

    def test_assemble_nop_halt(self):
        prog = assemble("NOP\nHALT")
        assert len(prog) == 2
        assert prog[0].opcode == Opcode.NOP
        assert prog[1].opcode == Opcode.HALT

    def test_assemble_ignores_comments(self):
        text = "# This is a comment\nACC core_0 weight_bank_0 spike_in_[0:4]\n// Another comment"
        prog = assemble(text)
        assert len(prog) == 1

    def test_assemble_with_address_prefix(self):
        text = "0000: ACC core_0 weight_bank_0 spike_in_[0:4]\n0001: HALT"
        prog = assemble(text)
        assert len(prog) == 2
        assert prog[0].opcode == Opcode.ACC
        assert prog[1].opcode == Opcode.HALT

    def test_roundtrip_disassemble_assemble(self):
        original = [
            Instruction(Opcode.ACC, 3, (0, 0, 8)),
            Instruction(Opcode.FIRE, 3, (64,)),
            Instruction(Opcode.LEAK, 3, (240,)),
            Instruction(Opcode.HALT),
        ]
        text = disassemble(original)
        reassembled = assemble(text)
        assert len(reassembled) == len(original)
        for a, b in zip(original, reassembled):
            assert a.opcode == b.opcode
            assert a.core_id == b.core_id
            assert a.operands == b.operands

    def test_bad_instruction_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            assemble("BADOP core_0")


# ---- Simulator tests ----

class TestNeuronCore:
    def test_initial_state(self):
        core = NeuronCore(core_id=0)
        assert core.accumulator == 0
        assert core.membrane_potential == 0
        assert core.has_spiked is False

    def test_reset(self):
        core = NeuronCore(core_id=0)
        core.accumulator = np.int16(100)
        core.membrane_potential = np.int16(50)
        core.has_spiked = True
        core.reset()
        assert core.accumulator == 0
        assert core.membrane_potential == 0
        assert core.has_spiked is False


class TestSpikeCoreCPU:
    def test_load_program(self):
        cpu = SpikeCoreCPU(num_cores=4)
        prog = [Instruction(Opcode.ACC, 0, (0, 0, 4)), Instruction(Opcode.HALT)]
        cpu.load_program(prog)
        assert len(cpu.program) == 2

    def test_load_weights(self):
        cpu = SpikeCoreCPU(num_cores=4)
        w = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int8)
        cpu.load_weights(layer_id=0, weights_int8=w)
        np.testing.assert_array_equal(cpu.cores[0].weights[:4], [1, 2, 3, 4])
        np.testing.assert_array_equal(cpu.cores[1].weights[:4], [5, 6, 7, 8])

    def test_acc_instruction(self):
        """Test that ACC accumulates weighted input correctly."""
        cpu = SpikeCoreCPU(num_cores=4)
        # Set weights for core 0
        cpu.cores[0].weights[:4] = np.array([2, 3, 1, -1], dtype=np.int8)

        prog = [
            Instruction(Opcode.ACC, 0, (0, 0, 4)),
            Instruction(Opcode.HALT),
        ]
        cpu.load_program(prog)

        # Input: [1, 1, 1, 1] → dot product = 2+3+1-1 = 5
        input_spikes = np.array([1, 1, 1, 1], dtype=np.int8)
        cpu.run(input_spikes, timesteps=1)
        # After ACC but no FIRE, accumulator holds the result
        # (membrane potential absorbs it via FIRE's else branch only if FIRE runs)
        # Since we only run ACC then HALT, accumulator should be 5
        assert cpu.cores[0].accumulator == 5

    def test_fire_with_spike(self):
        """Test FIRE emits spike when threshold is exceeded."""
        cpu = SpikeCoreCPU(num_cores=4)
        cpu.cores[0].weights[:2] = np.array([50, 50], dtype=np.int8)

        prog = [
            Instruction(Opcode.ACC, 0, (0, 0, 2)),
            Instruction(Opcode.FIRE, 0, (64,)),
            Instruction(Opcode.HALT),
        ]
        cpu.load_program(prog)

        # Input [1, 1] → acc = 100 >= 64 → spike
        input_spikes = np.array([1, 1], dtype=np.int8)
        counts = cpu.run(input_spikes, timesteps=1)
        assert counts[0] == 1
        assert len(cpu.get_spike_log()) == 1
        assert cpu.get_spike_log()[0] == (0, 0)

    def test_fire_no_spike(self):
        """Test FIRE does not emit when below threshold."""
        cpu = SpikeCoreCPU(num_cores=4)
        cpu.cores[0].weights[:2] = np.array([10, 10], dtype=np.int8)

        prog = [
            Instruction(Opcode.ACC, 0, (0, 0, 2)),
            Instruction(Opcode.FIRE, 0, (64,)),
            Instruction(Opcode.HALT),
        ]
        cpu.load_program(prog)

        # Input [1, 1] → acc = 20 < 64 → no spike
        input_spikes = np.array([1, 1], dtype=np.int8)
        counts = cpu.run(input_spikes, timesteps=1)
        assert counts[0] == 0
        assert len(cpu.get_spike_log()) == 0

    def test_leak_decays_membrane(self):
        """Test LEAK decays membrane potential (direct method call)."""
        cpu = SpikeCoreCPU(num_cores=4)
        cpu.cores[0].membrane_potential = 256

        inst = Instruction(Opcode.LEAK, 0, (128,))  # decay = 128/256 = 0.5
        cpu._exec_leak(inst)
        # 256 * 128 / 256 = 128
        assert cpu.cores[0].membrane_potential == 128

    def test_leak_in_program(self):
        """Test LEAK works within a full program (ACC → FIRE miss → LEAK)."""
        cpu = SpikeCoreCPU(num_cores=4)
        # Set weights so ACC produces an accumulation below threshold
        cpu.cores[0].weights[:2] = np.array([30, 30], dtype=np.int8)

        prog = [
            # ACC produces 60, FIRE threshold 100 → no spike, membrane = 60
            Instruction(Opcode.ACC, 0, (0, 0, 2)),
            Instruction(Opcode.FIRE, 0, (100,)),
            # LEAK with decay 128 → 60 * 128 / 256 = 30
            Instruction(Opcode.LEAK, 0, (128,)),
            Instruction(Opcode.HALT),
        ]
        cpu.load_program(prog)

        input_spikes = np.array([1, 1, 0, 0], dtype=np.int8)
        cpu.run(input_spikes, timesteps=1)
        assert cpu.cores[0].membrane_potential == 30

    def test_two_neuron_network(self):
        """Integration test: hand-assembled 2-neuron network.

        Neuron 0: ACC from inputs → FIRE (threshold 30)
        Neuron 1: ACC from inputs → FIRE (threshold 30)
        """
        cpu = SpikeCoreCPU(num_cores=4)

        # Both neurons have strong positive weights
        cpu.cores[0].weights[:3] = np.array([20, 20, 20], dtype=np.int8)
        cpu.cores[1].weights[:3] = np.array([10, 10, 5], dtype=np.int8)

        prog = [
            Instruction(Opcode.ACC, 0, (0, 0, 3)),
            Instruction(Opcode.FIRE, 0, (30,)),
            Instruction(Opcode.ACC, 1, (0, 0, 3)),
            Instruction(Opcode.FIRE, 1, (30,)),
            Instruction(Opcode.HALT),
        ]
        cpu.load_program(prog)

        # Input [1, 1, 1]: core0 = 60 >= 30 → spike, core1 = 25 < 30 → no spike
        input_spikes = np.array([1, 1, 1], dtype=np.int8)
        counts = cpu.run(input_spikes, timesteps=1)
        assert counts[0] == 1  # core 0 fires
        assert counts[1] == 0  # core 1 doesn't

    def test_get_output_activations(self):
        """Test reading accumulator values from output cores."""
        cpu = SpikeCoreCPU(num_cores=4)
        cpu.cores[0].weights[:2] = np.array([30, 40], dtype=np.int8)
        cpu.cores[1].weights[:2] = np.array([10, 20], dtype=np.int8)

        prog = [
            Instruction(Opcode.ACC, 0, (0, 0, 2)),
            Instruction(Opcode.ACC, 1, (0, 0, 2)),
            Instruction(Opcode.HALT),
        ]
        cpu.load_program(prog)

        input_spikes = np.array([1, 1], dtype=np.int8)
        cpu.run(input_spikes, timesteps=1)
        outputs = cpu.get_output_activations([0, 1])
        assert outputs[0] == 70  # 30 + 40
        assert outputs[1] == 30  # 10 + 20
