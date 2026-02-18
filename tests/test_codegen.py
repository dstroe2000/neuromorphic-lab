"""Tests for SpikeCore BYOC code generation (standalone path)."""

import numpy as np
import pytest

from spikecore.assembly import Opcode, disassemble
from spikecore.byoc_codegen import compile_from_torch, compile_nn_to_spikecore
from spikecore.quantize import manual_quantize_weights


class TestManualQuantization:
    def test_symmetric_quantize(self):
        w = np.array([[1.0, -0.5], [0.25, -1.0]], dtype=np.float32)
        q, scale, zp = manual_quantize_weights(w)
        assert q.dtype == np.int8
        assert zp == 0
        # Largest magnitude is 1.0, scale = 1.0 / 127
        assert abs(scale - 1.0 / 127) < 1e-5
        # 1.0 / scale = 127
        assert q[0, 0] == 127
        assert q[1, 1] == -127

    def test_zero_weights(self):
        w = np.zeros((3, 3), dtype=np.float32)
        q, scale, zp = manual_quantize_weights(w)
        assert np.all(q == 0)
        assert scale == 1.0


class TestCompileFromTorch:
    def test_single_layer_with_relu(self):
        state_dict = {
            "layers.0.weight": np.random.randn(10, 4).astype(np.float32),
            "layers.0.bias": np.zeros(10, dtype=np.float32),
        }
        layer_configs = [
            {"name": "layers.0", "in_features": 4, "out_features": 10, "activation": "relu"},
        ]
        program, q_weights = compile_from_torch(state_dict, layer_configs)

        # Should have: 10 * (ACC + FIRE + LEAK) + HALT = 31 instructions
        assert program[-1].opcode == Opcode.HALT
        acc_count = sum(1 for i in program if i.opcode == Opcode.ACC)
        fire_count = sum(1 for i in program if i.opcode == Opcode.FIRE)
        assert acc_count == 10
        assert fire_count == 10

    def test_output_layer_no_activation(self):
        state_dict = {
            "layers.0.weight": np.random.randn(5, 3).astype(np.float32),
            "layers.0.bias": np.zeros(5, dtype=np.float32),
        }
        layer_configs = [
            {"name": "layers.0", "in_features": 3, "out_features": 5, "activation": None},
        ]
        program, q_weights = compile_from_torch(state_dict, layer_configs)

        # No activation → ACC only, no FIRE/LEAK
        fire_count = sum(1 for i in program if i.opcode == Opcode.FIRE)
        assert fire_count == 0
        acc_count = sum(1 for i in program if i.opcode == Opcode.ACC)
        assert acc_count == 5

    def test_two_layer_mlp(self):
        state_dict = {
            "layers.0.weight": np.random.randn(8, 4).astype(np.float32),
            "layers.0.bias": np.zeros(8, dtype=np.float32),
            "layers.1.weight": np.random.randn(3, 8).astype(np.float32),
            "layers.1.bias": np.zeros(3, dtype=np.float32),
        }
        layer_configs = [
            {"name": "layers.0", "in_features": 4, "out_features": 8, "activation": "relu"},
            {"name": "layers.1", "in_features": 8, "out_features": 3, "activation": None},
        ]
        program, q_weights = compile_from_torch(state_dict, layer_configs)

        # Layer 0: 8 * (ACC+FIRE+LEAK) = 24; Layer 1: 3 * ACC = 3; + HALT = 28
        assert program[-1].opcode == Opcode.HALT
        assert len(q_weights) == 2

        # Verify core IDs are sequential
        acc_cores = [i.core_id for i in program if i.opcode == Opcode.ACC]
        assert acc_cores == list(range(8)) + list(range(8, 11))

    def test_disassemble_output_readable(self):
        state_dict = {
            "layers.0.weight": np.eye(3, dtype=np.float32),
            "layers.0.bias": np.zeros(3, dtype=np.float32),
        }
        layer_configs = [
            {"name": "layers.0", "in_features": 3, "out_features": 3, "activation": "relu"},
        ]
        program, _ = compile_from_torch(state_dict, layer_configs)
        text = disassemble(program)
        assert "ACC" in text
        assert "FIRE" in text
        assert "HALT" in text


class TestCompileNNToSpikecore:
    def test_mnist_mlp_shape(self):
        """Test compilation of an MNIST-sized MLP (784→64→10)."""
        weights = [
            np.random.randn(64, 784).astype(np.float32) * 0.01,
            np.random.randn(10, 64).astype(np.float32) * 0.01,
        ]
        biases = [
            np.zeros(64, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
        ]
        activations = ["relu", None]

        program, q_w, q_b = compile_nn_to_spikecore(weights, biases, activations)

        # 64 * 3 (ACC+FIRE+LEAK) + 10 * 1 (ACC) + 1 (HALT) = 203
        assert program[-1].opcode == Opcode.HALT
        assert len(q_w) == 2
        assert q_w[0][0].shape == (64, 784)
        assert q_w[1][0].shape == (10, 64)
