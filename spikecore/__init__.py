"""SpikeCore — A fictional neuromorphic hardware target for TVM BYOC demonstration."""

from .hardware_model import SpikeCoreCPU, NeuronCore, Instruction
from .assembly import assemble, disassemble, Opcode
from .byoc_codegen import register_spikecore_target
from .relay_patterns import spikecore_pattern_table
from .quantize import quantize_for_spikecore
from .visualize import plot_spike_raster, plot_compilation_graph, plot_weight_distribution
