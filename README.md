# SpikeCore Lab

**PyTorch → TVM BYOC → Neuromorphic Target — A complete compilation pipeline demo.**

## What is this?

SpikeCore Lab is a self-contained development environment that compiles a trained PyTorch neural network through TVM's BYOC (Bring Your Own Codegen) framework to a fictional neuromorphic hardware target called "SpikeCore." Every compilation stage — from floating-point model to integer-only neuromorphic assembly — is visible and inspectable in a Jupyter notebook.

This project bridges the gap between ML framework development and hardware-targeted compilation, demonstrating the exact workflow used to bring new accelerators (like Intel's Loihi) into a compiler ecosystem.

## SpikeCore Hardware Model

SpikeCore is a fictional neuromorphic chip designed to be simple enough to understand completely, yet realistic enough to demonstrate real compilation challenges:

| Property | Specification |
|---|---|
| Neuron cores | 128 |
| Weight memory | 256 bytes per core (int8) |
| Accumulator | int32 (32-bit signed) |
| Execution model | Event-driven (spike-based) |

**Instruction Set (3 primitives + 2 control):**

| Opcode | Operation | Description |
|---|---|---|
| `ACC` | Weighted accumulate | Dot product of input spikes × local weights → accumulator |
| `FIRE` | Threshold + emit spike | If accumulator ≥ threshold, right-shift by scale factor, emit spike, reset membrane |
| `LEAK` | Membrane decay | Multiply membrane potential by decay factor (fixed-point) |
| `NOP` | No operation | Pipeline bubble |
| `HALT` | Stop execution | End of program |

## Quick Start

```bash
# Build and run (TVM from source + PyTorch + Jupyter)
docker compose up

# Open notebook
# → http://localhost:8888/?token=spikecore
# → notebooks/01_spikecore_tvm.ipynb
# → Run All Cells
```

**Without Docker** (no TVM, uses standalone compilation path):
```bash
pip install torch torchvision numpy matplotlib jupyter pytest --extra-index-url https://download.pytorch.org/whl/cpu
cd neuromorphic-lab
jupyter notebook notebooks/01_spikecore_tvm.ipynb
```

**Run tests:**
```bash
# Inside Docker
docker compose run lab pytest tests/ -v

# Or locally
cd neuromorphic-lab
python -m pytest tests/ -v
```

## Project Structure

```
neuromorphic-lab/
├── README.md                           ← You are here
├── docs/plans/
│   └── 2026-02-17-spikecore-lab-design.md  ← Full design document
├── Dockerfile                          ← TVM from source + PyTorch + Jupyter
├── docker-compose.yml                  ← `docker compose up` → localhost:8888
├── requirements.txt                    ← Python deps pinned
├── notebooks/
│   └── 01_spikecore_tvm.ipynb          ← Main notebook (10 cells)
├── spikecore/                          ← Core library
│   ├── __init__.py                     ← Public API
│   ├── hardware_model.py               ← SpikeCoreCPU simulator (~150 lines)
│   ├── assembly.py                     ← ISA definition, assembler, disassembler
│   ├── byoc_codegen.py                 ← TVM BYOC target + standalone compiler
│   ├── relay_patterns.py               ← Relay pattern matching for BYOC
│   ├── quantize.py                     ← int8 quantization (TVM + manual paths)
│   └── visualize.py                    ← Spike rasters, graph viz, histograms
└── tests/
    ├── test_simulator.py               ← ACC/FIRE/LEAK unit tests
    ├── test_codegen.py                 ← Compilation & quantization tests
    └── test_roundtrip.py               ← PyTorch → SpikeCore accuracy test
```

## Notebook Walkthrough

| Cell | Stage | What it does |
|---|---|---|
| 1 | Setup | Import libraries, verify TVM installation |
| 2 | Model | Define & train 2-layer MLP on MNIST (784→64→10) |
| 3 | Export | Convert PyTorch model to TVM Relay IR |
| 4 | Quantize | float32 → int8 weights, int32 accumulators |
| 5 | Register | Define SpikeCore BYOC patterns + codegen callback |
| 6 | Partition | Split graph into host vs. SpikeCore subgraphs |
| 7 | Codegen | Emit SpikeCore assembly listing |
| 8 | Simulate | Run assembly on SpikeCore hardware model |
| 9 | Compare | PyTorch vs. SpikeCore predictions side-by-side |
| 10 | Visualize | Spike raster plots + weight distributions |

## How It Maps to Real Neuromorphic Compilation

| SpikeCore Concept | Real-World Equivalent |
|---|---|
| `ACC` instruction | Loihi dendritic accumulation / synaptic integration |
| `FIRE` instruction | Loihi axon compartment / spike generation |
| `LEAK` instruction | Loihi compartment leak current / membrane decay |
| BYOC pattern matching | Lava compiler's operator fusion & mapping |
| Graph partitioning | Lava's host ↔ neuromorphic chip splitting |
| int8 quantization | Loihi's native 1–9 bit weight precision |
| Core-local weight memory | Loihi's synapse memory per neurocore |
| Event-driven simulation | Loihi's asynchronous spike-based execution |

## Phase 2 Roadmap

- **MLIR extension**: torch-mlir → custom `spikecore` MLIR dialect → same simulator
- **Convolutional models**: Event-driven convolutions for vision tasks
- **Multi-chip simulation**: Spike routing between SpikeCore chips
- **Power/latency estimation**: Energy model based on spike counts and memory access patterns
- **Lava comparison**: Side-by-side with actual Intel Lava/Loihi workflow

## References

- [TVM BYOC Documentation](https://tvm.apache.org/docs/dev/how_to/relay_bring_your_own_codegen.html)
- [Intel Loihi 2 Architecture](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html)
- [Lava Software Framework](https://github.com/lava-nc/lava)
- [TVM Relay IR Specification](https://tvm.apache.org/docs/reference/langref/relay_expr.html)
- [Quantization in TVM](https://tvm.apache.org/docs/how_to/deploy_models/deploy_quantized.html)
- Davies et al., "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning" (IEEE Micro, 2018)
- Orchard et al., "Efficient Neuromorphic Signal Processing with Loihi 2" (IEEE, 2021)
